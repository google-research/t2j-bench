# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logit transforms and greedy selection, mirroring torch_orig/sample/sampler.py.

DELIBERATELY NO RNG DRAW. Torch's Philox and JAX's threefry are different PRNG
families, so a sampled token is not cross-framework comparable -- the same trap
TEST_PLAN §2.5 documents for PPO rollouts. What IS comparable, and what this
class grades, is the deterministic transform chain (temperature -> top-k ->
top-p) and the resulting log-probabilities. A candidate is free to draw from
the processed distribution; nothing compares the draw.

Order matters and is vLLM's: temperature, then top-k, then top-p. Applying
top-p before top-k would change which tokens survive.

Four op-level substitutions, and nothing else changes:

  * `masked_fill(cond, -inf)`  ->  `jnp.where(cond, -inf, x)`.
  * `logits.topk(k).values`    ->  `jax.lax.top_k(logits, k)[0]`.
  * `logits.sort(descending=True)` -> an ASCENDING argsort of `-logits`. Sorting
    the negation rather than reversing an ascending sort is what preserves
    torch's tie order: `jnp.argsort` is stable, so equal logits keep ascending
    index order, and `[::-1]` on an ascending sort would reverse that. The
    masked entries end up last either way, because -(-inf) is +inf.
  * `drop.scatter(-1, order, drop)` -> `take_along_axis(drop, argsort(order))`.
    `order` is a permutation, so its argsort is its inverse and the gather is
    exactly the scatter. Written this way because jnp has no in-place scatter
    along an axis and `.at[].set()` with an index array would need the batch
    dimension spelled out by hand.

ONE ACCUMULATOR CONVENTION IS RESTATED RATHER THAN SUBSTITUTED, in `apply_top_p`
and already in `topn_logprobs`: torch's reductions accumulate in fp32 and store
in the input dtype, and JAX's do neither, so the mirror has to say so out loud.
It is invisible at fp32 and it is a surviving-token-set disagreement at bf16 --
see the measurement on that line.

`float("-inf")` is written as a Python float on purpose: it is weakly typed, so
`jnp.where` keeps the logits' dtype instead of promoting a bf16 row to fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0        # 0 disables
    top_p: float = 1.0    # 1.0 disables

    def __post_init__(self) -> None:
        # Ranges mirror vLLM's SamplingParams._verify_args. top_p == 0.0 is the
        # one that matters here and it is rejected, not clamped: the inclusive
        # prefix in apply_top_p drops token i when the mass BEFORE it reaches
        # p, and at p = 0.0 that is true of the argmax too, so every entry
        # becomes -inf and `greedy` then returns index 0 -- a wrong token that
        # reads exactly like a real one. Nothing downstream could tell.
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables), got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be in (0, 1], got {self.top_p}. At top_p <= 0 the "
                "inclusive-prefix rule masks every token, including the argmax."
            )


def apply_temperature(logits: jax.Array, temperature: float) -> jax.Array:
    # 0.0 means greedy; scaling by it would divide by zero, and the argmax is
    # invariant to any positive scale, so pass through.
    if temperature in (0.0, 1.0):
        return logits
    return logits / temperature


def apply_top_k(logits: jax.Array, k: int) -> jax.Array:
    """Mask everything below the k-th largest logit.

    This is a THRESHOLD mask, exactly as vLLM does it, not a "keep k entries"
    mask: entries equal to the k-th largest are kept, so a tie straddling the
    cut leaves MORE than k finite entries. That is a real, reachable case at
    low precision and this side has to reproduce it, so it is pinned in
    test_top_k_keeps_every_entry_tied_at_the_threshold rather than left to
    whichever behaviour a port happens to land on. `jax.lax.top_k` returns
    exactly k entries, so implementing this by scattering its INDICES would
    quietly drop the ties -- the threshold is taken from its VALUES instead.
    """
    vocab = logits.shape[-1]
    if k <= 0 or k >= vocab:
        return logits
    kth = jax.lax.top_k(logits, k)[0][..., -1:]
    return jnp.where(logits < kth, _NEG_INF, logits)


def apply_top_p(logits: jax.Array, p: float) -> jax.Array:
    """Mask the tail beyond the smallest prefix whose mass reaches p.

    For any p > 0 the argmax always survives, however peaked the distribution.
    At p <= 0 it would not -- see SamplingParams.__post_init__, which rejects
    that rather than let it through.
    """
    if p >= 1.0:
        return logits
    order = jnp.argsort(-logits, axis=-1)
    ordered = jnp.take_along_axis(logits, order, axis=-1)
    # ACCUMULATE IN FP32, STORE IN THE LOGITS' DTYPE -- which is what torch's
    # `Tensor.softmax` and `Tensor.cumsum` do on a reduced-precision tensor, and
    # what `jax.nn.softmax` / `jnp.cumsum` do NOT. Both halves are needed and
    # both are measured:
    #   * the accumulator. `jax.nn.softmax(ordered)` normalises in the INPUT
    #     dtype, so over a 128 256-wide bf16 row the normaliser itself is wrong:
    #     MEASURED on the frozen row, the all-bf16 probabilities sum to 1.001559
    #     against float64's 1.000000, and that missing 1.6e-03 of mass is exactly
    #     what moves a `top_p` cut. Torch's accumulator is fp32 -- a bf16
    #     `cumsum` of 100 000 copies of 1e-4 returns exactly 10.0 -- and its
    #     probabilities sum to 1.000310 and sit 2.96e-04 from float64 truth,
    #     which is what the form below reproduces to the digit.
    #   * the storage. Torch's outputs are bf16 elements of an fp32 sum, so
    #     keeping fp32 here is not "more accurate agreement", it is a different
    #     cut point: it left one surviving-set disagreement at `top_p 0.9`.
    # MEASURED end to end through tests_parity, on instance.yaml's six-row grid
    # at the native dtype: 8 mask disagreements before this line, 1 with the
    # upcast alone, 0 with both halves. The fp32 tier is bit-unchanged either way
    # -- every cast below is a no-op on an fp32 row -- so no frozen golden moves,
    # and `topn_logprobs` below already carries the same upcast on both sides.
    probs = jax.nn.softmax(ordered.astype(jnp.float32),
                           axis=-1).astype(logits.dtype)
    cumulative = jnp.cumsum(probs.astype(jnp.float32),
                            axis=-1).astype(logits.dtype)
    # Drop entries whose PRECEDING cumulative mass already reached p. Shifting
    # by one is what makes the prefix inclusive and keeps the argmax.
    drop = cumulative - probs >= p
    remove = jnp.take_along_axis(drop, jnp.argsort(order, axis=-1), axis=-1)
    return jnp.where(remove, _NEG_INF, logits)


def process_logits(logits: jax.Array, params: SamplingParams) -> jax.Array:
    logits = apply_temperature(logits, params.temperature)
    logits = apply_top_k(logits, params.top_k)
    return apply_top_p(logits, params.top_p)


def greedy(logits: jax.Array) -> jax.Array:
    return jnp.argmax(logits, axis=-1)


def topn_logprobs(logits: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Top-n log-probabilities and their token ids, descending."""
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    values, indices = jax.lax.top_k(logprobs, n)
    return values, indices
