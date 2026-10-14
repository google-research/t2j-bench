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

"""Logit transforms and greedy selection, de-vendored from vllm/v1/sample/.

DELIBERATELY NO RNG DRAW. Torch's Philox and JAX's threefry are different PRNG
families, so a sampled token is not cross-framework comparable -- the same trap
TEST_PLAN §2.5 documents for PPO rollouts. What IS comparable, and what this
class grades, is the deterministic transform chain (temperature -> top-k ->
top-p) and the resulting log-probabilities. A candidate is free to draw from
the processed distribution; nothing compares the draw.

Order matters and is vLLM's: temperature, then top-k, then top-p. Applying
top-p before top-k would change which tokens survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # 0.0 means greedy; scaling by it would divide by zero, and the argmax is
    # invariant to any positive scale, so pass through.
    if temperature in (0.0, 1.0):
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask everything below the k-th largest logit.

    This is a THRESHOLD mask, exactly as vLLM does it, not a "keep k entries"
    mask: entries equal to the k-th largest are kept, so a tie straddling the
    cut leaves MORE than k finite entries. That is a real, reachable case at
    low precision and the JAX twin has to reproduce it, so it is pinned in
    test_top_k_keeps_every_entry_tied_at_the_threshold rather than left to
    whichever behaviour a port happens to land on.
    """
    vocab = logits.shape[-1]
    if k <= 0 or k >= vocab:
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, _NEG_INF)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Mask the tail beyond the smallest prefix whose mass reaches p.

    For any p > 0 the argmax always survives, however peaked the distribution.
    At p <= 0 it would not -- see SamplingParams.__post_init__, which rejects
    that rather than let it through.
    """
    if p >= 1.0:
        return logits
    ordered, order = logits.sort(dim=-1, descending=True)
    probs = ordered.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Drop entries whose PRECEDING cumulative mass already reached p. Shifting
    # by one is what makes the prefix inclusive and keeps the argmax.
    drop = cumulative - probs >= p
    remove = drop.scatter(-1, order, drop)
    return logits.masked_fill(remove, _NEG_INF)


def process_logits(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    logits = apply_temperature(logits, params.temperature)
    logits = apply_top_k(logits, params.top_k)
    return apply_top_p(logits, params.top_p)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def topn_logprobs(logits: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-n log-probabilities and their token ids, descending."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    top = logprobs.topk(n, dim=-1)
    return top.values, top.indices
