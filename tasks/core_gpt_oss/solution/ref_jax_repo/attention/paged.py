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

"""XLA-native ragged paged attention -- the substitution that makes this class
runnable on an H100.

Upstream's tpu_inference.layers.common.attention_interface.attention calls a TPU
Pallas kernel. Upstream ALSO ships the semantics as pure jnp, at
kernels/ragged_paged_attention/v3/kernel.py::ref_ragged_paged_attention -- but
that reference loops over sequences in Python with dynamic slicing, so it does
not jit. This is the same contract, vectorised over a padded [T, max_kv] score
matrix so it traces to a single fused graph.

The harness is what makes "same contract" a checked claim rather than a comment.

ATTENTION SINKS (`sinks`, optional) are this instance's one addition to the op.
gpt-oss carries a learned per-head logit -- `model.layers.*.self_attn.sinks`, 64
of them -- that joins the softmax denominator with no matching value, so it
removes probability mass from the real positions rather than redistributing it.
The keyword defaults to None, which is byte-for-byte the contract the other
families' copies of this file already have.

Formulation, identical to environment/torch_orig/attention/paged.py:
  q_pos[t] = seq_lens[s] - q_len[s] + (t - query_start_loc[s])
  valid[t, j] = (j <= q_pos[t]) and (j < seq_lens[s])
Scores and softmax in fp32; output cast back to the query dtype. The fp32
discipline is load-bearing rather than tidy: accumulating in the query dtype
reproduces every fp32 test in the suite while diverging ~0.6% relative under
bf16, which is the serving dtype for this class.

The `j < seq_lens[s]` conjunct is provably implied by `j <= q_pos[t]` for every
conforming metadata. It is kept because that is what upstream writes, and
because it becomes load-bearing the moment an implementation sources q_pos from
input_positions instead of deriving it from seq_lens. Do NOT "simplify" it away.

Two preconditions, both refused rather than assumed, because this file defines
what "correct" means for the whole task class and its failure mode must be an
exception rather than a plausible-looking wrong tensor:

  * Padded query tokens are out of scope for `core`. Upstream pads the token
    array to a compiled bucket (padded_total_num_scheduled_tokens) and masks the
    tail; here query.shape[0] must equal query_start_loc[-1]. `core` never needs
    the padding -- max_num_seqs=1 with a fixed prompt length is already
    statically shaped -- so support is deferred to the `ragged` mode, where
    bucketing is actually load-bearing. Unchecked, a padded token gets
    seq_of_tok == num_seqs, and JAX does not raise for that: out-of-bounds
    gathers CLAMP, so the token would silently be attributed to the last
    sequence. That is worse than the torch side's IndexError.
  * seq_lens must fit inside the gathered pages_per_seq * page_size window.
    Under-allocated pages are a WRONG ANSWER, not a slow one: the gather only
    reaches max_kv positions, so a longer sequence silently loses its oldest KV,
    and with a small sliding_window a row can lose every valid position and come
    back as a uniform-softmax average.

Both guards read metadata VALUES, so they are EAGER-ONLY: they fire on every
unjitted call and are skipped inside any trace. Under jit,
`query_start_loc[-1]` is a tracer whether the metadata arrives as an
argument, is closed over, or is even constructed inside the traced function, so
there is no arrangement that keeps them live. Skipping is the only honest
option: `jax.jit(paged_attention)` is a supported call, and raising
TracerIntegerConversionError there would trade a checked precondition for an
unusable function.

That is a real gap, named rather than papered over: a candidate whose whole
forward pass is jitted end to end never sees these two messages. What they buy
is the reference's own contract -- eager calls, this file's test suite, and
anything that validates metadata on the host before compiling -- and, above all,
that neither condition can ever come back as a plausible-looking wrong tensor
from an eager call. Making them live under jit needs jax.experimental.checkify,
which changes the function's signature and is not worth it here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from attention.metadata import AttentionMetadata


def _concrete_int(value) -> int | None:
    """int(value) if it is known now, None under trace. See the module docstring.

    Deliberately narrow: only a ConcretizationTypeError -- "this is a tracer" --
    returns None. Anything else is a real bug and propagates.
    """
    try:
        return int(value)
    except jax.errors.ConcretizationTypeError:
        return None


def paged_attention(
    query: jax.Array,
    kv_cache_layer: jax.Array,
    attn_metadata: AttentionMetadata,
    num_kv_heads: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: jax.Array | None = None,
) -> jax.Array:
    """query [T, N, D]; kv_cache_layer [P, page_size, 2, K, D] -> [T, N, D].

    ``sinks`` is [N], one learned logit per QUERY head, or None. It defaults to
    None so the other families' copies of this file keep the exact contract they
    had.
    """
    num_tokens, num_heads, head_dim = query.shape
    page_size = kv_cache_layer.shape[1]
    num_seqs = attn_metadata.num_seqs
    pages_per_seq = attn_metadata.pages_per_seq
    max_kv = pages_per_seq * page_size

    qsl = attn_metadata.query_start_loc.astype(jnp.int32)
    seq_lens = attn_metadata.seq_lens.astype(jnp.int32)

    # Padded query tokens are out of scope for `core` (see the module docstring).
    ragged_tokens = _concrete_int(qsl[-1])
    if ragged_tokens is not None and num_tokens != ragged_tokens:
        raise ValueError(
            f"paged_attention got {num_tokens} query tokens but "
            f"query_start_loc[-1] is {ragged_tokens}: padded query tokens are "
            f"not supported in the `core` mode (deferred to `ragged`)."
        )

    # Under-allocated pages are a wrong answer, not a slow one. Refuse instead.
    longest = _concrete_int(seq_lens.max()) if seq_lens.size else 0
    if longest is not None and longest > max_kv:
        raise ValueError(
            f"paged_attention: seq_lens max is {longest} but the block table "
            f"only reaches pages_per_seq * page_size = {pages_per_seq} * "
            f"{page_size} = {max_kv}; this request is under-allocated."
        )

    # Which sequence each query token belongs to.
    tok = jnp.arange(num_tokens)
    seq_of_tok = jnp.searchsorted(qsl[1:], tok, side="right")

    # Gather every sequence's pages, in block-table order.
    bt = attn_metadata.block_tables.astype(jnp.int32).reshape(num_seqs, pages_per_seq)
    gathered = kv_cache_layer[bt]                       # [S, pps, ps, 2, K, D]
    gathered = gathered.reshape(num_seqs, max_kv, 2, num_kv_heads, head_dim)
    keys = gathered[:, :, 0][seq_of_tok]                # [T, max_kv, K, D]
    values = gathered[:, :, 1][seq_of_tok]

    # Absolute position of each query token inside its sequence.
    q_start = qsl[seq_of_tok]
    q_len = qsl[seq_of_tok + 1] - q_start
    q_pos = seq_lens[seq_of_tok] - q_len + (tok - q_start)

    reps = num_heads // num_kv_heads
    keys = jnp.repeat(keys, reps, axis=2)               # [T, max_kv, N, D]
    values = jnp.repeat(values, reps, axis=2)

    scores = jnp.einsum(
        "tnd,tknd->tnk", query.astype(jnp.float32), keys.astype(jnp.float32)
    ) * sm_scale

    kv_pos = jnp.arange(max_kv)[None, :]
    valid = (kv_pos <= q_pos[:, None]) & (kv_pos < seq_lens[seq_of_tok][:, None])
    if sliding_window is not None:
        # `>` and not `>=`: the window is [q_pos - w + 1, q_pos], exactly w
        # positions with the query itself included.
        valid = valid & (kv_pos > (q_pos[:, None] - sliding_window))
    scores = jnp.where(valid[:, None, :], scores, jnp.finfo(jnp.float32).min)

    # ATTENTION SINKS. One learned logit per query head, concatenated onto the
    # score row before the softmax and dropped afterwards: it has no matching
    # value, so all it does is take mass OUT of the real positions. A head whose
    # sink dominates outputs ~0 instead of being forced to attend somewhere,
    # which is the point -- it is a learned "attend to nothing" escape.
    #
    # Concatenate-then-drop rather than an equivalent rescale of the softmax
    # denominator: the two are algebraically identical, and this is the form
    # both upstreams write, so a reader can line them up. In fp32, like the rest
    # of this file's accumulation. The shape check is STATIC, so unlike the two
    # guards above it stays live under jit.
    if sinks is not None:
        if sinks.shape != (num_heads,):
            raise ValueError(
                f"paged_attention expects one sink logit per query head, i.e. "
                f"shape ({num_heads},), got {tuple(sinks.shape)}. A K-length "
                "sink would broadcast one kv head's logit over its whole GQA "
                "group.")
        column = jnp.broadcast_to(sinks.astype(jnp.float32)[None, :, None],
                                  (num_tokens, num_heads, 1))
        probs = jax.nn.softmax(jnp.concatenate([scores, column], axis=-1),
                               axis=-1)[..., :-1]
    else:
        probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("tnk,tknd->tnd", probs, values.astype(jnp.float32))
    return out.astype(query.dtype)
