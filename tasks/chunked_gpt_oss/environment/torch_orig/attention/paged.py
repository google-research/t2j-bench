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

"""Reference paged attention -- the semantic definition this class grades against.

vLLM dispatches to FlashAttention with a paged KV cache; that kernel is not a
legible equivalence target and does not exist on the JAX side. What IS shared
between vLLM and tpu-inference is the CONTRACT, which upstream states twice:
once as a Pallas kernel and once as
tpu_inference/kernels/ragged_paged_attention/v3/kernel.py::ref_ragged_paged_attention.
This is that contract in torch. Task I1-9 differential-tests it against real
vLLM on GPU, which is what makes "reference" a claim rather than an assertion.

Formulation, identical on both sides:
  q_pos[t] = seq_lens[s] - q_len[s] + (t - query_start_loc[s])
  valid[t, j] = (j <= q_pos[t]) and (j < seq_lens[s])
Scores and softmax in fp32; output cast back to the query dtype. Cost is
O(pages_per_seq * page_size) per token, which is the point -- clarity over
speed, at a probe geometry chosen so it does not matter.

PADDED QUERY TOKENS ARE SUPPORTED HERE, and ATTENTION SINKS are this family's one
addition to the op. The two are independent and this file is their only meeting point.

Padding: upstream pads the token array to a compiled bucket
(padded_total_num_scheduled_tokens) and masks the tail. `core` refused that outright;
`ragged` is where bucketing is load-bearing, so the boundary moves and the masking
lands. A token array SHORTER than the batch it describes is still refused -- that is
not padding, it is a metadata that disagrees with its own tensor.

Sinks: gpt-oss carries a learned per-head logit (`model.layers.*.self_attn.sinks`, 64
of them) that joins the softmax denominator with no matching value, so it removes
probability mass from the real positions rather than redistributing it. The keyword
defaults to None, which is byte-for-byte the contract the other families' copies have.

THEIR INTERACTION, which neither column's comment covers. `ragged_llama3` justifies the
`is_pad` zeroing by saying a fully-masked row's softmax is uniform over the parked
sequence's window, i.e. a plausible-looking number from another request's KV. WITH A
SINK COLUMN THAT IS NO LONGER THE FAILURE MODE: a fully-masked row puts essentially all
its mass on the sink and the output is already ~0. The zeroing is kept anyway, and not
as belt-and-braces -- it is the graded contract on the ragged side, it is exactly zero
rather than merely small, and it must not depend on `sinks` being non-None, because the
same file serves a sink-free call whenever `sinks=None`.

Both preconditions here (that one, and seq_lens fitting inside the gathered
pages_per_seq * page_size window) are checked rather than assumed, for the same
reason: this file defines what "correct" means for the whole task class, so its
failure mode has to be an exception, never a plausible-looking wrong tensor.
"""

from __future__ import annotations

import torch

from attention.metadata import AttentionMetadata


def paged_attention(
    query: torch.Tensor,
    kv_cache_layer: torch.Tensor,
    attn_metadata: AttentionMetadata,
    num_kv_heads: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
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

    qsl = attn_metadata.query_start_loc.to(torch.long)
    seq_lens = attn_metadata.seq_lens.to(torch.long)

    # `ragged` buckets the token array to a compiled shape and masks the tail
    # (see the module docstring). A SHORTER array is still refused: padding adds
    # rows past the batch, it never removes rows the metadata claims exist, and
    # without this the missing rows would silently not be computed.
    ragged_tokens = int(qsl[-1])
    if num_tokens < ragged_tokens:
        raise ValueError(
            f"paged_attention got {num_tokens} query tokens but "
            f"query_start_loc[-1] is {ragged_tokens}: the token array is SHORTER "
            f"than the ragged batch it describes."
        )
    is_pad = torch.arange(num_tokens, device=query.device) >= ragged_tokens

    # Under-allocated pages are a WRONG ANSWER, not a slow one: the gather only
    # reaches max_kv positions, so a longer sequence silently loses its oldest
    # KV, and with a small sliding_window a row can lose every valid position
    # and come back as a uniform-softmax average. Refuse instead.
    longest = int(seq_lens.max()) if seq_lens.numel() else 0
    if longest > max_kv:
        raise ValueError(
            f"paged_attention: seq_lens max is {longest} but the block table "
            f"only reaches pages_per_seq * page_size = {pages_per_seq} * "
            f"{page_size} = {max_kv}; this request is under-allocated."
        )

    # Which sequence each query token belongs to.
    tok = torch.arange(num_tokens, device=query.device)
    # Clamped for the GATHER only. A padded token searchsorts to num_seqs, which
    # is off the end of every per-sequence array; it is parked on the last real
    # row, reads nothing (see `valid` below) and is zeroed before the return.
    seq_of_tok = torch.searchsorted(qsl[1:].contiguous(), tok, right=True)
    seq_of_tok = seq_of_tok.clamp(max=num_seqs - 1)

    # Gather every sequence's pages, in block-table order.
    bt = attn_metadata.block_tables.to(torch.long).reshape(num_seqs, pages_per_seq)
    gathered = kv_cache_layer[bt]                       # [S, pps, ps, 2, K, D]
    gathered = gathered.reshape(num_seqs, max_kv, 2, num_kv_heads, head_dim)
    keys = gathered[:, :, 0][seq_of_tok]                # [T, max_kv, K, D]
    values = gathered[:, :, 1][seq_of_tok]

    # Absolute position of each query token inside its sequence.
    q_start = qsl[seq_of_tok]
    q_len = qsl[seq_of_tok + 1] - q_start
    q_pos = seq_lens[seq_of_tok] - q_len + (tok - q_start)

    reps = num_heads // num_kv_heads
    keys = keys.repeat_interleave(reps, dim=2)          # [T, max_kv, N, D]
    values = values.repeat_interleave(reps, dim=2)

    scores = torch.einsum(
        "tnd,tknd->tnk", query.float(), keys.float()
    ) * sm_scale

    kv_pos = torch.arange(max_kv, device=query.device).unsqueeze(0)
    valid = (kv_pos <= q_pos.unsqueeze(1)) & (kv_pos < seq_lens[seq_of_tok].unsqueeze(1))
    if sliding_window is not None:
        valid = valid & (kv_pos > (q_pos.unsqueeze(1) - sliding_window))
    scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(torch.float32).min)

    # ATTENTION SINKS. One learned logit per query head, concatenated onto the
    # score row before the softmax and dropped afterwards: it has no matching
    # value, so all it does is take mass OUT of the real positions. A head whose
    # sink dominates outputs ~0 instead of being forced to attend somewhere,
    # which is the point -- it is a learned "attend to nothing" escape.
    #
    # Concatenate-then-drop rather than an equivalent rescale of the softmax
    # denominator: the two are algebraically identical, and this is the form
    # both upstreams write (`transformers`' eager_attention_forward for gpt-oss,
    # and vLLM's `Attention(..., sinks=...)`), so a reader can line them up. In
    # fp32, like the rest of this file's accumulation.
    if sinks is not None:
        if sinks.shape != (num_heads,):
            raise ValueError(
                f"paged_attention expects one sink logit per query head, i.e. "
                f"shape ({num_heads},), got {tuple(sinks.shape)}. A K-length "
                "sink would broadcast one kv head's logit over its whole GQA "
                "group.")
        column = sinks.float().reshape(1, num_heads, 1).expand(num_tokens, -1, 1)
        probs = torch.softmax(torch.cat([scores, column], dim=-1), dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("tnk,tknd->tnd", probs, values.float())
    # The padded tail returns zeros. A fully-masked row's softmax is uniform over
    # the parked sequence's window, i.e. a plausible-looking number computed from
    # another request's KV -- exactly the misattribution the `core` refusal
    # existed to prevent. Identity when nothing is padded.
    out = out.masked_fill(is_pad.reshape(-1, 1, 1), 0.0)
    return out.to(query.dtype)
