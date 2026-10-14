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

Scope boundary -- padded query tokens are NOT supported. Upstream pads the token
array to a compiled bucket (padded_total_num_scheduled_tokens) and masks the
tail; this reference requires query.shape[0] == query_start_loc[-1] and raises
otherwise. The `core` mode never needs the padding -- max_num_seqs=1 with a
fixed prompt length is already statically shaped -- so padded-query support is
deferred to the `ragged` mode, where bucketing is actually load-bearing. This is
a named boundary, not a TODO: it is checked, and moving it is a contract change
both sides make together. Unchecked, a padded token gets seq_of_tok == num_seqs
and dies in an IndexError on seq_lens, a long way from the cause.

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
) -> torch.Tensor:
    """query [T, N, D]; kv_cache_layer [P, page_size, 2, K, D] -> [T, N, D]."""
    num_tokens, num_heads, head_dim = query.shape
    page_size = kv_cache_layer.shape[1]
    num_seqs = attn_metadata.num_seqs
    pages_per_seq = attn_metadata.pages_per_seq
    max_kv = pages_per_seq * page_size

    qsl = attn_metadata.query_start_loc.to(torch.long)
    seq_lens = attn_metadata.seq_lens.to(torch.long)

    # Padded query tokens are out of scope for `core` (see the module docstring).
    # Checked, not assumed: without this a padded token lands on seq_of_tok ==
    # num_seqs and raises an opaque IndexError inside the gather.
    ragged_tokens = int(qsl[-1])
    if num_tokens != ragged_tokens:
        raise ValueError(
            f"paged_attention got {num_tokens} query tokens but "
            f"query_start_loc[-1] is {ragged_tokens}: padded query tokens are "
            f"not supported in the `core` mode (deferred to `ragged`)."
        )

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
    seq_of_tok = torch.searchsorted(qsl[1:].contiguous(), tok, right=True)

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

    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("tnk,tknd->tnd", probs, values.float())
    return out.to(query.dtype)
