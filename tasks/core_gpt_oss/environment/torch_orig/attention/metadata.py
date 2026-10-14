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

"""Attention metadata, de-vendored from tpu_inference's AttentionMetadata.

Kept: input_positions, block_tables, seq_lens, query_start_loc,
request_distribution, padded_num_reqs. Dropped: mamba_state_indices, pcp,
pcp_cache_pages, GroupedAttentionMetadata -- all belong to subsystems this task
class puts out of scope.

The two sides share this shape exactly; test_attention_metadata_schema grades a
candidate's version of it against the frozen scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionMetadata:
    # (total_num_scheduled_tokens,) absolute position of each query token, or
    # (num_rope_sections, total_num_scheduled_tokens) for an M-RoPE family whose
    # position is a tuple rather than a scalar. layers/rotary.py accepts both
    # ranks; every family in this class today builds the rank-1 form, and its
    # shape is graded, so nothing upstream of the rope may normalise the rank.
    input_positions: torch.Tensor
    # (max_num_seqs * pages_per_seq,) flattened page indirection.
    block_tables: torch.Tensor
    # (max_num_seqs,) total KV length per request AFTER this step's write.
    seq_lens: torch.Tensor
    # (max_num_seqs + 1,) cu_seqlens over the ragged query.
    query_start_loc: torch.Tensor
    # (3,) [n_decode, n_prefill, n_mixed]. Upstream calls these RpaCase.
    request_distribution: torch.Tensor
    # The compiled bucket. Static, not a tensor -- it shapes the trace.
    padded_num_reqs: int

    @property
    def num_seqs(self) -> int:
        return int(self.seq_lens.shape[0])

    @property
    def pages_per_seq(self) -> int:
        return int(self.block_tables.shape[0]) // self.num_seqs
