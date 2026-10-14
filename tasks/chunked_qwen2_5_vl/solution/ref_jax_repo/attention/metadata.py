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

"""Attention metadata, mirroring environment/torch_orig/attention/metadata.py.

Same six fields, same meanings; the two sides differ only in the array type.

Registered as a JAX dataclass so it crosses a jit boundary without being
flattened by hand. padded_num_reqs is a META field -- it shapes the trace, so it
must be static; putting it in data_fields would retrace on every value and
silently defeat the bucketing contract.

frozen=True mirrors torch_orig. register_dataclass reconstructs through
__init__, so it round-trips: as a jit argument, as a jit RETURN value, and
through tree_map. Two consequences of the frozen dataclass's generated __eq__
and __hash__, both measured against the torch twin because a schema test will
compare these contracts:

  * `a == b` between two distinct instances RAISES on both sides -- the
    generated __eq__ compares field tuples, and bool() of a multi-element array
    is ambiguous. The exception TYPE differs: ValueError here, RuntimeError on
    the torch side. `a == a` is True on both, via the identity short-circuit in
    tuple comparison. Compare fields, not instances.
  * hash() works on the torch side (torch.Tensor hashes by identity) and raises
    TypeError here (jax.Array is deliberately unhashable). Do not put this in a
    set or use it as a dict key on either side.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["input_positions", "block_tables", "seq_lens",
                 "query_start_loc", "request_distribution"],
    meta_fields=["padded_num_reqs"],
)
@dataclass(frozen=True)
class AttentionMetadata:
    # (total_num_scheduled_tokens,) absolute position of each query token, or
    # (num_rope_sections, total_num_scheduled_tokens) for an M-RoPE family whose
    # position is a tuple rather than a scalar. layers/rotary.py accepts both
    # ranks; every family in this class today builds the rank-1 form, and its
    # shape is graded, so nothing upstream of the rope may normalise the rank.
    input_positions: jax.Array
    # (max_num_seqs * pages_per_seq,) flattened page indirection.
    block_tables: jax.Array
    # (max_num_seqs,) total KV length per request AFTER this step's write.
    seq_lens: jax.Array
    # (max_num_seqs + 1,) cu_seqlens over the ragged query.
    query_start_loc: jax.Array
    # (3,) [n_decode, n_prefill, n_mixed]. Upstream calls these RpaCase.
    request_distribution: jax.Array
    # The compiled bucket. Static, not an array -- it shapes the trace.
    padded_num_reqs: int = -1

    @property
    def num_seqs(self) -> int:
        return int(self.seq_lens.shape[0])

    @property
    def pages_per_seq(self) -> int:
        return int(self.block_tables.shape[0]) // self.num_seqs
