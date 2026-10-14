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

"""Linear layers, de-vendored from vLLM at tensor-parallel size 1.

vLLM's Column/Row/QKV/MergedColumn parallel linears exist to shard a matmul
across TP ranks and to route HuggingFace checkpoint shards into the sharded
parameter. At TP=1 the sharding is the identity, so what remains -- and what
the equivalence target actually is -- is the FUSED PARAMETER LAYOUT: q, k and v
live in one `qkv_proj.weight`, gate and up in one `gate_up_proj.weight`.

That layout is load-bearing, not cosmetic. test_param_correspondence grades the
candidate against the POST-FUSION torch state dict, so a port that keeps three
separate q/k/v tensors fails -- correctly, because it does not mirror vLLM.

vLLM's linears return `(output, bias)`; the caller discards the bias. We return
a bare tensor and keep the bias folded in, which is the same computation with
one less thing for a port to get wrong.
"""

from __future__ import annotations

import torch
from torch import nn


class _LinearBase(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class ColumnParallelLinear(_LinearBase):
    """TP=1 degenerate case of vLLM's column-sharded linear."""


class ReplicatedLinear(_LinearBase):
    """Not sharded at any TP size -- every rank holds the whole weight.

    vLLM uses it for the MoE router (`gpt_oss.py::MLPBlock.router`), which is
    2880 -> 32 and produces a value every rank needs before it can decide which
    experts it owns. At TP=1 it is arithmetically the same as
    :class:`ColumnParallelLinear`; the separate name is kept because it is the
    class the equivalence target names and because a reader looking for the
    router should find the reason it is not sharded stated here.
    """


class RowParallelLinear(_LinearBase):
    """TP=1 degenerate case of vLLM's row-sharded linear."""


class MergedColumnParallelLinear(_LinearBase):
    """gate_proj and up_proj fused into one [2 * I, H] weight.

    `output_sizes` records the split so the caller (and the loader) can address
    the halves; SiluAndMul consumes the fused output directly.
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool) -> None:
        super().__init__(input_size, sum(output_sizes), bias)
        self.output_sizes = list(output_sizes)


class QKVParallelLinear(_LinearBase):
    """q_proj, k_proj and v_proj fused into one [(N + 2K) * D, H] weight."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool,
    ) -> None:
        # Sizes into locals first so super().__init__() -- and with it
        # nn.Module.__init__() -- runs before any attribute lands on self.
        # Assigning before it happens to work for plain ints, but a Tensor
        # would be silently absent from state_dict() and a Parameter would
        # raise; there is no reason to leave that trap set.
        q_size = total_num_heads * head_size
        kv_size = total_num_kv_heads * head_size
        super().__init__(hidden_size, q_size + 2 * kv_size, bias)
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.q_size = q_size
        self.kv_size = kv_size

    def split(self, qkv: torch.Tensor):
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
