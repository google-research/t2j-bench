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

"""De-vendored vLLM layers, tensor-parallel size 1, no CUDA custom ops.

gemma-4 needs NO new layer module. Its four heterogeneities are derived in
`config.py` and applied in `modeling.py`; the only file here that is not a
sibling instance's byte for byte is `rotary.py`, which gains the
`proportional` rope_type, and `activation.py`, which gains `GeluAndMul`."""

from layers.activation import GeluAndMul, SiluAndMul
from layers.layernorm import RMSNorm
from layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding

__all__ = [
    "GeluAndMul",
    "SiluAndMul",
    "RMSNorm",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "RotaryEmbedding",
    "ParallelLMHead",
    "VocabParallelEmbedding",
]
