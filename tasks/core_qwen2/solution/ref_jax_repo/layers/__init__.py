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

"""The vLLM layers as flax nnx modules, tensor-parallel size 1.

Exactly torch_orig's nine names, so `layers` is a mirror of `layers` and not a
re-spelling of it.
"""

from layers.activation import SiluAndMul
from layers.layernorm import RMSNorm
from layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding

__all__ = [
    "SiluAndMul",
    "RMSNorm",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
    "RotaryEmbedding",
    "ParallelLMHead",
    "VocabParallelEmbedding",
]
