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

"""SiluAndMul, mirroring environment/torch_orig/layers/activation.py."""

from __future__ import annotations

import jax
from flax import nnx


class SiluAndMul(nnx.Module):
    """y = silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.

    The gate and up projections are fused into one matmul upstream
    (MergedColumnParallelLinear), so this splits the fused output.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        d = x.shape[-1] // 2
        return jax.nn.silu(x[..., :d]) * x[..., d:]
