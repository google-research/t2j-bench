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

"""RMSNorm, mirroring environment/torch_orig/layers/layernorm.py.

Same two-argument residual form, same fp32 accumulation, same parameter name
(`weight`) -- test_param_correspondence keys off it.

The cast ORDER is load-bearing and fp32 hides it: the reciprocal-sqrt is
accumulated in fp32, the result is cast back to the input dtype, and only THEN
multiplied by `self.weight`. Multiplying in fp32 and casting at the very end is
bit-identical in fp32 and differs in bf16, which is this class's serving dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root-mean-square layer norm, optionally fused with a residual add.

    The two-argument form returns ``(normed, new_residual)`` and is what the
    decoder layer uses, so the residual stream is threaded explicitly rather
    than recomputed -- exactly as vLLM does it.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, *,
                 dtype=jnp.float32) -> None:
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=dtype))
        self.variance_epsilon = eps

    def __call__(self, x: jax.Array, residual: jax.Array | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.variance_epsilon)
        x = x.astype(orig_dtype) * self.weight[...]
        return x if residual is None else (x, residual)
