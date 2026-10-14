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

"""SiluAndMul and ClampedSwiGLU, mirroring torch_orig/layers/activation.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class SiluAndMul(nnx.Module):
    """y = silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.

    The gate and up projections are fused into one matmul upstream
    (MergedColumnParallelLinear), so this splits the fused output.

    UNUSED BY THIS FAMILY and deliberately unchanged: gpt-oss uses
    :class:`ClampedSwiGLU`. It stays here byte for byte because it is the shared
    `layers/` surface `core_llama3` and `core_qwen2` are graded against, and a
    family that needs a different activation adds one beside it.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        d = x.shape[-1] // 2
        return jax.nn.silu(x[..., :d]) * x[..., d:]


class ClampedSwiGLU(nnx.Module):
    """gpt-oss's routed-expert activation. vLLM calls it ``swigluoai``.

    THREE DIFFERENCES FROM :class:`SiluAndMul`, and each one changes the answer:

    1. **The interleave.** gate is ``x[..., ::2]`` and up is ``x[..., 1::2]`` --
       EVEN and ODD, not first half and second half. That is how the
       checkpoint's fused ``gate_up_proj`` is laid out (tpu-inference splits it
       the same way, ``blocks_u8[..., ::2, :]`` for gate), and reading it as
       halves gives a finite, fluent, wrong model.
    2. **The clamp.** ``swiglu_limit`` is 7.0 in config.json. gate is clamped
       from ABOVE only; up is clamped on BOTH sides. The asymmetry is not a typo
       -- gate feeds a sigmoid, where a large negative value is already
       saturated, while up multiplies the result and a large negative one flips
       its sign.
    3. **The gate function.** ``glu = gate * sigmoid(alpha * gate)`` with
       ``alpha`` = 1.702, and the output is ``(up + 1) * glu``, not
       ``glu * up``. ``alpha`` = 1.702 makes this the GELU approximation rather
       than SiLU (which is alpha = 1), and the ``+1`` is a residual path
       through the MLP that a plain SwiGLU does not have.

    Both constants are the released definition: ``alpha`` is hard-coded to 1.702
    in `transformers`' ``GptOssExperts`` and is not a config key, so it is a
    module constant here too rather than a ModelConfig field that no checkpoint
    would populate. ``limit`` IS a config key and is passed in.
    """

    #: Not a config key anywhere in gpt-oss -- see the class docstring.
    ALPHA: float = 1.702

    def __init__(self, limit: float) -> None:
        self.limit = float(limit)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate, up = x[..., ::2], x[..., 1::2]
        gate = jnp.minimum(gate, self.limit)
        up = jnp.clip(up, -self.limit, self.limit)
        return (up + 1) * (gate * jax.nn.sigmoid(gate * self.ALPHA))
