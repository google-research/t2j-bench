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

"""SiluAndMul and GeluAndMul, mirroring torch_orig/layers/activation.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

#: The one `hidden_activation` this reference implements, and the one gemma-4
#: declares. Named rather than inlined so `GeluAndMul` can refuse anything else
#: by value -- see its docstring for why a silent fallback is the expensive
#: failure here.
GEMMA4_HIDDEN_ACTIVATION = "gelu_pytorch_tanh"


class SiluAndMul(nnx.Module):
    """y = silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.

    The gate and up projections are fused into one matmul upstream
    (MergedColumnParallelLinear), so this splits the fused output.

    UNUSED BY THIS FAMILY and deliberately unchanged: gemma-4 uses
    :class:`GeluAndMul`. It stays here byte for byte because it is the shared
    `layers/` surface `core_llama3` and `core_qwen2` are graded against, and a
    family that needs a different activation adds one beside it -- which is what
    `core_gpt_oss` did with `ClampedSwiGLU`.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        d = x.shape[-1] // 2
        return jax.nn.silu(x[..., :d]) * x[..., d:]


class GeluAndMul(nnx.Module):
    """gemma-4's MLP activation. vLLM resolves it from `hidden_activation`.

    TWO THINGS ARE DIFFERENT FROM :class:`SiluAndMul` AND ONLY ONE OF THEM IS
    OBVIOUS.

    1. **The gate function is GELU, not SiLU**, and it is the *tanh
       approximation* rather than the exact erf form. ``config.text_config``
       declares ``hidden_activation: "gelu_pytorch_tanh"``, vLLM maps that string
       to ``GeluAndMul(approximate="tanh")`` (`gemma4.py:249`
       ``get_act_and_mul_fn(hidden_activation)``) and tpu-inference writes
       ``partial(nnx.gelu, approximate=True)`` (`gemma4.py:94`). The two
       approximations differ by up to ~1e-3 relative near |x| ~ 2, which is a
       real value error and not a rounding one -- large enough to move a logit,
       small enough that the text still reads.
    2. **The split is HALVES, not the interleave `core_gpt_oss` uses.** gate is
       ``x[..., :d]`` and up is ``x[..., d:]``. gpt-oss's fused
       ``gate_up_proj`` is interleaved (even rows gate, odd rows up) and this
       one is not: gemma-4 ships ``gate_proj`` and ``up_proj`` as SEPARATE
       checkpoint tensors, and `model/loader.py` concatenates them along dim 0
       in that order -- so the halves are halves by construction.

    ``limit``-style clamping is absent because gemma-4 has none.
    """

    def __init__(self, hidden_activation: str = GEMMA4_HIDDEN_ACTIVATION) -> None:
        # BY VALUE, NOT BY DEFAULT. A checkpoint declaring `silu` or `gelu`
        # (the exact erf form) is a different model, and running it through the
        # tanh approximation anyway produces a finite, fluent, wrong answer that
        # no shape check and no parity check between two equally-wrong sides can
        # see. This is the same argument `config.py` makes about `layer_types`:
        # the config is the authority.
        if hidden_activation != GEMMA4_HIDDEN_ACTIVATION:
            raise ValueError(
                f"hidden_activation is {hidden_activation!r}; this reference "
                f"implements {GEMMA4_HIDDEN_ACTIVATION!r} only. Silently using "
                "the tanh GELU for a model trained with a different gate is a "
                "wrong answer nothing downstream can attribute.")
        self.hidden_activation = hidden_activation

    def __call__(self, x: jax.Array) -> jax.Array:
        d = x.shape[-1] // 2
        # `approximate=True` IS the tanh form -- jax spells the same choice with
        # a bool where torch spells it with the string "tanh". The two are
        # measured against each other in tests_parity/test_layers_parity.py.
        return jax.nn.gelu(x[..., :d], approximate=True) * x[..., d:]
