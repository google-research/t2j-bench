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

"""MXFP4 dequantization, mirroring environment/torch_orig/quant/mxfp4.py.

RE-AUTHORED, NOT DE-VENDORED, and this file is where that costs the most.
`tpu_inference/models/jax/gpt_oss.py` imports torch at module scope (`:21`) and
its whole MXFP4 path is torch-typed -- `_build_mxfp4_pool` (`:406`) pools
`torch.Tensor` pairs, `_load_mxfp4` (`:449`) calls `.detach().cpu().numpy()`,
`_load_regular_param` (`:492`) does `loaded_weight.to(torch.float32).numpy()`.
None of it can be vendored into a torch-free reference. Upstream also keeps the
result as a qwix `QArray(qvalue, scale)`, and `qwix` is not in the inference
base and is not being added for one family.

TWO THINGS UPSTREAM DOES THAT THIS DELIBERATELY DOES NOT
---------------------------------------------------------
* `u8_unpack_e2m1` bitcasts to `jnp.float4_e2m1fn`. That is elegant and it is a
  dependency on a jax dtype whose availability is a version fact; a 16-entry
  table is the same function, is exact, and reads the same on both sides of
  this port. The table is the DEFINITION here (see the torch twin's docstring
  for the four things it pins), and the two sides are proved bit-identical by
  the harness against a fixture the torch side produced.
* upstream's `swap_last2` transform operates on its own QArray layout, where
  the packed dimension is kept most-minor. It is NOT the same operation as the
  final transpose here, which is the one `transformers`'
  `convert_moe_packed_tensors` performs and the one that puts the decoded
  weights in `[experts, in, out]`. Same name, different tensor.

JITTABLE. Everything below is `jnp` on traced arrays -- a gather, a shift, a
mask, an `ldexp` and a transpose -- so `layers/moe.py` can call it inside the
compiled forward, which is the whole point: the weights stay 4-bit resident and
only the layer being executed is ever materialised.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: OCP FP4 (E2M1), indexed by the 4-bit code. Sign in bit 3, 2-bit exponent,
#: 1-bit mantissa; code 0/8 are the two zeros, codes 1/9 the subnormal +-0.5.
FP4_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

#: E8M0 stores an exponent only, biased. Byte b is the scale 2 ** (b - 127).
#: The same number `jnp.finfo(jnp.float8_e8m0fnu).minexp` negates, which is how
#: tpu-inference spells it; written out because the constant is the contract.
E8M0_BIAS: int = 127

#: One shared scale per this many FP4 values -- the "MX" of MXFP4.
VALUES_PER_BLOCK: int = 32

#: ...which is this many bytes, two codes to a byte.
BYTES_PER_BLOCK: int = VALUES_PER_BLOCK // 2


def dequantize_mxfp4(
    blocks: jax.Array,
    scales: jax.Array,
    *,
    dtype,
    swap_last2: bool = True,
) -> jax.Array:
    """``blocks [*prefix, G, B]`` + ``scales [*prefix, G]`` -> the real weights.

    Returns ``[*prefix, G * B * 2]``, or that with its last two axes swapped
    when ``swap_last2`` (the default, and what the checkpoint's expert tensors
    need).

    Shape and dtype preconditions are checked EAGERLY -- they are static
    properties of the arrays, available under trace as well, so unlike
    `attention/paged.py`'s value-dependent guards these stay live inside jit.
    """
    if blocks.dtype != jnp.uint8:
        raise ValueError(
            f"blocks must be uint8, got {blocks.dtype}. The packing is defined "
            "on BYTES: a signed type sign-extends, and `byte >> 4` then reads "
            "codes 8..15 as negative table indices.")
    if scales.dtype != jnp.uint8:
        raise ValueError(
            f"scales must be uint8 (E8M0 exponent bytes), got {scales.dtype}")
    if blocks.ndim < 2:
        raise ValueError(
            f"blocks must be at least [G, B], got shape {tuple(blocks.shape)}")
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            f"MXFP4 carries one E8M0 scale per block: blocks {tuple(blocks.shape)} "
            f"implies scales {tuple(blocks.shape[:-1])}, but scales is "
            f"{tuple(scales.shape)}. Broadcasting the difference away would "
            "apply one exponent to the wrong rows.")

    table = jnp.asarray(FP4_VALUES, dtype=dtype)
    low = table[blocks & 0x0F]
    high = table[blocks >> 4]
    # Interleave back: [low(byte0), high(byte0), low(byte1), ...]. The one line
    # where reversing two names silently transposes adjacent weights.
    out = jnp.stack((low, high), axis=-1).reshape(*blocks.shape[:-1], -1)

    exponents = scales.astype(jnp.int32) - E8M0_BIAS
    out = jnp.ldexp(out, exponents[..., None]).astype(dtype)

    out = out.reshape(*blocks.shape[:-2], -1)
    if swap_last2:
        if out.ndim < 2:
            raise ValueError(
                "swap_last2 needs at least two axes after decoding; blocks "
                f"{tuple(blocks.shape)} decodes to {tuple(out.shape)}")
        out = jnp.swapaxes(out, -2, -1)
    return out
