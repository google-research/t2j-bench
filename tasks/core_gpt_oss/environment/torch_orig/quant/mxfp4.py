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

"""MXFP4 dequantization -- the definition this instance's experts decode with.

AUTHORED, NOT DE-VENDORED, and that is the one place in this reference tree where
neither upstream could be followed:

  * vLLM never materialises the dequantised tensor in Python. It dispatches to a
    CUDA kernel (`csrc/libtorch_stable/quantization/fp4/`), which is outside the
    vendored subtrees and is not a legible equivalence target.
  * tpu-inference keeps the weights as a qwix ``QArray(qvalue, scale)`` and hands
    the unpacking to that library's quantization machinery. `qwix` is not in the
    inference base and this class does not grow it -- the dependency would be
    load-bearing for exactly one family. Its loader is also `torch.Tensor`-typed
    end to end (`gpt_oss.py::_build_mxfp4_pool`, `::_load_mxfp4`), so it cannot
    be vendored into a torch-free reference at all.

So this file is a DEFINITION. A definition with no independent check is an
assertion, which is the argument `attention/paged.py` makes for differential-
testing itself against real vLLM; the same discipline applies here.
the harness compares this, EXACTLY, against
`transformers.integrations.mxfp4.convert_moe_packed_tensors` -- the unpacker
`from_pretrained` itself runs on this checkpoint.

THE LAYOUT, read off the checkpoint rather than recalled
--------------------------------------------------------
`openai/gpt-oss-20b` ships each expert projection as two uint8 tensors::

    model.layers.{i}.mlp.experts.gate_up_proj_blocks  [32, 5760, 90, 16]  uint8
    model.layers.{i}.mlp.experts.gate_up_proj_scales  [32, 5760, 90]      uint8
    model.layers.{i}.mlp.experts.down_proj_blocks     [32, 2880, 90, 16]  uint8
    model.layers.{i}.mlp.experts.down_proj_scales     [32, 2880, 90]      uint8

`blocks` is ``[*prefix, G, 16]``: G groups of 16 bytes, each byte holding two
4-bit codes, so each group is :data:`VALUES_PER_BLOCK` = 32 values. `scales` is
``[*prefix, G]``: one shared exponent per group. 90 groups x 32 = 2880 = the
hidden size, so the packed axis is the CONTRACTED one.

Four facts, each of which is a separate way to get this wrong:

1. **The code table is OCP FP4 (E2M1)**, :data:`FP4_VALUES`. Note 6.0, not 7.0 --
   E2M1's largest finite is 1.5 x 2**2. Note also the two signed zeros: code 8 is
   -0.0, and a table that folds it to +0.0 is right for every arithmetic use and
   wrong for a bit-exact comparison.
2. **The low nibble comes first.** Byte j decodes to values (2j, 2j+1) =
   (byte & 0x0F, byte >> 4). Reversing this transposes pairs of adjacent weights,
   which is invisible in any norm and changes every product.
3. **The scale is E8M0 with bias 127**: the stored byte b means 2**(b - 127).
   An off-by-one bias scales every expert by 2 -- and, because the router and the
   residual stream are untouched, it produces a model that still generates fluent
   text.
4. **The decoded groups are laid out along the LAST axis and then the last two
   axes are swapped** (:func:`dequantize_mxfp4`'s ``swap_last2``). Decoding gives
   ``[32, 5760, 2880]``; the swap gives ``[32, 2880, 5760]`` = ``[experts, in,
   out]``, which is the orientation `layers/moe.py` contracts with and the
   orientation HF's own `GptOssExperts.gate_up_proj` parameter has. For
   `down_proj` the two shapes are both ``[32, 2880, 2880]``, so a port that drops
   the swap is SHAPE-CORRECT and numerically wrong on half the model -- which is
   why the swap has its own test.

Everything here is exactly representable: the 16 table entries need three
mantissa bits, and scaling by a power of two is exact in float32 and in bfloat16
alike (`torch.ldexp`, no rounding). That is why the tests use rtol=0, atol=0 and
why the native-dtype tier does not lose anything the fp32 tier keeps.
"""

from __future__ import annotations

import torch

#: OCP FP4 (E2M1), indexed by the 4-bit code. Sign in bit 3, 2-bit exponent,
#: 1-bit mantissa; code 0/8 are the two zeros, codes 1/9 the subnormal +-0.5.
FP4_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

#: E8M0 stores an exponent only, biased. Byte b is the scale 2 ** (b - 127).
E8M0_BIAS: int = 127

#: One shared scale per this many FP4 values -- the "MX" of MXFP4.
VALUES_PER_BLOCK: int = 32

#: ...which is this many bytes, two codes to a byte.
BYTES_PER_BLOCK: int = VALUES_PER_BLOCK // 2


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype,
    swap_last2: bool = True,
) -> torch.Tensor:
    """``blocks [*prefix, G, B]`` + ``scales [*prefix, G]`` -> the real weights.

    Returns ``[*prefix, G * B * 2]``, or that with its last two axes swapped when
    ``swap_last2`` (the default, and what the checkpoint's expert tensors need --
    see the module docstring).

    ``dtype`` is the OUTPUT dtype and is applied to the table, not to the result:
    the scale is then a `ldexp`, which is exact, so the decode introduces no
    rounding at any dtype. Every precondition is checked rather than assumed,
    because this function defines what the expert weights ARE and its failure
    mode must be an exception, never a plausible-looking wrong tensor.
    """
    if blocks.dtype != torch.uint8:
        raise ValueError(
            f"blocks must be uint8, got {blocks.dtype}. The packing is defined on "
            "BYTES: an int8 tensor sign-extends, and `byte >> 4` then reads codes "
            "8..15 as negative table indices.")
    if scales.dtype != torch.uint8:
        raise ValueError(
            f"scales must be uint8 (E8M0 exponent bytes), got {scales.dtype}")
    if blocks.ndim < 2:
        raise ValueError(
            f"blocks must be at least [G, B], got shape {tuple(blocks.shape)}")
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            f"MXFP4 carries one E8M0 scale per block: blocks {tuple(blocks.shape)} "
            f"implies scales {tuple(blocks.shape[:-1])}, but scales is "
            f"{tuple(scales.shape)}. Broadcasting the difference away would apply "
            "one exponent to the wrong rows.")

    table = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    low = table[(blocks & 0x0F).long()]
    high = table[(blocks >> 4).long()]
    # Interleave back: [low(byte0), high(byte0), low(byte1), ...]. `stack` on a
    # new trailing axis followed by a flatten IS that interleave, and it is the
    # one line where reversing two names silently transposes adjacent weights.
    out = torch.stack((low, high), dim=-1).flatten(start_dim=-2)

    exponents = scales.to(torch.int32) - E8M0_BIAS
    out = torch.ldexp(out, exponents.unsqueeze(-1))

    out = out.flatten(start_dim=-2)
    if swap_last2:
        if out.ndim < 2:
            raise ValueError(
                "swap_last2 needs at least two axes after decoding; blocks "
                f"{tuple(blocks.shape)} decodes to {tuple(out.shape)}")
        out = out.transpose(-2, -1)
    return out.contiguous()
