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

"""Rotary position embedding, mirroring environment/torch_orig/layers/rotary.py.

Covers the neox style (rotate-half), plus two rope_scaling families: Llama-3's
piecewise NTK-by-parts, and YaRN, which is what gpt-oss uses. The cos/sin cache
is built eagerly at construction to max_position, as vLLM does, so the forward
is a pure gather.

YARN IS TWO SEPARATE CHANGES, and a port that makes only the first is subtly
wrong at every position rather than obviously wrong at long ones:

  1. The inverse frequencies are a per-dimension BLEND of interpolated
     (``inv_freq / factor``) and extrapolated (``inv_freq``) values, with a
     linear ramp between two "correction" dimensions derived from beta_fast and
     beta_slow.
  2. cos and sin are both multiplied by an ATTENTION FACTOR,
     ``0.1 * ln(factor) + 1`` -- 1.34657 at gpt-oss's factor of 32. A uniform
     gain on the rotated q and k, so it changes every attention logit by ~1.81x
     and is invisible to any "is this still a rotation" test.

``truncate`` is read from rope_scaling and defaults to True, which is what every
other YaRN checkpoint wants. gpt-oss sets it to FALSE, so its correction range
is the raw fractional (8.093, 17.398) rather than (8, 18).

`positions` CARRIES A ROPE-SECTION AXIS. It is accepted at (S,) or (R, S), where
R is the number of rope sections: 1 for every text-only family, and more for an
M-RoPE family whose position is a tuple rather than a scalar. `mrope_section`
says how the rotary_dim // 2 frequency band is partitioned between the sections
and defaults to a single section covering all of it, which is the text-only
case. See __call__ for why that default is bit-identical to the rank-1 gather it
replaces rather than merely close to it.

THE cos/sin CACHE IS NOT AN nnx.Param, and that is deliberate. It is derived
from (base, rotary_dim, max_position, rope_scaling) -- three of which are static
ints and one a dict -- so it is a buffer, not a weight. torch_orig says the same
thing with `register_buffer(..., persistent=False)`: present in the module,
absent from `parameters()` and from `state_dict()`. Promoting it to nnx.Param
here would put two tensors into the parameter pytree that have no counterpart in
the torch state dict, and test_param_correspondence grades that correspondence
1:1.

It is stored in fp32 and cast at gather time. torch_orig ends up equivalent by a
different route: `model.to(bfloat16)` casts float buffers too, so its table is
bf16 and `.to(query.dtype)` is a no-op. Casting elementwise before or after an
index gather is the same value either way, so the two sides agree bit for bit.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _apply_llama3_scaling(inv_freq: jax.Array, scaling: dict) -> jax.Array:
    """Llama-3's piecewise NTK-by-parts rescale of the inverse frequencies."""
    factor = float(scaling["factor"])
    low_factor = float(scaling["low_freq_factor"])
    high_factor = float(scaling["high_freq_factor"])
    orig_len = float(scaling["original_max_position_embeddings"])

    low_wavelen = orig_len / low_factor
    high_wavelen = orig_len / high_factor
    wavelen = 2 * math.pi / inv_freq

    scaled = inv_freq / factor
    smooth = (orig_len / wavelen - low_factor) / (high_factor - low_factor)
    inter = (1 - smooth) * scaled + smooth * inv_freq
    out = jnp.where(wavelen > low_wavelen, scaled, inv_freq)
    return jnp.where((wavelen >= high_wavelen) & (wavelen <= low_wavelen), inter, out)


def _yarn_correction_dim(rotations: float, dim: int, base: float,
                         original_max: float) -> float:
    """The rotary dimension that completes `rotations` turns over `original_max`."""
    return (dim * math.log(original_max / (rotations * 2 * math.pi))) / (
        2 * math.log(base))


def _yarn_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float,
                           original_max: float, truncate: bool) -> tuple[float, float]:
    low = _yarn_correction_dim(beta_fast, dim, base, original_max)
    high = _yarn_correction_dim(beta_slow, dim, base, original_max)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(low: float, high: float, dim: int) -> jax.Array:
    if low == high:
        high += 0.001                       # prevent a singularity
    ramp = (jnp.arange(dim, dtype=jnp.float32) - low) / (high - low)
    return jnp.clip(ramp, 0.0, 1.0)


def _apply_yarn_scaling(inv_freq: jax.Array, scaling: dict, rotary_dim: int,
                        base: float) -> tuple[jax.Array, float]:
    """``(blended inv_freq, mscale)``. See the module docstring for both halves."""
    factor = float(scaling["factor"])
    original_max = float(scaling["original_max_position_embeddings"])
    beta_fast = float(scaling.get("beta_fast") or 32)
    beta_slow = float(scaling.get("beta_slow") or 1)
    truncate = bool(scaling.get("truncate", True))

    low, high = _yarn_correction_range(beta_fast, beta_slow, rotary_dim, base,
                                       original_max, truncate)
    # 1 on the extrapolating (high-frequency) dimensions, 0 on the interpolating
    # ones, ramping linearly between the two correction dimensions.
    extrapolation = 1.0 - _yarn_linear_ramp(low, high, rotary_dim // 2)
    blended = (inv_freq / factor) * (1 - extrapolation) + inv_freq * extrapolation
    mscale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return blended, mscale


class RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool = True,
        rope_scaling: dict | None = None,
        mrope_section: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        assert is_neox_style, "ref_jax_repo covers the neox (rotate-half) style only"
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # How the rotary_dim // 2 frequency band is partitioned between rope
        # sections. ONE section spanning all of it is the default and the
        # text-only case; an M-RoPE family passes its own widths (Qwen2.5-VL:
        # (16, 24, 24) over a 64-wide band). A tuple of ints, so it stays part
        # of the module's static structure rather than its pytree.
        # `is None`, not `or`: an EMPTY `mrope_section` is a caller error, not an
        # omission, and `or` would silently resolve `()` and `[]` to the whole
        # band instead of letting the partition check below refuse them.
        half = rotary_dim // 2
        self.mrope_section = tuple(
            int(w) for w in ((half,) if mrope_section is None else mrope_section))
        if sum(self.mrope_section) != half:
            raise ValueError(
                f"mrope_section {list(self.mrope_section)} sums to "
                f"{sum(self.mrope_section)}, not rotary_dim // 2 = {half}. The "
                "sections PARTITION the frequency band: a short cover would drop "
                "frequencies and a long one would not fit, and either would be a "
                "quietly wrong rotation rather than an error.")
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
        )
        # 1.0 unless YaRN says otherwise. Kept as an attribute rather than
        # folded away, because it is the half of YaRN that a rotation-preserving
        # test cannot see and a reader has to be able to find it.
        self.mscale = 1.0
        rope_type = (rope_scaling or {}).get("rope_type")
        if rope_type == "llama3":
            inv_freq = _apply_llama3_scaling(inv_freq, rope_scaling)
        elif rope_type == "yarn":
            inv_freq, self.mscale = _apply_yarn_scaling(
                inv_freq, rope_scaling, rotary_dim, base)
        elif rope_type is not None:
            raise ValueError(
                f"unsupported rope_type {rope_type!r}. This reference implements "
                "the two the class's checkpoints use (llama3, yarn); silently "
                "ignoring an unknown one would build a plain-RoPE table for a "
                "model that was not trained with one.")
        self.inv_freq = inv_freq
        t = jnp.arange(max_position, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        # Plain attributes, not nnx.Param: buffers, not weights. `rope_scaling`
        # itself is NOT stored -- it is an unhashable dict and would become part
        # of the module's static structure.
        self.cos_cached = jnp.cos(freqs) * self.mscale
        self.sin_cached = jnp.sin(freqs) * self.mscale

    def _rotate(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        # x: [T, H, D]. cos/sin: [T, rotary_dim // 2] -> broadcast over heads.
        rot, pass_through = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x1, x2 = jnp.split(rot, 2, axis=-1)
        cos = jnp.expand_dims(cos, 1)
        sin = jnp.expand_dims(sin, 1)
        out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return jnp.concatenate([out, pass_through], axis=-1) if pass_through.size else out

    def _sections(self, table: jax.Array) -> jax.Array:
        """[R, T, rotary_dim // 2] -> [T, rotary_dim // 2].

        Section r contributes its own slice of the frequency band, taken from its
        own row of positions. With one section the slice is the whole band and
        the row is the only row, so this is a full-width gather of a single row:
        data movement, no arithmetic, hence bit-identical and not merely close.

        THE ROW COUNT IS CHECKED, and that is not defensive padding. It is a
        SHAPE check, not a behaviour branch -- shapes are static, so it fires
        under jit too -- and both wrong-shape cases are silent on at least one
        side. Measured: `(S,)` positions into a 3-section rope raises IndexError
        on torch_orig but returns a WRONG ROTATION here, because JAX lowers a
        static out-of-bounds index to a CLAMP and rows 1 and 2 both read row 0.
        `(R, S)` positions into a single-section rope silently drops rows 1..R-1
        on BOTH sides. These two files are behavioural twins and a divergence
        that only shows up on the family this axis was added for is the worst
        possible place to keep one.
        """
        if table.shape[0] != len(self.mrope_section):
            raise ValueError(
                f"positions has {table.shape[0]} rope section row(s) but this "
                f"RotaryEmbedding was built with {len(self.mrope_section)} "
                f"(mrope_section={list(self.mrope_section)}). The row count and "
                "the section count are the same number: row r carries section "
                "r's positions.")
        out, start = [], 0
        for r, width in enumerate(self.mrope_section):
            out.append(table[r, :, start:start + width])
            start += width
        return jnp.concatenate(out, axis=-1)

    def __call__(self, positions: jax.Array, query: jax.Array, key: jax.Array):
        # RANK-AGNOSTIC, NOT VL-BRANCHED. `positions` arrives as (S,) for a
        # text-only family and (R, S) for one with M-RoPE, where R is the number
        # of rope sections (Qwen2.5-VL: 3, for t/h/w). Normalising to (R, S) with
        # R == 1 here means every downstream expression is written once, and it
        # is why the text-only instances' goldens do not move: with the default
        # single section the gather-and-slice below reproduces the rank-1 form
        # this replaces exactly, bit for bit. `query` and `key` are NOT reshaped
        # -- only `positions` is -- because an extra reshape of the rotated
        # tensors would still "work" and would still move the last bit. There is
        # deliberately NO `if multimodal` in this file: a branch here would make
        # the shared column per-family, which is the cost model the whole class
        # rests on.
        positions = jnp.atleast_2d(positions)   # (S,) -> (1, S); (R, S) unchanged
        cos = self._sections(self.cos_cached[positions].astype(query.dtype))
        sin = self._sections(self.sin_cached[positions].astype(query.dtype))
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)
