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

Covers the neox style (rotate-half), plus three rope_scaling families: Llama-3's
piecewise NTK-by-parts, YaRN, and -- new for this family -- gemma-4's
``proportional``. The cos/sin cache is built eagerly at construction to
max_position, as vLLM does, so the forward is a pure gather.

===========================================================================
GEMMA-4'S PARTIAL ROPE PAIRS DIFFERENT DIMENSIONS THAN A TRUNCATED
``rotary_dim`` WOULD, AND NOTHING ABOUT THE SHAPES SAYS SO.
===========================================================================

This is the sharpest trap in the family. ``partial_rotary_factor: 0.25`` on a
head_dim of 512 rotates 64 dimension PAIRS either way, so every tensor in the
forward has the shape it would have had, every assertion in this repo passes,
and the logits come back finite and fluent and wrong.

The two readings::

    rotary_dim = 128 (the truncate-and-pass-through idiom this class already
    has, and which `core_gpt_oss` uses for its full-width rope):
        pairs (i, i + 64) for i < 64        dims 128..511 untouched

    gemma-4, and BOTH upstreams:
        pairs (i, i + 256) for i < 64       dims 64..255 and 320..511 untouched

Same count of rotated dimensions. Different dimensions. The fix is not a
different ``rotary_dim``: it is ``rotary_dim = head_size`` -- so ``_rotate``
splits at ``head_size // 2`` and pairs ``i`` with ``i + 256`` -- plus an
``inv_freq`` that is ZERO-PADDED out to ``head_size // 2``, which makes
``cos = 1`` and ``sin = 0`` on the pairs that must not move. That is an identity
rotation expressed as data rather than as a slice, and it is what makes the
"which dimensions" question have the right answer.

Both upstreams do exactly this and neither uses a truncated rotary_dim:

  * vLLM ``layers/rotary_embedding/gemma4_rope.py:40-77`` sets
    ``rope_angles = rotary_dim // 2``, ``nope_angles = head_size // 2 -
    rope_angles``, calls ``super().__init__(head_size, head_size, ...)`` -- note
    the second argument -- and ``torch.cat``s ``nope_angles`` zeros onto
    ``inv_freq``.
  * tpu-inference ``layers/jax/rope_interface.py:182-195`` computes
    ``rope_angles = int(rope_proportion * head_dim // 2)``, builds ``timescale``
    over ``2 * arange(rope_angles) / head_dim``, ``jnp.pad``s it with
    ``nope_angles`` zeros, and splits the input at ``head_dim // 2``.

THE DENOMINATOR IS ``head_size``, NOT ``rotary_dim``, AND THAT IS THE WHOLE
CONTENT OF ``rope_type: "proportional"``. The exponents are
``arange(0, 2 * rope_angles, 2) / head_size``, so the 64 surviving frequencies
are the FIRST 64 of the frequency ladder a full-width rope over 512 dimensions
would have had -- the low-frequency (long-wavelength) end is kept and the
high-frequency end is dropped. Dividing by ``2 * rope_angles`` instead would
give 64 frequencies spanning the whole ladder, which is a different position
encoding with the same shape.

``proportional`` ALSO MEANS "APPLY NO ROPE SCALING", and that is not an
omission here. tpu-inference gates it explicitly
(``rope_interface.py:190``: ``if rope_scaling and rope_scaling.get("rope_type")
!= "proportional"``), and the sliding layers' ``{"rope_theta": 10000.0,
"rope_type": "default"}`` is collapsed to ``None`` by ``normalize_rope_scaling``,
so NEITHER gemma-4 layer type gets any frequency rescale at all. The two layer
types differ only in ``base`` (10000 vs 1000000), in ``head_size`` (256 vs 512)
and in how many pairs rotate (all 128, vs 64 of 256).

WHY THE LLAMA-3 AND YARN BRANCHES ARE STILL HERE. Neither fires on this
checkpoint. They are carried forward from `core_gpt_oss`, which is where this
file came from, because this class's `layers/rotary.py` is additive per family
and deleting a branch a sibling instance needs is how a "shared" file stops
being shareable. The unified ``inv_freq`` below is bit-identical to the one they
were written against whenever ``partial_rotary_factor`` is 1.0 and
``rotary_dim == head_size``, which is every call either of them ever sees.

`positions` CARRIES A ROPE-SECTION AXIS. It is accepted at (S,) or (R, S), where
R is the number of rope sections: 1 for every text-only family, and more for an
M-RoPE family whose position is a tuple rather than a scalar. `mrope_section`
says how the rotary_dim // 2 frequency band is partitioned between the sections
and defaults to a single section covering all of it, which is the text-only
case. See __call__ for why that default is bit-identical to the rank-1 gather it
replaces rather than merely close to it.

THE cos/sin CACHE IS NOT AN nnx.Param, and that is deliberate. It is derived
from (base, rotary_dim, max_position, rope_scaling, partial_rotary_factor) --
static ints, a float and a dict -- so it is a buffer, not a weight. torch_orig
says the same thing with `register_buffer(..., persistent=False)`: present in
the module, absent from `parameters()` and from `state_dict()`. Promoting it to
nnx.Param here would put two tensors into the parameter pytree that have no
counterpart in the torch state dict, and test_param_correspondence grades that
correspondence 1:1.

It is stored in fp32 and cast at gather time. torch_orig ends up equivalent by a
different route: `model.to(bfloat16)` casts float buffers too, so its table is
bf16 and `.to(query.dtype)` is a no-op. Casting elementwise before or after an
index gather is the same value either way, so the two sides agree bit for bit.

ONE TABLE PER LAYER TYPE, SHARED BY EVERY LAYER OF THAT TYPE -- see
`modeling.py::Gemma4Model`, which constructs exactly two of these and hands them
to the 35 attention modules. The arithmetic that forces it: at
``max_position_embeddings`` 131072 a sliding table is 131072 x 128 x 4 B x 2
(cos and sin) = 134 MB and a full one is 268 MB, so 35 private tables would be
5.6 GB PER MODEL COPY and the grader holds three. Two shared tables are 402 MB.
Nothing about the values changes -- a table is a pure function of (base,
head_size, max_position) and every layer of a type asks for the same one.
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
        partial_rotary_factor: float = 1.0,
    ) -> None:
        assert is_neox_style, "ref_jax_repo covers the neox (rotate-half) style only"
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # HOW MANY PAIRS ACTUALLY ROTATE, and how many are padded to the
        # identity. `int(f * head_size // 2)` and not `int(f * head_size) // 2`:
        # the floor division binds after the multiply in both upstreams
        # (tpu-inference `rope_interface.py:182`), and at f = 0.25 on head_size
        # 512 the two agree at 64 -- but they do not agree in general and the
        # one that runs upstream is this one.
        self.rope_angles = int(partial_rotary_factor * head_size // 2)
        self.nope_angles = head_size // 2 - self.rope_angles
        if self.rope_angles <= 0 or self.nope_angles < 0:
            raise ValueError(
                f"partial_rotary_factor {partial_rotary_factor} gives "
                f"{self.rope_angles} rotated pair(s) of a possible "
                f"{head_size // 2}: a rope that rotates nothing, or more than "
                "the head has, is a configuration error rather than a limit.")
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
        # THE DENOMINATOR IS head_size. See the module docstring -- this is the
        # whole content of `rope_type: "proportional"`, and it is the one line
        # a reader coming from the other families' copies of this file must not
        # skim. When nothing is partial (`rope_angles == head_size // 2`) and
        # `rotary_dim == head_size`, this is bit-identical to the
        # `arange(0, rotary_dim, 2) / rotary_dim` form it replaces.
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, 2 * self.rope_angles, 2, dtype=jnp.float32)
                     / head_size)
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
        elif rope_type in ("proportional", "default", None):
            # BOTH OF gemma-4'S LAYER TYPES LAND HERE, and neither gets a
            # frequency rescale. `default` is the sliding layers' declaration
            # and carries no `factor`, so there is nothing to apply;
            # `proportional` is the full layers' and means, in as many words,
            # "the partial rope IS the scaling -- do not also rescale". Both
            # upstreams gate it exactly this way.
            pass
        else:
            raise ValueError(
                f"unsupported rope_type {rope_type!r}. This reference implements "
                "the three the class's checkpoints use (llama3, yarn, "
                "proportional); silently ignoring an unknown one would build a "
                "plain-RoPE table for a model that was not trained with one.")
        # ZERO-PAD THE FREQUENCIES THAT MUST NOT MOVE. cos(0 * pos) = 1 and
        # sin(0 * pos) = 0, so the padded pairs pass through unchanged -- an
        # identity rotation written as data. Done AFTER any scaling, because a
        # scaling function reading a zero frequency would divide by it
        # (`wavelen = 2 * pi / inv_freq`).
        if self.nope_angles:
            inv_freq = jnp.pad(inv_freq, (0, self.nope_angles),
                               mode="constant", constant_values=0.0)
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
        #
        # THE SPLIT IS AT rotary_dim // 2 AND THAT IS WHERE THE PAIRING LIVES.
        # gemma-4 passes rotary_dim == head_size, so the pairs are
        # (i, i + head_size // 2) -- (i, i + 256) on a full layer -- and the
        # dimensions that do not rotate are the ones whose FREQUENCY is zero,
        # not the ones past a slice boundary. See the module docstring.
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
