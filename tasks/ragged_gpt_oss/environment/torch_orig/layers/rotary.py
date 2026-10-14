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

"""Rotary position embedding, de-vendored from vLLM's forward_native path.

Covers the neox style (rotate-half), plus two rope_scaling families: Llama-3's
piecewise NTK-by-parts, and YaRN, which is what gpt-oss uses. The cos/sin cache
is built eagerly at construction to max_position, as vLLM does, so the forward is
a pure gather.

YARN IS TWO SEPARATE CHANGES, and a port that makes only the first is subtly
wrong at every position rather than obviously wrong at long ones:

  1. The inverse frequencies are a per-dimension BLEND of interpolated
     (``inv_freq / factor``) and extrapolated (``inv_freq``) values, with a
     linear ramp between two "correction" dimensions derived from beta_fast and
     beta_slow. Low-frequency dimensions interpolate, high-frequency ones
     extrapolate.
  2. cos and sin are both multiplied by an ATTENTION FACTOR,
     ``0.1 * ln(factor) + 1`` -- 1.3466 at gpt-oss's factor of 32. This is a
     uniform gain on the rotated q and k, so it changes every attention logit by
     ~1.81x. It is invisible to any "is this a rotation" test, which is why
     :func:`RotaryEmbedding.__init__` exposes it as ``self.mscale`` and
     the harness asserts it against transformers' own value.

``truncate`` is read from rope_scaling and defaults to True, which is what every
other YaRN checkpoint wants. gpt-oss sets it to FALSE, so its correction range is
the raw fractional (8.093, 17.398) rather than (8, 18).

`positions` CARRIES A ROPE-SECTION AXIS. It is accepted at (S,) or (R, S), where
R is the number of rope sections: 1 for every text-only family, and more for an
M-RoPE family whose position is a tuple rather than a scalar. `mrope_section`
says how the rotary_dim // 2 frequency band is partitioned between the sections
and defaults to a single section covering all of it, which is the text-only
case. See forward for why that default is bit-identical to the rank-1 gather it
replaces rather than merely close to it.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _apply_llama3_scaling(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
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
    out = torch.where(wavelen > low_wavelen, scaled, inv_freq)
    return torch.where((wavelen >= high_wavelen) & (wavelen <= low_wavelen), inter, out)


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


def _yarn_linear_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001                       # prevent a singularity
    ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(ramp, 0.0, 1.0)


def _apply_yarn_scaling(inv_freq: torch.Tensor, scaling: dict, rotary_dim: int,
                        base: float) -> tuple[torch.Tensor, float]:
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


class RotaryEmbedding(nn.Module):
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
        super().__init__()
        assert is_neox_style, "torch_orig covers the neox (rotate-half) style only"
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # How the rotary_dim // 2 frequency band is partitioned between rope
        # sections. ONE section spanning all of it is the default and the
        # text-only case; an M-RoPE family passes its own widths (Qwen2.5-VL:
        # (16, 24, 24) over a 64-wide band). A tuple of ints, so it is a plain
        # attribute and never reaches state_dict().
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
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        # 1.0 unless YaRN says otherwise. Kept as an attribute rather than folded
        # away, because it is the half of YaRN that a rotation-preserving test
        # cannot see and a reader has to be able to find it.
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
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos() * self.mscale, persistent=False)
        self.register_buffer("sin_cached", freqs.sin() * self.mscale, persistent=False)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: [T, H, D]. cos/sin: [T, rotary_dim // 2] -> broadcast over heads.
        rot, pass_through = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x1, x2 = rot.chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([out, pass_through], dim=-1) if pass_through.numel() else out

    def _sections(self, table: torch.Tensor) -> torch.Tensor:
        """[R, T, rotary_dim // 2] -> [T, rotary_dim // 2].

        Section r contributes its own slice of the frequency band, taken from its
        own row of positions. With one section the slice is the whole band and
        the row is the only row, so this is a full-width gather of a single row:
        data movement, no arithmetic, hence bit-identical and not merely close.

        THE ROW COUNT IS CHECKED, and that is not defensive padding. Torch would
        raise IndexError here on its own for too FEW rows -- but the JAX twin
        would not: JAX lowers a static out-of-bounds index to a CLAMP, so rows 1
        and 2 would both read row 0 and it would return a WRONG ROTATION with no
        error. The check exists so the two sides refuse the same input with the
        same message instead of one raising and one lying. Too MANY rows is
        silent on both sides -- `(R, S)` into a single-section rope just drops
        rows 1..R-1 -- and this catches that too.
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
        return torch.cat(out, dim=-1)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
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
        positions = torch.atleast_2d(positions)  # (S,) -> (1, S); (R, S) unchanged
        cos = self._sections(self.cos_cached[positions].to(query.dtype))
        sin = self._sections(self.sin_cached[positions].to(query.dtype))
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)
