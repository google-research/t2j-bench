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

"""Phi-3/Phi-4 configuration (mirror of torch_orig/transformers_impl/configuration.py).

JAX adjustments (as elsewhere): plain ``@dataclass``; extra ``dtype``/``with_value_head``.
Phi3 differs from Llama: **fused** ``qkv_proj`` and ``gate_up_proj`` projections, and a
**partial rotary** (``partial_rotary_factor`` = 0.75 -> only the first ``round(head_dim*
factor)`` dims are rotated). No attention bias; tied embeddings; RMSNorm; SwiGLU.
(``longrope`` scaling is read straight from the checkpoint's ``config.json``: its per-dim
``short_factor`` is 1.0 at the graded sequence lengths, but its ``attention_factor``
mscale is ~1.19 and does move the logits.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


def _rope_extras(cfg: dict):
    """(partial_rotary_factor, longrope short_factor, attention_factor) from ``config.json``.

    Phi-4's rope_scaling is ``longrope`` (keyed under ``type``): the per-dim ``short_factor``
    (~1.0 at short sequences) scales ``inv_freq``, and the ``attention_factor`` mscale
    (~1.19 for Phi-4-mini) scales cos/sin — the latter matters even at the graded lengths.
    """
    text = cfg.get("text_config", cfg)
    partial = float(text.get("partial_rotary_factor", 1.0) or 1.0)
    short: tuple = ()
    attn = 1.0
    rs = text.get("rope_scaling") or {}
    if (rs.get("rope_type") or rs.get("type")) == "longrope":
        short = tuple(float(x) for x in (rs.get("short_factor") or []))
        if rs.get("attention_factor") is not None:
            attn = float(rs["attention_factor"])
        else:
            maxp = text.get("max_position_embeddings")
            orig = text.get("original_max_position_embeddings")
            if maxp and orig and maxp > orig:
                attn = math.sqrt(1.0 + math.log(maxp / orig) / math.log(orig))
    return partial, short, attn


@dataclass
class Phi3Config:
    """Phi3 config. Canonical fields mirror torch ``Phi3Config.__init__``."""

    model_type: str = field(default="phi3", init=False)

    vocab_size: int = 200064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.75
    rope_short_factor: tuple = ()  # longrope per-dim inv_freq scaling (len rotary_dim/2)
    rope_attention_factor: float = 1.0  # longrope cos/sin mscale
    bos_token_id: int = 199999
    eos_token_id: int = 199999
    pad_token_id: Optional[int] = None
    _head_dim: Optional[int] = None

    # --- JAX-only extras -------------------------------------------------- #
    dtype: Any = jnp.bfloat16
    with_value_head: bool = False
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)

    @property
    def head_dim(self) -> int:
        return self._head_dim or (self.hidden_size // self.num_attention_heads)

    @property
    def rotary_dim(self) -> int:
        return int(round(self.head_dim * self.partial_rotary_factor))


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> Phi3Config:
    """Build a ``Phi3Config`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``Phi3Config.from_pretrained``). ``compute_dtype`` selects the flax compute dtype,
    and ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in
    any checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    heads = int(cfg["num_attention_heads"])
    partial, short, attn = _rope_extras(cfg)
    return Phi3Config(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(cfg.get("num_key_value_heads") or heads),
        hidden_act=str(cfg.get("hidden_act", "silu")),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 131072)),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5) or 1e-5),
        rope_theta=float(cfg.get("rope_theta", 10000.0) or 10000.0),
        partial_rotary_factor=partial,
        rope_short_factor=short,
        rope_attention_factor=attn,
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        _head_dim=int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads),
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        with_value_head=with_value_head,
    )

__all__ = ["Phi3Config", "config_from_pretrained"]
