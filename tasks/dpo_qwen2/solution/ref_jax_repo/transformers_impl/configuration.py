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

"""Qwen2 configuration (mirror of torch_orig/transformers_impl/configuration.py).

JAX adjustments:
- Plain ``@dataclass`` instead of HF ``PreTrainedConfig`` (torch-free).
- Adds ``dtype`` (the flax compute dtype) and ``with_value_head`` (PPO head, unused
  for DPO) — the only extra fields.
- ``head_dim`` is exposed as a read-only property (``hidden_size // num_attention_heads``)
  mirroring the torch config's derived attribute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass
class Qwen2Config:
    """Qwen2 config. Canonical fields mirror torch ``Qwen2Config.__init__``."""

    model_type: str = field(default="qwen2", init=False)

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    pad_token_id: Optional[int] = None
    _head_dim: Optional[int] = None

    # --- JAX-only extras -------------------------------------------------- #
    dtype: Any = jnp.bfloat16  # flax compute dtype (resolved from compute_dtype)
    with_value_head: bool = False  # PPO value head (unused for DPO)
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)

    @property
    def head_dim(self) -> int:
        return self._head_dim or (self.hidden_size // self.num_attention_heads)


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> Qwen2Config:
    """Build a ``Qwen2Config`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``Qwen2Config.from_pretrained``). ``compute_dtype`` selects the flax compute dtype,
    and ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in
    any checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    heads = int(cfg["num_attention_heads"])
    return Qwen2Config(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(cfg.get("num_key_value_heads") or heads),
        hidden_act=str(cfg.get("hidden_act", "silu")),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 32768)),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6) or 1e-6),
        rope_theta=float(cfg.get("rope_theta", 10000.0) or 10000.0),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        _head_dim=int(cfg["head_dim"]) if cfg.get("head_dim") else None,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        with_value_head=with_value_head,
    )


__all__ = ["Qwen2Config", "config_from_pretrained"]
