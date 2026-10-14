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

"""Starcoder2 configuration (mirror of torch_orig/transformers_impl/configuration.py).

JAX adjustments (as elsewhere): plain ``@dataclass``; extra ``dtype``/``with_value_head``.
Starcoder2 differs from the Llama family: **LayerNorm** (scale+bias) instead of RMSNorm,
a **GELU dense** MLP (``c_fc``/``c_proj`` with bias) instead of SwiGLU, and **bias on
q/k/v/o**. Embeddings are tied; plain NeoX RoPE with ``rope_theta``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass
class Starcoder2Config:
    """Starcoder2 config. Canonical fields mirror torch ``Starcoder2Config.__init__``."""

    model_type: str = field(default="starcoder2", init=False)

    vocab_size: int = 49152
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 30
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    hidden_act: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 16384
    norm_epsilon: float = 1e-5
    use_bias: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 999999.4420358813
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: Optional[int] = None
    # starcoder2-3b's config.json sets all three to 0.1; torch applies them via
    # ``nn.functional.dropout(..., training=self.training)`` (active in train, off in eval).
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
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
    def gelu_approximate(self) -> bool:
        return "tanh" in self.hidden_act


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> Starcoder2Config:
    """Build a ``Starcoder2Config`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``Starcoder2Config.from_pretrained``). ``compute_dtype`` selects the flax compute dtype,
    and ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in
    any checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    heads = int(cfg["num_attention_heads"])
    return Starcoder2Config(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(cfg.get("num_key_value_heads") or heads),
        hidden_act=str(cfg.get("hidden_act", "gelu_pytorch_tanh")),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 16384)),
        norm_epsilon=float(cfg.get("norm_epsilon") or cfg.get("layer_norm_epsilon") or 1e-5),
        use_bias=bool(cfg.get("use_bias", True)),
        rope_theta=float(cfg.get("rope_theta", 999999.4420358813) or 999999.4420358813),
        # starcoder2-3b's config.json omits the flag; torch's Starcoder2Config ties by default.
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        attention_dropout=float(cfg.get("attention_dropout", 0.0) or 0.0),
        residual_dropout=float(cfg.get("residual_dropout", 0.0) or 0.0),
        embedding_dropout=float(cfg.get("embedding_dropout", 0.0) or 0.0),
        _head_dim=int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads),
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        with_value_head=with_value_head,
    )

__all__ = ["Starcoder2Config", "config_from_pretrained"]
