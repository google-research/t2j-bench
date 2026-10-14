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

"""BLOOM configuration (mirror of torch_orig/transformers_impl/configuration.py).

JAX adjustments:
- Plain ``@dataclass`` instead of HF ``PreTrainedConfig`` (torch-free).
- Adds ``dtype`` (the flax compute dtype) and ``with_value_head`` (PPO head, unused
  for DPO) — the only extra fields.
- The ``attribute_map`` aliases (``num_hidden_layers`` <-> ``n_layer``,
  ``num_attention_heads`` <-> ``n_head``) are exposed as read-only properties so the
  flax modeling can use the descriptive names while the canonical fields match the
  torch config / the checkpoint ``config.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp

# Same alias mapping as torch BloomConfig.attribute_map.
attribute_map = {"num_hidden_layers": "n_layer", "num_attention_heads": "n_head"}

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass
class BloomConfig:
    """BLOOM config. Canonical fields mirror torch ``BloomConfig.__init__``."""

    model_type: str = field(default="bloom", init=False)

    vocab_size: int = 250880
    hidden_size: int = 64
    n_layer: int = 2
    n_head: int = 8
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int | None = None
    apply_residual_connection_post_layernorm: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    pretraining_tp: int = 1
    slow_but_exact: bool = False
    tie_word_embeddings: bool = True

    # --- JAX-only extras -------------------------------------------------- #
    dtype: Any = jnp.bfloat16  # flax compute dtype (resolved from compute_dtype)
    with_value_head: bool = False  # PPO value head (unused for DPO/bloom)
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)

    # --- attribute_map aliases (read-only, mirror torch's attribute_map) --- #
    @property
    def num_hidden_layers(self) -> int:
        return self.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self.n_head

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_head

    @property
    def layer_norm_eps(self) -> float:
        return self.layer_norm_epsilon


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> BloomConfig:
    """Build a ``BloomConfig`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``BloomConfig.from_pretrained``). Bloom's config.json still uses the pre-4.21 field
    names — ``n_embed`` for the width, ``n_layer`` for the depth — which HF maps through
    ``attribute_map``; the same mapping is spelled out here. ``compute_dtype`` selects
    the flax compute dtype, and ``with_value_head`` is the caller's ``add_valuehead`` —
    the PPO head is not in any checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return BloomConfig(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg.get("hidden_size") or cfg["n_embed"]),
        n_layer=int(cfg.get("num_hidden_layers") or cfg["n_layer"]),
        n_head=int(cfg.get("n_head") or cfg["num_attention_heads"]),
        layer_norm_epsilon=float(cfg.get("layer_norm_epsilon", 1e-5) or 1e-5),
        apply_residual_connection_post_layernorm=bool(
            cfg.get("apply_residual_connection_post_layernorm", False)
        ),
        pretraining_tp=int(cfg.get("pretraining_tp", 1) or 1),
        # bloom-560m's config.json omits the flag; torch's BloomConfig ties by default,
        # and the checkpoint carries no ``lm_head`` tensor of its own.
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        with_value_head=with_value_head,
    )

__all__ = ["BloomConfig", "config_from_pretrained", "attribute_map"]
