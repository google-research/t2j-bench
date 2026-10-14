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

"""Falcon configuration (mirror of torch_orig/transformers_impl/configuration.py).

falcon-rw: ALiBi (no rotary), two LayerNorms/layer, interleaved fused ``query_key_value``
(MHA), biased projections, GELU MLP. Sizes come from the checkpoint's ``config.json`` (with a
``hidden // 64`` fallback for the head count, which the rw checkpoints omit); ``dtype`` /
``with_value_head`` are the usual JAX extras.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass
class FalconConfig:
    """Falcon config. Canonical fields mirror torch ``FalconConfig``."""

    model_type: str = field(default="falcon", init=False)

    vocab_size: int = 50304
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
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


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> FalconConfig:
    """Build a ``FalconConfig`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``FalconConfig.from_pretrained``). ``compute_dtype`` selects the flax compute dtype,
    and ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in
    any checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    hidden = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    return FalconConfig(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=hidden,
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=n_heads,
        layer_norm_epsilon=float(cfg.get("layer_norm_epsilon", 1e-5) or 1e-5),
        # falcon-rw-1b's config.json omits the flag; torch's FalconConfig ties by
        # default, so the checkpoint's (bit-identical) ``lm_head.weight`` is dropped.
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        _head_dim=hidden // n_heads,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        with_value_head=with_value_head,
    )

__all__ = ["FalconConfig", "config_from_pretrained"]
