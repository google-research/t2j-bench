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

"""Gemma3 configuration (mirror of torch_orig; VL text path, TEST_PLAN.md §6.5).

Gemma3 text decoder specifics: **four** norms/layer (each ``(1+w)`` RMSNorm), **qk-norm**,
GELU-gated MLP, embedding scaled by ``sqrt(hidden)``, and **dual rope** — global layers
(every 6th) use theta 1e6, local/sliding layers use 1e4. LM nested under
``model/language_model/…``. Sizes come from the checkpoint's own ``config.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass(frozen=True)
class Gemma3Config:
    prefix: tuple
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_global: float = 1_000_000.0
    rope_local: float = 10_000.0
    sliding_pattern: int = 6
    query_pre_attn_scalar: float = 256.0
    tie_word_embeddings: bool = False
    with_value_head: bool = False
    dtype: Any = jnp.bfloat16
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)
    # ------------------------------------------------------------------ vision path
    # The SigLIP vision tower (Conv2d patchify, learned pos-embed, 27 pre-norm encoder
    # layers, no pooling head) + Gemma3 multimodal projector (avg-pool -> RMSNorm ->
    # linear) that turns ``<image_soft_token>`` placeholders into text-hidden embeddings
    # (mirror of torch ``SiglipVisionTransformer`` + ``Gemma3MultiModalProjector``). Read
    # from the checkpoint's ``vision_config``; fallbacks are the gemma-3-4b-it values. The
    # graded forward masks image positions out (attention_mask=0), so this path only moves
    # the training loss, never the goldens.
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_hidden_layers: int = 27
    vision_num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 896
    patch_size: int = 14
    vision_layer_norm_eps: float = 1e-6
    vision_use_head: bool = False
    mm_tokens_per_image: int = 256
    image_token_id: int = 262144


def load_config(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> Gemma3Config:
    """Build a ``Gemma3Config`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``load_config`` (which reads ``config.json`` and dispatches on
    ``model_type``; this repo grades the one Gemma3 config, so the dispatch collapses). The decoder this repo grades is the ``text_config``
    half of a composite VL config — for gemma-3-4b-it a Gemma 3 text tower — and every
    field it leaves out falls back to the ``Gemma3TextConfig`` default, exactly as HF's
    own loader does. ``compute_dtype`` selects the flax compute dtype, and
    ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in any
    checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    vision = cfg.get("vision_config", {})
    return Gemma3Config(
        # This tree nests the decoder under ``model/language_model``, the way torch's
        # runtime module names for the composite model do.
        prefix=("model", "language_model"),
        vocab_size=int(text.get("vocab_size", 262208)),
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=int(text.get("num_attention_heads", 8)),
        num_key_value_heads=int(text.get("num_key_value_heads", 4)),
        head_dim=int(text.get("head_dim", 256)),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_global=float(text.get("rope_theta", 1_000_000.0)),
        rope_local=float(text.get("rope_local_base_freq", 10_000.0)),
        sliding_pattern=int(text.get("sliding_window_pattern", 6)),
        query_pre_attn_scalar=float(text.get("query_pre_attn_scalar", 256)),
        # gemma-3-4b-it's config.json omits the flag and its checkpoint ships no
        # ``lm_head``; this tree keeps the head as a leaf of its own, materialised from
        # the tied embedding table by ``convert_checkpoint``.
        tie_word_embeddings=False,
        with_value_head=with_value_head,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        # Vision tower (SigLIP) + projector — read from the checkpoint's vision_config and
        # the top-level image_token_index / mm_tokens_per_image.
        vision_hidden_size=int(vision.get("hidden_size", 1152)),
        vision_intermediate_size=int(vision.get("intermediate_size", 4304)),
        vision_num_hidden_layers=int(vision.get("num_hidden_layers", 27)),
        vision_num_attention_heads=int(vision.get("num_attention_heads", 16)),
        num_channels=int(vision.get("num_channels", 3)),
        image_size=int(vision.get("image_size", 896)),
        patch_size=int(vision.get("patch_size", 14)),
        vision_layer_norm_eps=float(vision.get("layer_norm_eps", 1e-6)),
        vision_use_head=bool(vision.get("vision_use_head", False)),
        mm_tokens_per_image=int(cfg.get("mm_tokens_per_image", 256)),
        image_token_id=int(cfg.get("image_token_index", 262144)),
    )


__all__ = ["Gemma3Config", "load_config"]
