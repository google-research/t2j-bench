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

"""PaliGemmaForConditionalGeneration configuration (mirror of torch_orig/transformers_impl/configuration.py).

The graded forward is the **language-model path** (TEST_PLAN.md §6.5): a causal
decoder nested under the LM prefix ``model/language_model``. Sizes come from the
checkpoint's own ``config.json``; the switches (attention bias, qk-norm, gemma-family
gelu/(1+w)/embed-scale) are properties of the family rather than of the file.
``convert_checkpoint`` converts the FULL checkpoint (vision/audio towers included) so the
spec test matches; the module only reads the LM leaves. JAX extras: ``dtype``/``with_value_head``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass(frozen=True)
class PaliGemmaConfig:
    """PaliGemmaForConditionalGeneration LM-path config (canonical VL fields, read from the checkpoint's config.json)."""

    prefix: tuple           # e.g. ("model", "language_model")
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    attn_bias: bool = False       # q/k/v projection bias (qwen2-family)
    o_bias: bool = False          # output projection bias
    qk_norm: bool = False
    mlp_act: str = "silu"        # "silu" | "gelu"
    norm_add_one: bool = False   # gemma (1+weight) RMSNorm
    embed_scale: bool = False    # gemma: hidden *= sqrt(hidden_size)
    causal: bool = True          # paligemma is a prefix-LM: bidirectional over the text
    with_value_head: bool = False
    dtype: Any = jnp.bfloat16
    lora_rank: int = 0   # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)
    # ------------------------------------------------------------------ vision path
    # SigLIP vision tower (Conv2d patchify, learned pos-embed, 27 pre-norm encoder layers,
    # NO CLS token, NO pooling head) + PaliGemma multimodal projector (a single biased
    # Linear -- NO pooling / pixel-shuffle / RMSNorm / sqrt-normalization) that turns the
    # ``<image>`` placeholders (image_token_id) into text-hidden embeddings (mirror of torch
    # ``SiglipVisionTransformer`` + ``PaliGemmaMultiModalProjector``). Read from the
    # checkpoint's vision_config; fallbacks are paligemma-3b-pt-224's SiglipVisionConfig.
    # UNLIKE gemma3, the graded forward leaves the image tokens attended, so this path runs
    # on the goldens.
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_hidden_layers: int = 27
    vision_num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    vision_layer_norm_eps: float = 1e-6
    projection_dim: int = 2048
    image_token_id: int = 257152


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> PaliGemmaConfig:
    """Build a ``PaliGemmaConfig`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``PaliGemmaConfig.from_pretrained``). The decoder this repo grades is the
    ``text_config`` half of a composite VL config — for paligemma-3b-pt-224 a Gemma 1 2B —
    and every field it leaves out falls back to the ``GemmaConfig`` default, exactly as
    HF's own loader does. ``compute_dtype`` selects the flax compute dtype, and
    ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in any
    checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    vision = cfg.get("vision_config", {})
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    # PaliGemma's decoder is a Gemma 1: GeGLU MLP, (1 + weight) RMSNorm, and embeddings
    # scaled by sqrt(hidden_size).
    is_gemma = "gemma" in str(text.get("model_type", ""))
    return PaliGemmaConfig(
        # This tree nests the decoder under ``model/language_model``, the way torch's
        # runtime module names for the composite model do.
        prefix=("model", "language_model"),
        vocab_size=int(text["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(text.get("num_key_value_heads") or heads),
        head_dim=int(text.get("head_dim", 256)),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta=float(text.get("rope_theta", 10_000.0)),
        mlp_act="gelu" if is_gemma else "silu",
        norm_add_one=is_gemma,
        embed_scale=is_gemma,
        # PaliGemma is a prefix-LM: attention over the image + prompt is bidirectional.
        causal=str(cfg.get("model_type", "")) != "paligemma",
        # paligemma-3b-pt-224's config.json omits the flag and its checkpoint ships no
        # ``lm_head``; this tree keeps the head as a leaf of its own, materialised from
        # the tied embedding table by ``convert_checkpoint``.
        tie_word_embeddings=False,
        with_value_head=with_value_head,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        # SigLIP vision tower + projector -- read from the checkpoint's vision_config and the
        # top-level image_token_index / projection_dim (torch's PaliGemmaConfig copies the
        # top-level projection_dim onto vision_config).
        vision_hidden_size=int(vision.get("hidden_size", 1152)),
        vision_intermediate_size=int(vision.get("intermediate_size", 4304)),
        vision_num_hidden_layers=int(vision.get("num_hidden_layers", 27)),
        vision_num_attention_heads=int(vision.get("num_attention_heads", 16)),
        num_channels=int(vision.get("num_channels", 3)),
        image_size=int(vision.get("image_size", 224)),
        patch_size=int(vision.get("patch_size", 14)),
        vision_layer_norm_eps=float(vision.get("layer_norm_eps", 1e-6)),
        projection_dim=int(cfg.get("projection_dim", vision.get("projection_dim", 2048))),
        image_token_id=int(cfg.get("image_token_index", 257152)),
    )


__all__ = ["PaliGemmaConfig", "config_from_pretrained"]
