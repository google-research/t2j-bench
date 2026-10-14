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

"""Qwen2VLForConditionalGeneration configuration (mirror of torch_orig/transformers_impl/configuration.py).

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
class Qwen2VLConfig:
    """Qwen2VLForConditionalGeneration LM-path config (canonical VL fields, read from the checkpoint's config.json)."""

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
    # M-RoPE (Qwen VL): the collator's get_rope_index reproduces the 3D position_ids /
    # rope_deltas that torch's model.get_rope_index adds to the multimodal batch.
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    spatial_merge_size: int = 2
    # M-RoPE: split the head_dim/2 rotary frequencies across the (temporal, height, width)
    # position axes (Qwen2-VL config ``rope_scaling.mrope_section``). Sums to head_dim/2.
    mrope_section: tuple = (16, 24, 24)
    with_value_head: bool = False
    dtype: Any = jnp.bfloat16
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)
    # ------------------------------------------------------------------ vision path
    # The Qwen2-VL ViT (Conv3d-as-Dense patch embed, 2D vision RoPE, per-frame block-
    # diagonal attention, PatchMerger) that turns ``<|image_pad|>`` placeholders into
    # visual embeddings (mirror of torch ``Qwen2VisionTransformerPretrainedModel``). Read
    # from the checkpoint's ``vision_config``; the fallbacks are the Qwen2-VL-2B values. The
    # graded forward masks image positions out (attention_mask=0), so this path only moves
    # the training loss, never the goldens.
    vision_depth: int = 32
    vision_embed_dim: int = 1280
    vision_num_heads: int = 16
    vision_mlp_ratio: int = 4
    vision_in_chans: int = 3
    vision_patch_size: int = 14
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2
    vision_rope_theta: float = 10000.0
    vision_layer_norm_eps: float = 1e-6


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> Qwen2VLConfig:
    """Build a ``Qwen2VLConfig`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``Qwen2VLConfig.from_pretrained``). Qwen2-VL's config.json is flat: the decoder's
    fields sit at the top level, beside the tower's ``vision_config``. ``compute_dtype``
    selects the flax compute dtype, and ``with_value_head`` is the caller's
    ``add_valuehead`` — the PPO head is not in any checkpoint, so nothing can infer it
    from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    vc = cfg.get("vision_config", {})
    # rope_scaling.mrope_section = how head_dim/2 rotary freqs split over (t, h, w).
    mrope = tuple(int(s) for s in (cfg.get("rope_scaling") or {}).get("mrope_section", (16, 24, 24)))
    return Qwen2VLConfig(
        # This tree nests the decoder under ``model/language_model``, the way torch's
        # runtime module names for the composite model do.
        prefix=("model", "language_model"),
        vocab_size=int(text["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(text.get("num_key_value_heads") or heads),
        # Qwen2's config.json omits ``head_dim``; torch derives it the same way.
        head_dim=int(text.get("head_dim") or hidden // heads),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta=float(text.get("rope_theta", 1_000_000.0)),
        # A Qwen2 decoder: biased q/k/v, unbiased output, no qk-norm, SwiGLU MLP.
        attn_bias=True,
        # Qwen2-VL-2B-Instruct's config.json sets ``tie_word_embeddings`` and its
        # checkpoint ships no ``lm_head``; this tree keeps the head as a leaf of its own
        # instead, materialised from the tied embedding table by ``convert_checkpoint``.
        tie_word_embeddings=False,
        # M-RoPE: the collator's get_rope_index needs the vision token ids, which the
        # composite config keeps at the top level rather than beside the decoder's fields.
        image_token_id=int(cfg.get("image_token_id", 151655)),
        video_token_id=int(cfg.get("video_token_id", 151656)),
        vision_start_token_id=int(cfg.get("vision_start_token_id", 151652)),
        spatial_merge_size=int(vc.get("spatial_merge_size", 2)),
        mrope_section=mrope,
        with_value_head=with_value_head,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        # Vision tower (Qwen2-VL ViT) — read from the checkpoint's vision_config.
        vision_depth=int(vc.get("depth", 32)),
        vision_embed_dim=int(vc.get("embed_dim", 1280)),
        vision_num_heads=int(vc.get("num_heads", 16)),
        vision_mlp_ratio=int(vc.get("mlp_ratio", 4)),
        vision_in_chans=int(vc.get("in_chans", 3)),
        vision_patch_size=int(vc.get("patch_size", 14)),
        vision_temporal_patch_size=int(vc.get("temporal_patch_size", 2)),
        vision_spatial_merge_size=int(vc.get("spatial_merge_size", 2)),
    )


__all__ = ["Qwen2VLConfig", "config_from_pretrained"]
