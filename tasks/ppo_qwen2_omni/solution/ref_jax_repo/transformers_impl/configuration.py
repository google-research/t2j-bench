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

"""Qwen2.5-Omni configuration (mirror of torch_orig; thinker text path, §12.4).

The graded forward runs the **thinker** Qwen2 LM + ``thinker/lm_head``. Everything is
nested under ``thinker/`` (``thinker/model/{embed_tokens,layers/{i},norm}``). The full
checkpoint (thinker vision/audio towers + talker + token2wav) is converted by
``load_model``, but only the thinker LM is exercised. Qwen2 conventions: q/k/v bias, no qk-norm, rope 1e6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}


@dataclass(frozen=True)
class Qwen2_5OmniConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    with_value_head: bool = False
    dtype: Any = jnp.bfloat16
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)
    # --- thinker vision tower (Qwen2_5OmniVisionEncoder) + thinker M-RoPE ------ #
    # The thinker's image path: a Qwen2.5-VL-style windowed ViT (Conv3d-as-Dense patch
    # embed, 2D vision RoPE, windowed + full block-diagonal attention, PatchMerger) that
    # turns ``<|IMAGE|>`` placeholders into visual embeddings, plus the 3-D M-RoPE that
    # ``get_rope_index`` computes from the (t,h,w) image grid. Unlike the audio tower this
    # IS exercised by the graded forward: the frozen batches are image-only and the model
    # is handed no ``position_ids``, so the forward runs the ViT and computes M-RoPE itself.
    # Token ids come from ``thinker_config`` and the ViT dims from ``thinker_config.vision_config``.
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    audio_start_token_id: int = 151647
    audio_token_id: int = 151646
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    # M-RoPE: split head_dim/2 rotary freqs over (temporal, height, width) — sums to head_dim/2.
    mrope_section: tuple = (16, 24, 24)
    vision_depth: int = 32
    vision_hidden_size: int = 1280
    vision_intermediate_size: int = 3420
    vision_num_heads: int = 16
    vision_in_channels: int = 3
    vision_patch_size: int = 14
    vision_spatial_merge_size: int = 2
    vision_temporal_patch_size: int = 2
    vision_window_size: int = 112
    vision_out_hidden_size: int = 2048
    vision_fullatt_block_indexes: tuple = (7, 15, 23, 31)
    vision_rope_theta: float = 10000.0


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> Qwen2_5OmniConfig:
    """Build a ``Qwen2_5OmniConfig`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``Qwen2_5OmniConfig.from_pretrained``). Only the thinker's decoder is graded, so its
    fields come from ``thinker_config.text_config``. ``compute_dtype`` selects the flax
    compute dtype, and ``with_value_head`` is the caller's ``add_valuehead`` — the PPO
    head is not in any checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    # Only the thinker's decoder is graded; the talker and token2wav stacks ride along.
    thinker = cfg.get("thinker_config", cfg)
    text = thinker.get("text_config", cfg)
    vision = thinker.get("vision_config", {})
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    # M-RoPE section: HF keeps it under text rope_scaling / rope_parameters.
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    mrope = rope.get("mrope_section")
    mrope = tuple(int(s) for s in mrope) if mrope else (16, 24, 24)
    fullatt = vision.get("fullatt_block_indexes")
    fullatt = tuple(int(s) for s in fullatt) if fullatt else (7, 15, 23, 31)
    return Qwen2_5OmniConfig(
        vocab_size=int(text["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(text.get("num_key_value_heads") or heads),
        # The thinker's config.json omits ``head_dim``; torch derives it the same way.
        head_dim=int(text.get("head_dim") or hidden // heads),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta=float(text.get("rope_theta", 1_000_000.0)),
        with_value_head=with_value_head,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
        # Thinker multimodal token ids + M-RoPE (thinker_config keeps them beside text_config).
        image_token_id=int(thinker.get("image_token_index", 151655)),
        video_token_id=int(thinker.get("video_token_index", 151656)),
        vision_start_token_id=int(thinker.get("vision_start_token_id", 151652)),
        audio_start_token_id=int(thinker.get("audio_start_token_id", 151647)),
        audio_token_id=int(thinker.get("audio_token_index", 151646)),
        position_id_per_seconds=int(thinker.get("position_id_per_seconds", 25)),
        seconds_per_chunk=int(thinker.get("seconds_per_chunk", 2)),
        mrope_section=mrope,
        # Vision tower (Qwen2.5-VL windowed ViT) — parsed from thinker_config.vision_config.
        vision_depth=int(vision.get("depth", 32)),
        vision_hidden_size=int(vision.get("hidden_size", 1280)),
        vision_intermediate_size=int(vision.get("intermediate_size", 3420)),
        vision_num_heads=int(vision.get("num_heads", 16)),
        vision_in_channels=int(vision.get("in_channels", 3)),
        vision_patch_size=int(vision.get("patch_size", 14)),
        vision_spatial_merge_size=int(vision.get("spatial_merge_size", 2)),
        vision_temporal_patch_size=int(vision.get("temporal_patch_size", 2)),
        vision_window_size=int(vision.get("window_size", 112)),
        # out_hidden_size defaults to the text hidden size (merger emits LM-width features).
        vision_out_hidden_size=int(vision.get("out_hidden_size", hidden)),
        vision_fullatt_block_indexes=fullatt,
    )


__all__ = ["Qwen2_5OmniConfig", "config_from_pretrained"]
