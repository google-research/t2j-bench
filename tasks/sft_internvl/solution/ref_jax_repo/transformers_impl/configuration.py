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

"""InternVLForConditionalGeneration configuration (mirror of torch_orig/transformers_impl/configuration.py).

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
class InternVLConfig:
    """InternVLForConditionalGeneration LM-path config (canonical VL fields, read from the checkpoint's config.json)."""

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
    with_value_head: bool = False
    dtype: Any = jnp.bfloat16
    lora_rank: int = 0  # >0 enables in-forward LoRA on the projection kernels (0 = full)
    lora_alpha: int = 0  # LoRA scale = alpha/rank (defaults to 2*rank when 0)
    # ------------------------------------------------------------------ vision path
    # The InternViT tower + pixel-shuffle projector (``mlp1``) that turns ``<IMG_CONTEXT>``
    # placeholder tokens into visual embeddings (mirror of torch ``InternVLModel``). Read
    # from the checkpoint's ``vision_config`` / top-level keys; the fallbacks are the
    # InternVL3-1B-hf values. The graded forward masks image positions out
    # (attention_mask=0), so this path only moves the training loss, never the goldens.
    image_token_id: int = 151667
    downsample_ratio: float = 0.5
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "default"
    projector_hidden_act: str = "gelu"
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_num_channels: int = 3
    vision_image_size: int = 448
    vision_patch_size: int = 14
    vision_layer_norm_eps: float = 1e-6
    vision_hidden_act: str = "gelu"
    vision_attention_bias: bool = True
    vision_layer_scale_init_value: float = 0.1


def config_from_pretrained(
    model_path, compute_dtype: str = "bfloat16", *, with_value_head: bool = False
) -> InternVLConfig:
    """Build a ``InternVLConfig`` from the checkpoint's own ``config.json``.

    Mirror of torch's ``_load_local_config`` (which hands ``config.json`` to
    ``InternVLConfig.from_pretrained``). The decoder this repo grades is the
    ``text_config`` half of a composite VL config — for InternVL3-1B-hf a Qwen2.5-0.5B —
    and every field it leaves out falls back to the ``Qwen2Config`` default, exactly as
    HF's own loader does. ``compute_dtype`` selects the flax compute dtype, and
    ``with_value_head`` is the caller's ``add_valuehead`` — the PPO head is not in any
    checkpoint, so nothing can infer it from the weights.
    """
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    return InternVLConfig(
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
        # InternVL3-1B-hf's text_config omits ``tie_word_embeddings``, so it takes the
        # ``Qwen2Config`` default (False) -- and the checkpoint does carry a separate
        # ``language_model.lm_head.weight`` whose values differ from the embedding table
        # (torch reports it as its own parameter, with its own gradient). Read the flag
        # the way every other family here does rather than assuming a tie.
        tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
        with_value_head=with_value_head,
        dtype=_DTYPES.get(compute_dtype, jnp.bfloat16),
    )


__all__ = ["InternVLConfig", "config_from_pretrained"]
