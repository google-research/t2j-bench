# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone LLaVA-Next model implementation - decoupled from HuggingFace transformers.
Only depends on PyTorch.
Contains: CLIP Vision Encoder, Llama Language Model, LlavaNext Multi-Modal Model.
"""
import glob
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    ACT2FN, BaseModelOutput, BaseModelOutputWithPast,
    BaseModelOutputWithPooling, CausalLMOutputWithPast, ModelOutput,
)

logger = logging.getLogger(__name__)


# ===================== Output Dataclasses =====================

@dataclass
class LlavaNextCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


# ===================== PreTrainedModel Base =====================

class PreTrainedModel(nn.Module):
    """Minimal base class for pretrained models."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self._device = torch.device("cpu")

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        """Load pretrained model from directory."""
        if config is None:
            from .configuration import load_config
            config = load_config(model_path)

        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)

        model = cls(config)

        # Load weights
        weight_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if weight_files:
            try:
                from safetensors.torch import load_file
                state_dict = {}
                for wf in weight_files:
                    state_dict.update(load_file(wf))
            except ImportError:
                logger.warning("safetensors not available, trying .bin files")
                weight_files = []

        if not weight_files:
            weight_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
            state_dict = {}
            for wf in weight_files:
                state_dict.update(torch.load(wf, map_location="cpu", weights_only=True))

        if state_dict:
            # Map old-style keys to new structure
            mapped_state_dict = cls._remap_keys(state_dict)
            missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
            if missing:
                logger.info(f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                logger.info(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
            logger.info(f"Loaded {len(state_dict)} weight tensors from {model_path}")
        else:
            logger.warning(f"No weight files found in {model_path}")

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        if device_map == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

        return model

    @classmethod
    def _remap_keys(cls, state_dict):
        """Remap weight keys from HF checkpoint format to our format."""
        new_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Map: language_model.model.X -> model.language_model.X
            if new_key.startswith("language_model.model."):
                new_key = "model.language_model." + new_key[len("language_model.model."):]
            elif new_key.startswith("language_model.lm_head."):
                new_key = "lm_head." + new_key[len("language_model.lm_head."):]
            elif new_key.startswith("vision_tower."):
                new_key = "model." + new_key
            elif new_key.startswith("multi_modal_projector."):
                new_key = "model." + new_key
            elif new_key.startswith("image_newline"):
                new_key = "model." + new_key
            new_dict[new_key] = value
        return new_dict

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        if self.config:
            config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
            with open(os.path.join(save_directory, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
        logger.info(f"Model saved to {save_directory}")

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True


# ===================== CLIP Vision Encoder =====================

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False):
        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            hidden_states, output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class CLIPVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True, **kwargs):
        return self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ===================== Llama Language Model =====================

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    bsz, num_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.dim() == 2:
                # Expand 2D mask to 4D
                causal_mask = causal_mask[:, None, None, :]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


class LlamaModel(nn.Module):
    """Llama language model (without LM head)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device).unsqueeze(0)

        # Create causal mask
        if attention_mask is None:
            causal_mask = None
        elif attention_mask.dim() == 2:
            # Create 4D causal mask from 2D
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            total_len = past_len + seq_len
            causal_mask = torch.full((seq_len, total_len), torch.finfo(inputs_embeds.dtype).min,
                                     device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if seq_len > 1:
                causal_mask = torch.triu(causal_mask, diagonal=past_len + 1)
            else:
                causal_mask.fill_(0)
            # Apply padding mask
            pad_mask = (attention_mask == 0)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, total_len).clone()
            causal_mask.masked_fill_(pad_mask[:, None, None, :total_len], torch.finfo(inputs_embeds.dtype).min)
        else:
            causal_mask = attention_mask

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_kv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None
            hidden_states, present_kv = layer(
                hidden_states, attention_mask=causal_mask,
                position_ids=position_ids, past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                next_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
        )


# ===================== LLaVA-Next Multi-Modal =====================

def select_best_resolution(original_size, possible_resolutions):
    """Select the best resolution from candidates based on input image size."""
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """Calculate image patch grid shape for any-resolution images."""
    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size):
    """Calculate number of patches for an image."""
    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    num_patches += 1  # base patch
    return num_patches


def unpad_image(tensor, original_size):
    """Unpads a PyTorch tensor of a padded and resized image."""
    if not isinstance(original_size, (list, tuple)):
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaNextMultiModalProjector(nn.Module):
    """Two-layer MLP projector from vision to text space."""
    def __init__(self, config):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaNextModel(nn.Module):
    """LLaVA-Next model: vision tower + projector + language model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(config)

        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size) * embed_std)

        self.language_model = LlamaModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values, image_sizes, vision_feature_layer=None, vision_feature_select_strategy=None):
        """Extract and project image features."""
        vision_feature_layer = vision_feature_layer or self.config.vision_feature_layer
        vision_feature_select_strategy = vision_feature_select_strategy or self.config.vision_feature_select_strategy

        # Calculate patches per image
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        # Flatten batch of images
        if pixel_values.dim() == 5:
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)

        # Forward through vision tower
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)

        # Select feature layer
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[idx] for idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        # Remove CLS token for "default" strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]

        # Project to text space
        image_features = self.multi_modal_projector(selected_image_feature)

        # Split by image
        image_features = torch.split(image_features, image_num_patches, dim=0)

        # Pack with spatial unpadding
        image_features, feature_lens = self.pack_image_features(
            image_features, image_sizes, vision_feature_select_strategy, self.image_newline,
        )
        return image_features, feature_lens

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
        """Reshape, unpad and pack image features."""
        new_image_features = []
        feature_lens = []

        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]

                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx], self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])

                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )

                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)

            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))

        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features_flat = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features, feature_lens = self.get_image_features(pixel_values, image_sizes)
            image_features_flat = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # Replace image tokens with features
            special_image_mask = (input_ids == self.config.image_token_index)
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features_flat)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


class LlavaNextForConditionalGeneration(PreTrainedModel):
    """LLaVA-Next for conditional generation (with LM head)."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaNextModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.generation_config = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.config.text_config, "eos_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.config.text_config, "pad_token_id", eos_token_id)
        generated = input_ids
        current_attention_mask = attention_mask
        static_kwargs = dict(kwargs)
        if pixel_values is not None:
            static_kwargs["pixel_values"] = pixel_values
        if image_sizes is not None:
            static_kwargs["image_sizes"] = image_sizes

        for _ in range(max_new_tokens):
            outputs = self(
                input_ids=generated,
                attention_mask=current_attention_mask,
                use_cache=False,
                **static_kwargs,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(next_token, dtype=current_attention_mask.dtype)],
                dim=1,
            )
            if eos_token_id is not None and torch.all(next_token.eq(eos_token_id)):
                break

        if pad_token_id is not None and eos_token_id is not None and generated.shape[1] == input_ids.shape[1]:
            pad = torch.full((generated.shape[0], 1), pad_token_id, dtype=generated.dtype, device=generated.device)
            generated = torch.cat([generated, pad], dim=1)
        return generated

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
