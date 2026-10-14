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

"""Standalone Gemma3 model implementation - decoupled from HuggingFace transformers."""
import glob
import json
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ACT2FN, BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast, ModelOutput

logger = logging.getLogger(__name__)



@dataclass
class Gemma3CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    image_hidden_states: Optional[torch.FloatTensor] = None



class DynamicCache:
    """Simple dynamic KV cache for autoregressive generation."""

    def __init__(self):
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @property
    def is_initialized(self):
        return len(self.key_cache) > 0



def create_causal_mask(
    input_shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """Create a causal attention mask."""
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond[:, None] >= mask_cond[None, :], 0)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    if sliding_window is not None:
        total_len = tgt_len + past_key_values_length
        for i in range(tgt_len):
            abs_pos = i + past_key_values_length
            start = max(0, abs_pos - sliding_window + 1)
            if start > 0:
                mask[i, :start] = torch.finfo(dtype).min

    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)



class PreTrainedModel(nn.Module):
    """Minimal base class for pretrained models."""

    config_class = None
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass

    def _init_weights(self, module):
        pass

    def post_init(self):
        """Apply weight initialization."""
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        """Load a pretrained model from a checkpoint directory."""
        from .configuration import load_config

        model_path = pretrained_model_name_or_path
        device_map = kwargs.pop("device_map", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype_alias = kwargs.pop("dtype", None)
        if torch_dtype is None:
            torch_dtype = dtype_alias
        report_loading_info = kwargs.pop("output_loading_info", False)

        if config is None:
            config = load_config(model_path)

        model = cls(config)

        weight_files = _find_weight_files(model_path)
        if weight_files:
            missing = _load_weights_into_model(model, weight_files, dtype=torch_dtype)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        if device_map == "auto":
            if torch.cuda.is_available():
                model = model.to("cuda")
        elif device_map is not None and device_map != "cpu":
            model = model.to(device_map)

        if report_loading_info:
            return model, {"missing_keys": missing, "unexpected_keys": [], "mismatched_keys": []}
        return model

    def save_pretrained(self, save_directory):
        """Save model weights and config."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        if hasattr(self, "config") and self.config is not None:
            config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
            with open(os.path.join(save_directory, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)


def _find_weight_files(model_path):
    """Find weight files in model directory."""
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if safetensors_files:
        return safetensors_files

    bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
    if bin_files:
        return bin_files

    return []


def _remap_checkpoint_key(key: str) -> str:
    """Checkpoint name -> module name for the transformers>=5 multimodal layout."""
    for prefix, replacement in (
        ("language_model.model.", "model.language_model."),
        ("language_model.lm_head.", "lm_head."),
        ("vision_tower.", "model.vision_tower."),
        ("multi_modal_projector.", "model.multi_modal_projector."),
    ):
        if key.startswith(prefix):
            return replacement + key[len(prefix):]
    return key


def _load_weights_into_model(model, weight_files, dtype=None):
    """Load weights from files into model."""
    state_dict = {}
    for wf in weight_files:
        if wf.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                loaded = load_file(wf)
            except ImportError:
                logger.warning("safetensors not available, skipping %s", wf)
                continue
        else:
            loaded = torch.load(wf, map_location="cpu", weights_only=True)
        state_dict.update(loaded)

    model_state = model.state_dict()
    mapped_state = {}
    for key, value in state_dict.items():
        if dtype is not None:
            value = value.to(dtype=dtype)
        if key in model_state:
            mapped_state[key] = value
            continue
        new_key = _remap_checkpoint_key(key)
        if new_key in model_state:
            mapped_state[new_key] = value

    missing = [key for key in model_state if key not in mapped_state]
    model.load_state_dict(mapped_state, strict=False)
    if _tie_output_head(model, missing):
        missing.remove("lm_head.weight")
    if missing:
        logger.info(f"Missing keys ({len(missing)}): {missing[:10]}...")
    return missing


def _tie_output_head(model, missing) -> bool:
    """Point ``lm_head`` at the input embedding when the config says they are tied."""
    if "lm_head.weight" not in (missing or ()):
        return False
    config = getattr(model, "config", None)
    if not any(getattr(candidate, "tie_word_embeddings", False)
               for candidate in (getattr(config, "text_config", None), config)):
        return False
    head = getattr(model, "lm_head", None)
    embeddings = model.get_input_embeddings()
    if head is None or embeddings is None:
        return False
    if tuple(head.weight.shape) != tuple(embeddings.weight.shape):
        logger.warning("lm_head %s does not match embeddings %s; leaving it untied",
                       tuple(head.weight.shape), tuple(embeddings.weight.shape))
        return False
    head.weight = embeddings.weight
    return True



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_patches).unsqueeze(0), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=None)


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_head = getattr(config, "vision_use_head", True)
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(self, pixel_values, **kwargs):
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = self.head(last_hidden_state) if self.use_head else None
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values, **kwargs):
        return self.vision_model(pixel_values, **kwargs)



class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings

        layer_types = list(set(config.layer_types))

        rope_params = getattr(config, "rope_parameters", None)

        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                theta = 10000.0
            else:
                theta = 1000000.0

            if rope_params and isinstance(rope_params, dict):
                if layer_type in rope_params and isinstance(rope_params[layer_type], dict):
                    theta = rope_params[layer_type].get("rope_theta", theta)

            dim = config.head_dim
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Gemma3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.attention_dropout = config.attention_dropout
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx,
            embed_scale=config.hidden_size ** 0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        bsz, seq_len = inputs_embeds.shape[:2]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if not isinstance(attention_mask, dict):
            causal_mask_mapping = {}
            for layer_type in set(self.config.layer_types):
                sw = self.config.sliding_window if layer_type == "sliding_attention" else None
                mask = create_causal_mask(
                    (bsz, seq_len), inputs_embeds.dtype, inputs_embeds.device,
                    past_key_values_length=past_key_values_length,
                    sliding_window=sw,
                )
                if attention_mask is not None and not isinstance(attention_mask, dict):
                    padding_mask = attention_mask[:, None, None, :mask.shape[-1]]
                    padding_mask = (1.0 - padding_mask.to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
                    mask = mask + padding_mask
                causal_mask_mapping[layer_type] = mask
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer.attention_type],
                position_embeddings=position_embeddings[layer.attention_type],
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image ** 0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, hidden_size = vision_outputs.shape

        reshaped = vision_outputs.transpose(1, 2)
        reshaped = reshaped.reshape(batch_size, hidden_size, self.patches_per_image, self.patches_per_image)
        reshaped = reshaped.contiguous()

        pooled = self.avg_pool(reshaped)
        pooled = pooled.flatten(2).transpose(1, 2)

        normed = self.mm_soft_emb_norm(pooled)
        projected = torch.matmul(normed, self.mm_input_projection_weight)
        return projected.type_as(vision_outputs)



class Gemma3Model(nn.Module):
    """Gemma3 vision+language model (without LM head)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = Gemma3TextModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values):
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        last_hidden_state = vision_outputs.last_hidden_state
        image_features = self.multi_modal_projector(last_hidden_state)
        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        image_token_id = self.config.image_token_index

        if input_ids is not None and image_token_id >= self.vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            if input_ids is not None:
                special_image_mask = (input_ids == image_token_id)
            else:
                img_embed = self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = (inputs_embeds == img_embed).all(-1)

            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


class Gemma3ForConditionalGeneration(PreTrainedModel):
    """Full Gemma3 vision-language model with LM head."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Gemma3Model(config)
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
        token_type_ids: Optional[torch.LongTensor] = None,
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
        current_token_type_ids = token_type_ids
        static_kwargs = dict(kwargs)

        for _ in range(max_new_tokens):
            outputs = self(
                input_ids=generated,
                attention_mask=current_attention_mask,
                token_type_ids=current_token_type_ids,
                use_cache=False,
                **static_kwargs,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(next_token, dtype=current_attention_mask.dtype)],
                dim=1,
            )
            if current_token_type_ids is not None:
                current_token_type_ids = torch.cat([current_token_type_ids, current_token_type_ids[:, -1:]], dim=1)
            if eos_token_id is not None and torch.all(next_token.eq(eos_token_id)):
                break

        if pad_token_id is not None and eos_token_id is not None and generated.shape[1] == input_ids.shape[1]:
            pad = torch.full((generated.shape[0], 1), pad_token_id, dtype=generated.dtype, device=generated.device)
            generated = torch.cat([generated, pad], dim=1)
        return generated

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits_float = logits.float()
            shift_logits = logits_float[..., :-1, :]
            shift_labels = labels[..., 1:]

            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:]
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask[:, -shift_labels.shape[1]:] != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.text_config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
