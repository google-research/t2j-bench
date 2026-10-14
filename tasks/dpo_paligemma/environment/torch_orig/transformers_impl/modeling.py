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

"""Standalone PaliGemma model implementation - decoupled from HuggingFace transformers."""
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    ACT2FN,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    ModelOutput,
)

logger = logging.getLogger(__name__)



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



def _resolve_attn_implementation(config) -> str:
    """The attention backend for ``config``, defaulting to SDPA."""
    return getattr(config, "_attn_implementation", None) or "sdpa"


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
        attn_implementation = kwargs.pop("attn_implementation", None)

        if config is None:
            config = load_config(model_path)

        resolved_attn = attn_implementation or getattr(config, "_attn_implementation", None) or "sdpa"
        config._attn_implementation = resolved_attn
        for _sub_name in ("text_config", "vision_config"):
            _sub_config = getattr(config, _sub_name, None)
            if _sub_config is not None:
                _sub_config._attn_implementation = resolved_attn

        rng_state = torch.random.get_rng_state()
        model = cls(config)
        torch.random.set_rng_state(rng_state)

        weight_files = _find_weight_files(model_path)
        missing = []
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
    """Load weights from files into model, mapping checkpoint names onto the module tree."""
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
        self.config = config
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

        if _resolve_attn_implementation(self.config) == "sdpa":
            attn_output = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.embed_dim)
            return self.out_proj(attn_output)

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



class GemmaTextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class GemmaRMSNorm(nn.Module):
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


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        base = config.rope_theta
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
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


class GemmaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if _resolve_attn_implementation(self.config) == "sdpa":
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = GemmaTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx,
            embed_scale=config.hidden_size ** 0.5,
        )
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        return self.linear(image_features)



class PaliGemmaModel(PreTrainedModel):
    """PaliGemma vision+language model (without LM head)."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = GemmaTextModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_outputs.pooler_output = image_features
        return image_outputs

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        """Build the multimodal placeholder mask from ``input_ids`` (or ``inputs_embeds``)."""
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask

    def _build_causal_mask(self, inputs_embeds, attention_mask, past_key_values, token_type_ids):
        """Build the 4D additive attention mask."""
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        bsz, seq_len = inputs_embeds.shape[:2]
        min_val = torch.finfo(dtype).min

        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = seq_len + past_seen

        q_pos = torch.arange(past_seen, past_seen + seq_len, device=device)
        kv_pos = torch.arange(kv_len, device=device)
        causal = kv_pos[None, :] <= q_pos[:, None]
        combined = causal[None].expand(bsz, seq_len, kv_len).clone()

        if token_type_ids is not None:
            reversed_tti = 1 - token_type_ids
            is_image = reversed_tti == 1
            is_previous_image = F.pad(is_image, (1, 0), value=0)[:, :-1]
            new_image_start = is_image & ~is_previous_image
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))

            image_block = (
                is_image[:, :, None]
                & is_image[:, None, :]
                & (image_group_ids[:, :, None] == image_group_ids[:, None, :])
            )
            if kv_len != seq_len:
                image_block = F.pad(image_block, (kv_len - seq_len, 0), value=False)
            combined = combined | image_block

        if attention_mask is not None:
            padding = attention_mask.to(torch.bool)
            if padding.shape[-1] < kv_len:
                padding = F.pad(padding, (0, kv_len - padding.shape[-1]), value=False)
            combined = combined & padding[:, None, :]

        additive = torch.where(
            combined,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_val, dtype=dtype, device=device),
        )
        return additive.unsqueeze(1)

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
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0) + 1

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if not isinstance(causal_mask := attention_mask, dict):
            causal_mask = self._build_causal_mask(
                inputs_embeds, attention_mask, past_key_values, token_type_ids
            )

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


class PaliGemmaForConditionalGeneration(PreTrainedModel):
    """Full PaliGemma vision-language model with the LM head."""

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = PaliGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.generation_config = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs

    @staticmethod
    def _loss_function(logits, labels, vocab_size, ignore_index=-100, num_items_in_batch=None, **kwargs):
        logits = logits.float()
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(logits.device)
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

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
        logits_to_keep: int = 0,
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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self._loss_function(
                logits, labels, self.config.text_config.vocab_size,
                num_items_in_batch=kwargs.get("num_items_in_batch"),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


__all__ = ["PaliGemmaForConditionalGeneration", "PaliGemmaPreTrainedModel", "PaliGemmaModel"]

PaliGemmaPreTrainedModel = PreTrainedModel
