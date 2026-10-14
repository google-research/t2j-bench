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

"""Standalone Qwen3-VL model implementation - no transformers dependency."""

import json
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from .utils import ACT2FN

logger = logging.getLogger(__name__)


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


@dataclass
class BaseModelOutputWithPooling:
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


@dataclass
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    deepstack_features: Optional[list[torch.FloatTensor]] = None


@dataclass
class Qwen3VLModelOutputWithPast:
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class Qwen3VLCausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class DynamicCache:
    """Minimal KV cache for autoregressive decoding."""

    def __init__(self, **kwargs):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._seen_tokens = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs: Optional[dict] = None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def create_causal_mask(
    config, input_embeds, attention_mask=None, cache_position=None,
    past_key_values=None, position_ids=None,
):
    """Create a causal attention mask."""
    batch_size, seq_length = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device

    past_length = 0
    if past_key_values is not None:
        past_length = past_key_values.get_seq_length()

    total_length = past_length + seq_length

    causal_mask = torch.full((seq_length, total_length), torch.finfo(dtype).min, dtype=dtype, device=device)
    if seq_length > 1:
        causal_mask = torch.triu(causal_mask, diagonal=past_length + 1)
    else:
        causal_mask.fill_(0)

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

    if attention_mask is not None and attention_mask.dim() == 2:
        expanded_mask = attention_mask[:, None, None, :total_length].to(dtype)
        expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        if expanded_mask.shape[-1] < total_length:
            pad_size = total_length - expanded_mask.shape[-1]
            expanded_mask = F.pad(expanded_mask, (pad_size, 0), value=0)
        causal_mask = causal_mask + expanded_mask
    elif attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask

    return causal_mask



class PreTrainedModel(nn.Module):
    """Minimal PreTrainedModel base class for weight loading."""

    config_class = None
    base_model_prefix = ""
    _no_split_modules = []

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        from .ppo_compat import GenerationConfig
        self.generation_config = GenerationConfig()

    def post_init(self):
        """Initialize weights (no-op for loading from pretrained)."""
        pass

    @classmethod
    def _from_config(cls, config):
        """Create model from config without loading weights."""
        return cls(config)

    def get_input_embeddings(self):
        if hasattr(self, "embed_tokens"):
            return self.embed_tokens
        return None

    def set_input_embeddings(self, value):
        if hasattr(self, "embed_tokens"):
            self.embed_tokens = value

    @property
    def dtype(self):
        """Get the dtype of the first parameter."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _init_weights(self, module):
        """Default weight initialization."""
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load model from pretrained weights (memory-efficient)."""
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype_alias = kwargs.pop("dtype", None)
        if torch_dtype is None:
            torch_dtype = dtype_alias
        report_loading_info = kwargs.pop("output_loading_info", False)
        cache_dir = kwargs.pop("cache_dir", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)

        path = pretrained_model_name_or_path
        if not os.path.isdir(path):
            path = os.path.join("/data", pretrained_model_name_or_path)
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"Model {pretrained_model_name_or_path!r} not found at {path}. "
                    "Weights are baked into the image at build time; nothing is "
                    "fetched at run time."
                )

        config = kwargs.pop("config", None)
        if config is None:
            config = Qwen3VLConfig.from_pretrained(path)

        resolved_dtype = None
        if torch_dtype is not None:
            if torch_dtype == "auto":
                cfg_dtype = getattr(config, "torch_dtype", None)
                if isinstance(cfg_dtype, str):
                    resolved_dtype = getattr(torch, cfg_dtype, torch.float32)
                elif isinstance(cfg_dtype, torch.dtype):
                    resolved_dtype = cfg_dtype
                else:
                    resolved_dtype = torch.float32
            else:
                resolved_dtype = torch_dtype

        if low_cpu_mem_usage:
            with torch.device("meta"):
                model = cls(config)
        else:
            model = cls(config)
            if resolved_dtype is not None:
                model = model.to(resolved_dtype)

        missing = _load_state_dict_into_model(model, path, target_dtype=resolved_dtype)

        if device_map is not None:
            if device_map == "auto":
                if torch.cuda.is_available():
                    model = model.cuda()
            elif isinstance(device_map, str):
                model = model.to(device_map)

        model.eval()
        if report_loading_info:
            return model, {"missing_keys": missing, "unexpected_keys": [], "mismatched_keys": []}
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model weights and config."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        generation_config=None,
        max_new_tokens=None,
        **kwargs,
    ):
        """Simple autoregressive generation with sampling support."""
        try:
            from .ppo_compat import GenerationConfig as GC
        except (ImportError, ModuleNotFoundError):
            from dataclasses import dataclass as _dc

            @_dc
            class GC:
                do_sample: bool = False
                temperature: float = 1.0
                top_p: float = 1.0
                top_k: int = 50
                max_new_tokens: int = 256
                eos_token_id: int = None
                pad_token_id: int = None

        if generation_config is None:
            generation_config = getattr(self, "generation_config", GC())

        max_new = max_new_tokens or getattr(generation_config, "max_new_tokens", 256) or 256
        do_sample = getattr(generation_config, "do_sample", False)
        temperature = getattr(generation_config, "temperature", 1.0)
        top_p = getattr(generation_config, "top_p", 1.0)
        top_k = getattr(generation_config, "top_k", 50)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        elif eos_token_id is None:
            eos_token_id = []

        device = input_ids.device
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        first_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"generation_config", "position_ids", "rope_deltas"}
        }

        pixel_kwargs = {}
        grid_kwargs = {}
        for k, v in first_kwargs.items():
            if k in ("pixel_values", "pixel_values_videos"):
                pixel_kwargs[k] = v
            elif k in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                grid_kwargs[k] = v

        if hasattr(self, "model") and hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, generated.shape[1], device=device)
        cur_mask = attention_mask.clone()

        for step in range(max_new):
            if step == 0:
                outputs = self(
                    input_ids=generated, attention_mask=cur_mask,
                    use_cache=False, **first_kwargs,
                )
            else:
                outputs = self(
                    input_ids=generated, attention_mask=cur_mask,
                    use_cache=False, **grid_kwargs,
                )

            next_token_logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = next_token_logits.argmax(dim=-1)

            if pad_token_id is not None:
                next_tokens[finished] = pad_token_id

            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            cur_mask = torch.cat([cur_mask, torch.ones(batch_size, 1, device=device)], dim=-1)

            for eos_id in eos_token_id:
                finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        return generated


def _get_weight_files(model_path: str):
    """Return list of weight files to load, in order."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    single_safetensors = os.path.join(model_path, "model.safetensors")
    single_bin = os.path.join(model_path, "pytorch_model.bin")
    index_bin = os.path.join(model_path, "pytorch_model.bin.index.json")

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [os.path.join(model_path, f) for f in files], "safetensors"
    elif os.path.exists(single_safetensors):
        return [single_safetensors], "safetensors"
    elif os.path.exists(index_bin):
        with open(index_bin, "r") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [os.path.join(model_path, f) for f in files], "bin"
    elif os.path.exists(single_bin):
        return [single_bin], "bin"
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")


def _load_state_dict_into_model(model: nn.Module, model_path: str, target_dtype=None):
    """Load weights shard-by-shard directly into the model to minimize peak memory."""
    files, fmt = _get_weight_files(model_path)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set()

    for fpath in files:
        if fmt == "safetensors":
            from safetensors.torch import load_file
            shard = load_file(fpath, device="cpu")
        else:
            shard = torch.load(fpath, map_location="cpu", weights_only=True)

        if target_dtype is not None:
            shard = {k: v.to(target_dtype) for k, v in shard.items()}

        missing, unexpected = model.load_state_dict(shard, strict=False, assign=True)
        loaded_keys.update(k for k in shard.keys() if k in model_keys)

        del shard

    not_loaded = model_keys - loaded_keys

    tied_map = getattr(model, "_tied_weights_keys", None)
    if tied_map and isinstance(tied_map, dict):
        full_sd = model.state_dict()
        for dst_key, src_key in tied_map.items():
            if dst_key in not_loaded and src_key in full_sd:
                parts = dst_key.rsplit(".", 1)
                parent = model
                if len(parts) == 2:
                    for attr in parts[0].split("."):
                        parent = getattr(parent, attr)
                    attr_name = parts[1]
                else:
                    attr_name = parts[0]
                src_tensor = full_sd[src_key]
                setattr(parent, attr_name, nn.Parameter(src_tensor, requires_grad=True))
                not_loaded.discard(dst_key)
                loaded_keys.add(dst_key)

    missing = sorted(not_loaded)
    if missing:
        logger.warning(f"Missing keys after loading: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    dtype = target_dtype or torch.float32
    for name, module in model.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            if param.device == torch.device("meta"):
                new_param = nn.Parameter(
                    torch.empty(param.shape, dtype=dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
                setattr(module, pname, new_param)
        for bname, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.device == torch.device("meta"):
                new_buf = torch.empty(buf.shape, dtype=buf.dtype, device="cpu")
                setattr(module, bname, new_buf)

    return missing



class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = self._build_inv_freq(device=None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self, device) -> torch.Tensor:
        base = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        return 1.0 / (self.theta ** base)

    def _get_inv_freq(self, device) -> torch.Tensor:
        inv_freq = getattr(self, "inv_freq", None)
        needs_refresh = (
            inv_freq is None
            or inv_freq.device != device
            or inv_freq.numel() != (self.dim + 1) // 2
            or inv_freq.dtype != torch.float32
            or not torch.isfinite(inv_freq).all()
        )
        if needs_refresh:
            inv_freq = self._build_inv_freq(device=device)
            self.inv_freq = inv_freq
        return inv_freq

    def forward(self, seqlen: int) -> torch.Tensor:
        inv_freq = self._get_inv_freq(self.inv_freq.device)
        seq = torch.arange(seqlen, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(seq, inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x



def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(query, key, value, attention_mask, scaling, dropout=0.0,
                            num_key_value_groups=1, training=False):
    """Eager (manual) attention implementation."""
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = []
        for q, k, v in zip(*splits):
            out, _ = eager_attention_forward(
                q, k, v, attention_mask=None, scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                num_key_value_groups=self.num_key_value_groups, training=self.training,
            )
            attn_outputs.append(out)
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states



class Qwen3VLVisionModel(PreTrainedModel):
    config_class = Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config: Qwen3VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
             for _ in range(len(config.deepstack_visual_indexes))]
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor,
                **kwargs) -> BaseModelOutputWithDeepstackFeatures:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )



class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_params = config.rope_parameters
        self.base = rope_params.get("rope_theta", 1000000.0)
        self.dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.attention_scaling = 1.0
        inv_freq = self._build_inv_freq(device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = rope_params.get("mrope_section", [24, 20, 20])

    def _build_inv_freq(self, device) -> torch.Tensor:
        dim_range = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
        return 1.0 / (self.base ** (dim_range / self.dim))

    def _get_inv_freq(self, device) -> torch.Tensor:
        expected_size = (self.dim + 1) // 2
        inv_freq = getattr(self, "inv_freq", None)
        needs_refresh = (
            inv_freq is None
            or inv_freq.device != device
            or inv_freq.numel() != expected_size
            or inv_freq.dtype != torch.float32
            or not torch.isfinite(inv_freq).all()
        )
        if needs_refresh:
            inv_freq = self._build_inv_freq(device=device)
            self.inv_freq = inv_freq
        original_inv_freq = getattr(self, "original_inv_freq", None)
        if (
            original_inv_freq is None
            or original_inv_freq.device != device
            or original_inv_freq.numel() != expected_size
            or original_inv_freq.dtype != torch.float32
            or not torch.isfinite(original_inv_freq).all()
        ):
            self.original_inv_freq = inv_freq.clone()
        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self._get_inv_freq(x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = eager_attention_forward(
            query_states, key_states, value_states, attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            num_key_value_groups=self.num_key_value_groups, training=self.training,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3VLTextMLP(nn.Module):
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


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                cache_position=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            cache_position=cache_position, position_embeddings=position_embeddings, **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



class Qwen3VLTextModel(PreTrainedModel):
    config_class = Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                cache_position=None, visual_pos_masks=None,
                deepstack_visual_embeds=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config, input_embeds=inputs_embeds,
            attention_mask=attention_mask, cache_position=cache_position,
            past_key_values=past_key_values, position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=attention_mask,
                position_ids=text_position_ids, past_key_values=past_key_values,
                cache_position=cache_position, position_embeddings=position_embeddings, **kwargs,
            )
            hidden_states = layer_outputs

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states



class Qwen3VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen3VLConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    supports_gradient_checkpointing = True



class Qwen3VLModel(Qwen3VLPreTrainedModel):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(self, input_ids=None, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None):
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, input_ids_i in enumerate(total_input_ids):
                input_ids_i = input_ids_i[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain_images > 0) else len(input_tokens) + 1
                    ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens and remain_videos > 0) else len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t.item()
                    llm_grid_h = h.item() // spatial_merge_size
                    llm_grid_w = w.item() // spatial_merge_size
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
            return position_ids, mrope_position_deltas


    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            img_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))
            special_image_mask = (inputs_embeds == img_token_embed).all(-1)
            vid_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device))
            special_video_mask = (inputs_embeds == vid_token_embed).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            assert inputs_embeds[special_image_mask].numel() == image_features.numel(), \
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}"

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            assert inputs_embeds[special_video_mask].numel() == video_features.numel(), \
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}"
        return special_image_mask, special_video_mask

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None,
                pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None,
                cache_position=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask)
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            cache_position=cache_position, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds, **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        return self.model.get_image_features(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for autoregressive generation (required by PEFT)."""
        model_inputs = {"input_ids": input_ids}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if input_ids.shape[1] > 1:
                model_inputs["input_ids"] = input_ids[:, -1:]
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
        else:
            model_inputs["attention_mask"] = attention_mask

        for key in ["pixel_values", "pixel_values_videos", "image_grid_thw",
                     "video_grid_thw", "position_ids", "rope_deltas",
                     "cache_position", "second_per_grid_ts"]:
            if key in kwargs:
                model_inputs[key] = kwargs[key]

        return model_inputs

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None,
                cache_position=None, logits_to_keep=0, **kwargs):

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            last_hidden_state=hidden_states,
        )



class AutoModelForImageTextToText:
    """Simple registry that loads Qwen3VLForConditionalGeneration."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return Qwen3VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, **kwargs)
