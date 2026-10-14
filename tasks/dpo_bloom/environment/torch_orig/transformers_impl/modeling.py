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

"""PyTorch BLOOM model."""

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .configuration import BloomConfig


import logging as _logging
from safetensors.torch import load_file


class _WarnOnceLogger:
    """Stdlib logger with the ``warning_once`` method transformers' loggers expose."""

    def __init__(self, name):
        self._logger = _logging.getLogger(name)

    def warning_once(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._logger, item)


logger = _WarnOnceLogger(__name__)


def auto_docstring(obj=None, **kwargs):
    """No-op implementation of ``transformers.utils.auto_docstring``."""
    if obj is None:
        return lambda target: target
    return obj


class Cache:
    """Minimal implementation of ``transformers.cache_utils.Cache``."""

    is_compileable = False


class DynamicCache(Cache):
    def __init__(self, *args, **kwargs):
        self.key_cache = []
        self.value_cache = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states


class StaticCache(Cache):
    is_compileable = True

    def __init__(self, *args, **kwargs):
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self):
        return None


class ModelOutput(OrderedDict):
    """Minimal implementation of ``transformers.utils.ModelOutput``."""

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                super().__setitem__(field.name, value)

    def __setattr__(self, name, value):
        if any(name == field.name for field in fields(self)):
            if value is not None:
                super().__setitem__(name, value)
            elif name in self:
                super().__delitem__(name)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            return dict(self.items())[key]
        return self.to_tuple()[key]

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None
    cross_attentions: object = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None
    cross_attentions: object = None


@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None


@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: object = None
    attentions: object = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: object = None
    attentions: object = None


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing."""

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


def _load_checkpoint_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    """Read the local safetensors checkpoint (single file or sharded index) -> state dict."""

    model_path = Path(str(model_path))
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        with open(index, "r") as handle:
            weight_map = json.load(handle)["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state.update(load_file(str(model_path / shard), device="cpu"))
        return state
    single = model_path / "model.safetensors"
    if single.is_file():
        return load_file(str(single), device="cpu")
    raise FileNotFoundError(f"No safetensors checkpoint under {model_path}")


class PreTrainedModel(nn.Module):
    """Minimal implementation of ``transformers.modeling_utils.PreTrainedModel``."""

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _tied_weights_keys = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def post_init(self):
        self.tie_weights()

    def get_input_embeddings(self):
        base = getattr(self, self.base_model_prefix, self)
        if base is not self:
            return base.get_input_embeddings()
        raise NotImplementedError

    def set_input_embeddings(self, value):
        base = getattr(self, self.base_model_prefix, self)
        if base is not self:
            base.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None

    def tie_weights(self):
        tied = getattr(self, "_tied_weights_keys", None)
        if not tied or not getattr(self.config, "tie_word_embeddings", True):
            return
        for target_name, source_name in tied.items():
            source_param = self.get_parameter(source_name)
            parent_name, _, attr = target_name.rpartition(".")
            parent = self.get_submodule(parent_name) if parent_name else self
            setattr(parent, attr, source_param)

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        is_set = False
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_set = True
        if not is_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing."
            )

    def enable_input_require_grads(self):
        """Make the (frozen) input-embedding output require grad."""

        def make_inputs_require_grads(module, inputs, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=gradient_checkpointing_func
        )
        self.enable_input_require_grads()

    @property
    def loss_function(self):
        return ForCausalLMLoss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        config=None,
        local_files_only=True,
        output_loading_info=False,
        attn_implementation=None,
        dtype=None,
        **kwargs,
    ):
        path = Path(str(pretrained_model_name_or_path))
        if config is None:
            config = cls.config_class.from_pretrained(str(path))
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation

        rng_state = torch.random.get_rng_state()
        model = cls(config)
        torch.random.set_rng_state(rng_state)

        model.tie_weights()
        state_dict = _load_checkpoint_state_dict(path)
        missing, unexpected, mismatched = model._load_pretrained_state_dict(state_dict)
        if dtype is not None:
            model = model.to(dtype)
            model.tie_weights()
        model.eval()

        if output_loading_info:
            return model, {
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "mismatched_keys": mismatched,
                "error_msgs": [],
            }
        return model

    def _load_pretrained_state_dict(self, state_dict):
        """Load ``state_dict``, remapping the base-model prefix like ``from_pretrained``."""
        model_keys = set(self.state_dict().keys())
        prefix = self.base_model_prefix
        expects_prefix = bool(prefix) and any(k.startswith(prefix + ".") for k in model_keys)
        has_prefix = bool(prefix) and any(k.startswith(prefix + ".") for k in state_dict)
        if expects_prefix and not has_prefix:
            remapped = {f"{prefix}.{key}": value for key, value in state_dict.items()}
        elif has_prefix and not expects_prefix:
            plen = len(prefix) + 1
            remapped = {
                (key[plen:] if key.startswith(prefix + ".") else key): value
                for key, value in state_dict.items()
            }
        else:
            remapped = dict(state_dict)

        incompatible = self.load_state_dict(remapped, strict=False)
        tied_targets = set(getattr(self, "_tied_weights_keys", None) or {})
        missing = [key for key in incompatible.missing_keys if key not in tied_targets]
        unexpected = list(incompatible.unexpected_keys)
        return missing, unexpected, []


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """Link to paper: https://huggingface.co/papers/2108.12409 Alibi tensor is not causal as the original paper mentions, it relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value `softmax(l+a) = softmax(l)`."""
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """Dropout add function"""
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """Custom bias GELU function."""
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)"""
    x = x[0]
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


class BloomGelu(nn.Module):
    """Partly copied from Megatron-DeepSpeed code and adapted for our needs"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GeLUFunction.apply(x)


class BloomAttention(nn.Module):
    def __init__(self, config: BloomConfig, layer_idx: int | None = None):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _reshape(self, fused_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the last dimension into (num_heads, head_dim) and reshapes to (bs, heads, len, dim) shape without making any copies, results share same memory storage as `fused_qkv`"""
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        query_layer = fused_qkv[..., 0, :].transpose(1, 2)
        key_layer = fused_qkv[..., 1, :].transpose(1, 2)
        value_layer = fused_qkv[..., 2, :].transpose(1, 2)
        return query_layer, key_layer, value_layer

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads together over the last dimension"""
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        x = x.permute(0, 2, 1, 3)

        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
    ):
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        attention_probs = self.attention_dropout(attention_probs)

        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)

        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        context_layer = self._merge_heads(context_layer)

        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        return output_tensor, attention_probs


class BloomMLP(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


class BloomBlock(GradientCheckpointingLayer):
    def __init__(self, config: BloomConfig, layer_idx: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config, layer_idx)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
    ):

        layernorm_output = self.input_layernorm(hidden_states)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        attention_output, attn_weights = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        layernorm_output = self.post_attention_layernorm(attention_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        output = self.mlp(layernorm_output, residual)

        return output, attn_weights


@auto_docstring
class BloomPreTrainedModel(PreTrainedModel):
    config_class = BloomConfig
    config: BloomConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BloomBlock"]
    _skip_keys_device_placement = "past_key_values"
    _can_compile_fullgraph = True


@auto_docstring
class BloomModel(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.h = nn.ModuleList([BloomBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        self.post_init()

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
        """input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`): `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values.get_seq_length()` (`sequence_length` of input past key value states)."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        batch_size, seq_length, _ = inputs_embeds.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_length_with_past = seq_length + past_length
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + seq_length, device=inputs_embeds.device)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=past_key_values,
                attention_mask=causal_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing."""
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


@auto_docstring(
    custom_intro="""
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class BloomForCausalLM(BloomPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "transformer.word_embeddings.weight"}

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions:
        """input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`): `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values.get_seq_length()` (`sequence_length` of input past key value states)."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = transformer_outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=kwargs.get("num_items_in_batch"),
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Bloom Model transformer with a sequence classification head on top (linear layer).

    [`BloomForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class BloomForSequenceClassification(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast:
        """input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`): `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values.get_seq_length()` (`sequence_length` of input past key value states)."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class BloomForTokenClassification(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = BloomModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        """input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`): `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values.get_seq_length()` (`sequence_length` of input past key value states)."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class BloomForQuestionAnswering(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        end_positions: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | QuestionAnsweringModelOutput:
        """input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`): `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values.get_seq_length()` (`sequence_length` of input past key value states)."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "BloomForCausalLM",
    "BloomModel",
    "BloomPreTrainedModel",
    "BloomForSequenceClassification",
    "BloomForTokenClassification",
    "BloomForQuestionAnswering",
]
