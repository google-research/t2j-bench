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

"""PyTorch Mistral model."""

import json
import math
import re
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, fields
from functools import partial, wraps
from pathlib import Path
from typing import Optional, TypedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .configuration import MistralConfig, PreTrainedConfig


import logging as _logging
from safetensors.torch import load_file


class _WarnOnceLogger:
    """Stdlib logger exposing the ``warning_once`` method transformers' loggers provide."""

    def __init__(self, name):
        self._logger = _logging.getLogger(name)

    def warning_once(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._logger, item)


logger = _WarnOnceLogger(__name__)


def _torch_ge(major: int, minor: int) -> bool:
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return True
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


_is_torch_greater_or_equal_than_2_5 = _torch_ge(2, 5)
_is_torch_greater_or_equal_than_2_6 = _torch_ge(2, 6)
_is_torch_greater_or_equal_than_2_8 = _torch_ge(2, 8)
_is_torch_xpu_available = False
_is_torch_npu_available = False


def is_tracing(tensor=None) -> bool:
    """During eager freezing we are never compiling/tracing/exporting, so this is False."""
    return False


def auto_docstring(obj=None, **kwargs):
    """No-op implementation of ``transformers.utils.auto_docstring``."""
    if obj is None:
        return lambda target: target
    return obj


def use_kernel_forward_from_hub(*args, **kwargs):
    def decorator(cls):
        return cls

    return decorator


def use_kernel_func_from_hub(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def use_kernelized_func(module_names):
    """Attach the (already-defined) function(s) as a ``rotary_fn`` attribute on each instance."""
    if callable(module_names):
        module_names = [module_names]

    def decorator(cls):
        orig_init = cls.__init__

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            for fn in module_names:
                setattr(self, "rotary_fn", fn)

        cls.__init__ = new_init
        return cls

    return decorator


class Unpack:
    """Minimal ``typing.Unpack`` implementation: ``Unpack[X]`` evaluates to ``X`` (never introspected)."""

    def __class_getitem__(cls, item):
        return item


class BlockMask:
    """Implementation of flex-attention's ``BlockMask`` (used only in annotations / isinstance)."""


class TransformersKwargs(TypedDict, total=False):
    num_items_in_batch: torch.Tensor | None
    output_hidden_states: bool | None
    output_attentions: bool | None
    output_router_logits: bool | None
    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None
    position_ids: torch.LongTensor | None
    is_causal: bool | None


_dtype = torch.dtype


def maybe_autocast(
    device_type: str,
    dtype: _dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """Autocast only if it is already enabled in this context, or ``enabled=True`` here."""
    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()


class SiLUActivation(nn.Module):
    def forward(self, input):
        return nn.functional.silu(input)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier({"silu": SiLUActivation})


class Cache:
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
    """Minimal implementation of ``transformers.utils.ModelOutput`` (dataclass subclasses, ``None`` fields dropped from the ordered mapping while remaining reachable as attributes)."""

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
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: object = None
    hidden_states: object = None
    attentions: object = None


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


def deprecate_kwarg(old_name, version=None, new_name=None, **_dep_kwargs):
    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if old_name in kwargs and new_name is not None:
                if new_name in kwargs:
                    kwargs.pop(old_name)
                else:
                    kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper


def can_return_tuple(func):
    """Call ``output.to_tuple()`` if ``return_dict=False`` (passed or from config)."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper


def merge_with_config_defaults(func):
    """Use config fields (when present) as defaults for select args (chiefly ``use_cache``)."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args_with_config_defaults = [
            "use_cache",
            "vision_feature_layer",
            "vision_feature_select_strategy",
            "vision_aspect_ratio",
        ]
        for arg_name in args_with_config_defaults:
            arg_index = None
            if arg_name in func.__code__.co_varnames:
                arg_index = func.__code__.co_varnames.index(arg_name) - 1

            if arg_index is not None and len(args) > arg_index and args[arg_index] is not None:
                arg_value = args[arg_index]
            elif kwargs.get(arg_name) is not None:
                arg_value = kwargs[arg_name]
            else:
                arg_value = getattr(self.config, arg_name, None)

            if arg_value is not None:
                if arg_name == "use_cache":
                    if getattr(self, "gradient_checkpointing", False) and self.training and arg_value:
                        logger.warning_once(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                        )
                        arg_value = False
                elif arg_name == "vision_feature_select_strategy":
                    valid_strategies = ["default", "full"]
                    if arg_value not in valid_strategies:
                        raise ValueError(
                            f"`Unexpected select feature strategy: {arg_value}. Please select from {valid_strategies}."
                        )

                if arg_index is not None and len(args) > arg_index:
                    args = list(args)
                    args[arg_index] = arg_value
                    args = tuple(args)
                else:
                    kwargs[arg_name] = arg_value

        is_causal = kwargs.get("is_causal", getattr(self.config, "is_causal", None))
        if is_causal is not None:
            is_causal_in_config = hasattr(self.config, "is_causal")
            if is_causal_in_config:
                is_causal_original_value = self.config.is_causal
            self.config.is_causal = is_causal
            kwargs["is_causal"] = is_causal

        try:
            output = func(self, *args, **kwargs)
        finally:
            if is_causal is not None:
                if is_causal_in_config:
                    self.config.is_causal = is_causal_original_value
                else:
                    del self.config.is_causal

        return output

    return wrapper


def check_model_inputs(func):
    logger.warning_once("The `check_model_inputs` decorator is deprecated in favor of `merge_with_config_defaults`.")
    return merge_with_config_defaults(func)


class GeneralInterface:
    """Dict-like registry with a class-wide global mapping and an instance-local override map."""

    _global_mapping: dict = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __contains__(self, key):
        return key in self._local_mapping or key in self._global_mapping


def dynamic_rope_update(rope_forward):
    """Decorator to (re)compute RoPE frequencies in the forward for dynamic/longrope types."""

    def longrope_frequency_update(self, position_ids, device, layer_type=None):
        seq_len = torch.max(position_ids) + 1

        if layer_type is None:
            rope_type = self.rope_type
            original_inv_freq = self.original_inv_freq
            prefix = ""
            original_max_position_embeddings = self.config.rope_parameters["original_max_position_embeddings"]
        else:
            rope_type = self.rope_type[layer_type]
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"
            original_max_position_embeddings = self.config.rope_parameters[layer_type][
                "original_max_position_embeddings"
            ]

        if seq_len > original_max_position_embeddings:
            if not hasattr(self, f"{layer_type}_long_inv_freq"):
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                long_inv_freq, _ = rope_init_fn(
                    self.config,
                    device,
                    seq_len=original_max_position_embeddings + 1,
                    layer_type=layer_type,
                )
            self.register_buffer(f"{prefix}inv_freq", long_inv_freq, persistent=False)
            setattr(self, f"{prefix}long_inv_freq", long_inv_freq)
        else:
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)

    def dynamic_frequency_update(self, position_ids, device, layer_type=None):
        seq_len = torch.max(position_ids) + 1
        if layer_type is None:
            rope_type = self.rope_type
            max_seq_len_cached = self.max_seq_len_cached
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            rope_type = self.rope_type[layer_type]
            max_seq_len_cached = getattr(self, f"{layer_type}_max_seq_len_cached", self.max_seq_len_cached)
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        if seq_len > max_seq_len_cached:
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(
                self.config,
                device,
                seq_len=seq_len,
                layer_type=layer_type,
            )
            self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_max_seq_len_cached", seq_len)

        if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            setattr(self, f"{layer_type}_max_seq_len_cached", self.original_max_seq_len)

    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
        kwargs = {"layer_type": layer_type} if layer_type is not None else {}
        if "dynamic" in rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device, **kwargs)
        elif rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device, **kwargs)
        return rope_forward(self, x, position_ids, **kwargs)

    return wrapper


def _compute_llama3_parameters(
    config: "PreTrainedConfig",
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """Computes the inverse frequencies for llama 3.1 (verbatim math)."""
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    factor = rope_parameters_dict["factor"]
    low_freq_factor = rope_parameters_dict["low_freq_factor"]
    high_freq_factor = rope_parameters_dict["high_freq_factor"]
    old_context_len = rope_parameters_dict["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


ROPE_INIT_FUNCTIONS = {
    "llama3": _compute_llama3_parameters,
}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


ALL_ATTENTION_FUNCTIONS = {"sdpa": sdpa_attention_forward}


def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """This creates a basic lower-diagonal causal mask."""
    return kv_idx <= q_idx


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """This return the mask_function function corresponding to a 2D padding mask."""

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """This return the mask_function function corresponding to a 2D packed sequence mask."""

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask


def prepare_padding_mask(attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int) -> torch.Tensor | None:
    """From the 2D attention mask, prepare the correct padding mask to use by potentially padding it."""
    local_padding_mask = attention_mask
    if attention_mask is not None:
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
    return local_padding_mask


def _ignore_causal_mask_sdpa(
    padding_mask: torch.Tensor | None,
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: int | None = None,
) -> bool:
    """Whether SDPA's ``is_causal`` argument can be relied on instead of a materialized mask."""
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    if (
        not is_tracing(padding_mask)
        and (query_length == 1 or kv_length == query_length)
        and (local_attention_size is None or kv_length < local_attention_size)
        and (padding_mask is None or padding_mask.all())
    ):
        return True

    return False


def _non_vmap_expansion_sdpa(
    batch_indices: torch.Tensor, head_indices: torch.Tensor, q_indices: torch.Tensor, kv_indices: torch.Tensor
):
    """Broadcast the four index tensors over (b_idx, h_idx, q_idx, kv_idx) for index-based masks."""
    batch_indices = batch_indices[:, None, None, None]
    head_indices = head_indices[None, :, None, None]
    q_indices = q_indices[None, None, :, None]
    kv_indices = kv_indices[None, None, None, :]
    return batch_indices, head_indices, q_indices, kv_indices


def sdpa_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    allow_torch_fix: bool = True,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor | None:
    """Create a 4D boolean mask ``(batch_size, 1, q_length, kv_length)`` (non-vmap path)."""
    if isinstance(q_length, torch.Tensor):
        q_length, q_offset = q_length.shape[0], q_length[0].to(device)

    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(q_length, device=device) + q_offset
    kv_arange = torch.arange(kv_length, device=device) + kv_offset

    if not use_vmap:
        attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
        attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
    else:
        raise ValueError(
            "vmap-based mask creation is not supported in this build; use index-based masks."
        )

    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        attention_mask = attention_mask | torch.all(~attention_mask, dim=-1, keepdim=True)

    return attention_mask


def eager_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    allow_is_bidirectional_skip: bool = False,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """Float mask (0 / -inf). Present for completeness; not exercised under sdpa config."""
    _ = kwargs.pop("allow_is_causal_skip", None)
    _ = kwargs.pop("allow_torch_fix", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        allow_torch_fix=False,
        use_vmap=use_vmap,
        device=device,
        **kwargs,
    )
    if mask is not None:
        min_dtype = torch.finfo(dtype).min
        mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
    return mask


class AttentionMaskInterface(GeneralInterface):
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
    }


ALL_MASK_ATTENTION_FUNCTIONS = AttentionMaskInterface()


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor | None:
    """Find per-token sequence indices for packed batches (unused when a 2D mask is provided)."""
    first_dummy_value = position_ids[:, :1] - 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)
    if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
        return None
    return packed_sequence_mask


@deprecate_kwarg("input_embeds", version="5.6.0", new_name="inputs_embeds")
def _preprocess_mask_arguments(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    layer_idx: int | None,
    encoder_hidden_states: torch.Tensor | None = None,
):
    """Common mask-arg preprocessing: kv length/offsets and whether to early-exit."""
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None, None, None

    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None, None, None

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

    q_length = inputs_embeds.shape[1]
    if past_key_values is not None:
        q_offset = past_key_values.get_seq_length()
        q_offset = q_offset.to(inputs_embeds.device) if isinstance(q_offset, torch.Tensor) else q_offset
        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
    else:
        q_offset = 0
        if attention_mask is None:
            kv_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else q_length
            kv_offset = 0
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = inputs_embeds.shape[0]
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset


@deprecate_kwarg("input_embeds", version="5.6.0", new_name="inputs_embeds")
def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None = None,
    *,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
) -> torch.Tensor | BlockMask | None:
    """Create a standard causal (+padding) mask for the SDPA attention implementation."""
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False

    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype,
        config=config,
        use_vmap=use_vmap,
        device=device,
    )
    return causal_mask


def sliding_window_overlay(sliding_window: int) -> Callable:
    """This is an overlay depicting a sliding window pattern."""

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """This return the mask_function function to create a sliding window mask."""
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


@deprecate_kwarg("input_embeds", version="5.6.0", new_name="inputs_embeds")
def create_sliding_window_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None = None,
    *,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
) -> torch.Tensor | BlockMask | None:
    """Create a sliding-window causal (+padding) mask for the SDPA attention implementation."""
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False

    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        local_size=sliding_window,
        dtype=dtype,
        config=config,
        use_vmap=use_vmap,
        device=device,
    )
    return causal_mask


class FlashAttentionKwargs(TypedDict, total=False):
    """Keyword arguments for Flash Attention with Compile."""

    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None


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
    _supports_sdpa = False
    _tied_weights_keys = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def post_init(self):
        self.tie_weights()

    def get_input_embeddings(self):
        name = getattr(self, "_input_embed_layer", "embed_tokens")
        if (default_embedding := getattr(self, name, None)) is not None:
            return default_embedding
        if hasattr(self, "embeddings") and hasattr(self.embeddings, name):
            return getattr(self.embeddings, name)
        if hasattr(self, "model") and hasattr(self.model, name):
            return getattr(self.model, name)
        base = self.base_model
        if base is not None and base is not self:
            return base.get_input_embeddings()
        raise NotImplementedError(
            f"`get_input_embeddings` not auto-handled for {self.__class__.__name__}; "
            "please override in the subclass."
        )

    def set_input_embeddings(self, value):
        name = getattr(self, "_input_embed_layer", "embed_tokens")
        if hasattr(self, name):
            setattr(self, name, value)
        elif hasattr(self, "embeddings") and hasattr(self.embeddings, name):
            setattr(self.embeddings, name, value)
        elif hasattr(self, "model") and hasattr(self.model, name):
            setattr(self.model, name, value)
        elif self.base_model is not self:
            self.base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError(
                f"`set_input_embeddings` not auto-handled for {self.__class__.__name__}; "
                "please override in the subclass."
            )

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
        """Make the (frozen) input-embedding output require grad (transparent to forward values)."""

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
        elif config._attn_implementation is None:
            config._attn_implementation = "sdpa" if getattr(cls, "_supports_sdpa", False) else "eager"

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


class AutoModel:
    """Used only by the Generic head mixins' ``__init__``."""

    @classmethod
    def from_config(cls, config, **kwargs):
        raise NotImplementedError("AutoModel.from_config is not available in this build.")


@auto_docstring
class GenericForSequenceClassification:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
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
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GenericForQuestionAnswering:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.post_init()

    def get_input_embeddings(self):
        return getattr(self, self.base_model_prefix).embed_tokens

    def set_input_embeddings(self, value):
        getattr(self, self.base_model_prefix).embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        end_positions: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class GenericForTokenClassification:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class MistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """MistralRMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MistralDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class MistralPreTrainedModel(PreTrainedModel):
    config: MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MistralDecoderLayer,
        "attentions": MistralAttention,
    }


class MistralRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MistralConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: MistralConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """Computes the inverse frequencies according to the original RoPE implementation"""
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class MistralModel(MistralPreTrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MistralRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class MistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Example:"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        **kwargs,
    ) -> torch.LongTensor:
        """Greedy decoding, sufficient for the frozen generation probe."""
        ids = input_ids
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        for _ in range(int(max_new_tokens)):
            mask = torch.ones(ids.shape, dtype=torch.long, device=ids.device)
            logits = self(input_ids=ids, attention_mask=mask, use_cache=False).logits
            next_token = logits[:, -1, :].float().argmax(dim=-1)
            ids = torch.cat([ids, next_token[:, None]], dim=1)
        return ids


class MistralForTokenClassification(GenericForTokenClassification, MistralPreTrainedModel):
    pass


class MistralForSequenceClassification(GenericForSequenceClassification, MistralPreTrainedModel):
    pass


class MistralForQuestionAnswering(GenericForQuestionAnswering, MistralPreTrainedModel): ...


__all__ = [
    "MistralForCausalLM",
    "MistralForQuestionAnswering",
    "MistralModel",
    "MistralPreTrainedModel",
    "MistralForSequenceClassification",
    "MistralForTokenClassification",
]
