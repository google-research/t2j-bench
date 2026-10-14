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

import inspect
import itertools
import json
import math
import re
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, fields
from functools import partial, wraps
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm, Parameter
from torch.utils.checkpoint import checkpoint

from .configuration import PreTrainedConfig


import logging as _logging


class _WarnOnceLogger:
    """Stdlib logger exposing the ``warning_once`` method transformers' loggers provide."""

    def __init__(self, name):
        self._logger = _logging.getLogger(name)

    def warning_once(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._logger, item)


logger = _WarnOnceLogger(__name__)


class _LoggingModuleShim:
    """``logging`` module exposing ``get_logger`` (returns a warn-once logger)."""

    @staticmethod
    def get_logger(name=None):
        return _WarnOnceLogger(name or __name__)


logging = _LoggingModuleShim()


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


def torch_compilable_check(condition, message=""):
    """Eager implementation of transformers' compile-safe assert: raise on a False condition."""
    if not bool(condition):
        raise ValueError(message)


def is_flash_attention_requested(config):
    """This build only ships sdpa/eager, so flash attention is never requested."""
    return False


class SiLUActivation(nn.Module):
    def forward(self, input):
        return nn.functional.silu(input)


class QuickGELUActivation(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier({"silu": SiLUActivation, "quick_gelu": QuickGELUActivation})


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
class BaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
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
    _checkpoint_conversion_mapping = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def _from_config(cls, config, **kwargs):
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        return cls(config, **kwargs)

    def post_init(self):
        """Run weight init (composite-aware) then tie the output head."""
        self.init_weights()

    def init_weights(self):
        self.initialize_weights()
        self.tie_weights()

    @torch.no_grad()
    def initialize_weights(self):
        """Composite-aware ``self.apply(self._initialize_weights)`` -- see ``_smart_apply``."""
        _smart_apply(self, self._initialize_weights)

    def _initialize_weights(self, module):
        """Init ``module`` unless a parent already did (``_is_hf_initialized`` module flag)."""
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Default init, overridable per model."""
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            get_text_config = getattr(self.config, "get_text_config", None)
            text_config = get_text_config() if callable(get_text_config) else self.config
            std = getattr(text_config, "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if getattr(module, "weight", None) is not None:
                init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)

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
        mapping = getattr(self, "_checkpoint_conversion_mapping", None)
        if mapping:
            converted = {}
            for key, value in state_dict.items():
                new_key = key
                for pattern, replacement in mapping.items():
                    candidate = re.sub(pattern, replacement, new_key, count=1)
                    if candidate != new_key:
                        new_key = candidate
                        break
                converted[new_key] = value
            state_dict = converted

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


def capture_outputs(obj=None, *args, **kwargs):
    if callable(obj) and not args and not kwargs:
        return obj
    def _decorator(inner):
        return inner
    return _decorator


def _resolve_attention_interface(registry, name, default):
    getter = getattr(registry, "get_interface", None)
    if callable(getter):
        return getter(name, default)
    mapping_get = getattr(registry, "get", None)
    if callable(mapping_get):
        return mapping_get(name, default)
    return default


class _InitializationShim:
    """Implementation of ``transformers.initialization``."""

    @staticmethod
    def normal_(tensor, mean=0.0, std=1.0, generator=None):
        if not getattr(tensor, "_is_hf_initialized", False):
            return nn.init.normal_(tensor, mean=mean, std=std, generator=generator)
        return tensor

    @staticmethod
    def zeros_(tensor):
        if not getattr(tensor, "_is_hf_initialized", False):
            return nn.init.zeros_(tensor)
        return tensor

    @staticmethod
    def ones_(tensor):
        if not getattr(tensor, "_is_hf_initialized", False):
            return nn.init.ones_(tensor)
        return tensor

    @staticmethod
    def constant_(tensor, val):
        if not getattr(tensor, "_is_hf_initialized", False):
            return nn.init.constant_(tensor, val)
        return tensor

    @staticmethod
    def copy_(tensor, other):
        if not getattr(tensor, "_is_hf_initialized", False):
            with torch.no_grad():
                return tensor.copy_(other)
        return tensor


init = _InitializationShim()


def _smart_apply(module, fn):
    """Depth-first (children-then-self) apply that swaps ``fn`` to each nested ``PreTrainedModel``'s own ``_initialize_weights`` -- the torch-only equivalent of ``transformers``' ``smart_apply``."""
    for child in module.children():
        if isinstance(child, PreTrainedModel):
            _smart_apply(child, child._initialize_weights)
        else:
            _smart_apply(child, fn)
    fn(module)
    return module


class GenerationMixin:
    """Minimal implementation of ``transformers.generation.GenerationMixin``."""

    @classmethod
    def can_generate(cls):
        return True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        for key, value in kwargs.items():
            if value is not None:
                model_inputs[key] = value
        return model_inputs

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "This GenerationMixin does not implement generation; "
            "Qwen2_5OmniThinkerForConditionalGeneration provides a cache-free greedy `generate`."
        )


def check_torch_load_is_safe():
    """Mirror ``transformers.utils.check_torch_load_is_safe``: refuse ``torch.load`` on torch < 2.6 (CVE-2025-32434)."""
    parts = torch.__version__.split("+")[0].split(".")
    major, minor = int(parts[0]), int(parts[1])
    if (major, minor) < (2, 6):
        raise ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, "
            "we now require users to upgrade torch to at least v2.6 in order to use the function. "
            f"This version of torch is {torch.__version__}."
        )


def cached_file(path_or_repo_id, filename, **kwargs):
    """Local implementation of ``transformers.utils.hub.cached_file``."""
    subfolder = kwargs.get("subfolder") or ""
    full = Path(path_or_repo_id) / subfolder / filename
    if full.is_file():
        return str(full)
    if kwargs.get("_raise_exceptions_for_missing_entries", False):
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {filename}."
        )
    return None


class GELUActivation(nn.Module):
    """Erf-based GELU matching transformers' default ``GELUActivation`` (``F.gelu``)."""

    def forward(self, input):
        return nn.functional.gelu(input)


ACT2FN["gelu"] = GELUActivation


def auto_docstring(obj=None, *args, **kwargs):
    if callable(obj) and not args and not kwargs:
        return obj
    def _decorator(inner):
        return inner
    return _decorator

def can_return_tuple(fn=None, *args, **kwargs):
    if fn is None:
        def _decorator(obj):
            return obj
        return _decorator
    return fn

def merge_with_config_defaults(fn=None, *args, **kwargs):
    if fn is None:
        def _decorator(obj):
            return obj
        return _decorator
    return fn


def capture_outputs(obj=None, *args, **kwargs):
    if callable(obj) and not args and not kwargs:
        return obj
    def _decorator(inner):
        return inner
    return _decorator


def _resolve_attention_interface(registry, name, default):
    getter = getattr(registry, "get_interface", None)
    if callable(getter):
        return getter(name, default)
    mapping_get = getattr(registry, "get", None)
    if callable(mapping_get):
        return mapping_get(name, default)
    return default


from .configuration import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniVisionEncoderConfig,
)

from safetensors.torch import load_file

logger = logging.get_logger(__name__)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Generates a 1D Kaiser-windowed sinc filter."""
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


@auto_docstring
class Qwen2_5OmniPreTrainedModel(PreTrainedModel):
    config: Qwen2_5OmniConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SinusoidsPositionEmbedding):
            log_timescale_increment = np.log(module.max_timescale) / (module.channels // 2 - 1)
            inv_timescales = torch.exp(-log_timescale_increment * torch.arange(module.channels // 2).float())
            scaled_time = torch.arange(module.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
            init.copy_(module.positional_embedding, torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1))
        elif isinstance(module, UpSample1d):
            filter_tensor = kaiser_sinc_filter1d(0.5 / module.ratio, 0.6 / module.ratio, module.kernel_size)
            init.copy_(module.filter, filter_tensor)
        elif isinstance(module, DownSample1d):
            filter_tensor = kaiser_sinc_filter1d(module.cutoff, module.half_width, module.kernel_size)
            init.copy_(module.filter, filter_tensor)
        elif isinstance(module, Qwen2_5_VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
    input_modalities = ("image", "video", "audio", "text")

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: list[int],
        grid_ws: list[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:
        """Splits token index list into chunks based on token value ranges."""

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: torch.LongTensor | None = None,
        second_per_grids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the 3D rope index based on image and video's temporal, height and width in LLM."""
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_video and not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    elif min_ed == ed_video and use_audio_in_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

                if attention_mask is not None:
                    position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
                else:
                    position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas




@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen2.5OmniThinker causal language model (or autoregressive) outputs.
    """
)
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    """loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided): Language modeling loss (for next-token prediction)."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen2_5OmniAudioEncoderConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        attention_interface: Callable = _resolve_attention_interface(
            ALL_ATTENTION_FUNCTIONS, self.config._attn_implementation, eager_attention_forward
        )

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen2_5OmniAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args: hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`"""
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2_5OmniAudioEncoderLayer`].
    """
)
class Qwen2_5OmniAudioEncoder(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniAudioEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": Qwen2_5OmniAudioEncoderLayer,
        "attentions": Qwen2_5OmniAudioAttention,
    }

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if is_flash_attention_requested(self.config):
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs: Unpack[TransformersKwargs]):
        """feature_lens (`torch.LongTensor` of shape `(batch_size,)`): mel length"""
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.full((chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return BaseModelOutputWithPooling(last_hidden_state=token_audio)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """Pads a sequence of tensors to their maximum length on indicated `padding_side`."""
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2_5OmniVisionAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig = None) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.q = nn.Linear(self.dim, self.dim, bias=True)
        self.k = nn.Linear(self.dim, self.dim, bias=True)
        self.v = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = 1
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = _resolve_attention_interface(
            ALL_ATTENTION_FUNCTIONS, self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2_5OmniRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """Qwen2_5OmniRMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5OmniMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig) -> None:
        super().__init__()
        self.norm1 = Qwen2_5OmniRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2_5OmniRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5OmniVisionAttention(config=config)
        self.mlp = Qwen2_5OmniMLP(config, bias=True)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """cu_seqlens (`torch.Tensor`): Cumulative sequence lengths used for packed variable-length attention in Flash Attention kernels."""
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | list[int] | tuple[int, int] = 14,
        temporal_patch_size: int | list[int] | tuple[int, int] = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5OmniPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5OmniRMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5OmniVisionEncoder(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniVisionEncoderConfig
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]
    _input_embed_layer = "patch_embed"
    _can_record_outputs = {
        "hidden_states": Qwen2_5OmniVisionBlock,
        "attentions": Qwen2_5OmniVisionAttention,
    }
    input_modalities = ("image", "video")

    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

        self.post_init()

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        grid_thw_list = grid_thw.tolist()

        for grid_t, grid_h, grid_w in grid_thw_list:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        """Args: hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`): The final hidden states of the model."""
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Qwen2_5OmniRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=None):
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
        config: Qwen2_5OmniConfig | None = None,
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

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/)."""
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5OmniAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_parameters = config.rope_parameters
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = _resolve_attention_interface(
            ALL_ATTENTION_FUNCTIONS, self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2MLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_5OmniTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and not is_flash_attention_requested(config):
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2_5OmniAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2_5OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2_5OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        """Args: hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`"""

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
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
class Qwen2_5OmniThinkerTextModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniTextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "hidden_states": Qwen2_5OmniDecoderLayer,
        "attentions": Qwen2_5OmniAttention,
    }
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2_5OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
                "cache_position": cache_position,
            }
            full_attention_kwargs = {
                name: value
                for name, value in mask_kwargs.items()
                if name in inspect.signature(create_causal_mask).parameters
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**full_attention_kwargs),
            }
            if self.has_sliding_layers:
                sliding_attention_kwargs = {
                    name: value
                    for name, value in mask_kwargs.items()
                    if name in inspect.signature(create_sliding_window_causal_mask).parameters
                }
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**sliding_attention_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen2_5OmniThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(config.audio_config)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`): The tensors corresponding to the input videos."""
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        return self.visual(pixel_values_videos, grid_thw=video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`): The tensors corresponding to the input images."""
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """input_features (`torch.FloatTensor`): The tensors corresponding to the input audios."""
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
            **kwargs,
        )
        if audio_outputs.last_hidden_state.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError("length of audio_features should match audio_output_lengths")

        return audio_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is equal to the length of multimodal features."""
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        use_audio_in_video: bool | None = None,
        video_second_per_grid: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5OmniThinkerCausalLMOutputWithPast:
        """image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*): The temporal, height and width of feature shape of each image in LLM."""

        return_dict = kwargs.pop("return_dict", True)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                return_dict=True,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if past_key_values_length == 0 or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = (past_key_values_length + self.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        generation_config=None,
        max_new_tokens=None,
        **kwargs,
    ):
        """Torch-free greedy generation without a KV cache."""
        if generation_config is None:
            generation_config = getattr(self, "generation_config", None)

        def _cfg(name, default):
            if generation_config is not None and getattr(generation_config, name, None) is not None:
                return getattr(generation_config, name)
            return default

        max_new = max_new_tokens or _cfg("max_new_tokens", None) or 2048
        eos_token_id = _cfg("eos_token_id", getattr(self.config, "eos_token_id", None))
        pad_token_id = _cfg("pad_token_id", getattr(self.config, "pad_token_id", None))
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        elif eos_token_id is None:
            eos_token_id = []
        else:
            eos_token_id = list(eos_token_id)

        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, prompt_len, device=device, dtype=torch.long)

        generated = input_ids.clone()
        cur_mask = attention_mask.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        mm_keys = (
            "input_features",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "feature_attention_mask",
            "audio_feature_lengths",
            "video_second_per_grid",
        )
        mm_kwargs = {k: kwargs[k] for k in mm_keys if kwargs.get(k) is not None}
        use_audio_in_video = kwargs.get("use_audio_in_video", False)
        has_multimodal = any(
            mm_kwargs.get(k) is not None
            for k in ("pixel_values", "pixel_values_videos", "input_features")
        )

        self.rope_deltas = None
        if has_multimodal:
            feature_attention_mask = mm_kwargs.get("feature_attention_mask")
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            else:
                audio_feature_lengths = mm_kwargs.get("audio_feature_lengths")
            delta0 = (1 - attention_mask).sum(dim=-1, keepdim=True)
            prefill_pos, rope_deltas = self.get_rope_index(
                input_ids,
                mm_kwargs.get("image_grid_thw"),
                mm_kwargs.get("video_grid_thw"),
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                mm_kwargs.get("video_second_per_grid"),
            )
            rope_deltas = rope_deltas - delta0
        else:
            prefill_pos, rope_deltas = None, None

        for step in range(int(max_new)):
            if not has_multimodal:
                outputs = self(input_ids=generated, attention_mask=cur_mask, use_cache=False)
            elif step == 0:
                outputs = self(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    position_ids=prefill_pos,
                    use_cache=False,
                    use_audio_in_video=use_audio_in_video,
                    **mm_kwargs,
                )
            else:
                num_decoded = generated.shape[1] - prompt_len
                decode_idx = torch.arange(
                    prompt_len, prompt_len + num_decoded, device=device, dtype=prefill_pos.dtype
                ).view(1, -1)
                decode_pos = (decode_idx + rope_deltas.to(decode_idx.device)).unsqueeze(0).expand(3, -1, -1)
                position_ids = torch.cat([prefill_pos, decode_pos], dim=2)
                outputs = self(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1)
            if pad_token_id is not None:
                next_tokens[finished] = pad_token_id
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones(batch_size, 1, device=device, dtype=cur_mask.dtype)], dim=-1
            )
            for eos_id in eos_token_id:
                finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None

        return model_inputs




@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen2.5OmniTalker causal language model (or autoregressive) outputs.
    """
)
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):
    """loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided): Language modeling loss (for next-token prediction)."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    thinker_reply_part: torch.FloatTensor | None = None


@auto_docstring
class Qwen2_5OmniTalkerModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniTalkerConfig
    input_modalities = ("image", "video", "audio", "text")
    _can_record_outputs = {
        "hidden_states": Qwen2_5OmniDecoderLayer,
        "attentions": Qwen2_5OmniAttention,
    }

    _no_split_modules = ["Qwen2_5OmniTalkerDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2_5OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
                "cache_position": cache_position,
            }
            full_attention_kwargs = {
                name: value
                for name, value in mask_kwargs.items()
                if name in inspect.signature(create_causal_mask).parameters
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**full_attention_kwargs),
            }
            if self.has_sliding_layers:
                sliding_attention_kwargs = {
                    name: value
                    for name, value in mask_kwargs.items()
                    if name in inspect.signature(create_sliding_window_causal_mask).parameters
                }
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**sliding_attention_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen2_5OmniTalkerConfig
    base_model_prefix = "talker"
    output_modalities = ("audio",)

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)

        self.thinker_to_talker_proj = nn.Linear(config.embedding_size, config.hidden_size)

        self.model = Qwen2_5OmniTalkerModel(config)
        self.codebook_size = config.vocab_size
        self.codec_head = nn.Linear(config.hidden_size, self.codebook_size, bias=False)

        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_eos_token = config.tts_codec_end_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.codec_mask_token = config.tts_codec_mask_token_id

        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id

        self.spatial_merge_size = self.config.spatial_merge_size
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        thinker_reply_part: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        input_text_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_audio_in_video: bool | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        video_second_per_grid: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | Qwen2_5OmniTalkerCausalLMOutputWithPast:
        """thinker_reply_part (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*): Hidden states from the thinker model's output that represent the text reply part to be processed."""

        if attention_mask is not None and position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if past_key_values_length == 0 or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )

                if inputs_embeds is not None:
                    inputs_embeds[:, -1, :] += self.get_input_embeddings()(
                        torch.tensor([self.codec_bos_token], dtype=torch.long, device=inputs_embeds.device)
                    )
                    inputs_embeds[:, -2, :] += self.get_input_embeddings()(
                        torch.tensor([self.codec_pad_token], dtype=torch.long, device=inputs_embeds.device)
                    )
                self.rope_deltas = rope_deltas

            else:
                if inputs_embeds is not None:
                    batch_size, seq_length, _ = inputs_embeds.shape
                else:
                    batch_size, seq_length = input_ids.shape

                delta = (past_key_values_length + self.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            codec_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]

        talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_lm_input,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.codec_head(hidden_states)
        logits = logits.float()

        loss = None

        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part

        return model_kwargs


class Qwen2_5OmniDiTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen2_5OmniDiTConfig, device=None):
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
        config: Qwen2_5OmniDiTConfig | None = None,
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


class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel."""

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence."""

        if max_len is None:
            max_len = length.max().long().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)

        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        attention = self.conv(self.tanh(self.tdnn(attention)))

        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e., TDNN-Res2Net-TDNN-SqueezeExcitationBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual


class ECAPA_TimeDelayNet(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper."""

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        hidden_states = self.asp(hidden_states)

        hidden_states = self.fc(hidden_states)

        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


class DiTInputEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        self.spk_encoder = ECAPA_TimeDelayNet(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,
        drop_audio_cond: bool | None = False,
        code_embed_uncond: bool | None = None,
        apply_cfg: bool | None = True,
    ):
        if apply_cfg:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            speaker_embedding = torch.cat([speaker_embedding, torch.zeros_like(speaker_embedding)], dim=0)
            condition_vector = torch.cat([condition_vector, torch.zeros_like(condition_vector)], dim=0)
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:
            condition_vector = torch.zeros_like(condition_vector)
            speaker_embedding = torch.zeros_like(speaker_embedding)
        condition_vector = self.spk_encoder(condition_vector).unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        hidden_states = self.proj(torch.cat((hidden_states, condition_vector, code_embed, speaker_embedding), dim=-1))

        return hidden_states


class DiTCodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = torch.zeros_like(code)
        code_embed = self.codec_embed(code)

        code_embed = torch.repeat_interleave(code_embed, repeats=self.repeats, dim=1)
        return code_embed


class Qwen2_5_OmniAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class Qwen2_5_OmniAdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states


class DiTMLP(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = nn.ModuleList(
            [
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.ff:
            hidden_states = layer(hidden_states)
        return hidden_states


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""

    def rotate_half_codec(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class DiTAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self.is_causal = False

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)])

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_weights, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )

        attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)

        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)

        return attention_output


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, hidden_states, scale=1000):
        device = hidden_states.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * hidden_states.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.type_as(hidden_states)


class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.ModuleList([nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)])

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)
        return time_hidden


class DiTDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = Qwen2_5_OmniAdaLayerNormZero(config.hidden_size)

        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = DiTMLP(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(
        self, hidden_states, timestep, position_embeddings=None, block_diff=None
    ):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep)

        attn_output = self.attn(
            hidden_states=norm,
            position_embeddings=position_embeddings,
            attention_mask=(block_diff >= -float(self.look_backward_block))
            & (block_diff <= float(self.look_ahead_block)),
        )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states


class SnakeBeta(nn.Module):
    """A modified Snake function which uses separate parameters for the magnitude of the periodic components"""

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        """Forward pass of the function."""
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]

        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        self.cutoff = cutoff
        self.half_width = half_width
        self.kernel_size = kernel_size

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=self._get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=self._get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=self._get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
            ]
        )

        self.num_layers = len(self.convs1) + len(self.convs2)

        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = hidden_states
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni Token2WavBigVGAN model. Which take mel spectrogram as input and predict waveform.
    """
)
class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniBigVGANConfig
    input_modalities = "audio"

    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__(config)
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 7, 1, padding=3)

        ups = [
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**layer_idx),
                        config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ]
            )
            for layer_idx, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ]
        self.ups = nn.ModuleList(ups)

        self.resblocks = nn.ModuleList(
            [
                AMPBlock(config.upsample_initial_channel // (2 ** (layer_idx + 1)), kernel_size, dilation)
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(config.upsample_initial_channel // (2**self.num_upsample_layers))
        )
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2**self.num_upsample_layers), 1, 7, 1, padding=3, bias=False
        )

        self.post_init()

    def normalize_spectrogram(self, spectrogram, max_value, min_db):
        return torch.clamp((2 * max_value) * ((spectrogram - min_db) / (-min_db)) - max_value, -max_value, max_value)

    def amplitude_to_db(self, amplitude, min_db_level):
        min_level = torch.exp(
            torch.tensor(min_db_level / 20.0 * np.log(10), device=amplitude.device, dtype=amplitude.dtype)
        )
        return 20 * torch.log10(torch.clamp(amplitude, min=min_level))

    def process_mel_spectrogram(self, mel_spectrogram):
        amplitude_spectrum = torch.exp(mel_spectrogram)
        decibel_spectrum = self.amplitude_to_db(amplitude_spectrum, -115) - 20
        return self.normalize_spectrogram(decibel_spectrum, 1, -115)

    def forward(self, mel_spectrogram, **kwargs):
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output

        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze().cpu()


class RungeKutta4ODESolver:
    def __init__(self, function, initial_value):
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_step(self, function, time_start, time_step, time_end, value_start, function_value_start=None):
        k1 = function_value_start if function_value_start is not None else function(time_start, value_start)
        k2 = function(time_start + time_step * self._one_third, value_start + time_step * k1 * self._one_third)
        k3 = function(time_start + time_step * self._two_thirds, value_start + time_step * (k2 - k1 * self._one_third))
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        function_value_start = function(time_start, value_start)
        return self._rk4_step(
            function, time_start, time_step, time_end, value_start, function_value_start=function_value_start
        ), function_value_start

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points):
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            delta_value, _ = self._compute_step(self.function, time_start, time_step, time_end, current_value)
            next_value = current_value + delta_value

            while current_index < len(time_points) and time_end >= time_points[current_index]:
                solution[current_index] = self._linear_interpolation(
                    time_start, time_end, current_value, next_value, time_points[current_index]
                )
                current_index += 1

            current_value = next_value

        return solution


@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni Token2WavDiT model. Which take speech tokens as input and predict mel spectrogram.
    """
)
class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniDiTConfig
    input_modalities = "audio"
    _no_split_modules = ["DiTDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)

        self.text_embed = DiTCodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = DiTInputEmbedding(config)

        self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config=config)

        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )

        self.norm_out = Qwen2_5_OmniAdaLayerNormZero_Final(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

        self.post_init()

    def _create_block_diff(self, hidden_states):
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = torch.arange(seq_len, device=hidden_states.device) // self.block_size

        block_i = block_indices.unsqueeze(1)
        block_j = block_indices.unsqueeze(0)
        block_diff = block_j - block_i

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
        **kwargs,
    ):
        batch_size = hidden_states.shape[0]
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids[None, :].repeat(batch_size, 1)
        position_embeddings = self.rotary_embed(hidden_states, position_ids)
        blockwise_difference = self._create_block_diff(hidden_states)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)

        return output

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        maximum_duration = quantized_code.shape[1] * self.repeats
        batch_size = reference_mel_spectrogram.shape[0]
        if batch_size != 1:
            raise ValueError("Only batch size = 1 is currently supported")

        if maximum_duration > self.config.max_position_embeddings:
            raise ValueError(
                f"Requested mel length ({maximum_duration}) exceeds `dit_config.max_position_embeddings` "
                f"({self.config.max_position_embeddings}). Provide shorter `quantized_code`."
            )

        initial_state = torch.randn(
            [batch_size, maximum_duration, self.mel_dim],
            dtype=reference_mel_spectrogram.dtype,
            device=quantized_code.device,
        )
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                    apply_cfg=False,
                )
                return prediction

            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        time_embedding = torch.linspace(
            initial_time, 1, num_steps, device=quantized_code.device, dtype=conditioning_vector.dtype
        )

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        return generated_mel_spectrogram


@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni Token2Wav model. Consists a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.
    """
)
class Qwen2_5OmniToken2WavModel(Qwen2_5OmniPreTrainedModel):
    config: Qwen2_5OmniToken2WavConfig
    base_model_prefix = "model"
    input_modalities = "audio"
    _no_split_modules = ["Qwen2_5OmniToken2WavDiTModel", "Qwen2_5OmniToken2WavBigVGANModel"]

    def __init__(self, config: Qwen2_5OmniToken2WavConfig):
        super().__init__(config)
        attn_impl = config._attn_implementation
        if is_flash_attention_requested(config):
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel must inference with fp32, but Flash Attention only supports fp16 and bf16, "
                "attention implementation of Qwen2_5OmniToken2WavModel will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

        self.post_init()

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """Generates a waveform from input code and conditioning parameters."""

        mel_spectrogram = self.code2wav_dit_model.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        waveform = self.code2wav_bigvgan_model(mel_spectrogram)

        return waveform




@auto_docstring(
    custom_intro="""
    The full Qwen2.5Omni model, a multimodal model composed of 3 sub-models:
    - [`Qwen2_5OmniThinkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes text, audio, image, video as input and predict text tokens.
    - [`Qwen2_5OmniTalkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes thinker hidden states and response as input and predict speech tokens.
    - [`Qwen2_5OmniToken2WavModel`]:
    a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.
    """
)
class Qwen2_5OmniForConditionalGeneration(Qwen2_5OmniPreTrainedModel, GenerationMixin):
    config: Qwen2_5OmniConfig
    output_modalities = ("audio", "text")
    _no_split_modules = [
        "Qwen2_5OmniTalkerForConditionalGeneration",
        "Qwen2_5OmniToken2WavModel",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)

        self.has_talker = config.enable_audio_output
        self.speaker_map = {}
        if config.enable_audio_output:
            self.enable_talker()
        self.post_init()

    def enable_talker(self):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config)
        self.token2wav.float()
        self.has_talker = True

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    def forward(self, *args, **kwargs):
        return self.thinker(*args, **kwargs)

    def load_speakers(self, path):
        check_torch_load_is_safe()
        for key, value in torch.load(path, weights_only=True).items():
            self.speaker_map[key] = value
        logger.info(f"Speaker {list(self.speaker_map.keys())} loaded")

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        spk_path = cached_file(
            pretrained_model_name_or_path,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=token,
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{spk_path} not exists""")
        model.load_speakers(spk_path)

        return model

    @torch.no_grad()
    @deprecate_kwarg("return_audio", version="v5", new_name="generation_mode")
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        """Generate text response and audio from input."""
        generation_mode = kwargs.pop("generation_mode", None)
        return_audio = generation_mode != "text" and generation_mode is not False

        if return_audio and not self.has_talker:
            return_audio = False
        if return_audio is None:
            return_audio = self.has_talker
        if return_audio and speaker not in self.speaker_map:
            return_audio = False
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value

        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        speaker_params = self.speaker_map[speaker]

        embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(input_ids.device)
        if thinker_kwargs.get("input_features") is not None:
            audio_ids_mask = input_ids == self.config.thinker_config.audio_token_index
            audio_mask = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            audio_mask_tensor = torch.zeros(
                [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=input_ids.device,
            )
            embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
        if thinker_kwargs.get("pixel_values") is not None:
            image_ids_mask = input_ids == self.config.thinker_config.image_token_index
            image_mask = image_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            image_mask_tensor = torch.zeros(
                [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=input_ids.device,
            )
            embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
        if thinker_kwargs.get("pixel_values_videos") is not None:
            video_ids_mask = input_ids == self.config.thinker_config.video_token_index
            video_mask = video_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            video_mask_tensor = torch.zeros(
                [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=input_ids.device,
            )
            embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)

        processed_thinker_hidden = (
            (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
        ) + thinker_result.hidden_states[1:]
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(input_ids.device)
        thinker_token_embeds = [
            token_hidden_states[0].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
        ]
        thinker_hidden_states = [
            token_hidden_states[-1].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
        ]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids,
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=input_ids.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=input_ids.device),
            ],
            dim=1,
        )

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device)
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(input_ids.device)
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_token = torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=input_ids.device)
        eos_embedding = thinker_embed_tokens(eos_token).to(input_ids.device)

        pad_token = torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=input_ids.device)
        pad_embedding = thinker_embed_tokens(pad_token).to(input_ids.device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        talker_attention_mask = None
        if "attention_mask" in kwargs:
            talker_attention_mask = torch.cat(
                [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
            ).to(input_ids.device)

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(input_ids.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

        if self.token2wav.dtype != torch.float:
            self.token2wav.float()

        wav = self.token2wav(
            talker_generate_codes.to(input_ids.device),
            conditioning=speaker_params["cond"].to(input_ids.device).float(),
            reference_mel=speaker_params["ref_mel"].to(input_ids.device).float(),
            **token2wav_kwargs,
        )

        return thinker_result.sequences, wav.float()


__all__ = [
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniThinkerTextModel",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
]
