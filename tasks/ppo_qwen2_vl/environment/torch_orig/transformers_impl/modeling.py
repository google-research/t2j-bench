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

"""PyTorch Qwen2 model."""

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

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F
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
from .configuration import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig

from safetensors.torch import load_file

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen2VLModelOutputWithPast(ModelOutput):
    """past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`): It is a [`~cache_utils.Cache`] instance."""

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen2VL causal language model (or autoregressive) outputs.
    """
)
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided): Language modeling loss (for next-token prediction)."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2VLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """Qwen2VLRMSNorm is equivalent to T5LayerNorm"""
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


class Qwen2VLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen2VLConfig, device=None):
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
        config: Qwen2VLConfig | None = None,
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionRotaryEmbedding(nn.Module):
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


class PatchEmbed(nn.Module):
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


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


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


class VisionAttention(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

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


class Qwen2VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionAttention(config=config)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2MLP(nn.Module):
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


class Qwen2VLAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: Qwen2VLTextConfig, layer_idx: int | None = None):
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
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_parameters = config.rope_parameters
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
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


class Qwen2VLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and not is_flash_attention_requested(config):
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2VLAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
class Qwen2VLPreTrainedModel(PreTrainedModel):
    config: Qwen2VLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    config_class = Qwen2VLConfig


@auto_docstring
class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config: Qwen2VLVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen2VLVisionBlock"]
    _input_embed_layer = "patch_embed"
    _can_record_outputs = {
        "hidden_states": Qwen2VLVisionBlock,
        "attentions": VisionAttention,
    }

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )
        self.gradient_checkpointing = False

        self.post_init()

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
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

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """grid_thw (`torch.LongTensor` of shape `(num_images, 3)`): The temporal, height and width dimensions of feature shape for each image."""
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


@auto_docstring
class Qwen2VLTextModel(Qwen2VLPreTrainedModel):
    config: Qwen2VLTextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "hidden_states": Qwen2VLDecoderLayer,
        "attentions": Qwen2VLAttention,
    }

    def __init__(self, config: Qwen2VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

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


@auto_docstring
class Qwen2VLModel(Qwen2VLPreTrainedModel):
    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2VLTextModel._from_config(config.text_config)
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ):
        """Compute 3D positional indices for vision tokens derived from a single image or video input."""
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the 3D rope index based on image and video's sizes."""
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

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
        vision_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = video_embeds

        return vision_outputs

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
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

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
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

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
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            position_ids = None
        return position_ids

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2VLModelOutputWithPast:
        """image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*): The temporal, height and width of feature shape of each image in LLM."""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Qwen2VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _checkpoint_conversion_mapping = {
        r"^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
        r"^lm_head": "lm_head",
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`): The tensors corresponding to the input videos."""
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs
        )

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`): The tensors corresponding to the input images."""
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

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
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2VLCausalLMOutputWithPast:
        """labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): Labels for computing the masked language modeling loss."""

        outputs: Qwen2VLModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        generation_config=None,
        max_new_tokens=None,
        **kwargs,
    ):
        """Torch-free greedy/sampling generation (no KV cache)."""
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
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is None:
            do_sample = getattr(generation_config, "do_sample", False)
        temperature = kwargs.pop("temperature", None)
        if temperature is None:
            temperature = getattr(generation_config, "temperature", 1.0)
        top_p = kwargs.pop("top_p", None)
        if top_p is None:
            top_p = getattr(generation_config, "top_p", 1.0)
        top_k = kwargs.pop("top_k", None)
        if top_k is None:
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

        prompt_len = input_ids.shape[1]

        first_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"generation_config", "position_ids", "rope_deltas",
                         "token_type_ids", "mm_token_type_ids"}
        }
        grid_kwargs = {
            k: v
            for k, v in first_kwargs.items()
            if k in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts")
        }

        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)

        def _mm_token_type_ids(ids):
            mm = torch.zeros_like(ids)
            if image_token_id is not None:
                mm[ids == image_token_id] = 1
            if video_token_id is not None:
                mm[ids == video_token_id] = 2
            return mm

        image_grid_thw = first_kwargs.get("image_grid_thw")
        video_grid_thw = first_kwargs.get("video_grid_thw")
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, prompt_len, device=device)
        cur_mask = attention_mask.clone()

        mm0 = _mm_token_type_ids(input_ids)
        if has_multimodal:
            prefill_pos, rope_deltas = self.model.get_rope_index(
                input_ids,
                mm0,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            prefill_pos, rope_deltas = None, None
        if hasattr(self, "model") and hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None

        for step in range(max_new):
            if not has_multimodal:
                outputs = self(
                    input_ids=generated, attention_mask=cur_mask, use_cache=False,
                    **(first_kwargs if step == 0 else grid_kwargs),
                )
            elif step == 0:
                outputs = self(
                    input_ids=generated, attention_mask=cur_mask,
                    position_ids=prefill_pos, mm_token_type_ids=mm0,
                    use_cache=False, **first_kwargs,
                )
            else:
                num_decoded = generated.shape[1] - prompt_len
                decode_pos = torch.arange(
                    1, num_decoded + 1, device=device, dtype=prefill_pos.dtype
                ).view(1, 1, -1).expand(3, batch_size, -1)
                if rope_deltas is not None:
                    decode_pos = decode_pos + rope_deltas.view(1, batch_size, 1).to(decode_pos.device)
                position_ids = torch.cat([prefill_pos, decode_pos], dim=2)
                outputs = self(
                    input_ids=generated, attention_mask=cur_mask,
                    position_ids=position_ids, mm_token_type_ids=_mm_token_type_ids(generated),
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


__all__ = ["Qwen2VLForConditionalGeneration", "Qwen2VLModel", "Qwen2VLPreTrainedModel", "Qwen2VLTextModel"]
