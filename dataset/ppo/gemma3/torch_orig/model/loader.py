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

"""Repo-local pure-Torch model and tokenizer loading for Gemma3."""

import importlib
import importlib.util
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)
_IMPL_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_TRANSFORMERS_DIR = _IMPL_ROOT / "transformers_impl"
_LOCAL_TRANSFORMERS_PACKAGE = (
    f"_{_IMPL_ROOT.parent.parent.name}_{_IMPL_ROOT.parent.name}_torch_orig_transformers"
)


class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def _iter_checkpoint_files(model_path: Path) -> list[Path]:
    safe_index = model_path / "model.safetensors.index.json"
    pt_index = model_path / "pytorch_model.bin.index.json"
    if safe_index.is_file():
        weights_map = _read_json(safe_index)["weight_map"]
        return [model_path / shard for shard in sorted(set(weights_map.values()))]
    if pt_index.is_file():
        weights_map = _read_json(pt_index)["weight_map"]
        return [model_path / shard for shard in sorted(set(weights_map.values()))]
    for file_name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_path / file_name
        if candidate.is_file():
            return [candidate]
    raise FileNotFoundError(f"No checkpoint file found under {model_path}")


def _load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_safetensors(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for checkpoint_path in _iter_checkpoint_files(model_path):
        state_dict.update(_load_state_dict_file(checkpoint_path))
    return state_dict


def _ensure_local_transformers_package() -> str:
    if _LOCAL_TRANSFORMERS_PACKAGE in sys.modules:
        return _LOCAL_TRANSFORMERS_PACKAGE
    init_path = _LOCAL_TRANSFORMERS_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _LOCAL_TRANSFORMERS_PACKAGE,
        init_path,
        submodule_search_locations=[str(_LOCAL_TRANSFORMERS_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for local transformers package at {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LOCAL_TRANSFORMERS_PACKAGE] = module
    spec.loader.exec_module(module)
    return _LOCAL_TRANSFORMERS_PACKAGE


def _import_local_transformers_module(module_name: str):
    package_name = _ensure_local_transformers_package()
    return importlib.import_module(f"{package_name}.{module_name}")


@contextmanager
def _hf_transformers_context():
    import transformers as hf_transformers

    yield hf_transformers


def _resolve_runtime_dtype(config, requested_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    native_dtype = getattr(config, "torch_dtype", None)
    if (
        requested_dtype == torch.float16
        and native_dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        logger.warning(
            "Requested float16 for a Gemma3 checkpoint with native bfloat16 weights. "
            "Falling back to bfloat16 runtime for numerical stability."
        )
        return torch.bfloat16
    return requested_dtype


def _build_init_kwargs(model_args) -> dict[str, Any]:
    return {
        "trust_remote_code": getattr(model_args, "trust_remote_code", True),
        "cache_dir": getattr(model_args, "cache_dir", None),
        "revision": getattr(model_args, "model_revision", "main"),
        "token": getattr(model_args, "hf_hub_token", None),
    }


def _fallback_generation_config_cls():
    class _FallbackGenerationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def from_model_config(cls, config):
            payload = {
                "bos_token_id": getattr(config, "bos_token_id", None),
                "eos_token_id": getattr(config, "eos_token_id", None),
                "pad_token_id": getattr(config, "pad_token_id", None),
                "decoder_start_token_id": getattr(config, "decoder_start_token_id", None),
                "max_new_tokens": getattr(config, "max_new_tokens", 256),
                "do_sample": getattr(config, "do_sample", False),
                "temperature": getattr(config, "temperature", 1.0),
                "top_p": getattr(config, "top_p", 1.0),
                "top_k": getattr(config, "top_k", 50),
            }
            return cls(**payload)

    return _FallbackGenerationConfig


def _resolve_generation_config_cls():
    with _hf_transformers_context() as hf_transformers:
        generation_config = getattr(hf_transformers, "GenerationConfig", None)
    if generation_config is not None:
        return generation_config
    try:
        compat_mod = _import_local_transformers_module("ppo_compat")
        generation_config = getattr(compat_mod, "GenerationConfig", None)
        if generation_config is not None:
            return generation_config
    except Exception:
        pass
    return _fallback_generation_config_cls()


def _attach_generation_config(model) -> None:
    if hasattr(model, "generation_config"):
        return
    GenerationConfig = _resolve_generation_config_cls()
    model.generation_config = GenerationConfig.from_model_config(model.config)


def _maybe_move_model(model, device_map) -> None:
    if isinstance(device_map, str) and device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        model.to(device_map)


def load_tokenizer(model_args) -> TokenizerModule:
    tokenization_mod = _import_local_transformers_module("tokenization")
    processing_mod = _import_local_transformers_module("processing")
    image_processing_mod = _import_local_transformers_module("image_processing")

    tokenizer = tokenization_mod.Tokenizer(model_args.model_name_or_path)
    image_processor = image_processing_mod.Gemma3ImageProcessor.from_pretrained(model_args.model_name_or_path)
    processor = processing_mod.Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)

    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", getattr(model_args, "image_max_pixels", None))
    setattr(processor, "image_min_pixels", getattr(model_args, "image_min_pixels", None))

    return {"tokenizer": tokenizer, "processor": processor}


def load_model(tokenizer, model_args, finetuning_args, is_trainable=True):
    configuration_mod = _import_local_transformers_module("configuration")
    modeling_mod = _import_local_transformers_module("modeling")

    model_path_str = model_args.model_name_or_path
    if not Path(model_path_str).is_dir():
        try:
            from huggingface_hub import snapshot_download
            model_path_str = snapshot_download(model_path_str)
        except Exception:
            pass
    config = configuration_mod.load_config(model_path_str)
    resolved_dtype = _resolve_runtime_dtype(config, getattr(model_args, "compute_dtype", None))
    model = modeling_mod.Gemma3ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(_load_state_dict(Path(model_path_str)), strict=False)
    if missing:
        logger.info("Missing keys while loading model: %s", sorted(missing)[:20])
    if unexpected:
        logger.info("Unexpected keys while loading model: %s", sorted(unexpected)[:20])

    if getattr(model_args, "resize_vocab", False) and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if resolved_dtype is not None:
        model = model.to(dtype=resolved_dtype)
        model_args.compute_dtype = resolved_dtype

    if finetuning_args.finetuning_type == "lora" and is_trainable:
        model = _setup_lora(model, finetuning_args)

    if getattr(finetuning_args, "freeze_vision_tower", True):
        for name, parameter in model.named_parameters():
            if "vision_tower" in name or "vision_model" in name or "visual" in name:
                parameter.requires_grad = False

    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        for name, parameter in model.named_parameters():
            if "multi_modal_projector" in name or "projector" in name or "merger" in name:
                parameter.requires_grad = False

    if is_trainable:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()

    if not hasattr(model, "generate") and hasattr(model, "prepare_inputs_for_generation"):
        with _hf_transformers_context() as hf_transformers:
            GenerationMixin = hf_transformers.GenerationMixin
        generated_cls = type(
            f"{model.__class__.__name__}WithGenerationMixin",
            (model.__class__, GenerationMixin),
            {},
        )
        model.__class__ = generated_cls

    _attach_generation_config(model)
    _maybe_move_model(model, getattr(model_args, "device_map", None))
    _log_param_counts(model)
    return model


def _setup_lora(model, finetuning_args):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        logger.warning("peft is not installed; continuing without LoRA.")
        return model

    if finetuning_args.lora_target == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [token.strip() for token in finetuning_args.lora_target.split(",") if token.strip()]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=finetuning_args.lora_rank,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        use_rslora=getattr(finetuning_args, "use_rslora", False),
        use_dora=getattr(finetuning_args, "use_dora", False),
    )
    return get_peft_model(model, lora_config)


def _log_param_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")




def _ppo_call_with_supported_kwargs(fn, **kwargs):
    import inspect

    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**accepted)


def _ppo_value_head_hidden_size(model):
    config = getattr(model, "config", None)
    if config is None:
        return None
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return hidden_size
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config(decoder=True)
        except TypeError:
            text_config = get_text_config()
        if text_config is not None:
            hidden_size = getattr(text_config, "hidden_size", None)
            if hidden_size is not None:
                return hidden_size
    return None


def _ppo_find_lm_head(model):
    direct = getattr(model, "lm_head", None)
    if direct is not None:
        return direct
    for name, module in model.named_modules():
        if name.endswith("lm_head"):
            return module
    return None


def _ppo_output_has_values(output):
    value = None
    if isinstance(output, dict):
        value = output.get("values")
    else:
        value = getattr(output, "values", None)
    return value is not None and not callable(value) and hasattr(value, "shape")


def _ppo_inject_values(output, values):
    if isinstance(output, dict):
        result = dict(output)
        result["values"] = values
        return result
    try:
        setattr(output, "values", values)
        return output
    except Exception:
        pass
    if hasattr(output, "logits"):
        return {"logits": output.logits, "values": values, "raw_output": output}
    return {"values": values, "raw_output": output}


def _ppo_patch_forward(model):
    if getattr(model, "_ppo_values_forward_patched", False):
        return model
    lm_head = _ppo_find_lm_head(model)
    if lm_head is None:
        return model
    original_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        captured = {}

        def _capture(module, inputs):
            if inputs:
                captured["hidden"] = inputs[0]

        hook = lm_head.register_forward_pre_hook(_capture)
        try:
            output = original_forward(*args, **kwargs)
        finally:
            hook.remove()

        if _ppo_output_has_values(output):
            return output
        v_head = getattr(model, "v_head", None)
        hidden = captured.get("hidden")
        if v_head is None or hidden is None or not hasattr(hidden, "shape"):
            return output
        values = v_head(hidden.to(v_head.weight.device)).squeeze(-1)
        return _ppo_inject_values(output, values)

    model.forward = _wrapped_forward
    model._ppo_values_forward_patched = True
    return model


def _ppo_attach_value_head(model, model_args):
    import torch

    if hasattr(model, "v_head"):
        return model
    hidden_size = _ppo_value_head_hidden_size(model)
    if hidden_size is None:
        return model
    v_head = torch.nn.Linear(hidden_size, 1, bias=False)
    v_head.weight.data.zero_()
    try:
        first_param = next(model.parameters())
        v_head = v_head.to(device=first_param.device, dtype=first_param.dtype)
    except Exception:
        compute_dtype = getattr(model_args, "compute_dtype", None)
        if compute_dtype is not None:
            v_head = v_head.to(dtype=compute_dtype)
    model.v_head = v_head
    return model


_BASE_LOAD_MODEL = load_model


def load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=False, backend_device=None, **kwargs):
    model = _ppo_call_with_supported_kwargs(
        _BASE_LOAD_MODEL,
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=is_trainable,
        **kwargs,
    )
    if add_valuehead:
        model = _ppo_attach_value_head(model, model_args)
        model = _ppo_patch_forward(model)
    return model
