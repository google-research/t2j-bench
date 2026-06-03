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

"""Vendored pure-Torch model and tokenizer loading."""

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
_LOCAL_TRANSFORMERS_PACKAGE = "_paligemma_sft_torch_orig_transformers"
_DEFAULT_CONFIG_CLASS = 'PaliGemmaConfig'
_DEFAULT_ARCHITECTURE = 'PaliGemmaForConditionalGeneration'


class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def _resolve_model_path(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return candidate
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=model_name_or_path)).resolve()
    except Exception:
        return candidate


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


def _resolve_local_config_class():
    module = _import_local_transformers_module("configuration")
    if _DEFAULT_CONFIG_CLASS:
        config_class = getattr(module, _DEFAULT_CONFIG_CLASS, None)
        if config_class is not None:
            return config_class
    with _hf_transformers_context() as hf_transformers:
        base_class = hf_transformers.PretrainedConfig
    candidates = []
    for name in dir(module):
        value = getattr(module, name)
        if isinstance(value, type) and issubclass(value, base_class) and value is not base_class:
            candidates.append(value)
    if len(candidates) == 1:
        return candidates[0]
    preferred = [item for item in candidates if item.__name__.endswith("Config")]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(f"Could not uniquely resolve config class from local transformers: {[item.__name__ for item in candidates]}")


def _resolve_local_model_class(config):
    module = _import_local_transformers_module("modeling")
    architecture_names = list(getattr(config, "architectures", None) or [])
    if not architecture_names and _DEFAULT_ARCHITECTURE:
        architecture_names.append(_DEFAULT_ARCHITECTURE)
    expanded_architecture_names = []
    for arch_name in architecture_names:
        expanded_architecture_names.append(arch_name)
        if arch_name.endswith("Model"):
            stem = arch_name[:-5]
            expanded_architecture_names.extend(
                [
                    f"{stem}ForConditionalGeneration",
                    f"{stem}ForCausalLM",
                    f"{stem}LMHeadModel",
                ]
            )
    for arch_name in expanded_architecture_names:
        model_cls = getattr(module, arch_name, None)
        if model_cls is not None:
            return model_cls
    with _hf_transformers_context() as hf_transformers:
        base_class = hf_transformers.PreTrainedModel
    fallback = []
    for name in dir(module):
        value = getattr(module, name)
        if isinstance(value, type) and issubclass(value, base_class) and value is not base_class:
            fallback.append(value)
    preferred = [item for item in fallback if item.__name__.endswith(("ForCausalLM", "ForConditionalGeneration", "LMHeadModel"))]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(f"Could not resolve local model class for architectures={architecture_names!r} fallback={[item.__name__ for item in fallback]}")


def _get_init_kwargs(model_args) -> dict[str, Any]:
    return {
        "trust_remote_code": getattr(model_args, "trust_remote_code", True),
        "cache_dir": getattr(model_args, "cache_dir", None),
        "revision": getattr(model_args, "model_revision", "main"),
        "token": getattr(model_args, "hf_hub_token", None),
    }


def _get_local_init_kwargs(model_args) -> dict[str, Any]:
    return {
        "trust_remote_code": getattr(model_args, "trust_remote_code", True),
        "cache_dir": getattr(model_args, "cache_dir", None),
        "revision": getattr(model_args, "model_revision", "main"),
        "token": getattr(model_args, "hf_hub_token", None),
        "local_files_only": True,
    }


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _build_local_tokenizer_kwargs(tokenizer_cls, model_path: Path) -> dict[str, Any]:
    tokenizer_config = _load_json_if_exists(model_path / "tokenizer_config.json")
    kwargs: dict[str, Any] = {}
    for field_name, file_name in getattr(tokenizer_cls, "vocab_files_names", {}).items():
        candidate = model_path / Path(file_name).name
        if candidate.is_file():
            kwargs[field_name] = str(candidate)
    for field_name in (
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "cls_token",
        "sep_token",
        "mask_token",
        "add_bos_token",
        "add_eos_token",
        "clean_up_tokenization_spaces",
        "sp_model_kwargs",
    ):
        if field_name in tokenizer_config:
            kwargs[field_name] = tokenizer_config[field_name]
    return kwargs


def _load_local_tokenizer(tokenizer_cls, model_args):
    model_path = _resolve_model_path(model_args.model_name_or_path)
    try:
        return tokenizer_cls.from_pretrained(str(model_path), **_get_local_init_kwargs(model_args))
    except Exception:
        tokenizer_kwargs = _build_local_tokenizer_kwargs(tokenizer_cls, model_path)
        if not tokenizer_kwargs:
            raise
        return tokenizer_cls(**tokenizer_kwargs)


def _load_local_processor(processor_cls, model_args):
    model_path = _resolve_model_path(model_args.model_name_or_path)
    return processor_cls.from_pretrained(str(model_path), **_get_local_init_kwargs(model_args))


def _normalize_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    layer_types = config_dict.get("layer_types")
    num_hidden_layers = config_dict.get("num_hidden_layers")
    if isinstance(layer_types, list) and isinstance(num_hidden_layers, int) and layer_types and len(layer_types) != num_hidden_layers:
        config_dict["layer_types"] = [layer_types[0] for _ in range(num_hidden_layers)]
    rope_scaling = config_dict.get("rope_scaling")
    if config_dict.get("model_type") in {"minicpm"} and isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        if rope_type == "default":
            config_dict["rope_scaling"] = None
    if config_dict.get("model_type") == "yuan":
        config_dict.setdefault("eod_token", config_dict.get("eos_token_id"))
        config_dict.setdefault("sep_token", config_dict.get("eos_token_id"))
        config_dict.setdefault("use_loss_mask", False)
        config_dict.setdefault("max_position_embeddings", config_dict.get("model_max_length", 8192))
        config_dict.setdefault("causal_mask", True)
        config_dict.setdefault("use_flash_attention", False)
        config_dict.setdefault("use_shareqk", False)
        config_dict.setdefault("reset_attention_mask", False)
        config_dict.setdefault("reset_position_ids", False)
    if config_dict.get("model_type") == "mimo" and config_dict.get("rope_parameters") is None:
        config_dict["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": config_dict.get("rope_theta", 10000.0),
        }
    return config_dict


def _normalize_config(config):
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None and getattr(config, "model_type", None) in {"mimo"}:
        config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(config, "rope_theta", 10000.0),
        }
    rope_scaling = getattr(config, "rope_scaling", None)
    if getattr(config, "model_type", None) in {"minicpm"} and isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        if rope_type == "default":
            config.rope_scaling = None
    layer_types = getattr(config, "layer_types", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if isinstance(layer_types, list) and isinstance(num_hidden_layers, int) and len(layer_types) != num_hidden_layers and layer_types:
        config.layer_types = [layer_types[0] for _ in range(num_hidden_layers)]
    if getattr(config, "model_type", None) == "yuan":
        if not hasattr(config, "eod_token"):
            config.eod_token = getattr(config, "eos_token_id", None)
        if not hasattr(config, "sep_token"):
            config.sep_token = getattr(config, "eos_token_id", None)
        if not hasattr(config, "use_loss_mask"):
            config.use_loss_mask = False
        if not hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = getattr(config, "model_max_length", 8192)
        if not hasattr(config, "causal_mask"):
            config.causal_mask = True
        if not hasattr(config, "use_flash_attention"):
            config.use_flash_attention = False
        if not hasattr(config, "use_shareqk"):
            config.use_shareqk = False
        if not hasattr(config, "reset_attention_mask"):
            config.reset_attention_mask = False
        if not hasattr(config, "reset_position_ids"):
            config.reset_position_ids = False
    if getattr(config, "model_type", None) == "telechat":
        if not hasattr(config, "base_seqlen"):
            config.base_seqlen = getattr(config, "training_seqlen", getattr(config, "max_position_embeddings", 8192))
        if not hasattr(config, "flash_attn"):
            config.flash_attn = False
    return config


def _materialize_nested_configs(config):
    module = _import_local_transformers_module("configuration")
    with _hf_transformers_context() as hf_transformers:
        base_class = hf_transformers.PretrainedConfig
        AutoConfig = hf_transformers.AutoConfig

    def _instantiate_subconfig(config_cls, nested_config):
        if not isinstance(nested_config, dict):
            return nested_config
        normalized = _normalize_config_dict(dict(nested_config))
        if config_cls is AutoConfig or getattr(config_cls, "__name__", "") == "AutoConfig":
            model_type = normalized.get("model_type")
            if model_type:
                try:
                    return AutoConfig.for_model(model_type, **{key: value for key, value in normalized.items() if key != "model_type"})
                except Exception:
                    return nested_config
            return nested_config
        if hasattr(config_cls, "from_dict"):
            try:
                return config_cls.from_dict(normalized)
            except Exception:
                pass
        try:
            return config_cls(**normalized)
        except Exception:
            return nested_config

    def _finalize(current_config, seen):
        if not isinstance(current_config, base_class):
            return current_config
        config_id = id(current_config)
        if config_id in seen:
            return current_config
        seen.add(config_id)
        for name in dir(module):
            value = getattr(module, name)
            if not isinstance(value, type) or not issubclass(value, base_class) or value is base_class:
                continue
            base_config_key = getattr(value, "base_config_key", None)
            if not base_config_key or not hasattr(current_config, base_config_key):
                continue
            nested_config = getattr(current_config, base_config_key)
            setattr(current_config, base_config_key, _instantiate_subconfig(value, nested_config))
        sub_configs = getattr(current_config.__class__, "sub_configs", None)
        if isinstance(sub_configs, dict):
            for key, sub_config_cls in sub_configs.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, _instantiate_subconfig(sub_config_cls, getattr(current_config, key)))
        post_init = getattr(current_config, "__post_init__", None)
        if callable(post_init):
            try:
                post_init()
            except TypeError:
                try:
                    post_init(**{})
                except Exception:
                    pass
            except Exception:
                pass
        for value in list(vars(current_config).values()):
            if isinstance(value, base_class):
                _finalize(value, seen)
        return current_config

    return _finalize(config, set())


def _load_local_config(model_path: Path, config_class, model_args):
    try:
        config = config_class.from_pretrained(str(model_path), **_get_local_init_kwargs(model_args))
    except Exception:
        config = config_class.from_dict(_normalize_config_dict(_read_json(model_path / "config.json")))
    config = _normalize_config(config)
    return _materialize_nested_configs(config)


def _apply_lora(model, finetuning_args):
    if getattr(finetuning_args, "finetuning_type", "full") != "lora":
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        logger.warning("peft is not installed; continuing without LoRA.")
        return model
    if finetuning_args.lora_target == "all":
        target_modules = "all-linear"
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


def _freeze_multimodal_modules(model, finetuning_args):
    freeze_patterns = []
    if getattr(finetuning_args, "freeze_vision_tower", True):
        freeze_patterns.extend(["vision_tower", "vision_model", "visual", "vision_encoder", "image_tower"])
    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        freeze_patterns.extend(["multi_modal_projector", "projector", "mm_projector", "merger"])
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in freeze_patterns):
            parameter.requires_grad = False


def load_tokenizer(model_args) -> TokenizerModule:
    tokenizer = None
    processor = None
    with _hf_transformers_context() as hf_transformers:
        AutoTokenizer = hf_transformers.AutoTokenizer
        AutoProcessor = hf_transformers.AutoProcessor
        PreTrainedTokenizerBase = hf_transformers.PreTrainedTokenizerBase
        ProcessorMixin = hf_transformers.ProcessorMixin
        try:
            tokenization_module = _import_local_transformers_module("tokenization")
        except Exception:
            tokenization_module = None
        tokenizer_cls = None
        if tokenization_module is not None:
            tokenizer_candidates = []
            for name in dir(tokenization_module):
                value = getattr(tokenization_module, name)
                if (
                    isinstance(value, type)
                    and issubclass(value, PreTrainedTokenizerBase)
                    and value is not PreTrainedTokenizerBase
                    and value.__module__.startswith(_LOCAL_TRANSFORMERS_PACKAGE)
                ):
                    tokenizer_candidates.append(value)
            fast_candidates = [item for item in tokenizer_candidates if item.__name__.endswith("Fast")]
            if len(fast_candidates) == 1:
                tokenizer_cls = fast_candidates[0]
            elif len(tokenizer_candidates) == 1:
                tokenizer_cls = tokenizer_candidates[0]
        if tokenizer_cls is not None:
            tokenizer = _load_local_tokenizer(tokenizer_cls, model_args)
            tokenizer.padding_side = "right"
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=getattr(model_args, "use_fast_tokenizer", True),
                padding_side="right",
                **_get_init_kwargs(model_args),
            )
        try:
            processing_module = _import_local_transformers_module("processing")
        except Exception:
            processing_module = None
        processor_cls = None
        if processing_module is not None:
            processor_candidates = []
            for name in dir(processing_module):
                value = getattr(processing_module, name)
                if (
                    isinstance(value, type)
                    and issubclass(value, ProcessorMixin)
                    and value is not ProcessorMixin
                    and value.__module__.startswith(_LOCAL_TRANSFORMERS_PACKAGE)
                ):
                    processor_candidates.append(value)
            if len(processor_candidates) == 1:
                processor_cls = processor_candidates[0]
        try:
            if processor_cls is not None:
                processor = _load_local_processor(processor_cls, model_args)
            else:
                processor = AutoProcessor.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=getattr(model_args, "use_fast_tokenizer", True),
                    **_get_init_kwargs(model_args),
                )
        except Exception:
            processor = None
    if processor is not None and getattr(processor, "tokenizer", None) is None:
        setattr(processor, "tokenizer", tokenizer)
    return {"tokenizer": tokenizer, "processor": processor}


def load_model(tokenizer, model_args, finetuning_args, is_trainable=True):
    model_path = _resolve_model_path(model_args.model_name_or_path)
    config_class = _resolve_local_config_class()
    config = _load_local_config(model_path, config_class, model_args)
    model_cls = _resolve_local_model_class(config)
    model = model_cls(config)
    missing, unexpected = model.load_state_dict(_load_state_dict(model_path), strict=False)
    if missing:
        logger.info("Missing keys while loading model: %s", sorted(missing)[:20])
    if unexpected:
        logger.info("Unexpected keys while loading model: %s", sorted(unexpected)[:20])
    if getattr(model_args, "resize_vocab", False) and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    if getattr(model_args, "compute_dtype", None) is not None:
        model = model.to(dtype=model_args.compute_dtype)
    if is_trainable:
        model = _apply_lora(model, finetuning_args)
        _freeze_multimodal_modules(model, finetuning_args)
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
    if not hasattr(model, "generation_config"):
        with _hf_transformers_context() as hf_transformers:
            GenerationConfig = hf_transformers.GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)
    return model

def load_ref_model(tokenizer, model_args, finetuning_args, dpo_args):
    """Load the DPO reference model using the same repo-local loader path."""
    if not getattr(dpo_args, "use_ref_model", True):
        return None
    import copy

    ref_model_args = copy.deepcopy(model_args)
    ref_finetuning_args = copy.deepcopy(finetuning_args)
    ref_model_name = getattr(dpo_args, "ref_model", None)
    if ref_model_name:
        ref_model_args.model_name_or_path = ref_model_name
    return load_model(tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False)
