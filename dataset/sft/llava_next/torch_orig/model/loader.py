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

"""Repo-local pure-Torch model and tokenizer loading for LLaVA-Next."""

import importlib
import importlib.util
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

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


def _resolve_model_path(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(model_path), local_files_only=True))
    except Exception:
        return model_path


def _iter_checkpoint_files(model_path: Path) -> list[Path]:
    model_path = _resolve_model_path(model_path)
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
    if hasattr(GenerationConfig, "from_model_config"):
        try:
            model.generation_config = GenerationConfig.from_model_config(model.config)
            return
        except Exception:
            pass

    config_values = {
        "bos_token_id": getattr(model.config, "bos_token_id", None),
        "eos_token_id": getattr(model.config, "eos_token_id", None),
        "pad_token_id": getattr(model.config, "pad_token_id", None),
        "decoder_start_token_id": getattr(model.config, "decoder_start_token_id", None),
        "max_new_tokens": getattr(model.config, "max_new_tokens", 256),
        "do_sample": getattr(model.config, "do_sample", False),
        "temperature": getattr(model.config, "temperature", 1.0),
        "top_p": getattr(model.config, "top_p", 1.0),
        "top_k": getattr(model.config, "top_k", 50),
    }
    try:
        model.generation_config = GenerationConfig(**config_values)
        return
    except Exception:
        pass

    generation_config = GenerationConfig()
    for key, value in config_values.items():
        try:
            setattr(generation_config, key, value)
        except Exception:
            continue
    model.generation_config = generation_config


def _maybe_move_model(model, device_map) -> None:
    if isinstance(device_map, str) and device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        model.to(device_map)


def load_tokenizer(model_args) -> TokenizerModule:
    tokenization_mod = _import_local_transformers_module("tokenization")
    image_processing_mod = _import_local_transformers_module("image_processing")
    processing_mod = _import_local_transformers_module("processing")

    tokenizer = tokenization_mod.Tokenizer(model_args.model_name_or_path)
    image_processor = image_processing_mod.LlavaNextImageProcessor.from_pretrained(model_args.model_name_or_path)
    processor = processing_mod.LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return {"tokenizer": tokenizer, "processor": processor}


def load_model(tokenizer, model_args, finetuning_args, is_trainable=True):
    configuration_mod = _import_local_transformers_module("configuration")
    modeling_mod = _import_local_transformers_module("modeling")

    config = configuration_mod.load_config(model_args.model_name_or_path)
    model = modeling_mod.LlavaNextForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(_load_state_dict(Path(model_args.model_name_or_path)), strict=False)
    if missing:
        logger.info("Missing keys while loading model: %s", sorted(missing)[:20])
    if unexpected:
        logger.info("Unexpected keys while loading model: %s", sorted(unexpected)[:20])

    if getattr(model_args, "resize_vocab", False) and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model_args, "compute_dtype", None) is not None:
        model = model.to(dtype=model_args.compute_dtype)

    if finetuning_args.finetuning_type == "lora" and is_trainable:
        model = _setup_lora(model, finetuning_args)

    if getattr(finetuning_args, "freeze_vision_tower", True):
        for name, parameter in model.named_parameters():
            if "vision_tower" in name or "vision_model" in name or "visual" in name:
                parameter.requires_grad = False

    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        for name, parameter in model.named_parameters():
            if "multi_modal_projector" in name or "projector" in name or "image_newline" in name:
                parameter.requires_grad = False

    if is_trainable:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()

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
