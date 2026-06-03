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

"""Repo-local pure-Torch model and tokenizer loading for Qwen3-VL."""

import importlib
import importlib.util
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

import torch
from huggingface_hub import snapshot_download
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


def _resolve_model_path(model_name_or_path: str) -> Path:
    """Resolve a HuggingFace model ID or local path to a local directory."""
    path = Path(model_name_or_path)
    if path.is_dir():
        return path
    return Path(snapshot_download(repo_id=model_name_or_path))


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
    model.generation_config = GenerationConfig.from_model_config(model.config)


def _maybe_move_model(model, device_map) -> None:
    if isinstance(device_map, str) and device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        model.to(device_map)


def load_tokenizer(model_args) -> TokenizerModule:
    tokenization_mod = _import_local_transformers_module("tokenization")
    processing_mod = _import_local_transformers_module("processing")

    tokenizer = tokenization_mod.Tokenizer.from_pretrained(model_args.model_name_or_path)
    processor = processing_mod.Qwen3VLProcessor.from_pretrained(model_args.model_name_or_path)

    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", getattr(model_args, "image_max_pixels", None))
    setattr(processor, "image_min_pixels", getattr(model_args, "image_min_pixels", None))
    setattr(processor, "video_max_pixels", getattr(model_args, "video_max_pixels", None))
    setattr(processor, "video_min_pixels", getattr(model_args, "video_min_pixels", None))
    setattr(processor, "video_fps", getattr(model_args, "video_fps", None))
    setattr(processor, "video_maxlen", getattr(model_args, "video_maxlen", None))

    return {"tokenizer": tokenizer, "processor": processor}


def load_model(tokenizer, model_args, finetuning_args, is_trainable=False):
    configuration_mod = _import_local_transformers_module("configuration")
    modeling_mod = _import_local_transformers_module("modeling")

    config = configuration_mod.AutoConfig.from_pretrained(model_args.model_name_or_path)
    if getattr(model_args, "compute_dtype", None) is not None:
        config.torch_dtype = model_args.compute_dtype
    if getattr(model_args, "flash_attn", None) == "fa2":
        config._attn_implementation = "flash_attention_2"
    elif getattr(model_args, "flash_attn", None) == "sdpa":
        config._attn_implementation = "sdpa"
    elif getattr(model_args, "flash_attn", None) == "disabled":
        config._attn_implementation = "eager"

    model = modeling_mod.Qwen3VLForConditionalGeneration(config)
    resolved_path = _resolve_model_path(model_args.model_name_or_path)
    missing, unexpected = model.load_state_dict(_load_state_dict(resolved_path), strict=False)
    if missing:
        logger.info("Missing keys while loading model: %s", sorted(missing)[:20])
    if unexpected:
        logger.info("Unexpected keys while loading model: %s", sorted(unexpected)[:20])

    if getattr(model_args, "resize_vocab", False) and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if getattr(model_args, "compute_dtype", None) is not None:
        model = model.to(dtype=model_args.compute_dtype)

    if finetuning_args.finetuning_type == "lora" and is_trainable:
        model = _setup_lora(model, finetuning_args)
    elif is_trainable:
        if getattr(finetuning_args, "freeze_vision_tower", True):
            for name, parameter in model.named_parameters():
                if "visual" in name or "vision" in name:
                    parameter.requires_grad = False
        if getattr(finetuning_args, "freeze_multi_modal_projector", True):
            for name, parameter in model.named_parameters():
                if "multi_modal_projector" in name or "merger" in name:
                    parameter.requires_grad = False

    if is_trainable:
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
        target_modules = "all-linear"
    else:
        target_modules = [token.strip() for token in finetuning_args.lora_target.split(",") if token.strip()]

    modules_to_save = None
    if getattr(finetuning_args, "additional_target", None):
        modules_to_save = [m.strip() for m in finetuning_args.additional_target.split(",")]

    lora_alpha = (
        finetuning_args.lora_alpha
        if finetuning_args.lora_alpha is not None
        else finetuning_args.lora_rank * 2
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=finetuning_args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        use_rslora=finetuning_args.use_rslora,
        use_dora=finetuning_args.use_dora,
    )
    return get_peft_model(model, lora_config)


def _log_param_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")
