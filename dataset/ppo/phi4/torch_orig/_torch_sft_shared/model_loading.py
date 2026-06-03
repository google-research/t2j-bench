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

"""Pure-Torch model loading utilities without AutoModel loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


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
        path = model_path / file_name
        if path.is_file():
            return [path]
    raise FileNotFoundError(f"No supported checkpoint file found under {model_path}")


def _load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_safetensors(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for checkpoint_path in _iter_checkpoint_files(model_path):
        state_dict.update(_load_state_dict_file(checkpoint_path))
    return state_dict


def _resolve_model_class(config, family_slug: str):
    import transformers

    architecture_names = list(getattr(config, "architectures", None) or [])
    fallback_by_type = {
        "bloom": "BloomForCausalLM",
        "falcon": "FalconForCausalLM",
        "gemma": "GemmaForCausalLM",
        "gemma2": "Gemma2ForCausalLM",
        "gpt2": "GPT2LMHeadModel",
        "llama": "LlamaForCausalLM",
        "llava": "LlavaForConditionalGeneration",
        "llava_next": "LlavaNextForConditionalGeneration",
        "llava_next_video": "LlavaNextVideoForConditionalGeneration",
        "mistral": "MistralForCausalLM",
        "phi": "PhiForCausalLM",
        "phi3": "Phi3ForCausalLM",
        "qwen2_vl": "Qwen2VLForConditionalGeneration",
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "qwen3_vl": "Qwen3VLForConditionalGeneration",
        "paligemma": "PaliGemmaForConditionalGeneration",
        "qwen2_audio": "Qwen2AudioForConditionalGeneration",
        "starcoder2": "Starcoder2ForCausalLM",
    }
    if not architecture_names and getattr(config, "model_type", None) in fallback_by_type:
        architecture_names.append(fallback_by_type[config.model_type])
    for arch_name in architecture_names:
        model_cls = getattr(transformers, arch_name, None)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Could not resolve a torch model class for family `{family_slug}` "
        f"with architectures={architecture_names!r} model_type={getattr(config, 'model_type', None)!r}."
    )


def _apply_lora(model, finetuning_args):
    if getattr(finetuning_args, "finetuning_type", "full") != "lora":
        return model
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


def _freeze_multimodal_modules(model, finetuning_args):
    freeze_patterns = []
    if getattr(finetuning_args, "freeze_vision_tower", True):
        freeze_patterns.extend(["vision_tower", "vision_model", "visual", "vision_encoder", "image_tower"])
    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        freeze_patterns.extend(["multi_modal_projector", "projector", "mm_projector"])
    if not freeze_patterns:
        return
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in freeze_patterns):
            parameter.requires_grad = False


def load_tokenizer(model_args) -> TokenizerModule:
    from transformers import AutoProcessor, AutoTokenizer

    init_kwargs = {
        "trust_remote_code": getattr(model_args, "trust_remote_code", True),
        "cache_dir": getattr(model_args, "cache_dir", None),
        "revision": getattr(model_args, "model_revision", "main"),
        "token": getattr(model_args, "hf_hub_token", None),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=getattr(model_args, "use_fast_tokenizer", True),
        padding_side="right",
        **init_kwargs,
    )
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=getattr(model_args, "use_fast_tokenizer", True),
            **init_kwargs,
        )
    except Exception:
        processor = None
    if processor is not None and getattr(processor, "tokenizer", None) is None:
        setattr(processor, "tokenizer", tokenizer)
    return {"tokenizer": tokenizer, "processor": processor}


def load_model(tokenizer, model_args, finetuning_args, family_slug: str, is_trainable: bool = True):
    from transformers import AutoConfig

    model_path = Path(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
        cache_dir=getattr(model_args, "cache_dir", None),
        revision=getattr(model_args, "model_revision", "main"),
        token=getattr(model_args, "hf_hub_token", None),
    )
    model_cls = _resolve_model_class(config, family_slug)
    model = model_cls(config)
    missing, unexpected = model.load_state_dict(_load_state_dict(model_path), strict=False)
    if missing:
        logger.info("Missing keys while loading %s: %s", family_slug, sorted(missing)[:20])
    if unexpected:
        logger.info("Unexpected keys while loading %s: %s", family_slug, sorted(unexpected)[:20])

    if getattr(model_args, "resize_vocab", False) and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model_args, "compute_dtype", None) is not None:
        model = model.to(dtype=model_args.compute_dtype)

    if is_trainable:
        model = _apply_lora(model, finetuning_args)
        _freeze_multimodal_modules(model, finetuning_args)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    return model
