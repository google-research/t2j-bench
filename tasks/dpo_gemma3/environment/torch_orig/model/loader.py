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

"""Model and tokenizer loading for Gemma3."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors
import re
import copy

logger = logging.getLogger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


_MODEL_ROOT = Path("/data")


def _resolve_model_path(model_name_or_path) -> Path:
    """Resolve a model id to the directory its weights were baked into."""
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return candidate.resolve()
    local = _MODEL_ROOT / model_name_or_path
    if local.is_dir():
        return local
    raise FileNotFoundError(
        f"Model {str(model_name_or_path)!r} not found at {local}. Weights are baked into "
        "the image at build time; nothing is fetched at run time."
    )


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
    try:
        from transformers_impl import ppo_compat as compat_mod
        generation_config = getattr(compat_mod, "GenerationConfig", None)
        if generation_config is not None:
            return generation_config
    except Exception:
        pass
    return _fallback_generation_config_cls()


def _resolve_generation_mixin():
    """Torch-free GenerationMixin resolver."""
    try:
        from transformers_impl import generation as generation_mod
        return getattr(generation_mod, "GenerationMixin", None)
    except Exception:
        return None


def _attach_generation_config(model) -> None:
    if hasattr(model, "generation_config"):
        return
    GenerationConfig = _resolve_generation_config_cls()
    model.generation_config = GenerationConfig.from_model_config(model.config)


def _maybe_move_model(model, device_map) -> None:
    if isinstance(device_map, str) and device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        model.to(device_map)


def load_tokenizer(model_args) -> TokenizerModule:
    from transformers_impl import tokenization as tokenization_mod
    from transformers_impl import processing as processing_mod
    from transformers_impl import image_processing as image_processing_mod

    model_path = str(_resolve_model_path(model_args.model_name_or_path))
    tokenizer = tokenization_mod.Tokenizer(model_path)
    image_processor = image_processing_mod.Gemma3ImageProcessor.from_pretrained(model_path)
    processor = processing_mod.Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)

    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", getattr(model_args, "image_max_pixels", None))
    setattr(processor, "image_min_pixels", getattr(model_args, "image_min_pixels", None))

    return {"tokenizer": tokenizer, "processor": processor}


def _tied_to_input_embeddings(model, key: str) -> bool:
    """Is ``key`` a head that shares storage with the input embedding?"""
    if not key.endswith("lm_head.weight"):
        return False
    get_embeddings = getattr(model, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    embed_weight = getattr(embeddings, "weight", None)
    return (head_weight is not None and embed_weight is not None
            and head_weight.data_ptr() == embed_weight.data_ptr())


def _from_pretrained_or_raise(model_cls, config, model_path, dtype=None):
    """Instantiate ``model_cls`` and load the checkpoint at ``model_path`` into it."""
    model_path = Path(str(model_path))
    kwargs = {"local_files_only": True, "output_loading_info": True}
    attn_implementation = getattr(config, "_attn_implementation", None)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if dtype is not None:
        kwargs["dtype"] = dtype

    def _raw_state_dict():
        """Last resort: build the model and match checkpoint keys by raw name."""
        model = model_cls(config)
        missing, unexpected = model.load_state_dict(_load_state_dict(model_path), strict=False)
        return model, {"missing_keys": missing, "unexpected_keys": unexpected}

    attempts: list[str] = []
    for label, attempt in (
        ("from_pretrained()",
         lambda: model_cls.from_pretrained(str(model_path), **kwargs)),
        ("from_pretrained(config=...)",
         lambda: model_cls.from_pretrained(str(model_path), config=config, **kwargs)),
        ("raw state dict", _raw_state_dict),
    ):
        model = info = None
        try:
            model, info = attempt()
        except Exception as error:
            attempts.append(f"{label} raised {type(error).__name__}: {error}")
            continue
        still_random = [key for key in (info.get("missing_keys") or ())
                        if not _tied_to_input_embeddings(model, key)]
        mismatched = list(info.get("mismatched_keys") or ())
        if still_random or mismatched:
            attempts.append(
                f"{label} left {len(still_random)} parameter(s) at random init, e.g. "
                f"{sorted(still_random)[:5]}; mismatched={mismatched[:5]}")
            continue
        unexpected = list(info.get("unexpected_keys") or ())
        if unexpected:
            logger.info("Unexpected keys while loading model: %s", sorted(unexpected)[:20])
        return model

    raise ValueError(
        f"{model_cls.__name__} did not receive its pretrained weights from {model_path}. "
        + " | ".join(attempts)
    )


def load_model(tokenizer, model_args, finetuning_args, is_trainable=True):
    from transformers_impl import configuration as configuration_mod
    from transformers_impl import modeling as modeling_mod

    model_path_str = str(_resolve_model_path(model_args.model_name_or_path))
    config = configuration_mod.load_config(model_path_str)
    resolved_dtype = _resolve_runtime_dtype(config, getattr(model_args, "compute_dtype", None))
    model = _from_pretrained_or_raise(modeling_mod.Gemma3ForConditionalGeneration, config, Path(model_path_str), getattr(model_args, "compute_dtype", None))

    if getattr(model_args, "resize_vocab", False) and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if resolved_dtype is not None:
        model = model.to(dtype=resolved_dtype)
        model_args.compute_dtype = resolved_dtype

    if finetuning_args.finetuning_type == "lora" and is_trainable:
        model = _apply_lora(model, finetuning_args)

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
        GenerationMixin = _resolve_generation_mixin()
        if GenerationMixin is not None:
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


def _lora_exclude_pattern(model, finetuning_args, exclude_lm_head=False):
    """Modules LoRA must not adapt, as a regex for ``LoraConfig.exclude_modules``."""

    patterns = []
    if getattr(finetuning_args, "freeze_vision_tower", True):
        patterns.extend(["vision_tower", "vision_model", "visual", "vision_encoder", "image_tower"])
    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        patterns.extend(["multi_modal_projector", "projector", "mm_projector", "merger"])
    if exclude_lm_head:
        patterns.append("lm_head")
    module_names = [name for name, _ in model.named_modules()]
    present = [pattern for pattern in patterns
               if any(pattern in name for name in module_names)]
    if not present:
        return None
    return r".*(?:" + "|".join(re.escape(pattern) for pattern in present) + r").*"


class _LoraLinear(nn.Module):
    """Implementation of peft's ``lora.Linear``."""

    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: int, lora_dropout: float):
        super().__init__()
        self.base_layer = base_layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.r = {"default": r}
        self.lora_alpha = {"default": lora_alpha}
        self.scaling = {"default": lora_alpha / r}
        self.lora_dropout = nn.ModuleDict(
            {"default": nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()}
        )
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias=False)})
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.lora_magnitude_vector = nn.ModuleDict({})
        self._active_adapter = "default"
        self.reset_lora_parameters()
        self._align_adapter_dtype(base_layer.weight.dtype)

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B["default"].weight)

    def _align_adapter_dtype(self, base_dtype: torch.dtype):
        """Replicate peft's post-init dtype handling."""
        if not (base_dtype.is_floating_point or base_dtype.is_complex):
            return
        for linear in (self.lora_A["default"], self.lora_B["default"]):
            weight = linear.weight.data.to(base_dtype)
            if weight.dtype in (torch.float16, torch.bfloat16):
                weight = weight.to(torch.float32)
            linear.weight.data = weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        lora_A = self.lora_A["default"]
        lora_B = self.lora_B["default"]
        dropout = self.lora_dropout["default"]
        scaling = self.scaling["default"]
        x = x.to(lora_A.weight.dtype)
        result = result + lora_B(lora_A(dropout(x))) * scaling
        result = result.to(torch_result_dtype)
        return result


class _LoraModel(nn.Module):
    """Implementation of peft's ``LoraModel`` (a PeftModel's ``base_model``): holds the wrapped model under ``.model`` and delegates unknown attributes to it."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class _PeftModel(nn.Module):
    """Implementation of peft's ``PeftModelForCausalLM``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.base_model = _LoraModel(model)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model.model, name)


def _select_lora_targets(model, target_modules, exclude_pattern):
    """Names of the ``nn.Linear`` modules to adapt, in ``named_modules()`` traversal order."""

    exclude_re = re.compile(exclude_pattern) if exclude_pattern else None
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if exclude_re is not None and exclude_re.fullmatch(name):
            continue
        if target_modules == "all-linear":
            names.append(name)
        else:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in target_modules or name in target_modules:
                names.append(name)
    return names


def _get_submodule(model, name):
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr, getattr(parent, attr)


def _apply_lora(model, finetuning_args):
    if getattr(finetuning_args, "finetuning_type", "full") != "lora":
        return model
    r = int(finetuning_args.lora_rank)
    lora_alpha = int(finetuning_args.lora_alpha)
    lora_dropout = float(finetuning_args.lora_dropout)
    if finetuning_args.lora_target == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [token.strip() for token in finetuning_args.lora_target.split(",") if token.strip()]
    exclude_pattern = _lora_exclude_pattern(
        model, finetuning_args, exclude_lm_head=target_modules == "all-linear"
    )
    target_names = _select_lora_targets(model, target_modules, exclude_pattern)
    for name in target_names:
        parent, attr, module = _get_submodule(model, name)
        setattr(parent, attr, _LoraLinear(module, r, lora_alpha, lora_dropout))
    for param_name, param in model.named_parameters():
        param.requires_grad = (".lora_A." in param_name or ".lora_B." in param_name)
    return _PeftModel(model)


def _log_param_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")

def load_ref_model(tokenizer, model_args, finetuning_args, dpo_args):
    """Load the DPO reference model using the same loader path."""
    if not getattr(dpo_args, "use_ref_model", True):
        return None

    ref_model_args = copy.deepcopy(model_args)
    ref_finetuning_args = copy.deepcopy(finetuning_args)
    ref_model_name = getattr(dpo_args, "ref_model", None)
    if ref_model_name:
        ref_model_args.model_name_or_path = ref_model_name
    return load_model(tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False)
