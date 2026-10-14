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

"""Model and tokenizer loading."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors
from tokenizers import Tokenizer
import re
import copy
import shutil

logger = logging.getLogger(__name__)
_LOCAL_TRANSFORMERS_PACKAGE = "transformers_impl"
_DEFAULT_CONFIG_CLASS = 'LlamaConfig'
_DEFAULT_ARCHITECTURE = 'LlamaForCausalLM'


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


def _resolve_local_config_class():
    from transformers_impl import configuration as module
    config_class = getattr(module, _DEFAULT_CONFIG_CLASS, None)
    if config_class is not None:
        return config_class
    raise ValueError(
        f"Config class {_DEFAULT_CONFIG_CLASS!r} not found in {_LOCAL_TRANSFORMERS_PACKAGE}.configuration"
    )


def _resolve_local_model_class(config):
    from transformers_impl import modeling as module
    architecture_names = list(getattr(config, "architectures", None) or [])
    if not architecture_names and _DEFAULT_ARCHITECTURE:
        architecture_names.append(_DEFAULT_ARCHITECTURE)
    for arch_name in architecture_names:
        model_cls = getattr(module, arch_name, None)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Could not resolve local model class for architectures={architecture_names!r} "
        f"in {_LOCAL_TRANSFORMERS_PACKAGE}.modeling"
    )


def _get_local_init_kwargs(model_args) -> dict[str, Any]:
    return {
        "cache_dir": getattr(model_args, "cache_dir", None),
        "revision": getattr(model_args, "model_revision", "main"),
        "token": getattr(model_args, "hf_hub_token", None),
        "local_files_only": True,
    }


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


def _load_local_config(model_path: Path, config_class, model_args):
    try:
        config = config_class.from_pretrained(str(model_path), **_get_local_init_kwargs(model_args))
    except Exception:
        config = config_class.from_dict(_normalize_config_dict(_read_json(model_path / "config.json")))
    return _normalize_config(config)


def _special_token_str(value) -> Optional[str]:
    """Special tokens in tokenizer_config.json may be a str or an ``AddedToken`` dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("content")
    return str(value)


class _FastTokenizerAdapter:
    """Torch-free HF-compatible facade over a Rust ``tokenizers.Tokenizer``."""

    def __init__(self, tok, special_tokens: dict[str, Any], model_path: Path):
        self._tok = tok
        self._model_path = Path(model_path)
        self.padding_side = "right"
        self.bos_token = _special_token_str(special_tokens.get("bos_token"))
        self.eos_token = _special_token_str(special_tokens.get("eos_token"))
        self.pad_token = _special_token_str(special_tokens.get("pad_token"))
        self.unk_token = _special_token_str(special_tokens.get("unk_token"))

    def _id(self, token: Optional[str]):
        return self._tok.token_to_id(token) if token is not None else None

    @property
    def pad_token_id(self):
        return self._id(self.pad_token)

    @property
    def eos_token_id(self):
        return self._id(self.eos_token)

    @property
    def bos_token_id(self):
        return self._id(self.bos_token)

    @property
    def unk_token_id(self):
        return self._id(self.unk_token)

    def encode(self, text, add_special_tokens: bool = True) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self._tok.token_to_id(token) for token in tokens]
        return self._tok.token_to_id(tokens)

    def add_special_tokens(self, special_tokens_dict: dict[str, Any]) -> int:
        """HF-like: register special tokens (e.g. a pad token)."""
        added = 0
        for key, value in special_tokens_dict.items():
            token = _special_token_str(value)
            setattr(self, key, token)
            if token is not None and self._tok.token_to_id(token) is None:
                try:
                    added += self._tok.add_special_tokens([token])
                except Exception:
                    pass
        return added

    def save_pretrained(self, save_directory, **kwargs):
        """Persist the tokenizer files so a full training run can write its output dir."""

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        for file_name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            source = self._model_path / file_name
            if source.is_file():
                shutil.copy2(source, save_directory / file_name)
        return [str(save_directory / "tokenizer.json")]


def load_tokenizer(model_args) -> TokenizerModule:
    """Load the tokenizer as an adapter over the Rust ``tokenizers`` fast tokenizer."""
    tokenizer = None

    model_path = _resolve_model_path(model_args.model_name_or_path)
    tok_file = model_path / "tokenizer.json"
    if not tok_file.is_file():
        raise FileNotFoundError(f"No tokenizer.json under {model_path}")
    raw = Tokenizer.from_file(str(tok_file))
    special: dict[str, Any] = {}
    cfg_file = model_path / "tokenizer_config.json"
    if cfg_file.is_file():
        cfg = _read_json(cfg_file)
        special = {key: cfg.get(key) for key in ("bos_token", "eos_token", "pad_token", "unk_token")}
    tokenizer = _FastTokenizerAdapter(raw, special, model_path)
    return {"tokenizer": tokenizer, "processor": None}


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


def _lora_exclude_pattern(model, finetuning_args, exclude_lm_head=False):
    """Modules LoRA must not adapt, as a regex matched with ``re.fullmatch`` on module names."""

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
        target_modules = "all-linear"
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


def _freeze_multimodal_modules(model, finetuning_args):
    freeze_patterns = []
    if getattr(finetuning_args, "freeze_vision_tower", True):
        freeze_patterns.extend(["vision_tower", "vision_model", "visual", "vision_encoder", "image_tower"])
    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        freeze_patterns.extend(["multi_modal_projector", "projector", "mm_projector", "merger"])
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in freeze_patterns):
            parameter.requires_grad = False


def _log_param_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")


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
    model_path = _resolve_model_path(model_args.model_name_or_path)
    config_class = _resolve_local_config_class()
    config = _load_local_config(model_path, config_class, model_args)
    model_cls = _resolve_local_model_class(config)
    model = _from_pretrained_or_raise(model_cls, config, model_path, getattr(model_args, "compute_dtype", None))
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
    _log_param_counts(model)
    return model


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
