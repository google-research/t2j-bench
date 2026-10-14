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

"""Model and tokenizer loading for PaliGemma."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors
from tokenizers import AddedToken
import re
import inspect
import contextlib
import os
from safetensors.torch import save_file

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
            "Requested float16 for a checkpoint with native bfloat16 weights. "
            "Falling back to bfloat16 runtime for numerical stability."
        )
        return torch.bfloat16
    return requested_dtype


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
        if generation_config is not None and hasattr(generation_config, "from_model_config"):
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
    if getattr(model, "generation_config", None) is not None:
        return
    GenerationConfig = _resolve_generation_config_cls()
    model.generation_config = GenerationConfig.from_model_config(model.config)


def _maybe_move_model(model, device_map) -> None:
    if isinstance(device_map, str) and device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        model.to(device_map)


def _special_token_str(value) -> Optional[str]:
    """Special tokens in tokenizer_config.json may be a str or an ``AddedToken`` dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("content")
    return str(value)


class _FastTokenizerAdapter:
    """Torch-free HF-compatible facade over a Rust ``tokenizers.Tokenizer``."""

    def __init__(self, tok, special_tokens: dict[str, Any]):
        self._tok = tok
        self.padding_side = "right"
        self.bos_token = _special_token_str(special_tokens.get("bos_token"))
        self.eos_token = _special_token_str(special_tokens.get("eos_token"))
        self.pad_token = _special_token_str(special_tokens.get("pad_token"))
        self.unk_token = _special_token_str(special_tokens.get("unk_token"))

    def _id(self, token: Optional[str]):
        return self._tok.token_to_id(token) if token is not None else None

    def convert_tokens_to_ids(self, tokens):
        """HF-like: map a token (or a list of them) to id(s); ``None`` when unknown."""
        if isinstance(tokens, (list, tuple)):
            return [self._tok.token_to_id(token) for token in tokens]
        return self._tok.token_to_id(tokens)

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

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self._tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

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


def _register_added_tokens(raw, tokenizer_config: dict) -> None:
    """Mirror HF's merge of ``added_tokens_decoder`` into the fast tokenizer."""

    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    for token_id, spec in sorted(decoder.items(), key=lambda item: int(item[0])):
        spec = spec if isinstance(spec, dict) else {"content": spec}
        content = spec.get("content")
        if content is None or raw.token_to_id(content) is not None:
            continue
        token = AddedToken(
            content,
            single_word=spec.get("single_word", False),
            lstrip=spec.get("lstrip", False),
            rstrip=spec.get("rstrip", False),
            normalized=spec.get("normalized", False),
            special=spec.get("special", True),
        )
        if spec.get("special", True):
            raw.add_special_tokens([token])
        else:
            raw.add_tokens([token])
        actual = raw.token_to_id(content)
        if actual != int(token_id):
            raise ValueError(
                f"Added token {content!r} landed at id {actual}, but the checkpoint says "
                f"{token_id}; that id would index the wrong embedding row."
            )


def load_tokenizer(model_args) -> TokenizerModule:
    """Load the tokenizer + image processor."""
    tokenizer = None
    try:
        from tokenizers import Tokenizer

        model_path = _resolve_model_path(model_args.model_name_or_path)
        tok_file = model_path / "tokenizer.json"
        if tok_file.is_file():
            raw = Tokenizer.from_file(str(tok_file))
            special: dict[str, Any] = {}
            cfg_file = model_path / "tokenizer_config.json"
            if cfg_file.is_file():
                cfg = json.loads(cfg_file.read_text())
                special = {k: cfg.get(k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
                _register_added_tokens(raw, cfg)
            tokenizer = _FastTokenizerAdapter(raw, special)
    except Exception:
        tokenizer = None

    processor = None
    try:
        from transformers_impl import processing as processing_mod

        processor = processing_mod.PaliGemmaProcessor.from_pretrained(
            str(_resolve_model_path(model_args.model_name_or_path)), tokenizer=tokenizer
        )
        if getattr(processor, "tokenizer", None) is None:
            setattr(processor, "tokenizer", tokenizer)
        for attribute in ("image_max_pixels", "image_min_pixels"):
            if hasattr(model_args, attribute):
                setattr(processor, attribute, getattr(model_args, attribute))
    except Exception as error:
        logger.warning("Could not load a processor for %s: %s", model_args.model_name_or_path, error)
        processor = None

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
    config._attn_implementation = "eager"
    resolved_dtype = _resolve_runtime_dtype(config, getattr(model_args, "compute_dtype", None))
    model = _from_pretrained_or_raise(
        modeling_mod.PaliGemmaForConditionalGeneration, config, Path(model_path_str),
        getattr(model_args, "compute_dtype", None),
    )

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
        self._disable_adapters = False
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
        if self._disable_adapters:
            return result
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

    def disable_adapter(self):
        """Context manager mirroring peft's ``PeftModel.disable_adapter``."""

        @contextlib.contextmanager
        def _ctx():
            lora_layers = [m for m in self.modules() if isinstance(m, _LoraLinear)]
            previous = [m._disable_adapters for m in lora_layers]
            for m in lora_layers:
                m._disable_adapters = True
            try:
                yield self
            finally:
                for m, prev in zip(lora_layers, previous):
                    m._disable_adapters = prev

        return _ctx()

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



def _ppo_call_with_supported_kwargs(fn, **kwargs):

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


class _PPOForwardOutput(tuple):
    """``(logits, loss, value)`` output that is BOTH tuple-unpackable (the PPO trainer does ``logits, _, values = model(...)``) AND attribute-accessible via ``.logits`` / ``.loss`` / ``.value`` / ``.values`` (the evaluator's output readers look these up by attribute)."""

    def __new__(cls, logits, loss, value):
        return super().__new__(cls, (logits, loss, value))

    @property
    def logits(self):
        return self[0]

    @property
    def loss(self):
        return self[1]

    @property
    def value(self):
        return self[2]

    @property
    def values(self):
        return self[2]


class _ValueHeadModel(torch.nn.Module):
    """TRL-style value-head wrapper the PPO trainer expects."""

    def __init__(self, pretrained_model, v_head):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.v_head = v_head
        self.config = pretrained_model.config
        self.generation_config = getattr(pretrained_model, "generation_config", None)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        captured = {}
        lm_head = _ppo_find_lm_head(self.pretrained_model)
        hook = None
        if lm_head is not None:
            def _capture(_module, inputs):
                if inputs:
                    captured["hidden"] = inputs[0]
            hook = lm_head.register_forward_pre_hook(_capture)
        try:
            output = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=lm_head is None,
                return_dict=True,
                **kwargs,
            )
        finally:
            if hook is not None:
                hook.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            hs = getattr(output, "hidden_states", None)
            hidden = hs[-1] if hs else None
        value = None
        if hidden is not None:
            value = self.v_head(hidden.to(self.v_head.weight.dtype)).squeeze(-1)
        return _PPOForwardOutput(output.logits, getattr(output, "loss", None), value)

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.pretrained_model, "gradient_checkpointing_enable"):
            self.pretrained_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        if hasattr(self.pretrained_model, "gradient_checkpointing_disable"):
            self.pretrained_model.gradient_checkpointing_disable(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        """Write the base the way it wants to be written, plus the value head."""


        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.pretrained_model, "peft_config"):
            kwargs.setdefault("save_embedding_layers", False)
        try:
            self.pretrained_model.save_pretrained(save_directory, **kwargs)
        except TypeError:
            self.pretrained_model.save_pretrained(save_directory)
        save_file(
            {"v_head.weight": self.v_head.weight.detach().cpu().contiguous()},
            os.path.join(save_directory, "value_head.safetensors"),
        )


def _ppo_attach_value_head(model, model_args):

    if isinstance(model, _ValueHeadModel):
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
    return _ValueHeadModel(model, v_head)


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
    return model
