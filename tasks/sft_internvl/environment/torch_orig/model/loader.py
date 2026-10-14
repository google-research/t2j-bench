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

"""Model and tokenizer loading for InternVL."""

import json
import logging
import math
from pathlib import Path
from typing import Any, TypedDict

import torch
from safetensors.torch import load_file as load_safetensors
from torch import nn
from tokenizers import AddedToken
import re

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


def _load_checkpoint_files_into_model(model, checkpoint_files: list[Path]):
    """Load the shards one at a time, and report what the checkpoint did not cover."""
    model_keys = set(model.state_dict())
    supplied: set[str] = set()
    unexpected: list[str] = []
    for checkpoint_path in checkpoint_files:
        shard_state = _load_state_dict_file(checkpoint_path)
        _, shard_unexpected = model.load_state_dict(shard_state, strict=False)
        supplied.update(key for key in shard_state if key in model_keys)
        unexpected.extend(shard_unexpected)
        del shard_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sorted(model_keys - supplied), unexpected


def _validate_loaded_model_state(model, *, max_abs_threshold: float = 1.0e6, max_tensors: int = 64) -> None:
    checked = 0
    named_items = []
    if hasattr(model, "named_parameters"):
        named_items.extend((f"parameter:{name}", tensor) for name, tensor in model.named_parameters())
    if hasattr(model, "named_buffers"):
        named_items.extend((f"buffer:{name}", tensor) for name, tensor in model.named_buffers())
    for name, tensor in named_items:
        if checked >= max_tensors:
            break
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        if tensor.device.type == "meta":
            continue
        if tensor.numel() == 0:
            continue
        checked += 1
        probe = tensor.detach()
        if probe.device.type != "cpu":
            probe = probe.float().cpu()
        else:
            probe = probe.float()
        if not torch.isfinite(probe).all():
            raise RuntimeError(f"loaded model contains non-finite values in {name}")
        max_abs = float(probe.abs().max().item())
        if max_abs > max_abs_threshold:
            raise RuntimeError(
                f"loaded model contains implausibly large values in {name}: max_abs={max_abs}"
            )


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


def _from_pretrained_or_raise(model_cls, config, model_path: Path, dtype=None, target_device="cpu"):
    """Build the model and load the checkpoint, refusing to hand back a random network."""
    shared = {"local_files_only": True, "output_loading_info": True}
    attn_implementation = getattr(config, "_attn_implementation", None)
    if attn_implementation:
        shared["attn_implementation"] = attn_implementation
    if dtype is not None:
        shared["dtype"] = dtype

    def _build_then_load(on_meta: bool):
        checkpoint_files = _iter_checkpoint_files(model_path)
        if on_meta:
            original_dtype = torch.get_default_dtype()
            if dtype is not None and getattr(dtype, "is_floating_point", False):
                torch.set_default_dtype(dtype)
            try:
                with torch.device("meta"):
                    model = model_cls(config)
            finally:
                torch.set_default_dtype(original_dtype)
            model.to_empty(device=target_device)
        else:
            model = model_cls(config)
        missing, unexpected = _load_checkpoint_files_into_model(model, checkpoint_files)
        return model, {"missing_keys": missing, "unexpected_keys": unexpected}

    attempts: list[str] = []
    for label, attempt in (
        ("from_pretrained()",
         lambda: model_cls.from_pretrained(str(model_path), **shared)),
        ("from_pretrained(config=...)",
         lambda: model_cls.from_pretrained(str(model_path), config=config, **shared)),
        ("meta/to_empty", lambda: _build_then_load(on_meta=True)),
        ("eager", lambda: _build_then_load(on_meta=False)),
    ):
        try:
            model, info = attempt()
            _validate_loaded_model_state(model)
        except Exception as error:
            logger.warning("%s load failed (%s); trying the next path.", label, error)
            attempts.append(f"{label} raised {type(error).__name__}: {error}")
            continue
        still_random = [key for key in (info.get("missing_keys") or ())
                        if not _tied_to_input_embeddings(model, key)]
        if still_random:
            attempts.append(f"{label} left {len(still_random)} parameter(s) at random init, "
                            f"e.g. {sorted(still_random)[:5]}")
            continue
        unexpected = list(info.get("unexpected_keys") or ())
        if unexpected:
            logger.info("Unexpected keys while loading model: %s", sorted(unexpected)[:20])
        return model

    raise ValueError(
        f"{model_cls.__name__} did not receive its pretrained weights from {model_path}. "
        + " | ".join(attempts)
    )


def _fallback_generation_config_cls():
    class _FallbackGenerationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def from_model_config(cls, config):
            text_config = config.get_text_config() if hasattr(config, "get_text_config") else config

            def _resolve(name, default=None):
                value = getattr(config, name, None)
                if value is None:
                    value = getattr(text_config, name, None)
                return default if value is None else value

            payload = {
                "bos_token_id": _resolve("bos_token_id"),
                "eos_token_id": _resolve("eos_token_id"),
                "pad_token_id": _resolve("pad_token_id"),
                "decoder_start_token_id": _resolve("decoder_start_token_id"),
                "max_new_tokens": _resolve("max_new_tokens", 256),
                "do_sample": _resolve("do_sample", False),
                "temperature": _resolve("temperature", 1.0),
                "top_p": _resolve("top_p", 1.0),
                "top_k": _resolve("top_k", 50),
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


def _attach_generation_config(model) -> None:
    if hasattr(model, "generation_config"):
        return
    GenerationConfig = _resolve_generation_config_cls()
    model.generation_config = GenerationConfig.from_model_config(model.config)


def _special_token_str(value) -> Any:
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

    def _id(self, token):
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
                cfg = _read_json(cfg_file)
                special = {k: cfg.get(k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
                _register_added_tokens(raw, cfg)
            tokenizer = _FastTokenizerAdapter(raw, special)
    except Exception:
        tokenizer = None
    processor = None
    try:
        from transformers_impl.processing import InternVLProcessor

        processor = InternVLProcessor.from_pretrained(
            str(_resolve_model_path(model_args.model_name_or_path)), tokenizer=tokenizer
        )
        for attribute in ("image_max_pixels", "image_min_pixels"):
            if hasattr(model_args, attribute):
                setattr(processor, attribute, getattr(model_args, attribute))
    except Exception as error:
        logger.warning("Could not load a processor for %s: %s", model_args.model_name_or_path, error)
        processor = None
    if processor is not None and getattr(processor, "tokenizer", None) is None:
        setattr(processor, "tokenizer", tokenizer)
    return {"tokenizer": tokenizer, "processor": processor}


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

    def generate(self, *args, **kwargs):
        self.base_model.model.generation_config = self.generation_config
        return self.base_model.model.generate(*args, **kwargs)

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
    lora_dropout = float(getattr(finetuning_args, "lora_dropout", 0.0) or 0.0)
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


def load_model(tokenizer, model_args, finetuning_args, is_trainable=True):
    from transformers_impl import configuration as configuration_mod
    from transformers_impl import modeling as modeling_mod

    model_path = _resolve_model_path(model_args.model_name_or_path)
    config = configuration_mod.InternVLConfig.from_pretrained(str(model_path))
    model = _from_pretrained_or_raise(
        modeling_mod.InternVLForConditionalGeneration,
        config,
        model_path,
        dtype=getattr(model_args, "compute_dtype", None),
        target_device="cpu",
    )
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
    _attach_generation_config(model)
    _log_param_counts(model)
    return model
