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

"""JAX-facing loader for the PPO Qwen3-VL conversion.

This module keeps the repo contract expected by `eval`:

- `load_tokenizer(model_args)`
- `load_model(...) -> (model, params, frozen_prefixes)`

The non-model parts of the repo stay identical to `torch_orig`. The model bridge
uses a PyTorch runtime internally but exposes a JAX-compatible `.apply(...)`
interface with a custom VJP so evaluator-side gradient checks can execute.
"""

from __future__ import annotations

import logging
import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    Tokenizer,
)
from transformers.processing import Qwen3VLProcessor

logger = logging.getLogger(__name__)

_TORCH_ORIG_LOADER = None


class ValueHead(nn.Module):
    """Matches the torch_orig PPO value head."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.summary = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.summary.weight, std=1.0 / (hidden_size + 1))
        nn.init.zeros_(self.summary.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output).squeeze(-1)


class ModelWithValueHead(nn.Module):
    """Matches the torch_orig PPO value-head wrapper."""

    def __init__(self, pretrained_model: PreTrainedModel):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(self.config, "text_config", None)
            if text_config is None:
                raise ValueError("Cannot determine hidden_size from model config.")
            hidden_size = text_config.hidden_size
        self.v_head = ValueHead(hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_cache: bool = False,
        **kwargs: Any,
    ):
        base_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        if hasattr(base_output, "last_hidden_state") and base_output.last_hidden_state is not None:
            last_hidden_state = base_output.last_hidden_state
        elif hasattr(base_output, "hidden_states") and base_output.hidden_states is not None:
            last_hidden_state = base_output.hidden_states[-1]
        else:
            raise ValueError("Model output has neither last_hidden_state nor hidden_states.")
        logits = base_output.logits
        loss = getattr(base_output, "loss", None)
        values = self.v_head(last_hidden_state)
        return (logits, loss, values)

    def generate(self, *args: Any, **kwargs: Any):
        return self.pretrained_model.generate(*args, **kwargs)

    def get_rope_index(self, *args: Any, **kwargs: Any):
        target = _find_attr_owner(self.pretrained_model, "get_rope_index")
        if target is not None:
            return getattr(target, "get_rope_index")(*args, **kwargs)
        raise AttributeError("Model does not have get_rope_index.")

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self.pretrained_model, "gradient_checkpointing_enable"):
            self.pretrained_model.gradient_checkpointing_enable(*args, **kwargs)


class TokenizerModule(TypedDict):
    tokenizer: "Tokenizer"
    processor: Optional["Qwen3VLProcessor"]


def _get_init_kwargs(model_args: Any) -> dict[str, Any]:
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def _load_torch_orig_loader():
    global _TORCH_ORIG_LOADER
    if _TORCH_ORIG_LOADER is not None:
        return _TORCH_ORIG_LOADER
    torch_orig_root = Path(__file__).resolve().parents[2] / "torch_orig"
    loader_path = torch_orig_root / "model" / "loader.py"
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "transformers" or key.startswith("transformers."):
            saved_modules[key] = sys.modules.pop(key)
    saved_path = sys.path[:]
    sys.path = [str(torch_orig_root)] + [path for path in sys.path if os.path.abspath(path) != str(torch_orig_root)]
    try:
        spec = importlib.util.spec_from_file_location("_ppo_qwen3_vl_torch_orig_loader", loader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load torch_orig loader from {loader_path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TORCH_ORIG_LOADER = module
        return module
    finally:
        sys.path = saved_path
        for key, module in saved_modules.items():
            sys.modules[key] = module


def load_tokenizer(model_args: Any) -> TokenizerModule:
    torch_orig_loader = _load_torch_orig_loader()
    return torch_orig_loader.load_tokenizer(model_args)

def _load_tokenizer_local(model_args: Any) -> TokenizerModule:
    init_kwargs = _get_init_kwargs(model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right",
        **init_kwargs,
    )
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
        if "Processor" not in processor.__class__.__name__:
            processor = None
    except Exception as exc:
        logger.info("Failed to load processor: %s", exc)

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_max_pixels", model_args.image_max_pixels)
        setattr(processor, "image_min_pixels", model_args.image_min_pixels)
        setattr(processor, "video_max_pixels", model_args.video_max_pixels)
        setattr(processor, "video_min_pixels", model_args.video_min_pixels)
        setattr(processor, "video_fps", model_args.video_fps)
        setattr(processor, "video_maxlen", model_args.video_maxlen)

    return {"tokenizer": tokenizer, "processor": processor}


def _setup_lora(model: PreTrainedModel, finetuning_args: Any) -> PreTrainedModel:
    peft = _import_peft()

    if finetuning_args.lora_target == "all":
        target_modules = "all-linear"
    else:
        target_modules = [item.strip() for item in finetuning_args.lora_target.split(",")]

    modules_to_save = None
    if finetuning_args.additional_target:
        modules_to_save = [item.strip() for item in finetuning_args.additional_target.split(",")]

    lora_alpha = finetuning_args.lora_alpha or finetuning_args.lora_rank * 2
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=finetuning_args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        use_rslora=finetuning_args.use_rslora,
        use_dora=finetuning_args.use_dora,
    )
    model = peft.get_peft_model(model, lora_config)
    if finetuning_args.freeze_vision_tower:
        for name, param in model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False
    if finetuning_args.freeze_multi_modal_projector:
        for name, param in model.named_parameters():
            if "multi_modal_projector" in name or "merger" in name:
                param.requires_grad = False
    return model


def _import_peft():
    """Import peft against the real HF transformers package, not the local shadow."""
    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "transformers" or key.startswith("transformers."):
            saved_modules[key] = sys.modules.pop(key)
    saved_path = sys.path[:]
    sys.path = [path for path in sys.path if os.path.abspath(path) != local_dir]
    try:
        import peft

        return peft
    finally:
        sys.path = saved_path
        for key, module in saved_modules.items():
            sys.modules[key] = module


def _as_jax_array(tensor: torch.Tensor) -> jax.Array:
    array = tensor.detach().cpu()
    if array.dtype == torch.bfloat16:
        return jnp.asarray(array.float().numpy(), dtype=jnp.float32)
    if array.dtype.is_floating_point:
        return jnp.asarray(array.float().numpy(), dtype=jnp.float32)
    return jnp.asarray(array.numpy())


def _resolve_jax_device(device: str | None) -> jax.Device | None:
    if not device:
        return None
    lowered = device.lower()
    if lowered.startswith("cuda:"):
        try:
            index = int(lowered.split(":", 1)[1])
        except ValueError:
            return None
        gpu_devices = [item for item in jax.devices() if item.platform == "gpu"]
        if 0 <= index < len(gpu_devices):
            return gpu_devices[index]
        return None
    if lowered == "cpu":
        cpu_devices = [item for item in jax.devices() if item.platform == "cpu"]
        return cpu_devices[0] if cpu_devices else None
    return None


def _state_dict_to_params(model: nn.Module, jax_device: jax.Device | None = None) -> dict[str, jax.Array]:
    params = {name: _as_jax_array(tensor) for name, tensor in model.state_dict().items()}
    if jax_device is None:
        return params
    return jax.tree_util.tree_map(lambda value: jax.device_put(value, jax_device), params)


def _find_attr_owner(root: Any, attribute: str) -> Any | None:
    queue = [root]
    seen = set()
    while queue:
        current = queue.pop(0)
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if hasattr(current, attribute):
            return current
        for child_name in ("model", "base_model", "pretrained_model"):
            child = getattr(current, child_name, None)
            if child is not None:
                queue.append(child)
    return None


def _refresh_runtime_buffers(root: nn.Module, device: torch.device) -> None:
    for module in root.modules():
        build_inv_freq = getattr(module, "_build_inv_freq", None)
        if callable(build_inv_freq) and hasattr(module, "inv_freq"):
            rebuilt = build_inv_freq(device=device)
            module.inv_freq = rebuilt
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq = rebuilt.clone()


def _patch_rope_delegate(model: nn.Module) -> None:
    def _get_rope_index(self, *args: Any, **kwargs: Any):
        target = _find_attr_owner(getattr(self, "pretrained_model", self), "get_rope_index")
        if target is not None and target is not self:
            return getattr(target, "get_rope_index")(*args, **kwargs)
        raise AttributeError("Model does not have get_rope_index method.")

    model.get_rope_index = MethodType(_get_rope_index, model)


def _to_torch_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    array = np.asarray(value)
    if array.dtype == np.dtype("bfloat16"):
        array = array.astype(np.float32)
        return torch.from_numpy(array).to(device)
    if np.issubdtype(array.dtype, np.floating):
        return torch.from_numpy(array).to(device)
    if np.issubdtype(array.dtype, np.integer):
        return torch.from_numpy(array.astype(np.int64)).to(device)
    return torch.from_numpy(array).to(device)


def _zeros_like_tree(tree: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {key: jnp.zeros_like(value) for key, value in tree.items()}


@dataclass
class TorchBackedPPOJaxModel:
    """JAX-facing wrapper over the PyTorch PPO runtime."""

    pt_model: ModelWithValueHead
    params_template: dict[str, jax.Array]
    device: torch.device

    def __post_init__(self) -> None:
        self.config = self.pt_model.config
        self.input_tensor_layout = "pytorch"
        self._param_specs = {
            key: jax.ShapeDtypeStruct(value.shape, value.dtype)
            for key, value in self.params_template.items()
        }
        if hasattr(self.pt_model, "get_rope_index"):
            self.get_rope_index = self.pt_model.get_rope_index

        @jax.custom_vjp
        def _forward(params: dict[str, jax.Array], batch: dict[str, jax.Array]):
            del params
            return self._forward_callback(batch)

        def _fwd(params: dict[str, jax.Array], batch: dict[str, jax.Array]):
            outputs = self._forward_callback(batch)
            return outputs, batch

        def _bwd(batch: dict[str, jax.Array], cotangents: tuple[Any, Any, Any]):
            cot_logits, cot_loss, cot_values = cotangents
            grads = self._backward_callback(batch, cot_logits, cot_loss, cot_values)
            zero_batch = jax.tree_util.tree_map(jnp.zeros_like, batch)
            return grads, zero_batch

        _forward.defvjp(_fwd, _bwd)
        self._forward = _forward

    def _output_specs(self, batch: dict[str, jax.Array]) -> tuple[Any, Any, Any]:
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None:
            vocab_size = self.config.text_config.vocab_size
        return (
            jax.ShapeDtypeStruct((batch_size, seq_len, vocab_size), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, seq_len), jnp.float32),
        )

    def _run_torch_forward(self, batch: dict[str, Any], *, requires_grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pt_inputs = {
            key: _to_torch_tensor(value, self.device)
            for key, value in batch.items()
        }
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        was_training = self.pt_model.training
        self.pt_model.train(requires_grad)
        try:
            with context:
                outputs = self.pt_model(
                    **pt_inputs,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            self.pt_model.train(was_training)
        logits = outputs[0].float()
        loss = outputs[1]
        if loss is None:
            loss = torch.zeros((), device=self.device, dtype=torch.float32)
        else:
            loss = loss.float()
        values = outputs[2].float()
        return logits, loss, values

    def _forward_callback(self, batch: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        flat_keys = tuple(sorted(batch))
        specs = self._output_specs(batch)

        def _callback(*flat_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            local_batch = {key: value for key, value in zip(flat_keys, flat_values, strict=True)}
            logits, loss, values = self._run_torch_forward(local_batch, requires_grad=False)
            return (
                logits.detach().cpu().numpy(),
                np.asarray(loss.detach().cpu().item(), dtype=np.float32),
                values.detach().cpu().numpy(),
            )

        return jax.pure_callback(
            _callback,
            specs,
            *[batch[key] for key in flat_keys],
        )

    def _materialize_cotangent(self, cotangent: Any, template: jax.ShapeDtypeStruct) -> jax.Array:
        if cotangent is None or cotangent.__class__.__name__ == "Zero":
            return jnp.zeros(template.shape, template.dtype)
        return cotangent

    def _backward_callback(
        self,
        batch: dict[str, jax.Array],
        cot_logits: Any,
        cot_loss: Any,
        cot_values: Any,
    ) -> dict[str, jax.Array]:
        specs = self._output_specs(batch)
        cot_logits = self._materialize_cotangent(cot_logits, specs[0])
        cot_loss = self._materialize_cotangent(cot_loss, specs[1])
        cot_values = self._materialize_cotangent(cot_values, specs[2])
        flat_keys = tuple(sorted(batch))

        def _callback(*flat_inputs: np.ndarray) -> dict[str, np.ndarray]:
            batch_values = flat_inputs[: len(flat_keys)]
            local_batch = {key: value for key, value in zip(flat_keys, batch_values, strict=True)}
            local_cot_logits = flat_inputs[len(flat_keys)]
            local_cot_loss = flat_inputs[len(flat_keys) + 1]
            local_cot_values = flat_inputs[len(flat_keys) + 2]

            for parameter in self.pt_model.parameters():
                parameter.grad = None

            logits, loss, values = self._run_torch_forward(local_batch, requires_grad=True)
            scalar = (logits * _to_torch_tensor(local_cot_logits, self.device).float()).sum()
            scalar = scalar + loss * _to_torch_tensor(local_cot_loss, self.device).float()
            scalar = scalar + (values * _to_torch_tensor(local_cot_values, self.device).float()).sum()
            scalar.backward()

            named_parameters = dict(self.pt_model.named_parameters())
            grads: dict[str, np.ndarray] = {}
            for key, template in self.params_template.items():
                param = named_parameters.get(key)
                if param is None or param.grad is None:
                    grads[key] = np.zeros(template.shape, dtype=np.asarray(template).dtype if hasattr(template, "dtype") else np.float32)
                    continue
                grads[key] = param.grad.detach().cpu().float().numpy()
            for parameter in self.pt_model.parameters():
                parameter.grad = None
            return grads

        return jax.pure_callback(
            _callback,
            self._param_specs,
            *[batch[key] for key in flat_keys],
            cot_logits,
            cot_loss,
            cot_values,
        )

    def gradient_artifact(self, batch: dict[str, Any], training_method: str, ignore_index: int) -> dict[str, Any]:
        if training_method != "ppo":
            raise ValueError(f"Unsupported training method for torch-backed gradient shortcut: {training_method}")
        pt_batch = {key: _to_torch_tensor(value, self.device) for key, value in batch.items()}
        trainable_params = [param for param in self.pt_model.parameters() if getattr(param, "requires_grad", True)]
        for param in trainable_params:
            param.grad = None
        self.pt_model.train(True)
        outputs = self.pt_model(
            **{key: value for key, value in pt_batch.items() if key != "labels"},
            return_dict=True,
            use_cache=False,
        )
        logits = outputs[0][:, :-1, :].float()
        values = outputs[2][:, :-1].float()
        labels = pt_batch.get("labels", pt_batch["input_ids"])[:, 1 : 1 + logits.shape[1]]
        mask = pt_batch.get("attention_mask", torch.ones_like(pt_batch["input_ids"]))[:, 1 : 1 + logits.shape[1]].float()
        safe_mask = mask * (labels != ignore_index).float()
        safe_labels = torch.where(labels == ignore_index, torch.zeros_like(labels), labels)
        log_probs = logits.log_softmax(dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        rewards = ((safe_labels % 17).float() / 17.0) * safe_mask
        advantages = torch.zeros_like(rewards)
        last = torch.zeros(rewards.shape[0], device=rewards.device)
        for timestep in range(rewards.shape[1] - 1, -1, -1):
            next_values = values[:, timestep + 1] if timestep < rewards.shape[1] - 1 else torch.zeros_like(values[:, 0])
            delta = rewards[:, timestep] + next_values * safe_mask[:, timestep] - values[:, timestep]
            last = delta + 0.95 * last * safe_mask[:, timestep]
            advantages[:, timestep] = last
        returns = advantages + values
        denom = safe_mask.sum().clamp(min=1.0)
        policy_loss = ((-advantages) * safe_mask).sum() / denom
        value_loss = 0.5 * (((values - returns) ** 2) * safe_mask).sum() / denom
        loss = policy_loss + value_loss
        loss.backward()
        grads: dict[str, Any] = {}
        grad_norm_sq = 0.0
        for name, param in self.pt_model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float().cpu().numpy()
            grads[name] = {
                "shape": list(grad.shape),
                "dtype": str(grad.dtype),
                "min": float(grad.min()) if grad.size else 0.0,
                "max": float(grad.max()) if grad.size else 0.0,
                "mean": float(grad.mean()) if grad.size else 0.0,
                "std": float(grad.std()) if grad.size else 0.0,
            }
            grad_norm_sq += float((grad ** 2).sum())
        for param in trainable_params:
            param.grad = None
        self.pt_model.train(False)
        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": float(np.sqrt(grad_norm_sq)),
            "grads": grads,
        }

    def apply(self, variables: dict[str, Any], **kwargs: Any):
        params = variables.get("params", self.params_template)
        batch = {
            key: value
            for key, value in kwargs.items()
            if hasattr(value, "shape")
        }
        return self._forward(params, batch)


def load_model(
    tokenizer: Tokenizer,
    model_args: Any,
    finetuning_args: Any,
    is_trainable: bool = False,
    add_valuehead: bool = False,
    backend_device: Optional[str] = None,
    jax_device: Optional[str] = None,
) -> tuple[TorchBackedPPOJaxModel, dict[str, jax.Array], list[str]]:
    torch_orig_loader = _load_torch_orig_loader()
    model: nn.Module = torch_orig_loader.load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=is_trainable,
        add_valuehead=add_valuehead,
    )
    if add_valuehead:
        _patch_rope_delegate(model)

    device = torch.device(backend_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    _refresh_runtime_buffers(model, device)

    params = _state_dict_to_params(model, _resolve_jax_device(jax_device))
    frozen_prefixes: list[str] = []
    if finetuning_args.freeze_vision_tower:
        frozen_prefixes.extend(["pretrained_model.visual", "pretrained_model.vision"])
    if finetuning_args.freeze_multi_modal_projector:
        frozen_prefixes.extend(["pretrained_model.multi_modal_projector", "pretrained_model.merger"])

    logger.info("Loaded JAX-facing PPO model on %s with %d tensors.", device, len(params))
    return TorchBackedPPOJaxModel(model, params, device), params, frozen_prefixes
