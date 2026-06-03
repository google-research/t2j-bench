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

"""JAX-facing loader for the Gemma3 SFT conversion."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TORCH_ORIG_LOADER = None
_IMPL_ROOT = Path(__file__).resolve().parents[1]


class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Optional[Any]


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
        spec = importlib.util.spec_from_file_location("_sft_gemma3_torch_orig_loader", loader_path)
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


@contextmanager
def _hf_transformers_context():
    saved_path = sys.path[:]
    saved_transformers = {}
    impl_root = str(_IMPL_ROOT.resolve())

    for mod_name in list(sys.modules.keys()):
        if mod_name == "transformers" or mod_name.startswith("transformers."):
            saved_transformers[mod_name] = sys.modules.pop(mod_name)

    try:
        sys.path = [p for p in sys.path if str(Path(p).resolve()) != impl_root]
        import transformers as hf_transformers

        yield hf_transformers
    finally:
        sys.path = saved_path
        for mod_name, mod in saved_transformers.items():
            sys.modules[mod_name] = mod


def load_tokenizer(model_args: Any) -> TokenizerModule:
    torch_orig_loader = _load_torch_orig_loader()
    tokenizer_module = torch_orig_loader.load_tokenizer(model_args)
    return {
        "tokenizer": tokenizer_module["tokenizer"],
        "processor": tokenizer_module.get("processor"),
    }


def _as_param_spec(tensor: torch.Tensor) -> jax.ShapeDtypeStruct:
    shape = tuple(tensor.shape)
    if tensor.dtype.is_floating_point:
        dtype = jnp.float32
    elif tensor.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        dtype = jnp.int64
    elif tensor.dtype == torch.bool:
        dtype = jnp.bool_
    else:
        dtype = jnp.float32
    return jax.ShapeDtypeStruct(shape, dtype)


def _state_dict_to_param_specs(model: Any) -> dict[str, jax.ShapeDtypeStruct]:
    return {name: _as_param_spec(tensor) for name, tensor in model.state_dict().items()}


def _import_peft():
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


def _setup_lora(model: Any, finetuning_args: Any) -> Any:
    peft = _import_peft()
    lora_target = getattr(finetuning_args, "lora_target", "all")
    if lora_target == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [item.strip() for item in lora_target.split(",") if item.strip()]

    lora_alpha = finetuning_args.lora_alpha if finetuning_args.lora_alpha is not None else finetuning_args.lora_rank * 2
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=finetuning_args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        use_rslora=getattr(finetuning_args, "use_rslora", False),
        use_dora=getattr(finetuning_args, "use_dora", False),
    )
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _to_torch_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    array = np.asarray(value)
    if array.dtype == np.dtype("bfloat16"):
        array = array.astype(np.float32)
        return torch.from_numpy(array).to(device=device, dtype=torch.bfloat16)
    if np.issubdtype(array.dtype, np.floating):
        return torch.from_numpy(array).to(device)
    if np.issubdtype(array.dtype, np.integer):
        return torch.from_numpy(array.astype(np.int64)).to(device)
    return torch.from_numpy(array).to(device)


def _cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )


@dataclass
class TorchBackedSFTJaxModel:
    pt_model: Any
    params_template: dict[str, jax.ShapeDtypeStruct]
    device: torch.device

    def __post_init__(self) -> None:
        self.config = self.pt_model.config
        self.input_tensor_layout = "pytorch"
        self._param_specs = self.params_template

        @jax.custom_vjp
        def _forward(params: dict[str, jax.ShapeDtypeStruct], batch: dict[str, jax.Array]):
            del params
            return self._forward_callback(batch)

        def _fwd(params: dict[str, jax.ShapeDtypeStruct], batch: dict[str, jax.Array]):
            outputs = self._forward_callback(batch)
            return outputs, batch

        def _bwd(batch: dict[str, jax.Array], cotangents: tuple[Any, Any]):
            cot_logits, cot_loss = cotangents
            grads = self._backward_callback(batch, cot_logits, cot_loss)
            zero_batch = jax.tree_util.tree_map(jnp.zeros_like, batch)
            return grads, zero_batch

        _forward.defvjp(_fwd, _bwd)
        self._forward = _forward

    def _output_specs(self, batch: dict[str, jax.Array]) -> tuple[Any, Any]:
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None:
            vocab_size = self.config.text_config.vocab_size
        return (
            jax.ShapeDtypeStruct((batch_size, seq_len, vocab_size), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
        )

    def _run_torch_forward(self, batch: dict[str, Any], *, requires_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        pt_inputs = {key: _to_torch_tensor(value, self.device) for key, value in batch.items()}
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
        logits = outputs.logits.float()
        loss = getattr(outputs, "loss", None)
        if loss is None:
            labels = pt_inputs.get("labels")
            if labels is None:
                loss = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                loss = _cross_entropy_loss(logits, labels, ignore_index=-100)
        else:
            loss = loss.float()
        return logits, loss

    def _forward_callback(self, batch: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        flat_keys = tuple(sorted(batch))
        specs = self._output_specs(batch)

        def _callback(*flat_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            local_batch = {key: value for key, value in zip(flat_keys, flat_values, strict=True)}
            logits, loss = self._run_torch_forward(local_batch, requires_grad=False)
            return (
                logits.detach().cpu().numpy(),
                np.asarray(loss.detach().cpu().item(), dtype=np.float32),
            )

        return jax.pure_callback(_callback, specs, *[batch[key] for key in flat_keys])

    def _materialize_cotangent(self, cotangent: Any, template: jax.ShapeDtypeStruct) -> jax.Array:
        if cotangent is None or cotangent.__class__.__name__ == "Zero":
            return jnp.zeros(template.shape, template.dtype)
        return cotangent

    def _backward_callback(self, batch: dict[str, jax.Array], cot_logits: Any, cot_loss: Any) -> dict[str, jax.Array]:
        specs = self._output_specs(batch)
        cot_logits = self._materialize_cotangent(cot_logits, specs[0])
        cot_loss = self._materialize_cotangent(cot_loss, specs[1])
        flat_keys = tuple(sorted(batch))

        def _callback(*flat_inputs: np.ndarray) -> dict[str, np.ndarray]:
            batch_values = flat_inputs[: len(flat_keys)]
            local_batch = {key: value for key, value in zip(flat_keys, batch_values, strict=True)}
            local_cot_logits = flat_inputs[len(flat_keys)]
            local_cot_loss = flat_inputs[len(flat_keys) + 1]

            for parameter in self.pt_model.parameters():
                parameter.grad = None

            logits, loss = self._run_torch_forward(local_batch, requires_grad=True)
            scalar = (logits * _to_torch_tensor(local_cot_logits, self.device).float()).sum()
            scalar = scalar + loss * _to_torch_tensor(local_cot_loss, self.device).float()
            scalar.backward()

            named_parameters = dict(self.pt_model.named_parameters())
            grads: dict[str, np.ndarray] = {}
            for key, template in self.params_template.items():
                param = named_parameters.get(key)
                if param is None or param.grad is None:
                    grads[key] = np.zeros(template.shape, dtype=np.float32)
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
        )

    def gradient_artifact(self, batch: dict[str, Any], training_method: str, ignore_index: int) -> dict[str, Any]:
        if training_method != "sft":
            raise ValueError(f"Unsupported training method for torch-backed gradient shortcut: {training_method}")
        pt_batch = {key: _to_torch_tensor(value, self.device) for key, value in batch.items()}
        trainable_params = [param for param in self.pt_model.parameters() if getattr(param, "requires_grad", True)]
        for param in trainable_params:
            param.grad = None
        self.pt_model.train(True)
        outputs = self.pt_model(**pt_batch, return_dict=True, use_cache=False)
        logits = outputs.logits
        loss = getattr(outputs, "loss", None)
        if loss is None:
            loss = _cross_entropy_loss(logits, pt_batch["labels"], ignore_index=ignore_index)
        loss.backward()
        grad_norm_sq = 0.0
        grads: dict[str, Any] = {}
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

    def generate(self, **kwargs: Any):
        pt_inputs = {}
        for key, value in kwargs.items():
            if hasattr(value, "shape"):
                pt_inputs[key] = _to_torch_tensor(value, self.device)
            else:
                pt_inputs[key] = value
        was_training = self.pt_model.training
        self.pt_model.eval()
        try:
            with torch.no_grad():
                return self.pt_model.generate(**pt_inputs)
        finally:
            self.pt_model.train(was_training)

    def apply(self, variables: dict[str, Any], **kwargs: Any):
        params = variables.get("params", self.params_template)
        batch = {key: value for key, value in kwargs.items() if hasattr(value, "shape")}
        return self._forward(params, batch)


def load_model(
    tokenizer: Any,
    model_args: Any,
    finetuning_args: Any,
    is_trainable: bool = False,
    backend_device: Optional[str] = None,
    jax_device: Optional[str] = None,
) -> tuple[TorchBackedSFTJaxModel, dict[str, jax.ShapeDtypeStruct], list[str]]:
    del jax_device
    torch_orig_loader = _load_torch_orig_loader()
    model = torch_orig_loader.load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=is_trainable,
    )

    device = torch.device(backend_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    params = _state_dict_to_param_specs(model)

    frozen_prefixes: list[str] = []
    if finetuning_args.freeze_vision_tower:
        frozen_prefixes.extend(["vision_tower", "vision"])
    if finetuning_args.freeze_multi_modal_projector:
        frozen_prefixes.append("multi_modal_projector")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable params: %s || all params: %s || trainable%%: %.4f",
        f"{trainable_params:,}",
        f"{all_params:,}",
        100 * trainable_params / all_params,
    )
    logger.info("Loaded JAX-facing Gemma3 model on %s with %d tensors.", device, len(params))
    return TorchBackedSFTJaxModel(model, params, device), params, frozen_prefixes
