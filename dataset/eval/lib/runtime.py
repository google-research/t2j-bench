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

"""Repository runtime bridges and evaluator-owned training formulas."""

from __future__ import annotations

import copy
import gc
import inspect
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig, ImplementationConfig
from .discovery import RepoModules, repo_imports
from .utils import flatten_dict, jsonable, summarize_tensor, to_numpy, tree_summary


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import jax

        jax.random.PRNGKey(seed)
    except ImportError:
        pass


def _precision_overrides(profile: str) -> dict[str, Any]:
    if profile == "fp16_compare":
        return {"bf16": False, "fp16": True, "compute_dtype": "float16"}
    if profile == "bf16_compare":
        return {"bf16": True, "fp16": False, "compute_dtype": "bfloat16"}
    raise ValueError(f"Unsupported precision profile: {profile}")


def _prepare_repo_config(eval_config: EvalConfig, implementation: ImplementationConfig) -> dict[str, Any]:
    config = copy.deepcopy(eval_config.shared_config)
    config.update(_precision_overrides(eval_config.precision_profile))
    config.setdefault("do_train", True)
    config.setdefault("do_eval", False)
    config.setdefault("per_device_train_batch_size", eval_config.batch_size)
    config.setdefault("per_device_eval_batch_size", eval_config.batch_size)
    config.setdefault("max_samples", eval_config.max_eval_examples)
    config.setdefault("seed", eval_config.seed)
    config.update(copy.deepcopy(implementation.overrides))
    return config


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(fn)
    accepted = {}
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    for key, value in kwargs.items():
        if has_var_kwargs or key in signature.parameters:
            accepted[key] = value
    return fn(**accepted)


def _uses_default_torch_forward(target: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not isinstance(target, torch.nn.Module):
        return False
    forward = getattr(type(target), "forward", None)
    return forward is getattr(torch.nn.Module, "forward", None)


@contextmanager
def _repo_resolution_context(repo_dir: str):
    repo_path = str(Path(repo_dir).resolve())
    saved_modules = {}
    for key, module in list(sys.modules.items()):
        if key == "transformers" or key.startswith("transformers."):
            saved_modules[key] = module
            sys.modules.pop(key, None)
    sys.path.insert(0, repo_path)
    try:
        yield
    finally:
        sys.path = [item for item in sys.path if item != repo_path]
        for key in list(sys.modules.keys()):
            if key == "transformers" or key.startswith("transformers."):
                sys.modules.pop(key, None)
        for key, module in saved_modules.items():
            sys.modules[key] = module


def _get_template(modules: RepoModules, tokenizer: Any, data_args: Any) -> Any:
    if modules.data_template is None:
        return None
    builder = getattr(modules.data_template, "get_template_and_fix_tokenizer", None)
    if builder is None:
        return None
    return builder(tokenizer, data_args)


def _flatten_tree_raw(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_tree_raw(value, path))
        else:
            flat[path] = value
    return flat


def _summarize_param_leaf(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype") and value.__class__.__name__ == "ShapeDtypeStruct":
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": True,
            "sample": [],
            "sha256_16": "spec_only",
        }
    return summarize_tensor(value)


def _get_ignore_index(repo_dir: str) -> int:
    try:
        from importlib import import_module

        constants = import_module("data.constants")
        return int(getattr(constants, "IGNORE_INDEX", -100))
    except Exception:
        return -100


def _build_dataset_from_eval_examples(
    training_method: str,
    eval_examples: list[dict[str, Any]],
    template: Any,
    tokenizer: Any,
    processor: Any,
    data_args: Any,
) -> dict[str, Any]:
    from importlib import import_module

    processor_module = import_module("data.processor")
    selected = None
    processor_name = {
        "sft": "SupervisedDatasetProcessor",
        "dpo": "PairwiseDatasetProcessor",
        "ppo": "UnsupervisedDatasetProcessor",
    }
    for candidate in processor_name.values():
        if hasattr(processor_module, candidate):
            selected = candidate
    method_specific = processor_name[training_method]
    processor_cls = getattr(processor_module, method_specific, None)
    if processor_cls is None:
        processor_cls = getattr(processor_module, selected) if selected is not None else None
    if processor_cls is None:
        raise ValueError("Could not find a dataset processor for eval_examples.")
    proc = processor_cls(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )
    batched_examples = {
        key: [example.get(key) for example in eval_examples]
        for key in eval_examples[0]
    }
    processed = proc.preprocess_dataset(batched_examples)
    lengths = {len(value) for value in processed.values()}
    if len(lengths) != 1:
        raise ValueError("processor output has inconsistent field lengths")
    size = next(iter(lengths))
    dataset = [
        {key: value[index] for key, value in processed.items()}
        for index in range(size)
    ]
    return {"train_dataset": dataset}


def _first_batch(dataset: Any, collator: Any, batch_size: int) -> dict[str, Any]:
    examples = [dataset[idx] for idx in range(min(batch_size, len(dataset)))]
    batch = collator(examples) if collator is not None else examples
    if hasattr(batch, "data") and isinstance(getattr(batch, "data"), dict):
        batch = dict(batch.data)
    elif not isinstance(batch, dict) and hasattr(batch, "items"):
        batch = dict(batch.items())
    if not isinstance(batch, dict):
        raise ValueError("Collator must return a mapping batch.")
    return batch


def _to_torch(value: Any, device: str | None = None) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for evaluator execution.") from exc
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        elif tensor.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.uint8,
            torch.uint16 if hasattr(torch, "uint16") else torch.int16,
            torch.uint32 if hasattr(torch, "uint32") else torch.int32,
        }:
            tensor = tensor.long()
    else:
        array = np.asarray(value)
        if str(array.dtype) == "bfloat16":
            array = array.astype(np.float32)
        elif array.dtype.kind in {"i", "u"} and array.dtype != np.bool_:
            array = array.astype(np.int64, copy=False)
        tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _to_jax(value: Any) -> Any:
    import jax.numpy as jnp

    if isinstance(value, jnp.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                return jnp.asarray(tensor.float().numpy(), dtype=jnp.bfloat16)
            return jnp.asarray(tensor.numpy())
    except ImportError:
        pass
    array = np.asarray(value)
    if array.dtype == np.dtype("bfloat16"):
        return jnp.asarray(array.astype(np.float32), dtype=jnp.bfloat16)
    return jnp.asarray(array)


def _resolve_jax_device(device: str | None) -> Any | None:
    if not device:
        return None
    try:
        import jax
    except ImportError:
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


def _to_jax_on_device(value: Any, device: Any | None) -> Any:
    import jax

    array = _to_jax(value)
    if device is None:
        return array
    return jax.device_put(array, device)


def _normalize_wrapper_leaf(value: Any) -> Any:
    arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32)
    if arr.dtype.kind in {"i", "u"} and arr.dtype != np.bool_:
        return arr.astype(np.int64, copy=False)
    return value


def _normalize_wrapper_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_wrapper_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_normalize_wrapper_tree(v) for v in value]
        return type(value)(items)
    try:
        return _normalize_wrapper_leaf(value)
    except Exception:
        return value


def _normalize_gradient_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(artifact)
    grads = normalized.get("grads")
    if grads is None:
        grads = normalized.get("grad_summaries")
    if grads is None:
        grads = normalized.get("gradient_summaries")
    if grads is None:
        grads = normalized.get("gradients")
    if grads is not None and "grads" not in normalized:
        normalized["grads"] = grads
    if "grad_norm" not in normalized and isinstance(grads, dict):
        norm_sq = 0.0
        found = False
        for summary in grads.values():
            if not isinstance(summary, dict):
                continue
            norm = summary.get("norm")
            if norm is None:
                continue
            try:
                norm_sq += float(norm) ** 2
                found = True
            except Exception:
                continue
        if found:
            normalized["grad_norm"] = math.sqrt(norm_sq)
    if "loss" not in normalized:
        for candidate in ("total_loss", "objective", "scalar_loss", "lm_loss"):
            value = normalized.get(candidate)
            if isinstance(value, (int, float, np.generic)):
                normalized["loss"] = float(value)
                break
        if "loss" not in normalized:
            policy_loss = normalized.get("policy_loss")
            value_loss = normalized.get("value_loss")
            if isinstance(policy_loss, (int, float, np.generic)) and isinstance(value_loss, (int, float, np.generic)):
                normalized["loss"] = float(policy_loss) + float(value_loss)
    for scalar_name in ("loss", "grad_norm", "total_loss", "objective", "scalar_loss", "lm_loss", "policy_loss", "value_loss"):
        if scalar_name not in normalized:
            continue
        value = normalized[scalar_name]
        if isinstance(value, (int, float, np.generic)):
            continue
        try:
            arr = to_numpy(value)
            if arr.size == 1:
                normalized[scalar_name] = float(arr.reshape(()))
        except Exception:
            pass
    return normalized


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except Exception:
        return None


def _tensor_ndim(value: Any) -> int | None:
    shape = _tensor_shape(value)
    if shape is None:
        return None
    return len(shape)


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, (int, float, np.generic)):
        return True
    ndim = _tensor_ndim(value)
    return ndim == 0


def _as_output_mapping(output: Any) -> dict[str, Any]:
    if output is None:
        return {}
    if isinstance(output, dict):
        return output
    if hasattr(output, "_asdict"):
        try:
            data = output._asdict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if is_dataclass(output):
        try:
            fields = getattr(output, "__dataclass_fields__", {})
            return {name: getattr(output, name) for name in fields}
        except Exception:
            return {}
    if hasattr(output, "__dict__"):
        try:
            data = {key: value for key, value in vars(output).items() if not key.startswith("_")}
            if data:
                return data
        except Exception:
            pass
    return {}


def _extract_from_sequence(output: Any, name: str, default: Any = None) -> Any:
    if not isinstance(output, (tuple, list)):
        return default
    if name == "loss":
        for item in output:
            if _is_scalar_like(item):
                return item
        return default
    if name == "logits":
        for item in output:
            ndim = _tensor_ndim(item)
            if ndim is not None and ndim >= 3:
                return item
        return default
    if name == "hidden_states":
        for item in output:
            if isinstance(item, (tuple, list)) and item:
                return item
        return default
    if name == "values":
        seen_logits = False
        for item in output:
            ndim = _tensor_ndim(item)
            if ndim is None:
                continue
            if ndim >= 3 and not seen_logits:
                seen_logits = True
                continue
            if ndim in {2, 3}:
                return item
        return default
    if name == "last_hidden_state":
        logits = _extract_from_sequence(output, "logits")
        logits_shape = _tensor_shape(logits)
        for item in output:
            ndim = _tensor_ndim(item)
            if ndim != 3:
                continue
            if _tensor_shape(item) != logits_shape:
                return item
        return default
    return default


def _extract_output_field(output: Any, name: str, default: Any = None) -> Any:
    if output is None:
        return default
    mapping = _as_output_mapping(output)
    if name in mapping:
        return mapping.get(name, default)
    if hasattr(output, name):
        return getattr(output, name)
    extracted = _extract_from_sequence(output, name, default)
    if extracted is not default:
        return extracted
    return default


def _extract_output_field_exact(output: Any, name: str, default: Any = None) -> Any:
    if output is None:
        return default
    mapping = _as_output_mapping(output)
    if name in mapping:
        return mapping.get(name, default)
    if hasattr(output, name):
        return getattr(output, name)
    return default


def _detect_reference_value_field(output: Any) -> str | None:
    for candidate in ("values", "value", "state_values", "state_value"):
        marker = object()
        value = _extract_output_field_exact(output, candidate, marker)
        if value is marker or value is None or callable(value):
            continue
        if hasattr(value, "shape") or _tensor_ndim(value) is not None:
            return candidate
    return None


def _detect_reference_hidden_source(output: Any) -> str | None:
    for candidate in ("last_hidden_state", "hidden_states"):
        marker = object()
        value = _extract_output_field(output, candidate, marker)
        if value is marker or value is None or callable(value):
            continue
        if candidate == "last_hidden_state":
            if hasattr(value, "shape"):
                return candidate
            continue
        if candidate == "hidden_states":
            if isinstance(value, (tuple, list)) and value:
                last = value[-1]
                if hasattr(last, "shape"):
                    return candidate
            continue
    return None


def _should_retry_without_control_kwargs(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "multiple values for keyword argument" in text
        or "got multiple values for argument" in text
        or "unexpected keyword argument" in text
    )


def _call_model_with_control_kwargs(fn: Any, /, *args: Any, model_inputs: dict[str, Any], **control_kwargs: Any) -> Any:
    base_inputs = {key: value for key, value in model_inputs.items() if value is not None}
    attempts = [
        dict(control_kwargs),
        {key: value for key, value in control_kwargs.items() if key != "output_hidden_states"},
        {key: value for key, value in control_kwargs.items() if key not in {"output_hidden_states", "use_cache"}},
        {},
    ]
    last_exc: BaseException | None = None
    for extra in attempts:
        try:
            return fn(*args, **base_inputs, **extra)
        except TypeError as exc:
            last_exc = exc
            if not extra or not _should_retry_without_control_kwargs(exc):
                raise
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def _flatten_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return dict(state_dict.items())


def _reset_runtime_state(root: Any) -> None:
    queue = [root]
    seen = set()
    while queue:
        current = queue.pop(0)
        obj_id = id(current)
        if obj_id in seen or current is None:
            continue
        seen.add(obj_id)
        if hasattr(current, "rope_deltas"):
            try:
                setattr(current, "rope_deltas", None)
            except Exception:
                pass
        custom_reset = getattr(current, "reset_runtime_state", None)
        if callable(custom_reset):
            custom_reset()
        for child_name in ("model", "base_model", "pretrained_model", "module"):
            child = getattr(current, child_name, None)
            if child is not None:
                queue.append(child)


def _ensure_ppo_valuehead(model: Any) -> None:
    """Add a v_head Linear layer to the model if it does not already have one.

    For PPO, the evaluator needs a value head to compute values from
    hidden states.  If the repo's ``load_model`` did not attach one
    (e.g. because it does not accept ``add_valuehead``), this function
    adds a deterministic ``nn.Linear(hidden_size, 1, bias=False)``
    using the current random seed so both reference and candidate get
    identical initial weights when the seed is the same.
    """
    import torch

    backend = getattr(model, "pt_model", model)
    if backend is None:
        return
    if hasattr(backend, "v_head"):
        _patch_ppo_value_forward(backend)
        return
    config = getattr(backend, "config", None)
    if config is None:
        return
    hidden_size = getattr(getattr(config, "text_config", config), "hidden_size", None)
    if hidden_size is None:
        return
    v_head = torch.nn.Linear(hidden_size, 1, bias=False)
    # Move to same device / dtype as the model
    try:
        first_param = next(backend.parameters())
        v_head = v_head.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    backend.v_head = v_head
    _patch_ppo_value_forward(backend)


def _ppo_find_lm_head(model: Any) -> Any | None:
    direct = getattr(model, "lm_head", None)
    if direct is not None:
        return direct
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None
    for name, module in named_modules():
        if name.endswith("lm_head"):
            return module
    return None


def _ppo_output_has_values(output: Any) -> bool:
    value = _extract_output_field_exact(output, "values", None)
    return value is not None and not callable(value) and hasattr(value, "shape")


def _ppo_inject_values(output: Any, values: Any) -> Any:
    if isinstance(output, dict):
        result = dict(output)
        result["values"] = values
        return result
    try:
        setattr(output, "values", values)
        return output
    except Exception:
        pass
    logits = _extract_output_field(output, "logits", None)
    if logits is not None:
        return {"logits": logits, "values": values, "raw_output": output}
    return {"values": values, "raw_output": output}


def _patch_ppo_value_forward(model: Any) -> None:
    try:
        import torch
    except ImportError:
        return
    if model is None or getattr(model, "_ppo_values_forward_patched", False):
        return
    v_head = getattr(model, "v_head", None)
    if v_head is None:
        return
    lm_head = _ppo_find_lm_head(model)
    if lm_head is None:
        return
    original_forward = model.forward

    def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        captured: dict[str, Any] = {}

        def _capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            if inputs:
                captured["hidden"] = inputs[0]

        hook = lm_head.register_forward_pre_hook(_capture)
        try:
            output = original_forward(*args, **kwargs)
        finally:
            hook.remove()

        if _ppo_output_has_values(output):
            return output
        hidden = captured.get("hidden")
        if hidden is None or not hasattr(hidden, "shape"):
            return output
        values = v_head(hidden.to(v_head.weight.device)).squeeze(-1)
        return _ppo_inject_values(output, values)

    model.forward = _wrapped_forward
    model._ppo_values_forward_patched = True


def _repair_rotary_buffers(root: Any) -> None:
    try:
        import torch
    except ImportError:
        return

    target = getattr(root, "pt_model", root)
    if target is None or not hasattr(target, "modules"):
        return
    for module in target.modules():
        build_inv_freq = getattr(module, "_build_inv_freq", None)
        if not callable(build_inv_freq) or not hasattr(module, "inv_freq"):
            continue
        existing = getattr(module, "inv_freq", None)
        device = None
        if isinstance(existing, torch.Tensor) and existing.device.type != "meta":
            device = existing.device
        else:
            for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta":
                    device = tensor.device
                    break
        rebuilt = build_inv_freq(device=device)
        module.inv_freq = rebuilt
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = rebuilt.clone()


def _module_primary_device(module: Any) -> Any | None:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            for parameter in parameters():
                return parameter.device
        except Exception:
            pass
    buffers = getattr(module, "buffers", None)
    if callable(buffers):
        try:
            for buffer in buffers():
                return buffer.device
        except Exception:
            pass
    return None


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _causal_cross_entropy(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = np.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - _logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = np.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    token_losses = -gathered * mask
    denom = max(float(mask.sum()), 1.0)
    return float(token_losses.sum() / denom)


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))


def _batch_logps(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = np.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - _logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = np.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    return (gathered * mask).sum(axis=-1), mask.sum(axis=-1)


def _align_time_axis(*arrays: Any) -> tuple[Any, ...]:
    valid = [array for array in arrays if array is not None and hasattr(array, "shape") and len(array.shape) >= 2]
    if not valid:
        return arrays
    target = min(int(array.shape[1]) for array in valid)
    aligned = []
    for array in arrays:
        if array is None or not hasattr(array, "shape") or len(array.shape) < 2:
            aligned.append(array)
        else:
            aligned.append(array[:, :target, ...])
    return tuple(aligned)


def _normalize_ppo_time_axis(logits: Any, values: Any, labels: Any, mask: Any) -> tuple[Any, Any, Any, Any]:
    if hasattr(values, "ndim") and int(values.ndim) == 3 and int(values.shape[-1]) == 1:
        values = values.squeeze(-1)
    if hasattr(logits, "shape") and len(logits.shape) >= 3 and int(logits.shape[1]) > 1:
        logits = logits[:, :-1, :]
    if hasattr(values, "shape") and len(values.shape) >= 2 and int(values.shape[1]) > 1:
        if int(values.shape[1]) == int(labels.shape[1]) + 1:
            values = values[:, :-1]
        elif hasattr(logits, "shape") and len(logits.shape) >= 2 and int(values.shape[1]) == int(logits.shape[1]) + 1:
            values = values[:, :-1]
    return _align_time_axis(logits, values, labels, mask)


def _split_chosen_rejected(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = values.shape[0] // 2
    return values[:midpoint], values[midpoint:]


def _dpo_loss(
    policy_logps: np.ndarray,
    ref_logps: np.ndarray | None,
    beta: float,
    loss_type: str,
    label_smoothing: float,
    simpo_gamma: float,
) -> tuple[float, dict[str, Any]]:
    chosen_pi, rejected_pi = _split_chosen_rejected(policy_logps)
    pi_logratios = chosen_pi - rejected_pi
    if ref_logps is None:
        ref_logratios = np.zeros_like(pi_logratios)
    else:
        chosen_ref, rejected_ref = _split_chosen_rejected(ref_logps)
        ref_logratios = chosen_ref - rejected_ref
    logits = pi_logratios - ref_logratios
    if loss_type == "sigmoid":
        losses = -(
            (1.0 - label_smoothing) * np.log(1.0 / (1.0 + np.exp(-beta * logits)))
            + label_smoothing * np.log(1.0 / (1.0 + np.exp(beta * logits)))
        )
    elif loss_type == "orpo":
        losses = -np.log(1.0 / (1.0 + np.exp(-beta * pi_logratios)))
    elif loss_type == "simpo":
        losses = -np.log(1.0 / (1.0 + np.exp(-beta * (pi_logratios - simpo_gamma))))
    else:
        raise ValueError(f"Unsupported DPO loss type: {loss_type}")
    return float(np.mean(losses)), {
        "chosen_policy_logps": chosen_pi.tolist(),
        "rejected_policy_logps": rejected_pi.tolist(),
        "logits_margin": logits.tolist(),
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    denom = max(float(mask.sum()), 1.0)
    return float((values * mask).sum() / denom)


def _compute_gae(rewards: np.ndarray, values: np.ndarray, mask: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[0], dtype=np.float32)
    for timestep in range(rewards.shape[1] - 1, -1, -1):
        next_values = values[:, timestep + 1] if timestep < rewards.shape[1] - 1 else np.zeros_like(values[:, 0])
        delta = rewards[:, timestep] + gamma * next_values * mask[:, timestep] - values[:, timestep]
        last = delta + gamma * lam * last * mask[:, timestep]
        advantages[:, timestep] = last
    returns = advantages + values
    return advantages, returns


def _ppo_losses(
    logits: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    ignore_index: int,
    rewards: np.ndarray,
    gamma: float = 1.0,
    lam: float = 0.95,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
) -> dict[str, Any]:
    logits, values, labels, mask, rewards = _align_time_axis(logits, values, labels, mask, rewards)
    safe_mask = mask * (labels != ignore_index).astype(np.float32)
    safe_labels = np.where(labels == ignore_index, 0, labels)
    token_logps = logits - _logsumexp(logits, axis=-1, keepdims=True)
    gathered = np.take_along_axis(token_logps, safe_labels[..., None], axis=-1).squeeze(-1)
    old_logps = gathered.copy()
    advantages, returns = _compute_gae(rewards, values, safe_mask, gamma, lam)
    ratio = np.exp(gathered - old_logps)
    pg_1 = -advantages * ratio
    pg_2 = -advantages * np.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    policy_loss = _masked_mean(np.maximum(pg_1, pg_2), safe_mask)
    v_clipped = np.clip(values, values - cliprange_value, values + cliprange_value)
    vf_1 = (values - returns) ** 2
    vf_2 = (v_clipped - returns) ** 2
    value_loss = 0.5 * _masked_mean(np.maximum(vf_1, vf_2), safe_mask)
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "advantages": advantages,
        "returns": returns,
        "token_logprobs": gathered,
    }


def _torch_causal_loss(logits: Any, labels: Any, ignore_index: int) -> Any:
    import torch
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].contiguous().long()
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )


def _torch_batch_logps(logits: Any, labels: Any, ignore_index: int) -> tuple[Any, Any]:
    import torch

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].contiguous().long()
    mask = shifted_labels != ignore_index
    safe_labels = torch.where(mask, shifted_labels, torch.zeros_like(shifted_labels))
    log_probs = shifted_logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1), mask.sum(dim=-1)


def _jax_causal_loss(logits: Any, labels: Any, ignore_index: int) -> Any:
    import jax.scipy.special
    import jax.numpy as jnp

    shifted_logits = logits[:, :-1, :].astype(jnp.float32)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = jnp.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - jax.scipy.special.logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    token_losses = -gathered * mask
    denom = jnp.maximum(mask.sum(), 1)
    return token_losses.sum() / denom


def _jax_batch_logps(logits: Any, labels: Any, ignore_index: int) -> tuple[Any, Any]:
    import jax.scipy.special
    import jax.numpy as jnp

    shifted_logits = logits[:, :-1, :].astype(jnp.float32)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = jnp.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - jax.scipy.special.logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    return (gathered * mask).sum(axis=-1), mask.sum(axis=-1)


class RepoRuntime:
    """Uniform evaluator bridge over a repo implementation."""

    def __init__(
        self,
        eval_config: EvalConfig,
        implementation: ImplementationConfig,
        modules: RepoModules,
        repo_config: dict[str, Any],
    ) -> None:
        self.eval_config = eval_config
        self.implementation = implementation
        self.modules = modules
        self.repo_config = repo_config
        self.framework = implementation.framework
        self.jax_device = _resolve_jax_device(implementation.device) if implementation.framework == "flax" else None
        self.ignore_index = _get_ignore_index(modules.repo_dir)
        self.args = modules.config.get_train_args(repo_config)
        self.model_args = self.args[0]
        self.data_args = self.args[1]
        self.training_args = self.args[2]
        self.finetuning_args = self.args[3]
        self.extra_args = self.args[4:]
        tokenizer_module = modules.model_loader.load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module.get("processor")
        self.template = _get_template(modules, self.tokenizer, self.data_args)
        eval_examples = repo_config.get("eval_examples")
        if eval_examples is not None:
            if not isinstance(eval_examples, list) or not eval_examples:
                raise ValueError("shared_config.eval_examples must be a non-empty list.")
            dataset_module = _build_dataset_from_eval_examples(
                self.eval_config.training_method,
                eval_examples,
                self.template,
                self.tokenizer,
                self.processor,
                self.data_args,
            )
        else:
            dataset_module = _call_with_supported_kwargs(
                modules.data_loader.get_dataset,
                template=self.template,
                model_args=self.model_args,
                data_args=self.data_args,
                training_args=self.training_args,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        self.dataset_module = dataset_module
        self.train_dataset = dataset_module.get("train_dataset") if isinstance(dataset_module, dict) else None
        if self.train_dataset is None:
            raise ValueError(f"{implementation.name} did not expose train_dataset.")
        self.model, self.params, self.frozen_prefixes = self._load_model(trainable=True)
        self.ref_model, self.ref_params = self._load_reference_model()
        self.collator = self._build_collator()
        self.batch = _first_batch(self.train_dataset, self.collator, eval_config.batch_size)
        self.runtime_contract: dict[str, Any] = {}

    @classmethod
    def create(cls, eval_config: EvalConfig, implementation: ImplementationConfig) -> "RepoRuntime":
        _set_global_seed(eval_config.seed)
        repo_config = _prepare_repo_config(eval_config, implementation)
        with repo_imports(implementation.repo_dir) as modules:
            return cls(eval_config, implementation, modules, repo_config)

    def _load_model(self, trainable: bool) -> tuple[Any, Any | None, list[str]]:
        kwargs = {
            "tokenizer": self.tokenizer,
            "model_args": self.model_args,
            "finetuning_args": self.finetuning_args,
            "is_trainable": trainable,
        }
        if self.eval_config.training_method == "dpo" and self.extra_args:
            kwargs["disable_dropout"] = bool(getattr(self.extra_args[0], "disable_dropout", True))
        if self.eval_config.training_method == "ppo":
            kwargs["add_valuehead"] = True
        backend_device = self.implementation.backend_device or self.implementation.overrides.get("backend_device") or self.implementation.device
        if backend_device:
            kwargs["backend_device"] = backend_device
        if self.framework == "flax" and self.implementation.device:
            kwargs["jax_device"] = self.implementation.device
        loaded = _call_with_supported_kwargs(self.modules.model_loader.load_model, **kwargs)
        if isinstance(loaded, tuple):
            model = loaded[0]
            params = loaded[1] if len(loaded) > 1 else None
            frozen_prefixes = list(loaded[2]) if len(loaded) > 2 else []
            if self.framework == "pytorch" and self.implementation.device and hasattr(model, "to"):
                current_device = _module_primary_device(model)
                if str(current_device) != str(self.implementation.device):
                    model = model.to(self.implementation.device)
            _repair_rotary_buffers(model)
            _empty_cuda_cache()
            if self.eval_config.training_method == "ppo":
                _ensure_ppo_valuehead(model)
            return model, params, frozen_prefixes
        if self.framework == "pytorch" and self.implementation.device and hasattr(loaded, "to"):
            current_device = _module_primary_device(loaded)
            if str(current_device) != str(self.implementation.device):
                loaded = loaded.to(self.implementation.device)
        _repair_rotary_buffers(loaded)
        _empty_cuda_cache()
        if self.eval_config.training_method == "ppo":
            _ensure_ppo_valuehead(loaded)
        return loaded, None, []

    def _load_reference_model(self) -> tuple[Any | None, Any | None]:
        if self.eval_config.training_method == "dpo" and hasattr(self.modules.model_loader, "load_ref_model"):
            dpo_args = self.extra_args[0]
            loaded = _call_with_supported_kwargs(
                self.modules.model_loader.load_ref_model,
                tokenizer=self.tokenizer,
                model_args=self.model_args,
                finetuning_args=self.finetuning_args,
                dpo_args=dpo_args,
            )
            ref_device = self.implementation.overrides.get("reference_device")
            if ref_device is None:
                ref_device = self.implementation.backend_device or self.implementation.device
            if self.framework == "pytorch" and ref_device and loaded is not None and hasattr(loaded, "to"):
                loaded = loaded.to(ref_device)
            _repair_rotary_buffers(loaded)
            return loaded, None
        return None, None

    def _build_collator(self) -> Any:
        if self.modules.data_collator is None:
            return None
        module = self.modules.data_collator
        candidates = {
            "sft": "SFTDataCollatorWith4DAttentionMask",
            "dpo": "PairwiseDataCollatorWithPadding",
            "ppo": "MultiModalDataCollatorForSeq2Seq",
        }
        class_name = candidates[self.eval_config.training_method]
        if not hasattr(module, class_name):
            fallback = getattr(module, "MultiModalDataCollatorForSeq2Seq", None)
            if fallback is None:
                return None
            collator_cls = fallback
        else:
            collator_cls = getattr(module, class_name)
        kwargs = {
            "template": self.template,
            "model": self.model,
            "pad_to_multiple_of": getattr(self.training_args, "pad_to_multiple_of", None),
            "label_pad_token_id": self.ignore_index,
            "tokenizer": self.tokenizer,
            "processor": self.processor,
            "compute_dtype": getattr(self.model_args, "compute_dtype", None),
            "block_diag_attn": getattr(self.model_args, "block_diag_attn", False),
            "attn_implementation": getattr(getattr(self.model, "config", None), "_attn_implementation", "eager"),
        }
        return _call_with_supported_kwargs(collator_cls, **kwargs)

    def derive_runtime_contract(self) -> dict[str, Any]:
        contract: dict[str, Any] = {}
        if self.eval_config.training_method != "ppo":
            self.runtime_contract = contract
            return contract
        model = self.model
        params = self.params
        if self.framework == "pytorch":
            output = self._torch_forward(model, self.batch, output_hidden_states=False)
        else:
            output = self._jax_forward(model, params, self.batch, output_hidden_states=False)
        value_field = _detect_reference_value_field(output)
        if value_field is not None:
            contract["ppo_values_source"] = {"kind": "field", "name": value_field}
        else:
            hidden_source = _detect_reference_hidden_source(output)
            if hidden_source is not None and hasattr(self._ppo_backend_model(model), "v_head"):
                contract["ppo_values_source"] = {
                    "kind": "derived_from_hidden",
                    "hidden_source": hidden_source,
                }
            else:
                contract["ppo_values_source"] = {"kind": "missing"}
        self.runtime_contract = contract
        return contract

    def _wrapper_backend_model(self, model: Any | None = None) -> Any | None:
        target = self.model if model is None else model
        backend = getattr(target, "pt_model", None)
        return backend if backend is not None else target

    def _callable_model(self, model: Any | None = None) -> Any:
        target = self.model if model is None else model
        backend = getattr(target, "pt_model", None)
        if backend is not None and _uses_default_torch_forward(target):
            return backend
        if callable(target):
            return target
        if backend is not None and callable(backend):
            return backend
        return target

    def parameter_artifact(self) -> dict[str, Any]:
        return {key: _summarize_param_leaf(value) for key, value in self.flat_parameters().items()}

    def flat_parameters(self) -> dict[str, Any]:
        if self.params is not None:
            return _flatten_tree_raw(self.params)
        if hasattr(self.model, "state_dict"):
            return _flatten_state_dict(self.model.state_dict())
        raise ValueError(f"{self.implementation.name} does not expose parameters.")

    def batch_artifact(self) -> dict[str, Any]:
        return {key: summarize_tensor(value) for key, value in self.batch.items()}

    def batch_arrays(self) -> dict[str, np.ndarray]:
        return {key: to_numpy(value) for key, value in self.batch.items()}

    def _torch_forward(self, model: Any, batch: dict[str, Any], output_hidden_states: bool = False) -> Any:
        model = self._callable_model(model)
        _reset_runtime_state(model)
        primary_device = _module_primary_device(model)
        device = str(primary_device) if primary_device is not None else self.implementation.device
        prepared = {}
        for key, value in batch.items():
            if key == "labels" and self.eval_config.training_method == "ppo":
                continue
            prepared[key] = _to_torch(value, device=device)
        if hasattr(model, "eval"):
            model.eval()
        import torch

        with torch.no_grad():
            return _call_model_with_control_kwargs(
                model,
                model_inputs=prepared,
                return_dict=True,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )

    def _jax_forward(self, model: Any, params: Any, batch: dict[str, Any], output_hidden_states: bool = False) -> Any:
        import jax

        _reset_runtime_state(model)
        prepared = {}
        wrapper_mode = getattr(model, "pt_model", None) is not None
        for key, value in batch.items():
            if key == "labels" and self.eval_config.training_method == "ppo":
                continue
            converted = _to_jax_on_device(value, self.jax_device)
            prepared[key] = _normalize_wrapper_tree(converted) if wrapper_mode else converted
        variables = {"params": _normalize_wrapper_tree(params) if wrapper_mode else params} if params is not None else {}
        if self.jax_device is None:
            return _call_model_with_control_kwargs(
                model.apply,
                variables,
                model_inputs=prepared,
                return_dict=True,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )
        with jax.default_device(self.jax_device):
            return _call_model_with_control_kwargs(
                model.apply,
                variables,
                model_inputs=prepared,
                return_dict=True,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )

    def forward_outputs(self, batch: dict[str, Any] | None = None, *, reference_model: bool = False, output_hidden_states: bool | None = None) -> dict[str, Any]:
        batch = batch or self.batch
        model = self.ref_model if reference_model else self.model
        params = self.ref_params if reference_model else self.params
        if model is None:
            raise ValueError("Reference model is not available.")
        # For PPO, we need hidden states for value head computation
        need_hidden_for_ppo = (
            self.eval_config.training_method == "ppo"
            and self.runtime_contract.get("ppo_values_source", {}).get("kind") == "derived_from_hidden"
        )
        if output_hidden_states is None:
            output_hidden_states = self.eval_config.compare_hidden_states or need_hidden_for_ppo
        if self.framework == "pytorch":
            output = self._torch_forward(model, batch, output_hidden_states=output_hidden_states)
        else:
            output = self._jax_forward(model, params, batch, output_hidden_states=output_hidden_states)
        logits = _extract_output_field(output, "logits")
        loss = _extract_output_field(output, "loss")
        hidden_states = _extract_output_field(output, "hidden_states")
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
            if len(output) > 1:
                loss = output[1]
            if len(output) > 2:
                hidden_states = output[2]
        result = {
            "logits": to_numpy(logits),
            "loss": None if loss is None else float(np.asarray(to_numpy(loss)).reshape(())),
        }
        if hidden_states is not None and self.eval_config.compare_hidden_states:
            result["hidden_states"] = [to_numpy(item) for item in hidden_states]
        if self.eval_config.training_method == "ppo":
            values = self._extract_ppo_values(output, model, hidden_states=hidden_states)
            if values is not None and hasattr(values, "shape"):
                result["values"] = to_numpy(values)
        return result

    def _ppo_backend_model(self, model: Any) -> Any:
        """Get the backend model that might have v_head."""
        pt = getattr(model, "pt_model", None)
        if pt is not None:
            return pt
        return model

    def _compute_values_from_hidden(self, v_head: Any, last_hidden_state: Any) -> Any:
        import torch
        if isinstance(last_hidden_state, torch.Tensor):
            return v_head(last_hidden_state.to(v_head.weight.device)).squeeze(-1)
        # JAX/numpy path
        hidden = _to_torch(last_hidden_state, device=str(v_head.weight.device))
        return v_head(hidden).squeeze(-1)

    def _extract_ppo_values(self, output: Any, model: Any, hidden_states: Any = None) -> Any:
        source = self.runtime_contract.get("ppo_values_source", {})
        kind = source.get("kind")
        if kind == "field":
            name = source.get("name")
            if isinstance(name, str):
                values = _extract_output_field_exact(output, name, None)
                if values is not None and hasattr(values, "shape"):
                    return values
            return None
        if kind == "derived_from_hidden":
            backend = self._ppo_backend_model(model)
            v_head = getattr(backend, "v_head", None)
            if v_head is None:
                return None
            hidden_source = source.get("hidden_source")
            last_hs = None
            if hidden_source == "last_hidden_state":
                last_hs = _extract_output_field(output, "last_hidden_state")
            elif hidden_source == "hidden_states":
                if hidden_states is None:
                    hidden_states = _extract_output_field(output, "hidden_states")
                if hidden_states is not None:
                    last_hs = hidden_states[-1]
            if last_hs is not None and hasattr(last_hs, "shape"):
                return self._compute_values_from_hidden(v_head, last_hs)
            return None
        values = _extract_output_field(output, "values", None)
        if values is not None and hasattr(values, "shape"):
            return values
        backend = self._ppo_backend_model(model)
        v_head = getattr(backend, "v_head", None)
        if v_head is not None:
            last_hs = _extract_output_field(output, "last_hidden_state")
            if last_hs is None and hidden_states is not None:
                last_hs = hidden_states[-1]
            if last_hs is not None and hasattr(last_hs, "shape"):
                return self._compute_values_from_hidden(v_head, last_hs)
        return None

    def loss_artifact(self, batch: dict[str, Any] | None = None) -> dict[str, Any]:
        batch = batch or self.batch
        outputs = self.forward_outputs(batch)
        labels = to_numpy(batch["labels"]) if "labels" in batch else to_numpy(batch["input_ids"])
        logits = outputs["logits"]
        if self.eval_config.training_method == "sft":
            loss = outputs["loss"]
            if loss is None:
                loss = _causal_cross_entropy(logits, labels, self.ignore_index)
            return {"loss": float(loss)}
        if self.eval_config.training_method == "dpo":
            policy_logps, valid_len = _batch_logps(logits, labels, self.ignore_index)
            ref_logps = None
            if self.ref_model is not None:
                ref_outputs = self.forward_outputs(batch, reference_model=True, output_hidden_states=False)
                ref_logps, _ = _batch_logps(ref_outputs["logits"], labels, self.ignore_index)
            dpo_args = self.extra_args[0]
            loss, details = _dpo_loss(
                policy_logps=policy_logps,
                ref_logps=ref_logps,
                beta=float(getattr(dpo_args, "pref_beta", 0.1)),
                loss_type=str(getattr(dpo_args, "pref_loss", "sigmoid")),
                label_smoothing=float(getattr(dpo_args, "dpo_label_smoothing", 0.0)),
                simpo_gamma=float(getattr(dpo_args, "simpo_gamma", 0.5)),
            )
            return {
                "loss": loss,
                "policy_logps": policy_logps.tolist(),
                "valid_length": valid_len.tolist(),
                "reference_logps": None if ref_logps is None else ref_logps.tolist(),
                **details,
            }
        if self.eval_config.training_method == "ppo":
            values = outputs.get("values")
            if values is None:
                raise ValueError("PPO evaluation requires model outputs to include values.")
            labels = to_numpy(batch.get("labels", batch["input_ids"]))[:, 1:].astype(np.int64, copy=False)
            mask = to_numpy(batch.get("attention_mask", np.ones_like(labels)))[:, 1:].astype(np.float32)
            logits, values, labels, mask = _normalize_ppo_time_axis(logits, values, labels, mask)
            synthetic_rewards = ((labels % 17).astype(np.float32) / 17.0) * mask
            losses = _ppo_losses(logits, values, labels, mask, self.ignore_index, synthetic_rewards)
            return {
                "loss": float(losses["policy_loss"] + losses["value_loss"]),
                "policy_loss": float(losses["policy_loss"]),
                "value_loss": float(losses["value_loss"]),
                "advantages": losses["advantages"].tolist(),
                "returns": losses["returns"].tolist(),
                "token_logprobs": losses["token_logprobs"].tolist(),
            }
        raise ValueError(f"Unsupported training method: {self.eval_config.training_method}")

    def gradient_artifact(self) -> dict[str, Any]:
        if self.framework == "pytorch":
            return self._torch_gradient_artifact()
        return self._jax_gradient_artifact()

    def _torch_gradient_artifact(self) -> dict[str, Any]:
        model = self._callable_model(self.model)
        device = self._training_backend_device(model) or self.implementation.device
        batch = {key: _to_torch(value, device=device) for key, value in self.batch.items()}
        _reset_runtime_state(model)
        if hasattr(model, "train"):
            model.train()
        trainable_params = [param for param in model.parameters() if getattr(param, "requires_grad", True)]
        for param in trainable_params:
            param.grad = None
        inputs = {k: v for k, v in batch.items() if not (self.eval_config.training_method == "ppo" and k == "labels")}
        need_hidden = self.eval_config.training_method == "ppo" and hasattr(self._ppo_backend_model(model), "v_head")
        outputs = _call_model_with_control_kwargs(
            model,
            model_inputs=inputs,
            return_dict=True,
            use_cache=False,
            output_hidden_states=need_hidden,
        )
        loss_tensor = self._torch_scalar_loss(outputs, batch, model)
        loss_tensor.backward()
        grads = {}
        grad_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float().cpu().numpy()
            grads[name] = summarize_tensor(grad)
            grad_norm_sq += float((grad ** 2).sum())
        for param in trainable_params:
            param.grad = None
        return {"loss": float(loss_tensor.detach().cpu().item()), "grad_norm": math.sqrt(grad_norm_sq), "grads": grads}

    def _jax_gradient_artifact(self) -> dict[str, Any]:
        import jax

        custom_gradient = getattr(self.model, "gradient_artifact", None)
        if callable(custom_gradient):
            _reset_runtime_state(self.model)
            artifact = custom_gradient(
                batch=_normalize_wrapper_tree(self.batch),
                training_method=self.eval_config.training_method,
                ignore_index=self.ignore_index,
            )
            if isinstance(artifact, dict):
                normalized = _normalize_gradient_artifact(artifact)
                if "loss" in normalized and "grad_norm" in normalized:
                    return jsonable(normalized)
                backend_model = getattr(self.model, "pt_model", None)
                if backend_model is not None and hasattr(backend_model, "parameters"):
                    return self._torch_gradient_artifact()

        if self.params is None:
            backend_model = getattr(self.model, "pt_model", None)
            if backend_model is not None and hasattr(backend_model, "parameters"):
                return self._torch_gradient_artifact()
            raise ValueError(
                "flax candidate must expose params or provide a complete gradient_artifact() "
                "when used with evaluator gradient checks"
            )

        params = self.params
        batch = {key: _to_jax_on_device(value, self.jax_device) for key, value in self.batch.items()}

        def scalar_loss(p: Any) -> Any:
            param_input = _normalize_wrapper_tree(p) if getattr(self.model, "pt_model", None) is not None else p
            batch_input = {k: v for k, v in batch.items() if not (self.eval_config.training_method == "ppo" and k == "labels")}
            if getattr(self.model, "pt_model", None) is not None:
                batch_input = _normalize_wrapper_tree(batch_input)
            outputs = _call_model_with_control_kwargs(
                self.model.apply,
                {"params": param_input},
                model_inputs=batch_input,
                return_dict=True,
                use_cache=False,
            )
            return self._jax_scalar_loss(outputs, batch, p)

        if self.jax_device is None:
            loss_value, grads = jax.value_and_grad(scalar_loss)(params)
        else:
            with jax.default_device(self.jax_device):
                loss_value, grads = jax.value_and_grad(scalar_loss)(params)
        flat = flatten_dict(grads)
        grad_norm_sq = sum(float(np.square(value.astype(np.float32)).sum()) for value in flat.values())
        return {
            "loss": float(np.asarray(loss_value)),
            "grad_norm": math.sqrt(grad_norm_sq),
            "grads": {name: summarize_tensor(value) for name, value in flat.items()},
        }

    def _torch_scalar_loss(self, output: Any, batch: dict[str, Any], model: Any) -> Any:
        import torch

        logits = _extract_output_field(output, "logits")
        loss = _extract_output_field(output, "loss")
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
            if len(output) > 1:
                loss = output[1]
        if self.eval_config.training_method == "sft":
            return loss if loss is not None else _torch_causal_loss(logits, batch["labels"], self.ignore_index)
        if self.eval_config.training_method == "dpo":
            policy_logps, _ = _torch_batch_logps(logits, batch["labels"], self.ignore_index)
            ref_logps = None
            if self.ref_model is not None:
                ref_device_obj = _module_primary_device(self.ref_model)
                ref_device = str(ref_device_obj) if ref_device_obj is not None else self.implementation.backend_device or self.implementation.device
                ref_labels = _to_torch(batch["labels"], device=ref_device)
                with torch.no_grad():
                    ref_outputs = _call_model_with_control_kwargs(
                        self.ref_model,
                        model_inputs={
                            k: _to_torch(v, device=ref_device)
                            for k, v in batch.items()
                            if k != "labels"
                        },
                        return_dict=True,
                        use_cache=False,
                    )
                ref_logits = _extract_output_field(ref_outputs, "logits")
                ref_logps, _ = _torch_batch_logps(ref_logits, ref_labels, self.ignore_index)
                ref_logps = ref_logps.to(policy_logps.device)
            dpo_args = self.extra_args[0]
            chosen_pi, rejected_pi = torch.split(policy_logps, policy_logps.shape[0] // 2, dim=0)
            pi_logratios = chosen_pi - rejected_pi
            if ref_logps is None:
                ref_logratios = torch.zeros_like(pi_logratios)
            else:
                chosen_ref, rejected_ref = torch.split(ref_logps, ref_logps.shape[0] // 2, dim=0)
                ref_logratios = chosen_ref - rejected_ref
            beta = float(getattr(dpo_args, "pref_beta", 0.1))
            label_smoothing = float(getattr(dpo_args, "dpo_label_smoothing", 0.0))
            simpo_gamma = float(getattr(dpo_args, "simpo_gamma", 0.5))
            loss_type = str(getattr(dpo_args, "pref_loss", "sigmoid"))
            logits_margin = pi_logratios - ref_logratios
            if loss_type == "sigmoid":
                return -(
                    (1.0 - label_smoothing) * torch.nn.functional.logsigmoid(beta * logits_margin)
                    + label_smoothing * torch.nn.functional.logsigmoid(-beta * logits_margin)
                ).mean()
            if loss_type == "orpo":
                return -torch.nn.functional.logsigmoid(beta * pi_logratios).mean()
            if loss_type == "simpo":
                return -torch.nn.functional.logsigmoid(beta * (pi_logratios - simpo_gamma)).mean()
            raise ValueError(f"Unsupported DPO loss type: {loss_type}")
        if self.eval_config.training_method == "ppo":
            hidden_states = _extract_output_field(output, "hidden_states")
            values = self._extract_ppo_values(output, model, hidden_states=hidden_states)
            if values is None or not hasattr(values, "shape"):
                raise ValueError("PPO model must return values.")
            labels = batch.get("labels", batch["input_ids"])[:, 1:].long()
            mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))[:, 1:].float()
            logits, values, labels, mask = _normalize_ppo_time_axis(logits, values.float(), labels, mask)
            safe_mask = mask * (labels != self.ignore_index).float()
            safe_labels = torch.where(labels == self.ignore_index, torch.zeros_like(labels), labels)
            log_probs = logits.float().log_softmax(dim=-1)
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
            policy_loss = ((-advantages * torch.ones_like(gathered)) * safe_mask).sum() / safe_mask.sum().clamp(min=1.0)
            value_loss = 0.5 * (((values - returns) ** 2) * safe_mask).sum() / safe_mask.sum().clamp(min=1.0)
            return policy_loss + value_loss
        raise ValueError(f"Unsupported training method: {self.eval_config.training_method}")

    def _jax_scalar_loss(self, output: Any, batch: dict[str, Any], params: Any) -> Any:
        import jax
        import jax.numpy as jnp
        import jax.scipy.special

        logits = _extract_output_field(output, "logits")
        loss = _extract_output_field(output, "loss")
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
            if len(output) > 1:
                loss = output[1]
        if self.eval_config.training_method == "sft":
            return loss if loss is not None else _jax_causal_loss(logits, batch["labels"], self.ignore_index)
        if self.eval_config.training_method == "dpo":
            policy_logps, _ = _jax_batch_logps(logits, batch["labels"], self.ignore_index)
            ref_logps = None
            if self.ref_model is not None and self.ref_params is not None:
                ref_params = _normalize_wrapper_tree(self.ref_params) if getattr(self.ref_model, "pt_model", None) is not None else self.ref_params
                ref_batch = {k: v for k, v in batch.items() if k != "labels"}
                if getattr(self.ref_model, "pt_model", None) is not None:
                    ref_batch = _normalize_wrapper_tree(ref_batch)
                ref_output = _call_model_with_control_kwargs(
                    self.ref_model.apply,
                    {"params": ref_params},
                    model_inputs=ref_batch,
                    return_dict=True,
                    use_cache=False,
                )
                ref_logits = _extract_output_field(ref_output, "logits")
                ref_logps, _ = _jax_batch_logps(ref_logits, batch["labels"], self.ignore_index)
                ref_logps = jax.lax.stop_gradient(ref_logps)
            dpo_args = self.extra_args[0]
            chosen_pi, rejected_pi = jnp.split(policy_logps, 2, axis=0)
            pi_logratios = chosen_pi - rejected_pi
            if ref_logps is None:
                ref_logratios = jnp.zeros_like(pi_logratios)
            else:
                chosen_ref, rejected_ref = jnp.split(ref_logps, 2, axis=0)
                ref_logratios = chosen_ref - rejected_ref
            beta = float(getattr(dpo_args, "pref_beta", 0.1))
            label_smoothing = float(getattr(dpo_args, "dpo_label_smoothing", 0.0))
            simpo_gamma = float(getattr(dpo_args, "simpo_gamma", 0.5))
            loss_type = str(getattr(dpo_args, "pref_loss", "sigmoid"))
            logits_margin = pi_logratios - ref_logratios
            if loss_type == "sigmoid":
                return -(
                    (1.0 - label_smoothing) * jax.nn.log_sigmoid(beta * logits_margin)
                    + label_smoothing * jax.nn.log_sigmoid(-beta * logits_margin)
                ).mean()
            if loss_type == "orpo":
                return -jax.nn.log_sigmoid(beta * pi_logratios).mean()
            if loss_type == "simpo":
                return -jax.nn.log_sigmoid(beta * (pi_logratios - simpo_gamma)).mean()
            raise ValueError(f"Unsupported DPO loss type: {loss_type}")
        if self.eval_config.training_method == "ppo":
            values = self._extract_ppo_values(output, self.model)
            if values is None or not hasattr(values, "shape"):
                raise ValueError("PPO model must return values.")
            labels = batch.get("labels", batch["input_ids"])[:, 1:].astype(jnp.int32)
            mask = batch.get("attention_mask", jnp.ones_like(batch["input_ids"]))[:, 1:].astype(jnp.float32)
            logits, values, labels, mask = _normalize_ppo_time_axis(
                _to_jax(logits).astype(jnp.float32),
                _to_jax(values).astype(jnp.float32),
                labels,
                mask,
            )
            safe_mask = mask * (labels != self.ignore_index).astype(jnp.float32)
            safe_labels = jnp.where(labels == self.ignore_index, 0, labels)
            log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
            gathered = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
            rewards = ((safe_labels % 17).astype(jnp.float32) / 17.0) * safe_mask
            advantages = jnp.zeros_like(rewards)
            last = jnp.zeros(rewards.shape[0], dtype=jnp.float32)
            for timestep in range(rewards.shape[1] - 1, -1, -1):
                next_values = values[:, timestep + 1] if timestep < rewards.shape[1] - 1 else jnp.zeros_like(values[:, 0])
                delta = rewards[:, timestep] + next_values * safe_mask[:, timestep] - values[:, timestep]
                last = delta + 0.95 * last * safe_mask[:, timestep]
                advantages = advantages.at[:, timestep].set(last)
            returns = advantages + values
            policy_loss = ((-advantages) * safe_mask).sum() / jnp.maximum(safe_mask.sum(), 1.0)
            value_loss = 0.5 * (((values - returns) ** 2) * safe_mask).sum() / jnp.maximum(safe_mask.sum(), 1.0)
            return policy_loss + value_loss
        raise ValueError(f"Unsupported training method: {self.eval_config.training_method}")

    def lr_schedule_artifact(self, num_steps: int = 8) -> dict[str, Any]:
        warmup_steps = int(getattr(self.training_args, "warmup_steps", 0))
        if warmup_steps == 0:
            warmup_ratio_value = getattr(self.training_args, "warmup_ratio", 0.0)
            warmup_ratio = float(0.0 if warmup_ratio_value is None else warmup_ratio_value)
            warmup_steps = int(num_steps * warmup_ratio)
        learning_rate_value = getattr(self.training_args, "learning_rate", 1e-4)
        base_lr = float(1e-4 if learning_rate_value is None else learning_rate_value)
        scheduler_value = getattr(self.training_args, "lr_scheduler_type", "cosine")
        scheduler = str("cosine" if scheduler_value is None else scheduler_value)
        values = []
        for step in range(num_steps):
            if step < warmup_steps:
                scale = float(step) / float(max(1, warmup_steps))
            elif scheduler == "linear":
                scale = max(0.0, float(num_steps - step) / float(max(1, num_steps - warmup_steps)))
            else:
                progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
                scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            values.append(base_lr * scale)
        return {"learning_rates": values}

    def gradient_accumulation_invariant(self) -> dict[str, Any]:
        batch = self.batch
        batch_size = int(to_numpy(batch["input_ids"]).shape[0])
        if batch_size < 2:
            return {"supported": False, "reason": "batch_size < 2"}
        # For DPO, each micro-batch needs at least 2 examples (chosen+rejected pair)
        if self.eval_config.training_method == "dpo" and batch_size < 4:
            return {"supported": False, "reason": "DPO requires batch_size >= 4 for accumulation invariant"}
        midpoint = batch_size // 2
        full_loss = self.loss_artifact()["loss"]
        left = {key: to_numpy(value)[:midpoint] for key, value in batch.items()}
        right = {key: to_numpy(value)[midpoint:] for key, value in batch.items()}
        left_loss = self.loss_artifact(batch=left)["loss"]
        right_loss = self.loss_artifact(batch=right)["loss"]
        micro_loss = (left_loss + right_loss) / 2.0
        return {"supported": True, "full_batch_loss": full_loss, "micro_batch_loss": micro_loss}

    def _release_loaded_state(self) -> None:
        for attribute in ("model", "ref_model", "batch", "params", "ref_params", "train_dataset", "dataset_module"):
            if hasattr(self, attribute):
                try:
                    setattr(self, attribute, None)
                except Exception:
                    pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _training_backend_model(self) -> Any | None:
        candidate = getattr(self.model, "pt_model", self.model)
        if candidate is not None and hasattr(candidate, "parameters") and hasattr(candidate, "named_parameters"):
            return candidate
        return None

    def _training_backend_device(self, model: Any) -> str | None:
        model_device = getattr(self.model, "device", None)
        if callable(model_device):
            try:
                resolved = model_device()
                if resolved is not None:
                    return str(resolved)
            except Exception:
                pass
        elif model_device is not None:
            return str(model_device)
        candidate_device = self.implementation.backend_device or self.implementation.device
        if candidate_device:
            return candidate_device
        try:
            first_param = next(model.parameters())
            return str(first_param.device)
        except Exception:
            return None

    def _replay_step_schedule(self, num_steps: int) -> list[float]:
        return self.lr_schedule_artifact(num_steps=max(num_steps, 1))["learning_rates"][:num_steps]

    def _torch_step_loss(self, model: Any, batch: dict[str, Any]) -> Any:
        import torch

        device = self._training_backend_device(model)
        prepared = {key: _to_torch(value, device=device) for key, value in batch.items()}
        inputs = {k: v for k, v in prepared.items() if not (self.eval_config.training_method == "ppo" and k == "labels")}
        need_hidden = self.eval_config.training_method == "ppo" and hasattr(model, "v_head")
        outputs = _call_model_with_control_kwargs(
            model,
            model_inputs=inputs,
            return_dict=True,
            use_cache=False,
            output_hidden_states=need_hidden,
        )
        return self._torch_scalar_loss(outputs, prepared, model)

    def _torch_replay_artifact(self, num_steps: int) -> dict[str, Any]:
        import torch

        model = self._training_backend_model()
        if model is None:
            raise ValueError("runtime does not expose a replayable training backend")

        schedule = self._replay_step_schedule(num_steps)
        trainable_params = [param for param in model.parameters() if getattr(param, "requires_grad", True)]
        trajectory: list[dict[str, Any]] = []
        was_training = model.training
        model.train(True)
        try:
            for step_index, learning_rate in enumerate(schedule):
                _reset_runtime_state(model)
                for param in trainable_params:
                    param.grad = None
                loss = self._torch_step_loss(model, self.batch)
                loss.backward()
                grad_norm_sq = 0.0
                for param in trainable_params:
                    if param.grad is None:
                        continue
                    grad = param.grad.detach().float()
                    grad_norm_sq += float((grad * grad).sum().item())
                with torch.no_grad():
                    for param in trainable_params:
                        if param.grad is None:
                            continue
                        param.add_(param.grad, alpha=-float(learning_rate))
                trajectory.append(
                    {
                        "step": step_index,
                        "learning_rate": float(learning_rate),
                        "loss": float(loss.detach().cpu().item()),
                        "grad_norm": float(math.sqrt(grad_norm_sq)),
                    }
                )
                for param in trainable_params:
                    param.grad = None
        finally:
            model.train(was_training)
        return {"supported": True, "trajectory": trajectory}

    def single_step_artifact(self) -> dict[str, Any]:
        replay = RepoRuntime.create(self.eval_config, self.implementation)
        try:
            artifact = replay._torch_replay_artifact(num_steps=1)
            if not artifact["trajectory"]:
                raise ValueError("single-step replay produced an empty trajectory")
            return {"supported": True, **artifact["trajectory"][0]}
        finally:
            replay._release_loaded_state()

    def loss_trajectory_artifact(self) -> dict[str, Any]:
        replay = RepoRuntime.create(self.eval_config, self.implementation)
        try:
            return replay._torch_replay_artifact(num_steps=self.eval_config.loss_trajectory_steps)
        finally:
            replay._release_loaded_state()

    def generation_artifact(self) -> dict[str, Any]:
        import inspect

        backend_model = getattr(self.model, "pt_model", self.model)
        if not hasattr(self.model, "generate") or not hasattr(backend_model, "generate"):
            return {"supported": False, "reason": "model has no generate"}
        inputs = {}
        for key in (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "pixel_values",
            "image_grid_thw",
            "video_grid_thw",
            "position_ids",
            "rope_deltas",
        ):
            if key in self.batch:
                inputs[key] = self.batch[key]
        if self.framework == "pytorch":
            prepared = {key: _to_torch(value, device=self.implementation.device) for key, value in inputs.items()}
        else:
            prepared = {key: value for key, value in inputs.items()}
        generate_signature = inspect.signature(getattr(self.model, "generate"))
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in generate_signature.parameters.values()
        )
        config = getattr(self.model, "config", None)
        supports_audio_output = bool(
            getattr(self.model, "has_talker", False)
            or getattr(config, "enable_audio_output", False)
        )
        if (
            ("generation_mode" in generate_signature.parameters or supports_var_kwargs and supports_audio_output)
            and "generation_mode" not in prepared
        ):
            prepared["generation_mode"] = "text"
        if (
            ("return_audio" in generate_signature.parameters or supports_var_kwargs and supports_audio_output)
            and "return_audio" not in prepared
        ):
            prepared["return_audio"] = False
        with _repo_resolution_context(self.implementation.repo_dir):
            sequences = self.model.generate(**prepared, max_new_tokens=self.eval_config.generation_max_new_tokens)
        data = to_numpy(sequences)
        return {"supported": True, "sequences": data.tolist()}
