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

"""Vendored candidate-side (jax-path) formulas (mirrors the engine's runtime.py).

Only the helpers a native-jax candidate needs at grade time are vendored here —
no torch branches, no wrapper (``pt_model``) handling, no reference (torch)
formulas. Vendored from the authoring engine's ``runtime.py`` and frozen at
generation time. The torch ``import`` inside :func:`_to_jax` is guarded and
never fires in the hermetic grading environment.
"""

from __future__ import annotations

import math
from dataclasses import is_dataclass
from importlib import import_module
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Array conversion / device placement / module typing.
# --------------------------------------------------------------------------- #
def to_jax(value: Any) -> Any:
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


def resolve_jax_device(device: str | None) -> Any | None:
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


def to_jax_on_device(value: Any, device: Any | None) -> Any:
    import jax

    array = to_jax(value)
    if device is None:
        return array
    return jax.device_put(array, device)


def is_flax_module(module: Any) -> bool:
    """True if ``module`` is a flax.linen / flax.nnx module (checks the MRO)."""
    for klass in type(module).__mro__:
        mod = getattr(klass, "__module__", "") or ""
        if mod.startswith("flax.linen") or mod.startswith("flax.nnx") or mod == "flax":
            return True
    return False


def reset_runtime_state(root: Any) -> None:
    queue = [root]
    seen: set[int] = set()
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


def get_ignore_index(repo_dir: str) -> int:
    try:
        constants = import_module("data.constants")
        return int(getattr(constants, "IGNORE_INDEX", -100))
    except Exception:
        return -100


def flatten_tree_raw(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_tree_raw(value, path))
        else:
            flat[path] = value
    return flat


# --------------------------------------------------------------------------- #
# Model-output field extraction (handles dict / dataclass / namedtuple / tuple).
# --------------------------------------------------------------------------- #
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
    return None if shape is None else len(shape)


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, (int, float, np.generic)):
        return True
    return _tensor_ndim(value) == 0


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
            if _tensor_ndim(item) != 3:
                continue
            if _tensor_shape(item) != logits_shape:
                return item
        return default
    return default


def extract_output_field(output: Any, name: str, default: Any = None) -> Any:
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


def _should_retry_without_control_kwargs(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "multiple values for keyword argument" in text
        or "got multiple values for argument" in text
        or "unexpected keyword argument" in text
    )


def call_model_with_control_kwargs(fn: Any, /, *args: Any, model_inputs: dict[str, Any], **control_kwargs: Any) -> Any:
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


# --------------------------------------------------------------------------- #
# Numpy loss formulas (operate on already-materialized logits; torch-free).
# --------------------------------------------------------------------------- #
def grad_stats(value: Any) -> dict[str, Any]:
    """Layout-invariant per-parameter gradient summary (TEST_PLAN §6.3).

    Only transposition/permutation-invariant scalars so a JAX candidate (Dense
    kernel = torch weight.T, embedding/scale = weight, ...) is comparable to the
    torch reference without matching shape/layout. Vendored byte-identical from
    the authoring engine's ``grad_stats``."""
    arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        arr = arr.astype(np.float32)
    arr = arr.astype(np.float64, copy=False)
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {"norm": 0.0, "mean": 0.0, "std": 0.0, "abs_mean": 0.0, "count": 0, "shape_sorted": [], "finite": True}
    return {
        "norm": float(np.sqrt(np.sum(flat * flat))),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "abs_mean": float(np.abs(flat).mean()),
        "count": n,
        "shape_sorted": sorted(int(d) for d in arr.shape),
        "finite": bool(np.all(np.isfinite(arr))),
    }


def np_logsumexp(values: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))


def causal_cross_entropy(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> float:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = np.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - np_logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = np.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    token_losses = -gathered * mask
    denom = max(float(mask.sum()), 1.0)
    return float(token_losses.sum() / denom)


def batch_logps(logits: np.ndarray, labels: np.ndarray, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = np.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - np_logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = np.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    return (gathered * mask).sum(axis=-1), mask.sum(axis=-1)


def _align_time_axis(*arrays: Any) -> tuple[Any, ...]:
    valid = [a for a in arrays if a is not None and hasattr(a, "shape") and len(a.shape) >= 2]
    if not valid:
        return arrays
    target = min(int(a.shape[1]) for a in valid)
    aligned = []
    for a in arrays:
        if a is None or not hasattr(a, "shape") or len(a.shape) < 2:
            aligned.append(a)
        else:
            aligned.append(a[:, :target, ...])
    return tuple(aligned)


def normalize_ppo_time_axis(logits: Any, values: Any, labels: Any, mask: Any) -> tuple[Any, Any, Any, Any]:
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


def dpo_loss(
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


def _compute_gae(rewards, values, mask, gamma, lam):
    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[0], dtype=np.float32)
    for timestep in range(rewards.shape[1] - 1, -1, -1):
        next_values = values[:, timestep + 1] if timestep < rewards.shape[1] - 1 else np.zeros_like(values[:, 0])
        delta = rewards[:, timestep] + gamma * next_values * mask[:, timestep] - values[:, timestep]
        last = delta + gamma * lam * last * mask[:, timestep]
        advantages[:, timestep] = last
    returns = advantages + values
    return advantages, returns


def ppo_losses(
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
    token_logps = logits - np_logsumexp(logits, axis=-1, keepdims=True)
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


# --------------------------------------------------------------------------- #
# JAX loss formulas (differentiable; used by value_and_grad in candidate.py).
# --------------------------------------------------------------------------- #
def jax_causal_loss(logits: Any, labels: Any, ignore_index: int) -> Any:
    import jax.numpy as jnp
    import jax.scipy.special

    shifted_logits = logits[:, :-1, :].astype(jnp.float32)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = jnp.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - jax.scipy.special.logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    token_losses = -gathered * mask
    denom = jnp.maximum(mask.sum(), 1)
    return token_losses.sum() / denom


def jax_batch_logps(logits: Any, labels: Any, ignore_index: int) -> tuple[Any, Any]:
    import jax.numpy as jnp
    import jax.scipy.special

    shifted_logits = logits[:, :-1, :].astype(jnp.float32)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = jnp.where(mask, shifted_labels, 0)
    log_probs = shifted_logits - jax.scipy.special.logsumexp(shifted_logits, axis=-1, keepdims=True)
    gathered = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    return (gathered * mask).sum(axis=-1), mask.sum(axis=-1)


def dpo_trainer_loss(
    all_logits: Any,
    labels: Any,
    *,
    loss_type: str,
    beta: float,
    ftx_gamma: float,
    label_smoothing: float,
    simpo_gamma: float,
    ignore_index: int,
    use_ref_model: bool,
    ref_all_logits: Any = None,
) -> Any:
    """The FULL DPO-trainer objective (mean over the batch), differentiable (jax).

    This is the loss the ``train_trajectory`` golden was frozen from -- the reference
    ``DPOTrainer.get_batch_loss_metrics`` / ``_loss_and_aux`` -- reconstructed from the
    documented forward contract alone (``all_logits`` = the concatenated chosen+rejected
    forward; ``labels`` the matching label block). It is DISTINCT from the simplified,
    preference-only :func:`dpo_loss` the numeric ``loss_artifact`` golden uses: the trainer
    loss carries the length-normalized NLL terms (the ``-chosen_logps`` in ORPO and the
    ``ftx_gamma * (-chosen_logps_avg)`` auxiliary), which is why for ORPO the trajectory
    step-0 loss is O(NLL) (~4.5) rather than the O(0.4) pure odds-ratio term.

    Mirrors the reference exactly: shift+gather log-probs (:func:`jax_batch_logps`),
    length-normalize for orpo/simpo, split chosen/rejected at the batch midpoint, dispatch on
    ``loss_type`` (reference-free ORPO/SimPO or reference-based sigmoid), then add the
    ``ftx_gamma`` SFT term. ``ref_all_logits`` supplies the frozen reference policy's
    concatenated forward for the sigmoid loss (``None`` for the reference-free losses)."""
    import jax
    import jax.numpy as jnp

    all_logps, valid_length = jax_batch_logps(all_logits, labels, ignore_index)
    normalized = loss_type in ("orpo", "simpo")
    all_logps_norm = all_logps / valid_length if normalized else all_logps
    batch_size = all_logps.shape[0] // 2
    chosen_logps = all_logps_norm[:batch_size]
    rejected_logps = all_logps_norm[batch_size:]
    chosen_length = valid_length[:batch_size]
    chosen_logps_avg = chosen_logps if normalized else (all_logps[:batch_size] / chosen_length)

    if use_ref_model:
        if ref_all_logits is None:
            raise ValueError("reference-based DPO loss requires ref_all_logits")
        ref_logps, ref_valid = jax_batch_logps(ref_all_logits, labels, ignore_index)
        ref_norm = ref_logps / ref_valid if normalized else ref_logps
        ref_chosen = ref_norm[:batch_size]
        ref_rejected = ref_norm[batch_size:]
        logits = (chosen_logps - ref_chosen) - (rejected_logps - ref_rejected)
        if label_smoothing > 0:
            losses = (
                -jax.nn.log_sigmoid(beta * logits) * (1.0 - label_smoothing)
                - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
            )
        else:
            losses = -jax.nn.log_sigmoid(beta * logits)
    elif loss_type == "orpo":
        log_odds = (chosen_logps - rejected_logps) - (
            jnp.log1p(-jnp.exp(chosen_logps)) - jnp.log1p(-jnp.exp(rejected_logps))
        )
        losses = -chosen_logps + beta * (-jax.nn.log_sigmoid(log_odds))
    elif loss_type == "simpo":
        logits = (chosen_logps - rejected_logps) - (simpo_gamma / beta)
        losses = -jax.nn.log_sigmoid(beta * logits)
    else:
        raise ValueError(f"Unsupported DPO loss type: {loss_type}")

    if ftx_gamma > 1e-6:
        losses = losses + ftx_gamma * (-chosen_logps_avg)
    return jnp.mean(losses)


# --------------------------------------------------------------------------- #
# Evaluator-owned LR schedule (from training_args; mirrors lr_schedule_artifact).
# --------------------------------------------------------------------------- #
def lr_schedule(training_args: Any, num_steps: int = 8) -> list[float]:
    warmup_steps = int(getattr(training_args, "warmup_steps", 0))
    if warmup_steps == 0:
        warmup_ratio_value = getattr(training_args, "warmup_ratio", 0.0)
        warmup_ratio = float(0.0 if warmup_ratio_value is None else warmup_ratio_value)
        warmup_steps = int(num_steps * warmup_ratio)
    learning_rate_value = getattr(training_args, "learning_rate", 1e-4)
    base_lr = float(1e-4 if learning_rate_value is None else learning_rate_value)
    scheduler_value = getattr(training_args, "lr_scheduler_type", "cosine")
    # ``SchedulerType`` is a (str, Enum): ``v == "linear"`` is True but ``str(v)`` is
    # "SchedulerType.LINEAR", which missed the branch below. Resolve by value.
    scheduler = "cosine" if scheduler_value is None else str(
        getattr(scheduler_value, "value", scheduler_value)
    )
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
    return values
