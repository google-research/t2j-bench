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

"""Shared evaluator utilities."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


def failure_kind_from_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "resource exhausted" in text or "oom" in text:
        return "oom"
    if "nan" in text or "inf" in text or "nonfinite" in text:
        return "nonfinite"
    return "error"


def contract_hint_from_exception(exc: BaseException) -> str | None:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()

    if "get_text_config" in text:
        return (
            "config classes used by the repo-local model implementation must expose "
            "`get_text_config(decoder=None, **kwargs)` and keep that signature compatible "
            "with the runtime path used by both reference validation and cross-framework comparison"
        )
    if "ppo_compat" in lowered:
        return (
            "repo-local transformer packages must include every internally referenced helper "
            "module and import them relatively so the same package works in both preflight "
            "and comparison runtimes"
        )
    if "filenotfounderror" in lowered or "cannot import name" in lowered or "modulenotfounderror" in lowered:
        return (
            "the torch contract requires a self-contained repo-local package: every module needed by "
            "tokenizer loading, model loading, collator, loss, and generation must be present and importable"
        )
    if "get_rope_index" in text:
        return (
            "if the data pipeline or model forward relies on custom position-id helpers, the same helper "
            "must be exposed on the repo-local model path used by evaluation"
        )
    if "unexpected keyword argument" in lowered or "multiple values for keyword argument" in lowered:
        return (
            "forward/apply methods must accept the batch keys and control kwargs used by the training path, "
            "or accept a compatible `**kwargs` passthrough"
        )
    if "none type" in lowered or "nonetype" in lowered:
        return (
            "the torch contract requires tokenizer, processor, template, collator, and model-loading state "
            "to be initialized consistently before dataset batching, forward, and loss evaluation"
        )
    if "no checkpoint file found" in lowered or "config file not found" in lowered:
        return (
            "model_name_or_path must resolve to a local directory or snapshot containing the config and "
            "checkpoint files required by the repo-local loader"
        )
    return None


def with_contract_hint(summary: str, exc: BaseException) -> str:
    hint = contract_hint_from_exception(exc)
    if not hint:
        return summary
    return f"{summary} | contract_hint: {hint}"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot convert None to numpy.")
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            return tensor.cpu().numpy()
    except ImportError:
        pass
    try:
        import jax.numpy as jnp
        import jax

        if isinstance(value, (jax.Array, jnp.ndarray)):
            arr = np.asarray(value)
            if str(arr.dtype) == "bfloat16":
                return arr.astype(np.float32)
            return arr
    except ImportError:
        pass
    arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32)
    return arr


def flatten_dict(tree: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, path))
        else:
            flat[path] = to_numpy(value)
    return flat


def summarize_tensor(value: Any) -> dict[str, Any]:
    arr = to_numpy(value)
    flat = arr.reshape(-1) if arr.size else arr
    sample = flat[: min(8, flat.size)].tolist() if flat.size else []
    digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    finite = bool(np.all(np.isfinite(arr))) if np.issubdtype(arr.dtype, np.number) else True
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": finite,
        "sample": sample,
        "sha256_16": digest,
    }


def tree_summary(tree: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(tree)
    return {
        key: summarize_tensor(value)
        for key, value in sorted(flat.items())
    }


def is_nonfinite(value: Any) -> bool:
    arr = to_numpy(value)
    return bool(np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return to_numpy(value).tolist()
    except ImportError:
        pass
    try:
        import jax.numpy as jnp
        import jax
        if isinstance(value, (jax.Array, jnp.ndarray)):
            return to_numpy(value).tolist()
    except ImportError:
        pass
    return value
