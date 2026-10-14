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

"""Vendored array/JSON utilities (subset of the authoring engine's utils.py).

Torch-free at grade time: the ``import torch`` branch in :func:`to_numpy` /
:func:`jsonable` is a no-op guarded by ``try/except ImportError`` and never fires
in the hermetic grading environment. Vendored from the authoring engine's
``utils.py`` and frozen at generation time.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


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
        import jax
        import jax.numpy as jnp

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


def is_nonfinite(value: Any) -> bool:
    arr = to_numpy(value)
    return bool(np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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
        import jax
        import jax.numpy as jnp

        if isinstance(value, (jax.Array, jnp.ndarray)):
            return to_numpy(value).tolist()
    except ImportError:
        pass
    return value
