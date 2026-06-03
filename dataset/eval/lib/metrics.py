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

"""Numeric comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import to_numpy


@dataclass(slots=True)
class NumericThresholds:
    abs_tol: float
    rel_tol: float
    cosine_floor: float
    max_kl: float


FP16_THRESHOLDS = NumericThresholds(
    abs_tol=2e-2,
    rel_tol=2e-2,
    cosine_floor=0.995,
    max_kl=2e-2,
)

BF16_THRESHOLDS = NumericThresholds(
    abs_tol=4e-2,
    rel_tol=4e-2,
    cosine_floor=0.99,
    max_kl=4e-2,
)


def thresholds_for_profile(profile: str) -> NumericThresholds:
    if profile == "fp16_compare":
        return FP16_THRESHOLDS
    if profile == "bf16_compare":
        return BF16_THRESHOLDS
    raise ValueError(f"Unsupported precision profile: {profile}")


def compare_arrays(reference: Any, candidate: Any) -> dict[str, Any]:
    ref = to_numpy(reference).astype(np.float32, copy=False)
    cand = to_numpy(candidate).astype(np.float32, copy=False)
    if ref.shape != cand.shape:
        return {
            "same_shape": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
            "finite": bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(cand))),
        }
    finite = bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(cand)))
    diff = np.abs(ref - cand)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel = diff / denom
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    denom_cos = np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat)
    cosine = float(np.dot(ref_flat, cand_flat) / denom_cos) if denom_cos else 1.0
    return {
        "same_shape": True,
        "finite": finite,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "max_rel_diff": float(rel.max()) if rel.size else 0.0,
        "cosine_similarity": cosine,
    }


def compare_logits(reference: Any, candidate: Any) -> dict[str, Any]:
    metrics = compare_arrays(reference, candidate)
    if not metrics["same_shape"] or not metrics["finite"]:
        metrics["token_kl"] = float("inf")
        return metrics
    ref = to_numpy(reference).astype(np.float32, copy=False)
    cand = to_numpy(candidate).astype(np.float32, copy=False)
    ref_log_probs = ref - _logsumexp(ref, axis=-1, keepdims=True)
    cand_log_probs = cand - _logsumexp(cand, axis=-1, keepdims=True)
    ref_probs = np.exp(ref_log_probs)
    kl = np.sum(ref_probs * (ref_log_probs - cand_log_probs), axis=-1)
    metrics["token_kl"] = float(np.max(np.abs(kl))) if kl.size else 0.0
    return metrics


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))
