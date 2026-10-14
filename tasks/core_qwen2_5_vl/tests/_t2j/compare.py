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

"""Numeric comparison utilities used at grade time.

Reimplements the comparison math so grade time needs no engine.
Both inputs are cast to float32; a check passes only if all relevant metrics are
within the materialized threshold (see ``golden/meta.json``). Frozen at
generation time.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._arrays import to_numpy


@dataclass(frozen=True)
class NumericThresholds:
    abs_tol: float
    rel_tol: float
    cosine_floor: float
    max_kl: float
    # Strict fp32 gate only (check_logits_strict): a golden top1-top2 logit gap
    # ABOVE this marks a "decisive" position where argmax parity is required; near-ties
    # (<= margin) are excluded so a genuine fp32 tie-flip never fails a correct port.
    # Default 0.0 => every position is decisive. Ignored by the cosine/KL gates.
    argmax_margin: float = 0.0


FP16_THRESHOLDS = NumericThresholds(abs_tol=2e-2, rel_tol=2e-2, cosine_floor=0.995, max_kl=2e-2)
BF16_THRESHOLDS = NumericThresholds(abs_tol=4e-2, rel_tol=4e-2, cosine_floor=0.99, max_kl=4e-2)
# Native reimplementations execute independent kernels, so they legitimately
# diverge more than shared-kernel parity.
FP16_NATIVE_THRESHOLDS = NumericThresholds(abs_tol=5e-2, rel_tol=5e-2, cosine_floor=0.99, max_kl=5e-2)
BF16_NATIVE_THRESHOLDS = NumericThresholds(abs_tol=1e-1, rel_tol=1e-1, cosine_floor=0.98, max_kl=1e-1)
# FP32 exactness gate: both sides run identical fp32 weights, so
# agreement is limited only by independent-kernel fp32 accumulation noise. Much
# tighter than the native profile — catches reduction-order / dtype / scaling bugs
# that bf16-native tolerance (abs/rel 0.1) masks. The logit (forward) gate is
# effectively COSINE-ONLY: cosine_floor (0.999999) exceeds _KL_COSINE_OVERRIDE
# (0.9995), so max_kl never binds — it is the scalar-loss allclose gate that is the
# real exactness surface (calibrate per instance via meta["fp32_thresholds"] if a
# model's fp32 loss margin is legitimately looser, e.g. extreme-logit bloom). NOT
# registered in thresholds_for_profile (it is not a precision_profile). Kept
# byte-identical to the reference implementation's FP32_EXACT_THRESHOLDS.
FP32_EXACT_THRESHOLDS = NumericThresholds(abs_tol=1e-3, rel_tol=1e-3, cosine_floor=0.999999, max_kl=1e-4)


def thresholds_for_profile(profile: str) -> NumericThresholds:
    if profile == "fp16_compare":
        return FP16_THRESHOLDS
    if profile == "bf16_compare":
        return BF16_THRESHOLDS
    if profile == "fp16_native":
        return FP16_NATIVE_THRESHOLDS
    if profile == "bf16_native":
        return BF16_NATIVE_THRESHOLDS
    raise ValueError(f"Unsupported precision profile: {profile}")


def finite_difference_metric(
    analytic_directional: float,
    loss_base: float,
    loss_perturbed: float,
    epsilon: float,
    *,
    rel_tol: float = 2e-1,
    abs_tol: float = 1e-3,
) -> dict[str, Any]:
    """Compare an analytic directional derivative to a finite-difference estimate."""
    if epsilon == 0:
        return {"supported": False, "reason": "epsilon must be non-zero"}
    fd_estimate = (float(loss_perturbed) - float(loss_base)) / float(epsilon)
    analytic = float(analytic_directional)
    abs_diff = abs(fd_estimate - analytic)
    denom = max(abs(analytic), abs(fd_estimate), 1e-6)
    rel_diff = abs_diff / denom
    passed = abs_diff <= abs_tol or rel_diff <= rel_tol
    return {
        "supported": True,
        "analytic_directional": analytic,
        "finite_difference": fd_estimate,
        "epsilon": float(epsilon),
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "passed": bool(passed),
    }


def cosine_similarity(reference: Any, candidate: Any) -> float:
    ref = to_numpy(reference).astype(np.float32, copy=False).reshape(-1)
    cand = to_numpy(candidate).astype(np.float32, copy=False).reshape(-1)
    if ref.shape != cand.shape:
        return 0.0
    denom = np.linalg.norm(ref) * np.linalg.norm(cand)
    if not denom:
        return 1.0
    return float(np.dot(ref, cand) / denom)


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


def check_array(reference: Any, candidate: Any, thresholds: NumericThresholds) -> tuple[bool, dict[str, Any]]:
    """allclose-style elementwise parity: |ref-cand| <= abs_tol + rel_tol*|ref|.

    This is numpy.allclose semantics (the abs and rel terms combine, rather than
    being separate global maxima that must each hold). It is well-scaled for both
    small artifacts (abs term dominates) and large-magnitude ones such as
    grad_norm (rel term dominates) — the separate-maxima form is
    unsatisfiable for large values, so this is the calibrated grade-time gate.
    """
    metrics = compare_arrays(reference, candidate)
    if not metrics.get("same_shape", True) or not metrics.get("finite", True):
        return False, metrics
    ref = to_numpy(reference).astype(np.float32, copy=False)
    cand = to_numpy(candidate).astype(np.float32, copy=False)
    ok = bool(np.all(np.abs(ref - cand) <= thresholds.abs_tol + thresholds.rel_tol * np.abs(ref)))
    metrics["allclose"] = ok
    return ok, metrics


def check_lr_schedule(reference: Any, candidate: Any) -> tuple[bool, dict[str, Any]]:
    """Learning-rate schedule parity with a SCALE-RELATIVE band.

    LR values are tiny (~1e-6), so the precision profile's abs_tol (0.1 for
    bf16_native) is vacuous — ``check_array`` would pass any schedule, even all-zeros or
    a constant. This grades against the schedule's own peak instead:
    ``|ref-cand| <= 1e-9*peak + 1e-4*|ref|``. The peak term admits fp32 sampling noise on
    the largest LR (the golden is fp64, a candidate optax schedule is typically fp32);
    the relative term keeps every non-peak step (warmup ramp, cosine/linear tail)
    honest. Both sides are config-derived optax schedules, so a correct port matches to
    sampling noise while a wrong warmup/decay/peak — or an all-zeros/constant schedule —
    fails. Not the profile ``NumericThresholds``: this band is schedule-specific.
    """
    ref = to_numpy(reference).astype(np.float64, copy=False).reshape(-1)
    cand = to_numpy(candidate).astype(np.float64, copy=False).reshape(-1)
    if ref.shape != cand.shape:
        return False, {
            "same_shape": False, "reference_len": int(ref.size), "candidate_len": int(cand.size),
        }
    finite = bool(np.all(np.isfinite(cand)))
    peak = float(np.max(np.abs(ref))) if ref.size else 0.0
    abs_band = 1e-9 * peak
    rel_band = 1e-4
    diff = np.abs(ref - cand)
    tol = abs_band + rel_band * np.abs(ref)
    ok = bool(finite and ref.size > 0 and np.all(diff <= tol))
    return ok, {
        "same_shape": True,
        "finite": finite,
        "peak": peak,
        "abs_band": abs_band,
        "rel_band": rel_band,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "reference": [float(x) for x in ref[:12]],
        "candidate": [float(x) for x in cand[:12]],
    }


# A token_kl overshoot on hyper-peaked, extreme-scale logits (e.g. bloom, whose
# ln_f weight ~18 pushes logit std to ~10^2) is a softmax scale artifact, not a
# semantic divergence: at this cosine the logit vectors are near-identical (same
# argmax; the perpendicular part is fp16 peak jitter), so the distributions agree
# up to peak sharpness. Real breaks (disabled rotary, …) drop cosine far below the
# floor, so this override never masks them. Must match metrics.py (parity test).
_KL_COSINE_OVERRIDE = 0.9995


def check_logits(reference: Any, candidate: Any, thresholds: NumericThresholds) -> tuple[bool, dict[str, Any]]:
    """Distribution-level logit parity: cosine_floor + token_kl.

    Raw pre-softmax logits are large-magnitude and only defined up to a per-token
    additive constant, so elementwise abs/rel diff is not a meaningful equivalence
    measure (a tiny reference logit makes rel_diff explode). Cosine similarity and
    max per-token softmax KL are the calibrated logit metrics.
    """
    metrics = compare_logits(reference, candidate)
    if not metrics.get("same_shape", True) or not metrics.get("finite", True):
        return False, metrics
    cosine = metrics["cosine_similarity"]
    token_kl = metrics.get("token_kl", 0.0)
    metrics["kl_overridden_by_cosine"] = bool(token_kl > thresholds.max_kl and cosine >= _KL_COSINE_OVERRIDE)
    ok = cosine >= thresholds.cosine_floor and (token_kl <= thresholds.max_kl or cosine >= _KL_COSINE_OVERRIDE)
    return bool(ok), metrics


def check_logits_strict(
    reference: Any, candidate: Any, thresholds: NumericThresholds
) -> tuple[bool, dict[str, Any]]:
    """Strict fp32 logit gate: cosine + BINDING KL + argmax agreement.

    ``check_logits`` is effectively COSINE-ONLY on the fp32 profile -- its cosine_floor
    (0.999999) exceeds ``_KL_COSINE_OVERRIDE`` (0.9995), so the KL clause is bypassed on
    exactly the surface where both sides run identical fp32 weights and agreement should
    be tightest. This variant makes all three clauses load-bearing:

      (1) cosine >= cosine_floor  -- direction; catches gross breaks.
      (2) token_kl <= max_kl  (BINDING; the cosine override is NOT applied) -- a
          temperature / logit-scale / soft-cap bug is argmax- AND cosine-invariant, so
          only the per-token softmax KL sees the confidence shift it introduces.
      (3) argmax parity on every DECISIVE golden position (golden top1-top2 gap >
          argmax_margin) -- excluding near-ties drops genuine fp32 tie-flips while still
          forcing the candidate to pick the reference's token wherever it is decisive.

    ``max_kl`` and ``argmax_margin`` are calibrated per instance (the calibration
    tooling writes ``fp32_thresholds`` into golden/meta.json): ``max_kl`` admits the
    measured cross-kernel fp32 noise and ``argmax_margin`` excludes every observed
    tie-flip. With the defaults
    (profile max_kl, argmax_margin=0.0) every position is decisive and exact argmax
    parity is demanded.
    """
    metrics = compare_logits(reference, candidate)
    if not metrics.get("same_shape", True) or not metrics.get("finite", True):
        metrics["strict_ok"] = False
        return False, metrics
    ref = to_numpy(reference).astype(np.float32, copy=False)
    cand = to_numpy(candidate).astype(np.float32, copy=False)
    ref2 = ref.reshape(-1, ref.shape[-1])
    cand2 = cand.reshape(-1, cand.shape[-1])

    cosine = metrics["cosine_similarity"]
    token_kl = metrics.get("token_kl", 0.0)
    cosine_ok = cosine >= thresholds.cosine_floor
    kl_ok = token_kl <= thresholds.max_kl  # BINDING: no cosine override here.

    if ref2.shape[-1] >= 2:
        part = np.partition(ref2, -2, axis=-1)
        gap = part[:, -1] - part[:, -2]                 # golden decisiveness per position
    else:
        gap = np.full((ref2.shape[0],), np.inf, dtype=np.float32)
    decisive = gap > thresholds.argmax_margin
    disagree = ref2.argmax(-1) != cand2.argmax(-1)
    n_bad = int(np.count_nonzero(decisive & disagree))
    argmax_ok = n_bad == 0

    ok = bool(cosine_ok and kl_ok and argmax_ok)
    metrics.update({
        "cosine_ok": bool(cosine_ok),
        "kl_ok": bool(kl_ok),
        "argmax_ok": bool(argmax_ok),
        "threshold_max_kl": float(thresholds.max_kl),
        "argmax_margin": float(thresholds.argmax_margin),
        "n_decisive": int(decisive.sum()),
        "n_argmax_disagree_decisive": n_bad,
        "strict_ok": ok,
    })
    return ok, metrics


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))


# --- Per-parameter gradient comparison -------------------- #
# Map torch state-dict names and JAX pytree paths to a common canonical key so
# per-parameter gradients can be paired across frameworks. Values are compared via
# grad_stats (norm/mean/std), which are transposition/permutation-invariant, so the
# torch weight [out,in] vs flax kernel [in,out] layout difference does not matter.
_PARAM_PREFIX_RE = re.compile(r"^(transformer|gpt_neox|model)\.")
# Torch-side WRAPPER paths with no flax counterpart, stripped before anything else:
# TRL's value-head wrapper contributes a `pretrained_model.` prefix to every parameter
# of a PPO instance, and PEFT's injection contributes `base_model.model.` plus a
# `.base_layer.` infix on every adapted Linear of a LoRA instance. Both are facts about
# how the *torch reference* was assembled, not about the model, and a candidate's flax
# tree cannot and should not reproduce them -- left in, NOT ONE parameter of those
# instances maps to its counterpart.
_WRAPPER_PREFIX_RE = re.compile(r"^(?:pretrained_model\.|base_model\.model\.)+")
_WRAPPER_INFIX = ".base_layer."
# Params EXCLUDED from the strict per-param gate — never required to map or agree. Two
# reasons: (1) vision/multimodal subtrees whose torch<->flax naming legitimately
# diverges (or that a VL text-only artifact omits); (2) TIED embeddings — torch folds a
# tied lm_head's grad into embed_tokens (grad = embedding-lookup + lm_head-matmul paths
# summed), so an untied candidate's embed / lm_head grad norms legitimately differ.
# Both are covered elsewhere (forward parity, global grad_norm, gradient authenticity).
# The gate focuses on the transformer-layer params (attn/mlp/norms), where per-param
# backprop bugs are most likely and cross-framework grads compare cleanly.
_EXCLUDED_MARKERS = (
    "vision", "visual", "image", "patch", "audio",
    # Qwen-Omni: "thinker" is the TEXT LM (its model.layers.* linear weights ARE the
    # gated backprop surface and DO appear in the golden), so it must NOT be excluded.
    # "talker" (audio-token LM) and "token2wav" (BigVGAN vocoder) get no gradient from
    # the text loss the golden freezes -> no golden counterpart -> exclude them.
    "talker", "token2wav",
    "multi_modal", "mm_", "v_head", "lm_head", "embed",
    # LayerNorm/RMSNorm SCALE gradients: small-magnitude and numerically sensitive in
    # the norm backward, they are the noisiest per-param grads cross-framework
    # and are covered by the
    # aggregate checks (forward parity, global grad_norm, trajectory, authenticity). The
    # gate therefore focuses on the LINEAR projection weights (attn q/k/v/o, mlp
    # gate/up/down, query_key_value, dense, fc1/fc2) — the well-conditioned backprop
    # surface where per-param bugs matter and cross-framework grads compare cleanly.
    "norm",
)

# Per-parameter grad NORMS are inherently noisier cross-framework than aggregate
# metrics (each is a reduction over ONE tensor, not averaged over all params), so the
# profile's logit/loss rel_tol (10% native) is too tight: fp16 accumulation noise makes
# large attn/mlp weight grads diverge ~11-14% (uniform noise, not a bug).
# Use a dedicated, looser grad rel-tol; a real per-param backprop bug shifts a norm far
# more (a x2 grad = 100%), so this still catches bugs while tolerating fp16 noise. The
# abs floor (profile abs_tol) still covers tiny-norm (layernorm) grads.
_GRAD_REL_TOL = 0.3


def canonical_param_key(name: str) -> str:
    """Normalize a torch state-dict name or JAX pytree path to a common key."""
    s = str(name).replace("/", ".")
    s = _WRAPPER_PREFIX_RE.sub("", s)            # pretrained_model. / base_model.model. -> ''
    s = s.replace(_WRAPPER_INFIX, ".")           # q_proj.base_layer.weight -> q_proj.weight
    s = re.sub(r"_(\d+)(?=[.]|$)", r".\1", s)   # layers_0 / h_0 -> layers.0 (NOT dense_4h_to_h)
    s = _PARAM_PREFIX_RE.sub("model.", s)        # transformer./gpt_neox. -> model.
    parts = s.split(".")
    if parts and parts[-1] in ("weight", "kernel", "embedding", "scale"):
        parts[-1] = "W"                          # unify Linear/Embed/Norm leaf tokens
    return ".".join(parts)


def _is_gated(canon_key: str) -> bool:
    """True for a core transformer-layer param covered by the strict gate."""
    return not any(mk in canon_key for mk in _EXCLUDED_MARKERS)


def compare_grad_summaries(
    ref_grads: dict, cand_grads: dict, thresholds: NumericThresholds, grad_rel_tol: float | None = None
) -> tuple[bool, dict[str, Any]]:
    """Per-parameter gradient parity by canonical name.

    Over the core transformer-layer params (attn/mlp/norms), passes iff (a) every such
    candidate parameter maps to a reference parameter — so a broken canonicalization
    fails loudly rather than skipping — and (b) each mapped parameter's gradient L2 norm
    agrees within the profile tolerance (allclose semantics; the norm is
    layout-invariant). Tied-embedding and vision/multimodal params are excluded from the
    gate (see _EXCLUDED_MARKERS) but counted for visibility.
    """
    ref_by: dict[str, Any] = {}
    for k, v in ref_grads.items():
        ref_by.setdefault(canonical_param_key(k), v)   # tied weights: keep first
    cand_by: dict[str, Any] = {}
    for k, v in cand_grads.items():
        cand_by.setdefault(canonical_param_key(k), v)

    gated = [ck for ck in cand_by if _is_gated(ck)]
    unmapped = sorted(ck for ck in gated if ck not in ref_by)
    matched = [ck for ck in gated if ck in ref_by]
    failures: dict[str, Any] = {}
    for ck in matched:
        r_norm = float(ref_by[ck].get("norm", 0.0))
        c_norm = float(cand_by[ck].get("norm", 0.0))
        finite = bool(cand_by[ck].get("finite", True))
        rel_tol = max(thresholds.rel_tol, _GRAD_REL_TOL if grad_rel_tol is None else grad_rel_tol)
        ok = finite and abs(r_norm - c_norm) <= thresholds.abs_tol + rel_tol * abs(r_norm)
        if not ok:
            failures[ck] = {"golden_norm": r_norm, "candidate_norm": c_norm, "finite": finite}

    passed = (not unmapped) and (not failures) and (len(matched) > 0)
    details = {
        "candidate_params": len(cand_by),
        "gated": len(gated),
        "matched": len(matched),
        "excluded": len(cand_by) - len(gated),
        "num_failures": len(failures),
        "unmapped": unmapped[:10],
        "failures": dict(list(failures.items())[:10]),
    }
    return passed, details


# --- Per-parameter CONVERSION checks -------------------------- #
# Since the candidate converts the HF checkpoint itself, the grader no longer demands a
# verbatim pytree: it checks CORRESPONDENCE (does every parameter you built line up with
# a checkpoint tensor, at a compatible shape?) and per-parameter L2 norms (do they hold
# the right numbers?). Both are keyed by canonical_param_key, so the candidate's module
# naming is its own choice.
#
# The parameter gate below is deliberately WIDER than the gradient gate above. A
# parameter is READ from the checkpoint, not computed through a backward pass, so the
# numerical-noise argument that keeps `norm` and `embed` out of _EXCLUDED_MARKERS does
# not apply — those are exactly the tensors a conversion bug corrupts, so they stay in.
# What is excluded here, and why:
#   * vision/multimodal subtrees — their torch<->flax module naming legitimately
#     diverges (the paligemma / gemma3 / llava_next vision towers do not
#     canonicalize onto their torch names). Covered by forward parity, which runs on
#     real images.
#   * `v_head` — no checkpoint contains it; PPO synthesizes it zero-initialized.
#   * `lm_head` — the candidate may legitimately TIE it to the embedding and never
#     materialize a leaf. Its values are checked exactly by the logit parity gate.
#   * `lora` — adapters are randomly initialized, not converted.
#
# THE VISION GROUP IS NOW CONDITIONAL — see `_param_excluded`. It is split out
# rather than deleted because the argument for it still holds on a text-only
# instance, and the two groups are re-joined below so the old membership test is
# still expressible and still checkable.
_VISION_MARKERS = (
    "vision", "visual", "image", "patch", "audio", "talker", "token2wav",
    "multi_modal", "mm_",
)
#: Excluded on EVERY instance, for reasons that have nothing to do with vision.
_ALWAYS_EXCLUDED = ("v_head", "lm_head", "lora")
#: What the single unconditional tuple used to be, preserved exactly: the union,
#: in the original order. Nothing may enter or leave the exclusion silently as a
#: side effect of the split — the harness
#: asserts this is a partition of the two groups above.
_PARAM_EXCLUDED_MARKERS = _VISION_MARKERS + _ALWAYS_EXCLUDED

# Parameters are COPIED out of the checkpoint, so both sides hold the same numbers and
# the only legitimate gap between two L2 norms is float32 accumulation order plus, at
# worst, one extra rounding if the candidate casts. That is ~1e-6 relative. 1e-3 leaves
# three orders of margin and still catches every real conversion bug: an unloaded,
# duplicated, truncated, or wrongly-bitcast tensor moves its norm by percent to orders
# (a missed transpose keeps the norm, but fails the shape check instead). The precision
# profile's rel_tol — up to 0.1 for bf16_native — is far too loose to be useful for
# exactly-copied values and is deliberately NOT used here.
_PARAM_NORM_ABS_TOL = 1e-4
_PARAM_NORM_REL_TOL = 1e-3


def _param_excluded(name: str, *, multimodal: bool) -> bool:
    """Is this parameter outside the strict correspondence gate?

    THE VISION GROUP IS CONDITIONAL, and that is a correction rather than a
    feature. The exclusion was inherited from the training class, where its
    justification — for the paligemma / gemma3 / llava_next towers — was that those
    upstream HF towers do not canonicalize onto their torch names, and that
    forward parity covers them instead.

    NEITHER HALF OF THAT TRANSFERS TO THIS CLASS ON A VL FAMILY. Both sides'
    `modeling.py` (and `vision.py`) are authored here, so canonicalization works
    by construction — the same argument `test_reference_units` already rests on
    for every other family. On such a family the two towers'
    parameter name sets are IDENTICAL, with no rename table on either
    side. And on that family the tower is not an incidental subtree, it is the
    instance's entire subject: excluding it leaves the one thing the instance
    exists to grade with no VALUE gate at all.

    WHAT WAS ACTUALLY UNGATED, stated precisely because the obvious reading is
    wrong: `test_param_correspondence` asserts `set(got_by) == set(want_by)`
    over the FULL, un-excluded key set before it ever calls
    `compare_param_spec`, so a tower parameter that is MISSING, EXTRA or the
    wrong SHAPE already fails today. What no check could see was a tower tensor
    holding the wrong NUMBERS — `compare_param_norms` is the only check that
    reads a value, and it filtered the tower out. A perturbed tower tensor was
    therefore caught by nothing.

    Text-only instances keep the exclusion, unchanged: a stray VL tensor in a
    text checkpoint still has no counterpart to map onto, and no instance other
    than a declared-multimodal one ever passes `multimodal=True`.
    """
    lowered = name.lower()
    if any(marker in lowered for marker in _ALWAYS_EXCLUDED):
        return True
    if multimodal:
        return False
    return any(marker in lowered for marker in _VISION_MARKERS)


def _is_param_gated(canon_key: str, *, multimodal: bool = False) -> bool:
    """True for a parameter covered by the strict conversion gate.

    THE INVERTED, DEFAULT-TEXT-ONLY WRAPPER, kept because three call sites read
    it that way and one of them is not this class's: the TRAINING task class's
    param-norms tooling calls `compare._is_param_gated(key)` positionally
    against `tasks/*/tests/_t2j/compare.py`. That copy is untouched by this
    change, but the two copies are held to statement-level agreement outside a
    named exemption, so the inference copy must keep answering the same
    one-argument question the same way or the next sync breaks that tool
    silently.
    """
    return not _param_excluded(canon_key, multimodal=multimodal)


def _canon_first(entries: dict) -> dict[str, tuple[str, Any]]:
    """{canonical key: (original key, value)}, keeping the first of any collision."""
    out: dict[str, tuple[str, Any]] = {}
    for key, value in entries.items():
        out.setdefault(canonical_param_key(key), (key, value))
    return out


def _shape_of(entry: Any) -> tuple[int, ...]:
    if isinstance(entry, dict):
        entry = entry.get("shape", ())
    return tuple(int(d) for d in entry)


def compare_param_spec(ref_spec: dict, cand_spec: dict, *,
                       multimodal: bool = False) -> tuple[bool, dict[str, Any]]:
    """1:1 correspondence between the candidate pytree and the torch checkpoint.

    ``ref_spec`` is ``golden/params_spec.json`` — the frozen torch reference's
    ``{state_dict name: {shape, dtype}}``, i.e. the tensors the HF checkpoint actually
    holds. ``cand_spec`` is the candidate's own ``{pytree path: {shape, dtype}}``.

    Passes iff every GATED candidate parameter maps, by canonical key, onto exactly one
    reference parameter of a compatible shape. Compatible means equal, or — for rank-2 —
    the transpose, because a flax ``nn.Dense`` kernel is ``[in, out]`` where a torch
    ``nn.Linear`` weight is ``[out, in]``. dtype is not gated: the compute dtype
    is the candidate's choice and the values are gated by compare_param_norms.

    Only the candidate side is required to map. A candidate that ties or fuses fewer
    leaves is not penalised here; every leaf it *does* build is norm-checked, which is
    what makes an unloaded or garbage tensor impossible to hide.

    ``multimodal`` widens the gate onto the vision tower — see :func:`_param_excluded`.
    It is the REPORTING half here rather than a new gate:
    ``test_param_correspondence`` already asserts set equality over the full,
    un-excluded key set before it calls this. Threading it anyway keeps the
    diagnostics honest, because a candidate reading ``excluded: 326`` beside a tower
    failure is being told the tower is not graded, which on a VL instance is false.
    """
    ref_by = _canon_first(ref_spec)
    cand_by = _canon_first(cand_spec)

    gated = [ck for ck in cand_by if _is_param_gated(ck, multimodal=multimodal)]
    unmapped = sorted(ck for ck in gated if ck not in ref_by)
    shape_mismatch: dict[str, Any] = {}
    for ck in gated:
        if ck not in ref_by:
            continue
        ref_shape = _shape_of(ref_by[ck][1])
        cand_shape = _shape_of(cand_by[ck][1])
        ok = ref_shape == cand_shape or (len(ref_shape) == 2 and ref_shape == cand_shape[::-1])
        if not ok:
            shape_mismatch[ck] = {
                "checkpoint": {"key": ref_by[ck][0], "shape": list(ref_shape)},
                "candidate": {"key": cand_by[ck][0], "shape": list(cand_shape)},
            }

    passed = (not unmapped) and (not shape_mismatch) and bool(gated)
    details = {
        "candidate_params": len(cand_by),
        "reference_params": len(ref_by),
        "gated": len(gated),
        "matched": len(gated) - len(unmapped),
        "excluded": len(cand_by) - len(gated),
        "num_unmapped": len(unmapped),
        "num_shape_mismatch": len(shape_mismatch),
        # An unmapped key is nearly always a naming slip; show the candidate's own path
        # next to the canonical form it produced so the fix is obvious.
        "unmapped": [{"canonical": ck, "candidate_key": cand_by[ck][0]} for ck in unmapped[:10]],
        "shape_mismatch": dict(list(shape_mismatch.items())[:10]),
    }
    return passed, details


def compare_param_norms(
    ref_norms: dict, cand_norms: dict, rel_tol: float | None = None,
    abs_tol: float | None = None, *, multimodal: bool = False
) -> tuple[bool, dict[str, Any]]:
    """Per-parameter L2 norm parity against the frozen reference weights.

    ``ref_norms`` is ``golden/param_norms.json`` — ``{canonical key: {"norm": float,
    "shape": [...]}}`` computed from the reference's own converted weights. The L2 norm
    is layout-invariant, so this is indifferent to naming, to the [in,out]/[out,in]
    convention, and to how the tree is nested — while still catching a wrongly-ordered
    fused QKV, an unloaded or half-read shard, a duplicated tie, a misread bf16 bit
    pattern, or a tensor left at its random init.

    ``multimodal`` widens the gate onto the vision tower, and THIS is the call site
    that matters: it is the only check in the class that reads a parameter's VALUE, so
    before the flag existed a perturbed tower tensor changed no name and no shape and
    was caught by nothing. See :func:`_param_excluded`.

    ORDERING, for whoever freezes the golden: ``golden/param_norms.json`` itself is
    NOT filtered by this predicate — the param-norms freezer writes a
    norm for every converted leaf and uses the gate only for its coverage
    cross-check — so a golden frozen before this narrowing still contains the tower's
    numbers and does not have to be re-frozen for the widened gate to have something
    to compare against.
    """
    rel = _PARAM_NORM_REL_TOL if rel_tol is None else float(rel_tol)
    abs_ = _PARAM_NORM_ABS_TOL if abs_tol is None else float(abs_tol)
    ref_by = _canon_first(ref_norms)
    cand_by = _canon_first(cand_norms)

    gated = [ck for ck in cand_by if _is_param_gated(ck, multimodal=multimodal)]
    unmapped = sorted(ck for ck in gated if ck not in ref_by)
    failures: dict[str, Any] = {}
    for ck in gated:
        if ck not in ref_by:
            continue
        ref_entry = ref_by[ck][1]
        cand_entry = cand_by[ck][1]
        r_norm = float(ref_entry.get("norm", 0.0))
        c_norm = float(cand_entry.get("norm", 0.0))
        finite = bool(cand_entry.get("finite", True))
        if not (finite and abs(r_norm - c_norm) <= abs_ + rel * abs(r_norm)):
            failures[ck] = {
                "candidate_key": cand_by[ck][0],
                "golden_norm": r_norm,
                "candidate_norm": c_norm,
                "finite": finite,
            }

    passed = (not unmapped) and (not failures) and bool(gated)
    details = {
        "candidate_params": len(cand_by),
        "gated": len(gated),
        "matched": len(gated) - len(unmapped),
        "excluded": len(cand_by) - len(gated),
        "rel_tol": rel,
        "abs_tol": abs_,
        "num_unmapped": len(unmapped),
        "num_failures": len(failures),
        "unmapped": [{"canonical": ck, "candidate_key": cand_by[ck][0]} for ck in unmapped[:10]],
        "failures": dict(list(failures.items())[:10]),
    }
    return passed, details


# --------------------------------------------------------------------------- #
# Interface parity (spec stage). The candidate's per-module PUBLIC API must be a
# SUPERSET of the frozen contract: torch_orig's ported public interface plus the
# documented JAX-only extras (e.g. ``model/loader.convert_checkpoint``). This is static
# AST analysis of the candidate tree -- no import, no torch, no candidate run -- so a
# dropped entry point fails here loudly instead of surfacing as an inscrutable import
# error downstream. The freezer that produces ``golden/interface.json``
# uses the SAME name rule below so the two agree.
# --------------------------------------------------------------------------- #
def public_api(source: str) -> set[str]:
    """Top-level public classes, functions and UPPER_CASE constants of a module source.

    Underscore-prefixed names are private and excluded. A syntactically broken file
    yields the empty set, so its contract names then read as missing (a FAIL) rather than
    raising at collection time.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                out.add(node.target.id)
    return {n for n in out if not n.startswith("_")}


def candidate_public_api(repo_dir: str) -> dict[str, set[str]]:
    """``{repo-relative .py path -> public API}`` for every module in the candidate tree."""
    out: dict[str, set[str]] = {}
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                src = open(full, encoding="utf-8").read()
            except OSError:
                continue
            out[os.path.relpath(full, repo_dir)] = public_api(src)
    return out


def compare_interface(spec: dict, cand_api: dict[str, set[str]]) -> dict[str, list[str]]:
    """Contract names the candidate does not expose, per module (empty dict == parity).

    ``spec`` is ``golden/interface.json``: ``required`` (torch_orig's ported public API
    per shared module) plus ``required_extras`` (documented JAX-only names). Policy is
    SUPERSET -- every required name must be present (a missing one is a gap); any extra
    names the candidate defines are allowed. Names only, not signatures.
    """
    want: dict[str, set[str]] = {}
    for group in ("required", "required_extras"):
        for rel, names in (spec.get(group) or {}).items():
            want.setdefault(rel, set()).update(names)
    missing: dict[str, list[str]] = {}
    for rel, names in sorted(want.items()):
        gap = sorted(names - cand_api.get(rel, set()))
        if gap:
            missing[rel] = gap
    return missing


# --------------------------------------------------------------------------- #
# Interface KIND / METHODS / CONSTANTS / STRUCTURE / SIGNATURE (spec stage).
# All are static AST / file-walk analysis of the candidate tree -- no import, no
# torch, no candidate run -- so a required entry point that is the wrong kind, a
# class that dropped a method, a retuned load-bearing constant, a missing module,
# or a callable that will not accept the arguments the harness passes it FAILS
# here, loudly, rather than surfacing as an inscrutable error in a downstream
# numeric check. The freezer writes these
# same sections into ``golden/interface.json`` with the SAME rules used below, so
# the frozen contract and these checks agree by construction (exactly as the name
# rule in ``public_api`` is mirrored on both sides).
# --------------------------------------------------------------------------- #
#: Load-bearing named constants whose VALUE (or, for a dict, whose key set) the port
#: must reproduce. Mirrored in the freezer. Dict-valued constants
#: (``SCHEDULER_MAP`` -> function objects, ``DATA_CONFIG`` -> nested structure) are not
#: literals, so only their keys are comparable; scalar constants are compared by value.
WANTED_CONSTANTS: frozenset[str] = frozenset({
    "IGNORE_INDEX",       # label-mask sentinel; must be exactly -100 (HF / LlamaFactory)
    "DEFAULT_TEMPLATE",   # chat-template fallback name
    "SCHEDULER_MAP",      # lr-scheduler registry: KEYS gate which schedules are wired
    "DATA_CONFIG",        # dataset registry structure: KEYS gate the dataset contract
})


def _iter_py_sources(repo_dir: str):
    """Yield ``(repo-relative .py path, source text)`` for the candidate tree.

    Skips ``__pycache__`` and any file that cannot be read as UTF-8 (a binary blob or
    permission error is simply not part of the interface). Shared by the kind / methods
    walks so they see exactly the same module set as ``candidate_public_api``.
    """
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                src = open(full, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            yield os.path.relpath(full, repo_dir), src


# --- kinds ----------------------------------------------------------------- #
def public_api_kinds(source: str) -> dict[str, str]:
    """``{public top-level name -> "class" | "function" | "constant"}`` for a module.

    Same visibility rule as ``public_api`` (drop ``_``-prefixed, UPPER-only constants); a
    syntactically broken file yields ``{}`` so its names read as absent (a parity FAIL)
    rather than raising.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = "function"
        elif isinstance(node, ast.ClassDef):
            out[node.name] = "class"
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out.setdefault(t.id, "constant")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                out.setdefault(node.target.id, "constant")
    return {n: k for n, k in out.items() if not n.startswith("_")}


def candidate_public_api_kinds(repo_dir: str) -> dict[str, dict[str, str]]:
    """``{rel.py -> {public name -> kind}}`` for every module in the candidate tree."""
    return {rel: public_api_kinds(src) for rel, src in _iter_py_sources(repo_dir)}


def compare_interface_kinds(spec: dict, cand_kinds: dict[str, dict[str, str]]) -> dict[str, dict[str, dict]]:
    """Required names the candidate defines as the WRONG kind (empty dict == all correct).

    ``spec['kinds']`` freezes ``{rel: {name: kind}}`` for the reference's contract names. A
    candidate that shadows a required class with a same-named function (or vice versa)
    satisfies name-only ``compare_interface`` but is caught here. A required name the
    candidate does not define at all is a ``compare_interface`` miss, reported there, so it
    is skipped here to avoid double-counting.
    """
    kinds = spec.get("kinds") or {}
    mismatch: dict[str, dict[str, dict]] = {}
    for rel, want in sorted(kinds.items()):
        have = cand_kinds.get(rel, {})
        bad = {
            name: {"expected": kind, "candidate": have.get(name)}
            for name, kind in sorted(want.items())
            if name in have and have[name] != kind
        }
        if bad:
            mismatch[rel] = bad
    return mismatch


# --- class methods --------------------------------------------------------- #
def public_class_methods(source: str) -> dict[str, set[str]]:
    """``{top-level class name -> its non-underscore method names}`` for a module.

    Only methods DEFINED in the class body are seen (AST, no import), never inherited ones
    -- the same rule on both sides, so a method the reference inherits (and the candidate
    also inherits) is neither required nor credited. A broken file yields ``{}``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                m.name for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")
            }
    return out


def candidate_class_methods(repo_dir: str) -> dict[str, dict[str, set[str]]]:
    """``{rel.py -> {class name -> method names}}`` across the candidate tree."""
    return {rel: public_class_methods(src) for rel, src in _iter_py_sources(repo_dir)}


def compare_interface_methods(spec: dict, cand_methods: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    """Frozen methods a candidate class is missing, per ``rel::Class`` (empty == superset).

    ``spec['methods']`` freezes ``{rel: {Class: [methods]}}`` -- for each class present in
    BOTH torch_orig and the reference, the methods they share (torch ∩ ref, the same
    superset policy as ``compare_interface``). The candidate's same-named class at the same
    module must expose at least those (extra methods are fine). A class the candidate does
    not define reports every frozen method as missing -- a class-level parity gap is a
    methods gap too.
    """
    methods = spec.get("methods") or {}
    missing: dict[str, dict[str, list[str]]] = {}
    for rel, classes in sorted(methods.items()):
        have_classes = cand_methods.get(rel, {})
        rel_missing: dict[str, list[str]] = {}
        for cls, names in sorted(classes.items()):
            gap = sorted(set(names) - have_classes.get(cls, set()))
            if gap:
                rel_missing[cls] = gap
        if rel_missing:
            missing[rel] = rel_missing
    return missing


# --- constants ------------------------------------------------------------- #
def _constant_repr(value_node: ast.AST) -> dict | None:
    """Freeze-able view of a constant's VALUE from its AST, or None if not literal.

    A dict literal is reduced to its sorted string keys (``{"kind":"dict","keys":[...]}``):
    ``SCHEDULER_MAP`` maps names to FUNCTION objects and ``DATA_CONFIG`` to nested
    structures, neither literal, so only the keys are comparable. Everything else is
    ``literal_eval``'d (``{"kind":"value","value": -100}``). A non-literal scalar (a call,
    a Name) or a ``{**splat}`` dict yields None -> the constant is simply not frozen.
    """
    if isinstance(value_node, ast.Dict):
        keys: list[str] = []
        for k in value_node.keys:
            if k is None:  # ``{**base}`` splat -> keys unknown statically
                return None
            try:
                keys.append(str(ast.literal_eval(k)))
            except (ValueError, SyntaxError, TypeError):
                return None
        return {"kind": "dict", "keys": sorted(keys)}
    try:
        return {"kind": "value", "value": ast.literal_eval(value_node)}
    except (ValueError, SyntaxError, TypeError):
        return None


def module_constants(source: str, names) -> dict[str, dict]:
    """``{NAME -> repr}`` for each wanted top-level constant a module defines literally.

    First top-level binding wins. A wanted name that is absent, or bound to a non-literal
    (a computed value), is omitted -- the freeze is data-driven, pinning only what the
    reference actually states as a literal.
    """
    wanted = set(names)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, dict] = {}
    for node in tree.body:
        target = value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value_node = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value_node = node.target.id, node.value
        if target in wanted and target not in out:
            rep = _constant_repr(value_node)
            if rep is not None:
                out[target] = rep
    return out


def candidate_module_constants(repo_dir: str, wanted_by_rel: dict) -> dict[str, dict[str, dict]]:
    """``{rel -> {NAME -> repr}}`` for the candidate, restricted to the frozen (rel, name) set."""
    out: dict[str, dict[str, dict]] = {}
    for rel, names in wanted_by_rel.items():
        try:
            src = open(os.path.join(repo_dir, rel), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            out[rel] = {}
            continue
        out[rel] = module_constants(src, names)
    return out


def compare_constants(spec: dict, cand_constants: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Load-bearing constants the candidate states differently (empty dict == all match).

    ``spec['constants'] = {rel: {NAME: repr}}`` from the reference. Policy:
      * value constants (``IGNORE_INDEX``, ``DEFAULT_TEMPLATE``) must match EXACTLY;
      * dict constants (``SCHEDULER_MAP`` / ``DATA_CONFIG`` keys) must be a SUPERSET -- every
        frozen key present, extras allowed.
    A frozen constant the candidate does not define (or defines non-literally) is a miss.
    """
    constants = spec.get("constants") or {}
    problems: dict[str, dict[str, dict]] = {}
    for rel, wanted in sorted(constants.items()):
        have = cand_constants.get(rel, {})
        rel_problems: dict[str, dict] = {}
        for name, want in sorted(wanted.items()):
            got = have.get(name)
            if got is None:
                rel_problems[name] = {"error": "missing or non-literal", "expected": want}
            elif want.get("kind") == "dict":
                missing_keys = sorted(set(want.get("keys", [])) - set(got.get("keys", [])))
                if got.get("kind") != "dict" or missing_keys:
                    rel_problems[name] = {
                        "error": "dict keys not a superset",
                        "missing_keys": missing_keys,
                        "expected_keys": want.get("keys"),
                        "candidate_keys": got.get("keys"),
                    }
            elif got.get("kind") != "value" or got.get("value") != want.get("value"):
                rel_problems[name] = {
                    "error": "value mismatch",
                    "expected": want.get("value"),
                    "candidate": got.get("value"),
                }
        if rel_problems:
            problems[rel] = rel_problems
    return problems


# --- module / directory structure ------------------------------------------ #
def candidate_py_files(repo_dir: str) -> set[str]:
    """Repo-relative ``.py`` paths in the candidate tree (excludes ``__pycache__``)."""
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                out.add(os.path.relpath(os.path.join(dirpath, fn), repo_dir))
    return out


def candidate_empty_dirs(repo_dir: str) -> list[str]:
    """Sub-directories holding no ``.py`` at any depth (namespace-package hijack trap).

    Mirrors the repository structure check: a leftover package dir (kept alive by a
    stray ``__pycache__``) becomes a namespace package that loses to a same-named real
    package on ``sys.path`` -- how a leftover ``datasets/`` silently handed ``import
    datasets`` to the torch-importing HF one. The candidate must have none.
    """
    out: list[str] = []
    for dirpath, dirnames, _ in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if os.path.abspath(dirpath) == os.path.abspath(repo_dir):
            continue
        has_py = any(
            fn.endswith(".py")
            for _, _, fns in os.walk(dirpath)
            for fn in fns
        )
        if not has_py:
            out.append(os.path.relpath(dirpath, repo_dir))
    return out


def compare_structure(spec: dict, repo_dir: str) -> dict:
    """Candidate module tree vs the frozen torch_orig layout (all-empty == satisfies).

    ``spec['structure'] = {files:[torch .py], allow_missing:[...], allow_extra:[...]}``.
    Required = ``files - allow_missing`` (every torch module the port must carry; today
    ``allow_missing`` is empty -- every torch module has a counterpart). The candidate's
    ``.py`` set must be a SUPERSET (extra files, incl. the reference's documented
    ``allow_extra`` torch-free shims, are fine) with NO empty/namespace-package directory.
    """
    structure = spec.get("structure") or {}
    files = set(structure.get("files") or [])
    allow_missing = set(structure.get("allow_missing") or [])
    required = files - allow_missing
    cand_files = candidate_py_files(repo_dir)
    return {
        "missing_files": sorted(required - cand_files),
        "empty_dirs": sorted(candidate_empty_dirs(repo_dir)),
    }


# --- callable signatures --------------------------------------------------- #
def _callable_param_info(func_node) -> dict:
    """Parameter shape of a def node: positional slots, keyword-acceptable names, varargs."""
    a = func_node.args
    pos = [p.arg for p in getattr(a, "posonlyargs", [])] + [p.arg for p in a.args]
    by_keyword = {p.arg for p in a.args} | {p.arg for p in a.kwonlyargs}
    return {
        "positional_slots": len(pos),
        "accepts_by_keyword": sorted(by_keyword),
        "has_var_positional": a.vararg is not None,
        "has_var_keyword": a.kwarg is not None,
    }


def _bound_method_param_info(func_node) -> dict:
    """:func:`_callable_param_info` for a method, minus the bound receiver.

    The harness supplies ``self`` / ``cls`` by BINDING the call, so it is not one of
    the arguments it passes. Without dropping it every method reads as having one
    spare positional slot, and a method the harness calls with N arguments would be
    accepted at N-1 -- i.e. the check would be looser on methods than on functions,
    which is the opposite of what a contract wants.
    """
    info = dict(_callable_param_info(func_node))
    a = func_node.args
    leading = [p.arg for p in getattr(a, "posonlyargs", [])] + [p.arg for p in a.args]
    if leading and leading[0] in ("self", "cls"):
        info["positional_slots"] = max(0, info["positional_slots"] - 1)
        info["accepts_by_keyword"] = sorted(set(info["accepts_by_keyword"]) - {leading[0]})
    return info


def candidate_callable_params(repo_dir: str, rel: str, name: str) -> dict | None:
    """Parameter info for ``name`` in candidate module ``rel``.

    ``name`` is either a top-level function (``convert_checkpoint``) or a method
    (``Engine.prefill``). None when the module, the class or the callable is absent
    (a parity/structure miss, reported there). AST-only, so default values /
    annotations need not import.

    THE METHOD FORM IS AN INFERENCE-CLASS DIVERGENCE from the byte-copied training
    original, and it is a bug fix rather than a feature. The training class's four
    graded entry points (``get_train_args``, ``load_tokenizer``, ``load_model``,
    ``run_<method>``) are all module-level functions, so a lookup that scanned only
    ``tree.body`` was sufficient there. On this side ten of the nineteen frozen
    signature targets are methods -- ``Engine.prefill`` / ``decode_step`` /
    ``generate`` / ``reset``, the five ``BlockPool`` entry points and
    ``ModelConfig.from_hf`` -- and the top-level-only lookup returned
    ``{"ok": False, "reason": "callable not found"}`` for every one of them against
    the CORRECT reference. A contract section that reports ten false failures on the
    worked example is a section nobody can turn on.

    The same defect exists in ``tasks/*/tests/_t2j/compare.py`` across 45 instances;
    fixing it there is out of scope here, so this copy diverges. The
    divergence is machine-checked and confined: the harness compares the two files
    symbol by symbol and allows exactly this function and its helper to differ.
    """
    try:
        tree = ast.parse(open(os.path.join(repo_dir, rel), encoding="utf-8").read())
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None
    if "." in name:
        cls_name, method_name = name.split(".", 1)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != cls_name:
                continue
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and stmt.name == method_name:
                    return _bound_method_param_info(stmt)
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _callable_param_info(node)
    return None


def check_callable_signature(info: dict, *, positional: int = 0, keywords=()) -> dict:
    """Does a candidate callable ACCEPT what the harness passes it?

    ``positional`` = leading positional args the harness supplies (the callable needs at
    least that many positional slots, or ``*args``). ``keywords`` = keyword names the
    harness supplies (each must be a positional-or-keyword / keyword-only parameter, or the
    callable must take ``**kwargs``). Extra, reordered or renamed positional-only params are
    fine -- the port adapts parameter lists. ``info`` is ``candidate_callable_params(...)``;
    None (callable absent) returns ``ok=False`` so the reason is explicit.
    """
    if info is None:
        return {"ok": False, "reason": "callable not found", "missing_keywords": list(keywords)}
    positional_ok = info["has_var_positional"] or info["positional_slots"] >= positional
    accepts_kw = set(info["accepts_by_keyword"])
    missing_kw = [] if info["has_var_keyword"] else [k for k in keywords if k not in accepts_kw]
    return {
        "ok": bool(positional_ok and not missing_kw),
        "positional_ok": bool(positional_ok),
        "positional_slots": info["positional_slots"],
        "needs_positional": positional,
        "missing_keywords": missing_kw,
    }


def compare_signatures(spec: dict, repo_dir: str) -> dict[str, dict]:
    """Grader-entry callables that will not accept the harness's arguments (empty == ok).

    ``spec['signatures'] = {"rel::name": {"positional": N, "keywords": [...]}}`` freezes, for
    the fixed grader entry points, how the harness calls each. Every listed callable must
    ACCEPT that call shape (see ``check_callable_signature``). Existence itself is a parity
    concern; a callable that is absent still reports here (``ok=False``) so a dropped entry
    point is unambiguous.
    """
    signatures = spec.get("signatures") or {}
    problems: dict[str, dict] = {}
    for ref, want in sorted(signatures.items()):
        rel, name = ref.split("::", 1)
        info = candidate_callable_params(repo_dir, rel, name)
        result = check_callable_signature(
            info, positional=int(want.get("positional", 0)), keywords=tuple(want.get("keywords", ()))
        )
        if not result["ok"]:
            problems[ref] = result
    return problems
