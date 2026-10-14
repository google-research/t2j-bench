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

"""Continuous similarity primitives and task scoring functions.

Scores lie in [0, 1]. Non-finite or incompatible inputs score zero; EPS protects
denominators. Metrics normally compare float32 operands, with float64 used for
learning-rate schedules and finite-difference derivatives.

Fixed constants control scalar sensitivity, low-tail aggregation, tolerance-band
ramps, and near-zero gradient directions. They are shared across tasks, not fitted
to candidates. score_scalar_banded also uses tolerances frozen in the goldens.
Structural Spec checks remain in compare.py. Floating-point differences can keep
a faithful implementation slightly below a score of one.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ._arrays import to_numpy

# The denominator guard -- never a verdict tolerance (§4.2). The other fixed method constants
# (gain / aggregation exponent / band ramp / vacuous floor) are defined below with rationale.
EPS = 1e-12
_LN2 = float(np.log(2.0))

# P0.3: the symmetric relative error is DOUBLED before it is subtracted, so a scalar reports
# its TRUE relative gap (gain 2) instead of half of it -- 10% off -> 0.90 not 0.95, 20% off
# -> 0.80 not 0.91. gain 1 recovers the retired half-error basin exactly; a faithful fp32
# scalar (rel gap ~1e-3) still scores >= 0.998, so honesty is not punished.
_SCALAR_ERR_GAIN = 2.0

# P0.2: aggregation exponent for the low-tail-sensitive mean. ``p < 1`` makes a single
# broken element (score -> 0) bite super-linearly (a lone 0 among many 1s drags the mean
# far below 1 - 1/N), while a UNIFORM faithful set (all ~0.999) is unchanged -- power means
# are p-invariant on constant inputs, so honesty is never punished and only a genuinely
# broken element/check is penalized harder. ``p = 1`` recovers the arithmetic mean exactly.
_AGG_P = 0.5

# A gradient-authenticity probe direction is *vacuous* when the scalar loss genuinely does
# not depend on it (e.g. a PPO value-only loss along an lm_head direction): both the analytic
# directional derivative and its finite-difference estimate are ~0 -- they AGREE the direction
# is flat, which is faithful, not a severed backward. The analytic side is floored to ~1e-12
# upstream and fd is exactly 0, so raw ``S_scalar`` misreads that agreement as a 50% error;
# below this absolute tolerance the direction is scored 1.0 (:func:`score_grad_authenticity`).
# Real directional derivatives are O(0.1-10), so this never masks a genuine disagreement.
_GRAD_VACUOUS_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Casting / finiteness helpers.
# --------------------------------------------------------------------------- #
def _as_f32(value: Any) -> np.ndarray | None:
    """``value`` -> flat-friendly float32 ndarray, or ``None`` if unconvertible."""
    try:
        return to_numpy(value).astype(np.float32, copy=False)
    except (ValueError, TypeError):
        return None


def _as_f64(value: Any) -> np.ndarray | None:
    try:
        return to_numpy(value).astype(np.float64, copy=False)
    except (ValueError, TypeError):
        return None


def _finite(arr: np.ndarray) -> bool:
    return bool(arr.size) and bool(np.all(np.isfinite(arr)))


def _clip01(x: float) -> float:
    """Clamp to [0,1] (guards ULP overshoot / a ``NaN`` slipping through -> 0.0)."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _scalar_or_none(x: Any) -> float | None:
    """One finite python float from ``x`` (a number, 0-d, or size-1 array), else ``None``.
    Mirrors :func:`S_scalar`'s extractor for callers that must branch on the value."""
    try:
        v = float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
    except (ValueError, TypeError, IndexError):
        return None
    return v if np.isfinite(v) else None


# --------------------------------------------------------------------------- #
# The five primitives (§4.2). Each returns a scalar similarity in [0,1].
# --------------------------------------------------------------------------- #
def S_scalar(a: Any, b: Any) -> float:
    """Symmetric relative similarity: ``1 - GAIN*|a-b| / (|a| + |b| + EPS)`` (GAIN = 2).

    1 at equality, -> 0 as either value dwarfs the other, sign-aware
    (``S_scalar(1, -1) = 0``). Both-zero -> 1 (``0 == 0``). ``NaN``/``Inf`` -> 0.0. The
    ``GAIN`` (:data:`_SCALAR_ERR_GAIN`) makes the reported deficit the TRUE relative error,
    not half of it (P0.3), so a wrong loss/objective no longer sits in a lenient basin.
    """
    # Pull one float from each side (a scalar may arrive as a python number, a 0-d or a
    # size-1 array); anything unconvertible or empty scores 0.
    try:
        av = float(np.asarray(a, dtype=np.float64).reshape(-1)[0])
        bv = float(np.asarray(b, dtype=np.float64).reshape(-1)[0])
    except (ValueError, TypeError, IndexError):
        return 0.0
    if not (np.isfinite(av) and np.isfinite(bv)):
        return 0.0
    return _clip01(1.0 - _SCALAR_ERR_GAIN * abs(av - bv) / (abs(av) + abs(bv) + EPS))


def S_L2(a: Any, b: Any) -> float:
    """Magnitude-aware closeness: ``1 - ||a-b|| / (||a|| + ||b|| + EPS)`` (flattened).

    Rewards direction *and* magnitude; a uniform 2x scaling scores 2/3, not 1, so it
    catches scale bugs a pure cosine misses. Both-zero -> 1. Shape mismatch /
    non-finite -> 0.0.
    """
    ra, rb = _as_f32(a), _as_f32(b)
    if ra is None or rb is None:
        return 0.0
    ra, rb = ra.reshape(-1), rb.reshape(-1)
    if ra.shape != rb.shape or not (_finite(ra) and _finite(rb)):
        return 0.0
    num = float(np.linalg.norm(ra - rb))
    den = float(np.linalg.norm(ra) + np.linalg.norm(rb) + EPS)
    return _clip01(1.0 - num / den)


def S_cos(a: Any, b: Any) -> float:
    """Direction-only closeness: ``max(0, (a.b) / (||a|| ||b|| + EPS))`` (flattened).

    Scale-invariant, clamped at 0 (anti-aligned / orthogonal -> 0). A zero-norm operand
    -> 0 (no false "perfect" match). Used inside :func:`S_tensor`, rarely alone.
    """
    ra, rb = _as_f32(a), _as_f32(b)
    if ra is None or rb is None:
        return 0.0
    ra, rb = ra.reshape(-1), rb.reshape(-1)
    if ra.shape != rb.shape or not (_finite(ra) and _finite(rb)):
        return 0.0
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb) + EPS)
    return _clip01(float(np.dot(ra, rb)) / denom)


def S_tensor(a: Any, b: Any) -> float:
    """Array workhorse: ``sqrt(S_cos * S_L2)`` -- both direction and magnitude must agree.

    Either factor near 0 pulls the product down; a zero-norm candidate against a
    non-zero reference scores 0 (``S_cos = 0``). Shape mismatch / non-finite -> 0.0.
    """
    return _clip01((S_cos(a, b) * S_L2(a, b)) ** 0.5)


def S_dist(p: Any, q: Any, valid_mask: Any = None) -> float:
    """Distribution similarity over a vocab: ``1 - JSD(softmax(p), softmax(q)) / ln2``.

    Jensen-Shannon *divergence* in nats is bounded by ``ln2``, so the normalized score is
    in [0,1]. Runs on the softmax, so it is invariant to the per-token additive logit
    constant for free (no centering), it *sees* a temperature/sharpening change (unlike
    scale-blind cosine), and it needs no KL cap or argmax margin -- it replaces the
    retired centered-cosine + ``max_kl`` + ``argmax_margin`` triple (§4.2).

    ``p``/``q`` are reshaped to ``(-1, vocab)``; rows are restricted to
    ``valid_mask == 1`` positions (flattened over the leading axes) when a mask is given.
    The score is the low-tail-sensitive mean (:func:`power_mean`, P0.2) over the kept rows
    (a non-finite row scores 0), so a candidate that matches the easy/high-entropy rows
    while breaking a few cannot average the broken rows away. A faithful port's rows are
    uniformly ~0.999, and the power mean is p-invariant on a constant set, so honesty is
    unchanged. Empty -> 0.0.
    """
    rp, rq = _as_f32(p), _as_f32(q)
    if rp is None or rq is None:
        return 0.0
    if rp.shape != rq.shape or rp.ndim < 1 or rp.shape[-1] < 1:
        return 0.0
    vocab = rp.shape[-1]
    rp = rp.reshape(-1, vocab)
    rq = rq.reshape(-1, vocab)
    if valid_mask is not None:
        m = _as_f32(valid_mask)
        if m is not None:
            sel = m.reshape(-1) > 0
            if sel.shape[0] == rp.shape[0]:
                rp, rq = rp[sel], rq[sel]
    if rp.shape[0] == 0:
        return 0.0
    scores = np.empty(rp.shape[0], dtype=np.float64)
    for i in range(rp.shape[0]):
        scores[i] = _row_dist_similarity(rp[i], rq[i])
    # Low-tail-sensitive aggregation (P0.2, _AGG_P) instead of a raw arithmetic mean --
    # matches score_mean / power_mean everywhere else in the toolkit. power_mean already
    # sanitizes non-finite -> 0 and clips to [0,1]. (Forward-reference to power_mean, which
    # is defined below; resolved at call time.)
    return power_mean(scores)


def _row_dist_similarity(p_row: np.ndarray, q_row: np.ndarray) -> float:
    """``1 - JSD(softmax(p_row), softmax(q_row)) / ln2`` for one logit row."""
    if not (np.all(np.isfinite(p_row)) and np.all(np.isfinite(q_row))):
        return 0.0
    lp = _log_softmax(p_row.astype(np.float64))
    lq = _log_softmax(q_row.astype(np.float64))
    p = np.exp(lp)
    q = np.exp(lq)
    m = 0.5 * (p + q)
    lm = np.log(np.maximum(m, EPS))
    # KL(p||m) = sum p*(lp-lm); 0*log0 == 0, so mask on the support of each side.
    kl_pm = float(np.sum(np.where(p > 0, p * (lp - lm), 0.0)))
    kl_qm = float(np.sum(np.where(q > 0, q * (lq - lm), 0.0)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    if not np.isfinite(jsd):
        return 0.0
    return _clip01(1.0 - jsd / _LN2)


def _log_softmax(row: np.ndarray) -> np.ndarray:
    mx = float(np.max(row))
    shifted = row - mx
    return shifted - float(np.log(np.sum(np.exp(shifted))))


# --------------------------------------------------------------------------- #
# Scoring functions (§4.4). Thin, named wrappers over the primitives, matching the
# per-test map in §4.3. Every one returns a similarity in [0,1]; nothing is pass/fail.
# --------------------------------------------------------------------------- #
def _select_logit_rows(p: Any, q: Any, valid_mask: Any = None):
    """(rp, rq) as ``(rows, vocab)`` restricted to ``valid_mask == 1`` rows, mirroring
    :func:`S_dist`'s selection so the scale term scores the SAME rows as the distribution
    term. Returns ``(None, None)`` on shape mismatch / empty selection."""
    rp, rq = _as_f32(p), _as_f32(q)
    if rp is None or rq is None:
        return None, None
    if rp.shape != rq.shape or rp.ndim < 1 or rp.shape[-1] < 1:
        return None, None
    vocab = rp.shape[-1]
    rp = rp.reshape(-1, vocab)
    rq = rq.reshape(-1, vocab)
    if valid_mask is not None:
        m = _as_f32(valid_mask)
        if m is not None:
            sel = m.reshape(-1) > 0
            if sel.shape[0] == rp.shape[0]:
                rp, rq = rp[sel], rq[sel]
    if rp.shape[0] == 0:
        return None, None
    return rp, rq


def S_logit_scale(p: Any, q: Any, valid_mask: Any = None) -> float:
    """Scale/temperature sensitivity that ``S_dist`` lacks: ``S_L2`` over the row-CENTERED
    logits of the kept rows (P0.1).

    ``S_dist`` runs on the softmax, so it is lenient on a global logit scale/temperature (a
    15% scale shift barely moves a high-entropy softmax). Centering each row over the vocab
    removes the per-row additive constant that softmax -- and thus the loss/argmax -- is
    invariant to (so a pure additive gauge still scores 1.0, matching ``S_dist``), while the
    L2 over the centered logits drops under a scale/temperature change that distorts the
    spread. Non-finite / empty -> 0.0.
    """
    rp, rq = _select_logit_rows(p, q, valid_mask)
    if rp is None:
        return 0.0
    cp = rp - rp.mean(axis=-1, keepdims=True)
    cq = rq - rq.mean(axis=-1, keepdims=True)
    if not (_finite(cp) and _finite(cq)):
        return 0.0
    num = float(np.linalg.norm(cp - cq))
    den = float(np.linalg.norm(cp) + np.linalg.norm(cq) + EPS)
    return _clip01(1.0 - num / den)


def score_logits(ref: Any, cand: Any, valid_mask: Any = None) -> float:
    """Logit fidelity over ``valid_mask == 1`` positions: geometric mean of a DISTRIBUTION
    term and a SCALE term, ``sqrt(S_dist * S_logit_scale)`` (P0.1).

    ``S_dist`` (softmax-JSD) scores the per-row distribution shape; ``S_logit_scale``
    (centered-logit L2) scores the magnitude/spread that softmax-JSD is lenient on, so a
    global logit scale / temperature bug can no longer hide behind a high JSD. Both terms
    are invariant to the per-row additive gauge, and a faithful port keeps both ~0.999.
    Either factor 0 -> 0.0. Serves ``test_forward_logits``, ``test_fp32_forward``, the
    batteries, ``test_mm_attended_forward``, ``test_generation_logits`` (per step). No KL
    cap, no argmax clause."""
    dist = S_dist(ref, cand, valid_mask)
    scale = S_logit_scale(ref, cand, valid_mask)
    if dist <= 0.0 or scale <= 0.0:
        return 0.0
    return _clip01((dist * scale) ** 0.5)


def score_scalar(ref: Any, cand: Any) -> float:
    """``S_scalar``. Serves every scalar loss / logprob / grad-norm / validation loss."""
    return S_scalar(ref, cand)


# Faithful-tolerance band width for :func:`score_scalar_banded`. Beyond the dead-zone
# ``tol`` the score ramps linearly to 0 over ``_BAND_RAMP`` further tolerances. This is
# NOT a tunable verdict tolerance: ``tol`` is frozen as ``clamp(factor*torch_self_std,
# floor, cap)`` from torch's OWN run-to-run variance (O1) -- the oracle's oscillation
# around itself, NOT the ref_jax-vs-torch gap (an old design that widened the band exactly
# where the reference diverges, hiding it) -- so a faithful port lands in-band and scores 1.0.
_BAND_RAMP = 3.0


def score_scalar_banded(ref: Any, cand: Any, tol: Any = None, *, ramp: float = _BAND_RAMP) -> float:
    """Tolerance-banded scalar parity for the in-band real-training gates (§7.7 L6, §7.4 T3).

    A dead-zone + sharpened ramp that replaces the wide ``S_scalar`` basin for the in-band
    held-out / train-run losses. ``tol`` is the frozen tolerance
    (``validation_loss.json['tol']`` / ``train_run.json['train_tol']``), computed as
    ``clamp(factor*torch_self_std, floor, cap)`` from torch's OWN run-to-run variance (O1) --
    the oracle's oscillation around itself, NOT the ref_jax-vs-torch gap (an old design that
    widened the band exactly where the reference diverges, hiding it). The target is torch's:

      * ``|ref - cand| <= tol``                 -> 1.0  (indistinguishable from the faithful
                                                        port; honesty is not punished --
                                                        incl. a PPO neg-mean-reward that
                                                        sign-flips around 0, which the raw
                                                        ``S_scalar`` collapsed to 0.0)
      * ``tol < |ref-cand| <= (1+ramp)*tol``    -> linear 1.0 -> 0.0 (a drift a few faithful
                                                        gaps wide collapses; mediocrity is not
                                                        rewarded, unlike ``S_scalar``'s 0.67@2x)
      * ``|ref - cand| > (1+ramp)*tol``         -> 0.0

    ``tol`` None / non-finite / <= 0 -> falls back to :func:`S_scalar` (a golden with no
    frozen band grades exactly as before, so shipping this ahead of the re-freeze is safe).
    ``NaN``/``Inf`` operands -> 0.0.
    """
    try:
        av = float(np.asarray(ref, dtype=np.float64).reshape(-1)[0])
        bv = float(np.asarray(cand, dtype=np.float64).reshape(-1)[0])
    except (ValueError, TypeError, IndexError):
        return 0.0
    if not (np.isfinite(av) and np.isfinite(bv)):
        return 0.0
    try:
        t = float(tol)
    except (TypeError, ValueError):
        t = float("nan")
    if not np.isfinite(t) or t <= 0.0:
        return S_scalar(av, bv)
    d = abs(av - bv)
    if d <= t:
        return 1.0
    return _clip01(1.0 - (d - t) / (ramp * t))


def score_tensor(ref: Any, cand: Any) -> float:
    """``S_tensor``. Serves hidden states, module I/O, merged embeds, vision features,
    full-FT gradients."""
    return S_tensor(ref, cand)


def power_mean(values: Sequence[float], p: float = _AGG_P) -> float:
    """Low-tail-sensitive aggregator (§4.7, P0.2): ``(mean(x**p))**(1/p)`` over scores in
    [0,1].

    ``p = 1`` is the arithmetic mean; ``0 < p < 1`` penalizes low elements super-linearly
    (a single 0 among many 1s drags the mean well below ``1 - 1/N``), while a uniform set is
    p-invariant (a faithful ~0.999 set is unchanged). Values are clipped to [0,1] first.
    Empty -> 0.0.
    """
    vals = [float(x) for x in values]
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    # A degenerate element (nan/inf) scores 0.0, per the module's degenerate->0.0 convention:
    # map non-finite BEFORE the clip so +inf does not clamp to a perfect 1.0 and a lone nan
    # does not poison the whole mean. (Graded scores are pre-sanitized, so this is defensive.)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if p == 1.0:
        return _clip01(float(np.mean(arr)))
    return _clip01(float(np.mean(arr ** p) ** (1.0 / p)))


def score_mean(pairs: Sequence[tuple], fn: Callable[[Any, Any], float]) -> float:
    """Generic within-test aggregator (§4.4): low-tail-sensitive mean (:func:`power_mean`,
    P0.2) of ``fn(ref, cand)`` over a list of per-layer / per-unit / per-step / per-tensor
    element pairs (empty -> 0.0).

    This is how a multi-element test collapses to one score. A uniform faithful set is
    unchanged; a single broken element bites harder than the retired arithmetic mean. ``fn``
    is one of the primitives above (``S_scalar``/``S_tensor``/``S_dist``/...).
    """
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return power_mean([fn(a, b) for a, b in pairs])


def score_schedule(ref: Any, cand: Any) -> float:
    """``S_L2`` in float64 over the enumerated learning-rate sequence (§4.3)."""
    ra, rb = _as_f64(ref), _as_f64(cand)
    if ra is None or rb is None:
        return 0.0
    ra, rb = ra.reshape(-1), rb.reshape(-1)
    if ra.shape != rb.shape or ra.size == 0 or not (_finite(ra) and _finite(rb)):
        return 0.0
    num = float(np.linalg.norm(ra - rb))
    den = float(np.linalg.norm(ra) + np.linalg.norm(rb) + EPS)
    return _clip01(1.0 - num / den)


def score_grad_authenticity(analytic: Any, fd: Any) -> float:
    """``S_scalar(analytic, fd)`` -- analytic directional derivative vs its central
    finite-difference estimate (§4.4). A faked/severed backward drives ``fd`` -> 0 while the
    analytic side stays finite (or vice versa), so the score collapses.

    A genuine *vacuous* direction (the scalar loss does not depend on it -- e.g. a PPO
    value-only loss along an lm_head direction) has analytic and fd BOTH ~0: they AGREE the
    direction is flat, which is faithful. Upstream floors the analytic side to ~1e-12 and fd
    is exactly 0, so raw ``S_scalar`` would misread that agreement as a 50% error (0.5, or
    0.0 under the P0.3 GAIN=2) and punish a correct backward; when BOTH sides are within
    :data:`_GRAD_VACUOUS_TOL` of zero we honor the documented ``analytic == fd == 0 -> 1``
    contract and return 1.0. This is not hackable: a severed backward fails the caller's
    ``analytic > 1e-12`` precondition on the non-vacuous probe, and ``fd`` is measured from
    real forward passes, so a candidate cannot manufacture mutual flatness on a live direction.
    """
    av = _scalar_or_none(analytic)
    bv = _scalar_or_none(fd)
    if (
        av is not None
        and bv is not None
        and abs(av) <= _GRAD_VACUOUS_TOL
        and abs(bv) <= _GRAD_VACUOUS_TOL
    ):
        return 1.0
    return S_scalar(analytic, fd)


def score_generation_prefix(
    ref_seq: Any, cand_seq: Any, prompt_len: int, new_tokens: int
) -> float:
    """Fraction of structural generation facts that hold, each a term in [0,1], averaged
    (§4.4): batch size preserved; the prompt is an exact prefix; exactly ``new_tokens``
    were appended. Serves ``test_generation_shape``."""
    ra = _as_f32(ref_seq)
    rb = _as_f32(cand_seq)
    if ra is None or rb is None:
        return 0.0
    ref = np.asarray(ref_seq)
    cand = np.asarray(cand_seq)
    if ref.ndim < 2 or cand.ndim < 2:
        return 0.0
    terms: list[float] = []
    # (1) batch preserved.
    terms.append(1.0 if cand.shape[0] == ref.shape[0] else 0.0)
    # (2) exactly new_tokens appended (candidate length == prompt_len + new_tokens).
    terms.append(1.0 if cand.shape[1] == prompt_len + new_tokens else 0.0)
    # (3) the prompt is an exact prefix, over the rows that line up.
    nb = min(ref.shape[0], cand.shape[0])
    pl = min(prompt_len, ref.shape[1], cand.shape[1])
    if nb and pl:
        match = np.asarray(ref[:nb, :pl] == cand[:nb, :pl])
        terms.append(float(match.mean()))
    else:
        terms.append(0.0)
    return _clip01(float(np.mean(terms)))


def score_frozen_base(ref_base: Any, cand_base: Any) -> float:
    """Frozen-base drift (§4.3): ``1 - ||dP_base|| / (||P_base|| + EPS)``.

    ``ref_base`` is the reference (pre-train) base weight, ``cand_base`` the candidate's
    post-step base weight; a LoRA step must leave the base untouched, so a faithful port
    scores 1.0 and any leaked update lowers it. Shape mismatch / non-finite -> 0.0.
    """
    rp = _as_f32(ref_base)
    rc = _as_f32(cand_base)
    if rp is None or rc is None:
        return 0.0
    rp, rc = rp.reshape(-1), rc.reshape(-1)
    if rp.shape != rc.shape or not (_finite(rp) and _finite(rc)):
        return 0.0
    drift = float(np.linalg.norm(rc - rp))
    base = float(np.linalg.norm(rp) + EPS)
    return _clip01(1.0 - drift / base)


def score_softmax_normalization(probs: Any) -> float:
    """Softmax normalization (§4.3): ``1 - clamp(mean|sum(softmax) - 1|, 0, 1)``.

    ``probs`` is a candidate probability tensor whose last axis should sum to 1; the
    score falls as the rows drift off the simplex. Non-finite -> 0.0.
    """
    arr = _as_f32(probs)
    if arr is None or arr.ndim < 1 or not _finite(arr):
        return 0.0
    sums = np.sum(arr, axis=-1)
    err = float(np.mean(np.abs(sums - 1.0)))
    return _clip01(1.0 - min(1.0, max(0.0, err)))
