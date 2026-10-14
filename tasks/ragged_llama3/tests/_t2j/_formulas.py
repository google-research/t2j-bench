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

"""Inference reference math the grader needs INDEPENDENTLY of the candidate.

Numpy only (plus an optional, lazily-imported jax for the one device helper), so
it imports in the torch-free verifier and in the freezer alike.

Nothing here is a training formula: inference has no backward pass, so the DPO /
PPO / causal-CE family that ``tasks/*/tests/_t2j/_formulas.py`` carries has no
analogue and is deliberately absent rather than copied across and left unused.

What IS here, and why each earns its place:

  * ``centered_cosine`` -- logits are defined only up to a per-token additive
    constant, so the raw cosine of two logit matrices is dominated by whatever
    per-row offset each framework's lm_head happens to produce. Centring each row
    first is the measure ``compare.check_logits`` is calibrated against,
    expressed as a plain function, usable by the freezer and by
    a calibration sweep that has no ``NumericThresholds`` in hand.
  * ``softmax_kl`` / ``top1_agreement`` / ``logprob_of`` -- the three
    distribution-level statistics the L2/L4/L5 tiers report. ``compare.py`` folds
    the first into a pass/fail; these return the NUMBER, which is what a
    calibration run and a failure message need.
  * ``reference_paged_attention_numpy`` -- a numpy transcription of the ONE
    formulation both reference trees implement. It exists to
    sanity-check a frozen golden against itself -- ``paged_attention_units.npz``
    carries its own ``(q, k, v)`` inputs, so the freeze can be re-derived here,
    in a third implementation that shares no code with either side. It is NOT a
    grading oracle: the graded comparison is always candidate-vs-golden.
  * ``param_stats`` -- the layout-invariant per-parameter reduction
    ``golden/param_norms.json`` is frozen with and ``compare.compare_param_norms``
    reads back.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Array coercion. Torch-free by construction: nothing in this module imports
# torch, and a jax array converts through ``np.asarray`` without one.
# --------------------------------------------------------------------------- #
def as_f32(value: Any) -> np.ndarray:
    """Any array-like -> a float32 numpy array (bfloat16 included).

    ``np.asarray`` on a bf16 jax array yields an ml_dtypes bfloat16 numpy array,
    which numpy's own reductions handle badly; upcasting here means every metric
    below accumulates in float32 regardless of the serving dtype.
    """
    if value is None:
        raise ValueError("cannot convert None to an array")
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    return arr.astype(np.float32, copy=False)


def _rows(value: Any) -> np.ndarray:
    """[..., V] -> [N, V]: every metric below is per-token over the last axis."""
    arr = as_f32(value)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(-1, arr.shape[-1])


def _log_softmax(rows: np.ndarray) -> np.ndarray:
    peak = np.max(rows, axis=-1, keepdims=True)
    shifted = rows - peak
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


# --------------------------------------------------------------------------- #
# Logit statistics.
# --------------------------------------------------------------------------- #
def centered_cosine(reference: Any, candidate: Any) -> float:
    """Minimum per-token cosine after removing each row's mean.

    MINIMUM, not mean: a single broken position (one token attending the wrong
    page) is invisible in an average over a 12-token prompt, and the whole point
    of the paged tier is that a single position can be the bug. Returns 0.0 on a
    shape mismatch, mirroring ``compare.cosine_similarity``.

    A DEGENERATE ROW SCORES 1.0 ONLY IF **BOTH** SIDES ARE DEGENERATE. This used
    to be ``np.where(denom > 0, ..., 1.0)``, which returns perfect agreement
    whenever EITHER centred row has zero norm -- so a candidate emitting all
    zeros (or any per-token constant, which centres to zero) scored the maximum
    on this clause instead of the minimum.

    This is load-bearing: when a hidden-states abs clause is vacuous, the
    centred cosine is the only guard -- and a port emitting ``zeros_like``
    placeholders would then satisfy both clauses at once. Widening the band
    cannot fix that, because the second clause was rewarding the degenerate
    answer.

    The three cases are now distinct, and only the first is agreement:

      both degenerate   1.0   two constant rows genuinely carry the same (no)
                              direction, and the ABS clause still grades their
                              values -- this is the case the old comment meant
      one degenerate    0.0   ORTHOGONALITY, not the minimum -- an anti-correlated
                              pair scores -1.0. 0.0 is the honest reading (a row
                              with no direction cannot agree OR disagree with one
                              that has one) and it is below every floor in this
                              class (fp32 0.9999995, native 0.998, hidden_states
                              0.98), so it fails all of them. It does mean a
                              degenerate candidate scores strictly better than an
                              inverted one, which is correct: inversion is a
                              worse answer than silence.
      neither           dot / denom

    A NON-FINITE REFERENCE ROW ALSO CHANGED VERDICT here, and it is worth
    knowing: a row of ``nan`` or all ``-inf`` used to score 1.0 (``nan > 0`` is
    False, so it took the old fallback) and now scores 0.0. No shipped surface
    can produce one -- masked logits are handled by explicit ``isfinite`` and
    ``isneginf`` clauses before they reach here -- but a future family that
    freezes one will now see it fail rather than pass silently.
    """
    ref, cand = _rows(reference), _rows(candidate)
    if ref.shape != cand.shape or ref.size == 0:
        return 0.0
    ref = ref - ref.mean(axis=-1, keepdims=True)
    cand = cand - cand.mean(axis=-1, keepdims=True)
    ref_norm = np.linalg.norm(ref, axis=-1)
    cand_norm = np.linalg.norm(cand, axis=-1)
    denom = ref_norm * cand_norm
    dot = np.sum(ref * cand, axis=-1)
    both_flat = (ref_norm == 0) & (cand_norm == 0)
    per_row = np.where(
        denom > 0,
        dot / np.where(denom > 0, denom, 1.0),
        np.where(both_flat, 1.0, 0.0),
    )
    return float(np.min(per_row))


def softmax_kl(reference: Any, candidate: Any) -> float:
    """max_t KL(softmax(ref_t) || softmax(cand_t)).

    ``inf`` on a shape mismatch or a non-finite input, so a caller comparing
    against a tolerance never reads a broken artifact as agreement.
    """
    ref, cand = _rows(reference), _rows(candidate)
    if ref.shape != cand.shape or ref.size == 0:
        return float("inf")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(cand))):
        return float("inf")
    ref_lp = _log_softmax(ref)
    cand_lp = _log_softmax(cand)
    kl = np.sum(np.exp(ref_lp) * (ref_lp - cand_lp), axis=-1)
    return float(np.max(np.abs(kl)))


def top1_agreement(reference: Any, candidate: Any) -> float:
    """Fraction of positions where the two argmaxes agree (0.0 on a mismatch)."""
    ref, cand = _rows(reference), _rows(candidate)
    if ref.shape != cand.shape or ref.size == 0:
        return 0.0
    return float(np.mean(ref.argmax(-1) == cand.argmax(-1)))


def logprob_of(logits: Any, token_ids: Any) -> np.ndarray:
    """Log-probability of ``token_ids[t]`` under ``logits[t]``, per position.

    The float32 log_softmax is deliberate and matches both engines' step-logprob
    convention (``engine.generate`` casts to float32 before ``log_softmax``): a
    bf16 log_softmax over a 128k vocab loses two digits of the very quantity the
    L5 trajectory band is measured in.
    """
    rows = _rows(logits)
    ids = np.asarray(token_ids).reshape(-1).astype(np.int64)
    if ids.shape[0] != rows.shape[0]:
        raise ValueError(
            f"logprob_of: {rows.shape[0]} logit rows but {ids.shape[0]} token ids"
        )
    return _log_softmax(rows)[np.arange(rows.shape[0]), ids].astype(np.float32)


# --------------------------------------------------------------------------- #
# Paged attention, transcribed.
# --------------------------------------------------------------------------- #
def reference_paged_attention_numpy(
    query: Any,
    kv_cache_layer: Any,
    *,
    block_tables: Any,
    seq_lens: Any,
    query_start_loc: Any,
    num_kv_heads: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: Any | None = None,
) -> np.ndarray:
    """The shared formulation, in numpy.

    ``query``          [T, N, D]
    ``kv_cache_layer`` [P, page_size, 2, K, D]
    ``block_tables``   [S * pages_per_seq] flattened page indirection
    ``seq_lens``       [S] total KV length per request AFTER this step's write
    ``query_start_loc``[S + 1] cu_seqlens over the ragged query

    Identical to ``attention/paged.py`` on both sides:

        q_pos[t]    = seq_lens[s] - q_len[s] + (t - query_start_loc[s])
        valid[t, j] = (j <= q_pos[t]) and (j < seq_lens[s])

    Scores and the PV product accumulate in float32 and the result is cast back
    to the query dtype -- the fp32 discipline is load-bearing, not tidiness:
    accumulating in bf16 reproduces every fp32 test in the suite while diverging
    ~0.6% relative at the serving dtype.

    Refuses padded query tokens for the same reason both references do: numpy
    would clamp an out-of-range gather rather than raise, so the padded token
    would silently be attributed to the last sequence.
    """
    q = np.asarray(query)
    q_dtype = q.dtype
    q32 = as_f32(q)
    cache = as_f32(kv_cache_layer)
    qsl = np.asarray(query_start_loc).astype(np.int64).reshape(-1)
    lens = np.asarray(seq_lens).astype(np.int64).reshape(-1)
    bt = np.asarray(block_tables).astype(np.int64).reshape(-1)

    num_tokens, num_heads, head_dim = q32.shape
    page_size = cache.shape[1]
    num_seqs = lens.shape[0]
    if bt.size % num_seqs:
        raise ValueError(
            f"block_tables of size {bt.size} does not divide into {num_seqs} rows"
        )
    pages_per_seq = bt.size // num_seqs
    max_kv = pages_per_seq * page_size

    if num_tokens != int(qsl[-1]):
        raise ValueError(
            f"reference_paged_attention_numpy got {num_tokens} query tokens but "
            f"query_start_loc[-1] is {int(qsl[-1])}: padded query tokens are not "
            "supported (deferred to the `ragged` mode, as on both reference sides)."
        )
    longest = int(lens.max()) if lens.size else 0
    if longest > max_kv:
        raise ValueError(
            f"reference_paged_attention_numpy: seq_lens max is {longest} but the "
            f"block table only reaches pages_per_seq * page_size = {pages_per_seq} "
            f"* {page_size} = {max_kv}; this request is under-allocated."
        )

    tok = np.arange(num_tokens)
    seq_of_tok = np.searchsorted(qsl[1:], tok, side="right")

    gathered = cache[bt.reshape(num_seqs, pages_per_seq)]
    gathered = gathered.reshape(num_seqs, max_kv, 2, num_kv_heads, head_dim)
    keys = gathered[:, :, 0][seq_of_tok]                 # [T, max_kv, K, D]
    values = gathered[:, :, 1][seq_of_tok]

    q_start = qsl[seq_of_tok]
    q_len = qsl[seq_of_tok + 1] - q_start
    q_pos = lens[seq_of_tok] - q_len + (tok - q_start)

    reps = num_heads // num_kv_heads
    keys = np.repeat(keys, reps, axis=2)                 # [T, max_kv, N, D]
    values = np.repeat(values, reps, axis=2)

    scores = np.einsum("tnd,tknd->tnk", q32, keys) * float(sm_scale)

    kv_pos = np.arange(max_kv)[None, :]
    valid = (kv_pos <= q_pos[:, None]) & (kv_pos < lens[seq_of_tok][:, None])
    if sliding_window is not None:
        # `>` and not `>=`: the window is [q_pos - w + 1, q_pos], exactly w
        # positions with the query itself included.
        valid = valid & (kv_pos > (q_pos[:, None] - int(sliding_window)))
    scores = np.where(valid[:, None, :], scores, np.finfo(np.float32).min)

    # ATTENTION SINKS: one learned logit per head that joins the softmax
    # DENOMINATOR and has no value row, so the head can attend to "nothing" and
    # emit a small output instead of being forced to distribute a full unit of
    # probability mass over the real keys. gpt-oss's `attn.sinks` is [N]; a family
    # without them passes None and this block is inert, which is why core_llama3's
    # frozen unit is bit-unchanged by its presence.
    #
    # It is NOT a key with a zero value vector: that would also contribute to the
    # numerator's normalisation in the same way but would additionally be visible
    # as an extra column here. Folding it into the denominator only is what makes
    # `out` shrink toward zero as the sink logit grows, which is the behaviour the
    # checkpoint was trained with.
    peak = scores.max(axis=-1, keepdims=True)
    if sinks is not None:
        sink = as_f32(np.asarray(sinks)).reshape(1, -1, 1)
        if sink.shape[1] != num_heads:
            raise ValueError(
                f"reference_paged_attention_numpy got {sink.shape[1]} sink "
                f"logit(s) for {num_heads} query head(s): `sinks` is per-HEAD, "
                "one scalar for each of the query heads.")
        peak = np.maximum(peak, sink)
    exp = np.exp(scores - peak)
    denom = exp.sum(axis=-1, keepdims=True)
    if sinks is not None:
        denom = denom + np.exp(sink - peak)
    probs = exp / denom
    out = np.einsum("tnk,tknd->tnd", probs, values)
    return out.astype(q_dtype)


# --------------------------------------------------------------------------- #
# Parameter reduction.
# --------------------------------------------------------------------------- #
def param_stats(value: Any) -> dict[str, Any]:
    """Layout-invariant summary of one parameter tensor.

    ``norm`` is the L2 norm in float64 -- transposition- and
    permutation-invariant, so it is indifferent to the [in, out] / [out, in]
    convention and to how a candidate nests its tree, while still moving by
    percent-to-orders for an unloaded, duplicated, truncated or wrongly-bitcast
    tensor. ``shape_sorted`` is carried alongside ``shape`` so a shape comparison
    can be made transpose-tolerant without re-deriving it at every call site.

    Read back by ``compare.compare_param_norms`` (``norm`` + ``finite``) and by
    ``compare.compare_param_spec`` (``shape``).
    """
    arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        arr = arr.astype(np.float32)
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    finite = bool(np.all(np.isfinite(flat))) if flat.size else True
    # A non-finite tensor is a REPORTABLE state (``finite: False`` is what
    # compare_param_norms fails on), not a warning to emit: reducing over an inf
    # legitimately produces nan here, so silence the numpy warning rather than
    # letting a corrupt candidate spray RuntimeWarnings across the grade log.
    with np.errstate(invalid="ignore", over="ignore"):
        return {
            "norm": float(np.sqrt(np.sum(np.square(flat)))) if flat.size else 0.0,
            "mean": float(flat.mean()) if flat.size else 0.0,
            "std": float(flat.std()) if flat.size else 0.0,
            "abs_mean": float(np.abs(flat).mean()) if flat.size else 0.0,
            "count": int(flat.size),
            "finite": finite,
            "shape": [int(d) for d in arr.shape],
            "shape_sorted": sorted(int(d) for d in arr.shape),
        }


# --------------------------------------------------------------------------- #
# Device resolution (in-sandbox helper; imports jax lazily so this module stays
# importable in the torch-free, jax-free freezer toolbox).
# --------------------------------------------------------------------------- #
def resolve_jax_device(spec: str | None = None):
    """The device every probe runs on, or ``None`` to let JAX choose.

    ``spec`` is a ``"platform:index"`` string (``"gpu:0"``) or ``None``. An
    unmatched spec RAISES rather than falling back to CPU: the whole class is
    pinned to one H100 and a silent CPU fallback would produce
    numbers that look like a parity failure.
    """
    import jax

    if not spec:
        return None
    platform, _, index = str(spec).partition(":")
    idx = int(index) if index else 0
    devices = [d for d in jax.devices() if d.platform == platform]
    if not devices:
        raise RuntimeError(
            f"no JAX device of platform {platform!r} is available (have: "
            f"{sorted({d.platform for d in jax.devices()})}); refusing to fall "
            "back to another platform silently"
        )
    if idx >= len(devices):
        raise RuntimeError(
            f"device {spec!r} requested but only {len(devices)} {platform} "
            "device(s) are visible"
        )
    return devices[idx]
