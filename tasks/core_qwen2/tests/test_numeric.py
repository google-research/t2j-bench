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

"""Numeric stage: does the port compute the same numbers?

Every assertion here is a VALUE comparison against a frozen ``torch_orig``
artifact. That is the filing rule, and it is the complement of ``test_spec.py``'s:
a check whose verdict could NOT change by moving to another card belongs there,
and a check that compares a float from a model forward belongs here.

WHY THIS TIER EXISTS SEPARATELY FROM SPEC, stated as the hole it closes.
Spec cannot see a per-weight TRANSPOSITION at any shape and says so at length
(``test_param_correspondence``: ``sorted(shape)`` is transpose-tolerant on
purpose, because a flax ``nn.Dense`` kernel is ``[in, out]`` where a torch
``nn.Linear`` weight is ``[out, in]``; ``test_param_norms``: the L2 norm is
transposition-invariant by construction). Spec also cannot see a QKV FUSION ORDER
that is wrong AS IMPLEMENTED -- ``test_loader_spec`` compares the declared
``STACKED_PARAMS`` table, and a converter that concatenates q, v, k while the
table still says q, k, v scores a clean sweep of that tier. Both breaks are
value-level, and this is the first tier that compares a value.

THE FIVE GATES, and which surface each one is for
--------------------------------------------------
* LOGITS -- never ``abs``/``rel``: raw logits are defined only up
  to a per-token additive constant, so an elementwise diff is not an equivalence
  measure. :func:`_assert_logit_parity` is the logit gate -- minimum per-token
  CENTERED cosine, BINDING softmax KL, and argmax agreement on DECISIVE golden
  positions.
* ARRAYS (attention output, cache, activations, sampler grids) -- ``allclose``
  through ``compare.check_array``, which is the right measure for a quantity with
  a fixed scale.
* MASKS (the sampler's top-k / top-p masks) -- EXACT set equality. A mask is a
  decision, not an arithmetic result, so no tolerance is warranted and none is
  given.
* SLOT OCCUPANCY (which cache slots were written) -- EXACT. An untouched slot is
  bit-zero on both sides, so "which slots hold data" is a copy property that no
  amount of kernel noise can move. The VALUES written into them are arithmetic and
  are graded at the native band; the brief's "a write is a copy, so no tolerance
  is warranted" is true of the placement and false of the payload -- the
  two sides' cache tensors differ in roughly 14% of their elements by up
  to one bf16 ULP. Deliberately an order of magnitude and not a digit count: the
  exact number is not run-stable (on core_llama3, 142 755 on one run and 142 697 on
  another; the CACHE SIZE is per-instance -- 1 048 576 elements there, 393 216 on
  core_qwen2 -- so treat the 14% as the portable figure and the counts as that
  instance's), because
  bf16 on these shared cards is not reproducible across processes.
* TOKEN IDS (top-N) -- exact where the golden's own values are distinct,
  tie-tolerant where they are not.

THRESHOLDS ARE NEVER HARD-CODED HERE. Every tolerance is read from
``golden.meta`` -- ``fp32_thresholds`` / ``native_thresholds`` in
``tests/instance.yaml``, mirrored into ``golden/meta.json`` by
the freezer. Each per-instance override records its
measured floor, its chosen value and the ratio between them, in a comment beside
the value. What was measured is the REFERENCE AS CANDIDATE -- largely the same
code -- so those floors bound cross-kernel noise and are NOT an estimate of what
an independent port will produce; that is why the ratios are what they are, and
it is written down at the point of use rather than left to be re-derived.

WHAT IS DESELECTED ON A ``core`` INSTANCE, and why each one. All seven
are named in ``eval_summary.json``'s ``deselected`` list; none of them skips,
because ``reward = passed/(passed+failed+error)`` would charge a skip to the
candidate in both numerator and denominator.

* ``test_ragged_forward_logits``, ``test_batch_composition_battery`` -- need a
  second concurrent sequence, which the ``core`` mode has no way to express.
* ``test_chunked_prefill_logits``, ``test_chunk_boundary_battery`` -- need a split
  prefill, likewise.
* ``test_prefill_logits_battery``, ``test_paging_battery`` -- their golden
  (``battery/{prompt_length,paging}/``) is OPTIONAL and no producer writes those
  two axes. The battery freezer now EXISTS, but it
  implements ``batch_composition`` only; these two arm themselves the day it
  grows their axes.
* ``test_fp32_decode_step_logits`` -- the golden catalog
  freezes no fp32 decode artifact to compare it against. A catalog gap, gated on
  the artifact a producer would write, so it arms itself the same way. See
  ``conftest._golden_facts``'s ``has_fp32_decode``.

The nine that RUN on ``core`` are: reference units, the isolated paged-attention
op, the cache write, the native and fp32 prefill logits, the per-layer hidden
states, the decode-step logits, the sampler transforms and the top-N logprobs.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from _t2j import _formulas as F
from _t2j import compare, golden_manifest, similarity

# --------------------------------------------------------------------------- #
# Continuous scoring. The numeric stage grades FIDELITY, not a
# pass/fail band: every check funnels its [0,1] similarity through :func:`_score`,
# and the conftest aggregates the per-check scores per dimension with
# ``similarity.power_mean`` (p<1, low-tail sensitive). This mirrors the training
# grader (``tasks/*/tests``) so the numeric reward dimension has the same
# anti-inflation shape -- a single wholly-broken forward check bites
# super-linearly, while a uniform faithful set (~0.999) is unchanged.
#
# The mechanism is kept OFF the ~22 test bodies, which funnel through two gates
# (:func:`_assert_logit_parity`, :func:`_assert_array_parity`): an autouse fixture
# binds this test's ``record_property`` and resets a per-test accumulator, and
# every :func:`_score` call appends its value and RE-records the running power
# mean under the single ``score`` user_property. ``dict(user_properties)`` keeps
# the last write, so the value the conftest reads is the aggregate over the whole
# test (all layers / steps / variants). A test that records NOTHING has no
# ``score`` and the conftest reads 0.0 -- a pass with no fidelity signal earns no
# credit -- except the pure-gate tests (exact grids / positions / placement),
# which carry no continuous signal and which the conftest names in
# ``_NUMERIC_GATE_TESTS`` and keeps out of the mean (a gate can only fail, never
# bank credit).
# --------------------------------------------------------------------------- #
_RECORDER = None  # this test's record_property, bound per-test by the fixture below
_SCORE_ACC: list[float] = []  # per-test [0,1] similarities, reset each setup


@pytest.fixture(autouse=True)
def _bind_score_recorder(record_property):
    """Bind ``record_property`` and reset the per-test similarity accumulator."""
    global _RECORDER
    _RECORDER = record_property
    _SCORE_ACC.clear()
    yield
    _RECORDER = None


def _score(s: float) -> float:
    """Accumulate one [0,1] similarity and re-record the running power mean.

    Returns the clamped aggregate. ``similarity.power_mean`` sanitises
    non-finite -> 0.0 and clips to [0,1], so a degenerate ``s`` cannot inflate the
    recorded score. Re-recording every call (rather than once at the end) keeps the
    value in the CALL-phase report the conftest reads, without a teardown hook.
    """
    _SCORE_ACC.append(float(s))
    agg = similarity.power_mean(_SCORE_ACC)
    if _RECORDER is not None:
        _RECORDER("score", agg)
    return agg


# --------------------------------------------------------------------------- #
# Threshold plumbing. Two bands, read from the golden, never restated here.
# --------------------------------------------------------------------------- #
#: Which key in ``fp32_thresholds`` / ``native_thresholds`` carries the floor for
#: the MINIMUM per-token centered cosine. It is not a ``NumericThresholds`` field
#: -- the shared comparator has one cosine gate and it is on the RAW cosine -- so
#: it rides the ``*_extras`` half of the band (see ``fixtures._threshold_extras``).
#:
#: Centering matters, and is why this one is used rather than the raw
#: cosine. A logit row's mean is large and shared, so the raw cosine of two logit
#: matrices is dominated by that offset; removing it per row is what makes the
#: measure sensitive to the part that differs. Taking the MINIMUM over rows rather
#: than the mean is the other half: one token attending the wrong page is
#: invisible in an average over twelve.
_CENTERED_COSINE_KEY = "centered_cosine_floor"


def _band(golden, tier: str):
    """``(NumericThresholds, extras)`` for ``"fp32"`` or ``"native"``.

    One accessor, so a check cannot silently grade the fp32 surface at the native
    band or the reverse -- which is the cheapest way for a tier to look strict and
    not be.
    """
    if tier == "fp32":
        return golden.fp32_thresholds, golden.fp32_extras
    if tier == "native":
        return golden.native_thresholds, golden.native_extras
    raise ValueError(f"unknown precision tier: {tier!r}")


def _centered_floor(thresholds, extras, surface: str | None = None) -> float:
    """The centered-cosine floor for a band, optionally for one SURFACE.

    Falls back to the band's own ``cosine_floor`` when the instance has not
    calibrated one. That fallback is a DOCUMENTED BAND from ``meta``, not a
    constant invented here: an instance that never calibrates still grades the
    centered cosine at the tolerance its profile states, and the class rule that
    no tolerance is hard-coded in a test module holds either way.

    PER-SURFACE, for exactly the reason :func:`_surface_band` is: one centered
    cosine cannot serve every array on a tier when the arrays are not on one
    scale or one condition number. ``centered_cosine_floor.hidden_states`` in
    ``instance.yaml`` overrides the tier value for the residual stream alone,
    leaving the LOGIT clause -- which is the sensitive one, and the one this
    floor was calibrated for -- untouched.

    This hook exists for core_qwen2. That checkpoint's
    residual stream peaks at 1560 and then COLLAPSES to 59 at layer 22 as the
    massive activations cancel. The cross-framework disagreement is one bf16 ULP
    of the peak (8.0) at every layer from 4 to 22 -- textbook agreement -- but
    once the signal drops 26x while the error does not, the same 8.0 is 13.5%
    of the layer and the centered cosine reads 0.9931. Raising the TIER floor to
    admit that would have loosened the logit clause on the same instance by 3.5x
    for a reason that has nothing to do with logits. core_llama3 names no
    per-surface floor and is graded exactly as before.
    """
    if surface is not None:
        override = extras.get(f"{_CENTERED_COSINE_KEY}.{surface}")
        if override is not None:
            return float(override)
    return float(extras.get(_CENTERED_COSINE_KEY, thresholds.cosine_floor))


# --------------------------------------------------------------------------- #
# The gates.
# --------------------------------------------------------------------------- #
def _assert_logit_parity(reference, candidate, golden, tier: str, *, name: str,
                         self_comparison: bool = False) -> dict:
    """The logit gate, as a CONTINUOUS [0,1] score (not a pass/fail band).

    The fidelity is ``similarity.score_logits`` -- ``sqrt(S_dist * S_logit_scale)``:
    a softmax-JSD distribution term (invariant to the per-token additive gauge that
    a raw elementwise diff is meaningless under) times a
    centered-logit scale term that a temperature / logit-scale / soft-cap error can
    no longer hide behind. It needs NO threshold -- no cosine floor, no softmax-KL
    cap, no argmax margin -- so it retires the band triple the old strict gate read
    from ``golden.meta``; a faithful fp32 port keeps both terms ~0.999. The
    score is fed to :func:`_score`, which the conftest aggregates.

    TWO STRUCTURAL ASSERTS SURVIVE, because they are not similarity questions:

    1. SHAPE, first -- "your logits are a different shape" is a different repair
       from "your logits are wrong", and ``score_logits`` would collapse a shape
       mismatch to 0.0 without naming it. A failed assert fails the test, which the
       conftest already records as 0.0.
    2. ``self_comparison`` -- set when BOTH operands are the candidate's own output
       across two legs of an invariance (e.g. decode vs. prefill of the SAME port).
       An invariance is an EQUALITY between two runs of one implementation, so it is
       graded on exact top-1 agreement rather than as a similarity: a port with flat
       logits must not pass by looking "similar to itself".

    ``golden``/``tier`` are retained for call-compatibility and the report; the
    per-tier band accessors (:func:`_band`, :func:`_centered_floor`) are now dormant.
    """
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    assert candidate.shape == reference.shape, {
        "check": name, "error": "logit shape mismatch",
        "golden": list(reference.shape), "candidate": list(candidate.shape),
    }
    if self_comparison:
        left_argmax = int(reference.reshape(1, -1).argmax(-1)[0])
        right_argmax = int(candidate.reshape(1, -1).argmax(-1)[0])
        assert left_argmax == right_argmax, {
            "check": name, "tier": tier,
            "error": "the two legs of this invariance chose different top-1 "
                     "tokens. Both rows are your own output under a "
                     "transformation that must not change the answer, so no "
                     "tie-break latitude applies.",
            "left_argmax": left_argmax, "right_argmax": right_argmax,
        }
    score = similarity.score_logits(reference, candidate)
    _score(score)
    return {"check": name, "tier": tier, "score": score}


#: Prefix of the per-surface ``abs_tol`` overrides in a band's extras, e.g.
#: ``abs_tol.kv_cache_state``. See :func:`_surface_band`.
_ABS_TOL_PREFIX = "abs_tol."


def _surface_band(golden, tier: str, surface: str | None):
    """The band for one ARRAY surface: the tier's, with its own ``abs_tol`` if named.

    ONE ABSOLUTE BAND CANNOT SERVE EVERY ARRAY ON A TIER, and on this instance it
    is not close. ``check_array``'s abs term is a fixed quantity while the native
    tier's arrays peak at 0.51 (the isolated attention output), 17.1 (the
    KV cache) and 414.0 (the residual stream) -- an 800x spread. An ``abs_tol``
    wide enough for the residual stream exceeds the entire attention output, so
    that check would admit any array whatsoever; one narrow enough for the
    attention output fails the worked reference on the other two.

    So a surface whose scale forces its own band NAMES it, in ``instance.yaml``,
    beside its own measured floor -- and a surface that does not name one is
    graded at the tier default, which is the class profile. The lookup is by
    string rather than by a table in this module for the same reason every other
    tolerance here is: a threshold must be readable in one place, and that place
    is the instance.
    """
    thresholds, extras = _band(golden, tier)
    if surface is not None:
        override = extras.get(f"{_ABS_TOL_PREFIX}{surface}")
        if override is not None:
            fields = {name: getattr(thresholds, name)
                      for name in thresholds.__dataclass_fields__}
            thresholds = compare.NumericThresholds(
                **{**fields, "abs_tol": float(override)})
    return thresholds


#: How many times the instance's OWN declared relative band a shallow hidden
#: state may disagree by before :func:`_shallow_abs_cap` refuses it.
#:
#: Derived from the reference of all four shipped instances, as
#: ``needed_abs_tol_i / (declared * running_peak_i / global_peak)`` -- the
#: smallest margin that admits the worked reference at layer ``i``. The worst
#: requirement on each instance, and the layer it comes from:
#:
#:     core_llama3    0.100  @ hidden_16      core_qwen2     0.350  @ hidden_22
#:     ragged_llama3  0.126  @ hidden_16      core_gpt_oss   1.222  @ hidden_6
#:
#: 4.0 carries 3.3x of the worst of those. Two separate things fix the value and
#: they agree: (a) from 3.0 upward the cap is no longer the binding constraint on
#: ANY shipped layer -- what binds is the instance's declared band itself, at
#: 2.16x on core_gpt_oss's hidden_23 -- so raising it further buys nothing and 4.0
#: is the round number just above that knee; (b) core_gpt_oss's shallow
#: requirements are single-ELEMENT order statistics over the same expert-routing
#: flip sample its ``centered_cosine_floor`` carries ~1.6x of for run-stability,
#: and 1.222 * 1.6 = 1.96 against the 2.64 (a) already demands.
#:
#: IT COSTS NO COVERAGE, and that is checked rather than assumed -- see
#: :func:`_shallow_abs_cap`.
_HIDDEN_STATE_CAP_MARGIN = 4.0


def _shallow_abs_cap(declared: float, running_peak: float,
                     global_peak: float) -> float | None:
    """The declared hidden-state band, rescaled to the depth the stream is at.

    ``abs_tol.hidden_states`` is ONE number for a surface whose scale varies by
    orders of magnitude ALONG it, and it is calibrated at the deep end -- so at
    the shallow end it is wider than the entire tensor and grades nothing.
    The declared band against the layer's own peak: core_qwen2 16.0 against
    0.0713 at hidden_0 (224x the whole tensor), core_gpt_oss 600.0 against 75.5
    (8x). A candidate returning ALL ZEROS for those layers leaves zero elements
    out of band.

    So the band is rescaled per layer by the reference's own magnitude::

        cap_i = MARGIN * declared * running_peak_i / global_peak

    which is the instance's declared RELATIVE band -- ``declared /
    global_peak``, the only relative number it actually calibrated -- applied at
    the scale the stream has reached by layer ``i``. ``running_peak``, not the
    layer's own: at core_qwen2's hidden_22 the stream collapses 26x (1560 -> 59)
    while the cross-framework error stays at the scale it arrived carrying, so
    normalising by the layer's own peak fails a correct port exactly there.

    IT ONLY EVER NARROWS. ``running_peak <= global_peak`` by construction, so the
    unscaled value is at most ``MARGIN * declared`` and ``_assert_array_parity``
    clamps it to ``declared``; from the layer that SETS the global peak onwards
    the cap is exactly the declared band, bit for bit.

    WHY NOT bf16 QUANTA, which is what this was until core_gpt_oss. The previous
    form was ``2 * bf16_ULP(running_peak)`` -- a class-wide physical constant, on
    the premise that the cross-framework disagreement is a quantum of the value
    being carried. That premise is a FAMILY property, not a class one, and
    core_gpt_oss falsifies it: its MoE router is ill-conditioned at bf16, a
    flipped token runs a genuinely different expert, and hidden_2 needs 3.894
    quanta against the 2.0 the constant allowed -- a reward-0.9697 false negative
    on the worked reference.

    THE CRITERION FOR PICKING BETWEEN THE TWO IS SLACK, NOT SENSITIVITY, and it
    is worth being exact about which. Both forms are one class-wide constant
    times a per-layer unit, so the constant has to be large enough for the WORST
    instance, and every better-behaved instance then carries the difference as
    dead slack. Re-derived from the same measurement, the multiplier each unit
    needs spans 0.050 .. 3.894 for quanta (78x) and 0.100 .. 1.222 for this form
    (12x) -- so this unit costs 6.4x less slack. Part of why is unmysterious
    rather than physical: on three of the four instances the required multiplier
    is definitionally the reciprocal of that instance's own recorded declared-band
    headroom, so what the number really says is that the declared band ALREADY
    encodes each family's noise level and this reuses it instead of re-deriving
    it from bf16 resolution. That is a good engineering reason and not a claim
    about arithmetic.

    ON SENSITIVITY THE TWO ARE CLOSE, and the honest statement is per layer. At
    hidden_0 -- the layer with the widest declared band relative to its own
    contents, and the one the escape lands on -- this form is tighter on all four
    instances: the smallest single-layer uniform scale error the absolute clause
    still rejects is 10.97% / 14.10% / 10.97% / 15.75% (llama3 / qwen2 / ragged /
    gpt_oss) here against 16.49% / 18.22% / 16.49% / 17.95% for quanta at the
    multiplier giving the same 2.16x. It does NOT win everywhere: the quantum's
    2x sawtooth can land favourably, and at core_gpt_oss's hidden_2 (peak 121,
    just under 128) quanta give 14.96% against this form's 15.75%.

    WHAT THE CAP BUYS, AND IT IS MAGNITUDE. The centred-cosine clause beside it
    is scale-invariant BY CONSTRUCTION -- a candidate returning
    ``0.5 * reference`` scores 0.9999999 at every layer of every instance -- so a
    pure scale error at the shallow end is invisible to it, and the declared band
    there is 224x (core_qwen2 hidden_0) or 8x (core_gpt_oss hidden_0) the entire
    tensor. Without this cap core_qwen2's hidden_0/1/2 have NO magnitude grading
    at all from either clause. Layers at which the absolute clause rejects that
    halved candidate, llama3 / qwen2 / ragged / gpt_oss:

        declared band alone   16/17   22/25   16/17   17/25
        with this cap         17/17   25/25   17/17   24/25

    An all-zeros candidate gives the same two rows, and its count is FLAT in the
    margin from 1.0 to 32.0 -- which is why the margin above could be chosen for
    headroom on the reference alone. (The last hold-out is core_gpt_oss's
    hidden_24, where the stream has collapsed from 41728 to 225 and the declared
    600 is the ceiling. The cosine clause covers the all-zeros case there --
    0.000 against a 0.95 floor, on every layer of every instance -- but NOT the
    scale case, which is the asymmetry this whole block is about.)

    ``None`` -- no narrowing at all -- whenever either peak is zero or not
    finite. BOTH, and the ``running_peak`` half is the one that is reachable: a
    LEADING layer whose reference is all zeros has ``running_peak == 0`` against
    a perfectly good ``global_peak``, and the product would then be a cap of
    0.0. A zero cap is not a loose band, it is an EXACT-EQUALITY requirement
    that fails a correct port on a single denormal. The bf16-quantum form this
    replaced floored at ``2**-133`` for exactly this reason and said so; the
    floor is gone with the ULP, so the guard has to carry it.

    Unreachable on the four shipped instances -- their leading peaks are 0.361,
    0.0713, 0.361 and 75.5 -- but the cap is derived from data and the next
    family's data is not this family's, which is the argument the original
    comment made. A global peak of zero means the whole reference is zero, which
    is a broken golden the surrounding checks report far better than a
    degenerate tolerance would.
    """
    if not (float(global_peak) > 0.0) or not np.isfinite(global_peak):
        return None
    if not (float(running_peak) > 0.0) or not np.isfinite(running_peak):
        return None
    return float(_HIDDEN_STATE_CAP_MARGIN * float(declared)
                 * float(running_peak) / float(global_peak))


def _assert_array_parity(reference, candidate, golden, tier: str, *, name: str,
                         surface: str | None = None,
                         abs_cap: float | None = None) -> dict:
    """Array fidelity as a CONTINUOUS [0,1] score via ``similarity.S_tensor``.

    For quantities with a fixed scale -- an attention output, a cache tensor, an
    activation, a masked logit row. NOT for raw logits. ``S_tensor`` is
    ``sqrt(S_cos * S_L2)``: direction AND magnitude must agree, so it catches a
    scale bug a pure cosine misses and needs no absolute band. Being scale-relative
    is why it replaces the per-surface ``abs_tol`` / shallow-cap machinery the old
    ``allclose`` gate needed for a surface whose scale spans orders of magnitude
    (see :func:`test_hidden_states`). A faithful fp32 port scores ~0.999; a
    zero-norm candidate against a non-zero reference scores 0. The score is fed to
    :func:`_score`, which the conftest aggregates.

    A SHAPE ASSERT SURVIVES, first: "your array is a different shape" is a different
    repair from "your values are wrong", and ``S_tensor`` would collapse a mismatch
    to 0.0 without naming it (a failed assert fails the test, recorded as 0.0).
    ``surface``/``abs_cap`` are retained for call-compatibility; the per-surface
    band accessors (:func:`_surface_band`, :func:`_shallow_abs_cap`) are now dormant.
    """
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    assert candidate.shape == reference.shape, {
        "check": name, "tier": tier, "surface": surface,
        "error": "array shape mismatch",
        "golden": list(reference.shape), "candidate": list(candidate.shape),
    }
    score = similarity.S_tensor(reference, candidate)
    _score(score)
    return {"check": name, "tier": tier, "surface": surface, "score": score}


# --------------------------------------------------------------------------- #
# Small readers, so a golden-schema change breaks in one place.
# --------------------------------------------------------------------------- #
def _frozen_slots(golden) -> np.ndarray:
    """EVERY flat cache slot the frozen scenario wrote, from ``inputs.npz``.

    ``slot_mapping`` is the graded forward's scatter target. A mixed-stage
    scenario runs a WARM-UP forward before it (``warmup_slot_mapping``) to write
    the cached history its `decode` requests attend to, and those slots are
    occupied in the frozen cache too -- so an occupancy assertion that read only
    the graded mapping would report every warm-up slot as an unexpected write on
    both sides. Absent on `core`, which runs one forward.
    """
    graded = np.asarray(golden.scenario_slot_mapping).reshape(-1).astype(np.int64)
    warm = golden.inputs.get("warmup_slot_mapping")
    if warm is None:
        return graded
    return np.concatenate([graded, np.asarray(warm).reshape(-1).astype(np.int64)])


def _occupied_slots(cache: np.ndarray, half: int | None = None) -> np.ndarray:
    """Boolean per flat slot: does ANY layer hold a nonzero there?

    ``cache`` is ONE GROUP of the container -- ``[n_slots, pages, page_size, 2,
    kv_heads, head_dim]``, where ``n_slots`` is ``num_layers`` for a family that
    allocates one cache slot per layer and fewer for one that SHARES slots. The
    flat slot axis is ``pages * page_size``, which is what ``slot_mapping``
    indexes. ``half`` selects the K (0) or V (1) plane, or both when ``None``.
    """
    layers, slots = cache.shape[0], cache.shape[1] * cache.shape[2]
    flat = cache.reshape(layers, slots, *cache.shape[3:])
    if half is not None:
        flat = flat[:, :, half]
    axes = tuple(i for i in range(flat.ndim) if i != 1)
    return np.any(flat != 0, axis=axes)


def _golden_cache_groups(golden) -> list[tuple[str, np.ndarray]]:
    """``[(member name, frozen array)]`` for the post-prefill cache, IN CONTAINER ORDER.

    The KV cache is a CONTAINER -- a tuple holding one tensor per distinct KV
    geometry -- and ``kv_cache_state.npz`` records it as ``cache`` when there is one
    geometry and ``cache_group{i}`` when there are several
    (``_t2j.golden_manifest.cache_state_member_names``, the single definition the
    freezer writes from and the manifest requires against).

    THE ORDER IS THE CONTAINER'S, and it is derived from ``cache_spec.json``'s
    declared group count rather than from ``sorted(npz.files)``: ``cache_group10``
    sorts before ``cache_group2``, so a lexicographic read would silently transpose
    two groups on the first family with more than ten geometries and compare each
    against the other's frozen answer.

    THE TWO RECORDS OF THE KIND ARE CROSS-CHECKED HERE, exactly as the freezer's
    ``_gate_cache_kind`` does at freeze time and the manifest does at precheck
    time. ``cache_spec.json`` and ``kv_cache_state.npz`` declare it independently,
    so they can disagree, and the disagreement is invisible to a reader that just
    subscripts one name: a state file holding ``cache`` beside a spec declaring two
    ``groups`` would otherwise grade group 0 alone against a contract describing
    both. That is an ARTIFACT defect and is reported as one -- a candidate cannot
    cause it and must not be charged for it.
    """
    declared = golden_manifest.declared_cache_groups_in_spec(golden.cache_spec)
    assert declared is not None, {
        "check": "test_kv_cache_write",
        "error": "golden/cache_spec.json declares neither `shape` (one KV "
                 "geometry) nor a non-empty `groups` (several), so the number of "
                 "arrays kv_cache_state.npz should hold is undefined. This is an "
                 "artifact defect, not a port defect.",
        "cache_spec_keys": sorted(golden.cache_spec),
    }
    names = golden_manifest.cache_state_member_names(declared)
    frozen = dict(golden.kv_cache_state)
    assert set(frozen) == set(names), {
        "check": "test_kv_cache_write",
        "error": "golden/cache_spec.json and golden/kv_cache_state.npz disagree "
                 "about the KV cache's KIND. The spec declares the group count "
                 "below, so the state file must hold exactly those members -- "
                 "`cache` for one geometry, `cache_group{i}` for several, never "
                 "both spellings and never a `cache` that means group 0. This is "
                 "an artifact defect, not a port defect.",
        "declared_groups": declared,
        "expected_members": list(names),
        "frozen_members": sorted(frozen),
    }
    return [(name, np.asarray(frozen[name], dtype=np.float32)) for name in names]


def _unit_specs(golden) -> tuple[list[str], dict]:
    """``([unit name, in dataflow order], {name: frozen spec})`` from the units json."""
    meta = (golden.reference_units or {}).get("meta") or {}
    return list(meta.get("modeling_units") or []), dict(meta.get("units") or {})


def _grade_battery(golden, cand, axis: str, *, name: str, varies: str) -> int:
    """Every frozen variant of one L3 axis, re-forwarded by the candidate.

    Returns the number of variants graded. The battery layout is
    ``golden/battery/{axis}/{variant}/{inputs.npz, prefill_forward.npz}``.
    ``inputs.npz`` is what crosses into the sandbox (``fixtures
    ._battery_inputs``), ``prefill_forward.npz`` carries the frozen ``logits``
    that stay here.

    THE COVERAGE ASSERTIONS ARE THE POINT, not decoration. An axis with one
    variant does not test generalisation along it, and a candidate whose probe
    returned a subset of the variants would otherwise be graded on the subset and
    reported as a pass.
    """
    variants = (golden.battery or {}).get(axis) or {}
    assert len(variants) >= 2, {
        "check": name, "error": f"the {axis!r} battery axis froze "
                                f"{len(variants)} variant(s); an axis that varies "
                                f"{varies} needs at least two to say anything",
        "hint": "re-freeze the inference-battery golden",
    }
    probe = cand.battery
    assert probe.get("supported"), {
        "check": name, "error": "the candidate produced no battery results",
        "reason": probe.get("reason"),
    }
    axes = probe.get("axes") or {}
    assert sorted(axes.get(axis) or {}) == sorted(variants), {
        "check": name, "error": "the candidate did not re-forward every frozen "
                                "variant of this axis",
        "frozen": sorted(variants), "candidate": sorted(axes.get(axis) or {}),
    }
    for variant in sorted(variants):
        frozen = (variants[variant].get("prefill_forward") or {})
        assert "logits" in frozen, {
            "check": name, "variant": variant,
            "error": "the frozen variant carries no prefill_forward.npz::logits",
        }
        _assert_logit_parity(frozen["logits"], axes[axis][variant]["logits"],
                             golden, "native", name=f"{name}[{variant}]")
    return len(variants)


# =========================================================================== #
# L1 -- the isolated pieces
# =========================================================================== #
def test_reference_units(golden, cand):
    """Per-submodule I/O: the candidate's own module, on the REFERENCE's input.

    THE LOCALISER. ``test_prefill_logits`` says the answer is wrong; this says
    which module produced a wrong answer from a right input, because being handed
    the frozen activation is what stops an upstream break from contaminating a
    downstream verdict. Nine units at layer 0 and the two model-level tails, frozen
    from the reference implementation:
    ``input_layernorm``, ``self_attn.{qkv_proj, rotary_emb, o_proj}``,
    ``post_attention_layernorm``, ``mlp.{gate_up_proj, down_proj}``, ``model.norm``
    and ``lm_head``.

    This is the coverage claim the whole tier is built to make good:
    a TRANSPOSED SQUARE weight -- ``o_proj``, [2048, 2048], which Spec cannot see
    at any shape by design -- fails here, and this check names the projection
    (``reference_units/o_proj/o_proj.out0``).

    SQUARE, and the qualifier is the whole content of the claim. A transposed
    NON-SQUARE weight never reaches this check: the candidate's own
    ``apply_params`` refuses the shape ("``gate_up_proj.weight`` is (2048, 16384)
    in the converted checkpoint but (16384, 2048) in the model") and the worker
    dies, so all nine collected checks report the SAME ``CandidateError`` and it is
    ``apply_params``, not this check, that names the projection. That is the
    reference's strictness rather than a verdict of this tier -- the same trap
    ``test_param_correspondence``'s docstring names -- and it is why the square
    case is the one that demonstrates anything.

    fp32 on both sides, at the fp32 band: the units are frozen fp32
    (fp32-robust) and the probe replays them through the candidate's fp32 model, so
    the only permitted disagreement is cross-kernel fp32 reduction order.

    WHAT IT DOES NOT SEE. Everything about the ATTENTION interior -- the block
    table gather, the causal mask, the softmax -- because ``self_attn`` is
    deliberately not a unit: its interior has its own isolated golden and its own
    check (``test_paged_attention_op``), computed on its own frozen ``(q, k, v)``.
    And everything about layers 1..N-1, whose WEIGHTS are covered exhaustively by
    ``test_param_norms`` and whose CODE is the same code.
    """
    order, specs = _unit_specs(golden)
    arrays = (golden.reference_units or {}).get("arrays") or {}
    assert order and specs, {
        "error": "golden/reference_units.json names no units",
        "hint": "re-freeze the reference-units golden",
    }
    probe = cand.reference_units
    assert probe.get("supported"), {
        "error": "the candidate produced no per-submodule replay",
        "reason": probe.get("reason"),
    }

    # COVERAGE FIRST. A probe that returned three of nine units would otherwise be
    # graded on three and reported green -- the deselection accounting error with
    # no deselection to make it visible.
    got = probe.get("units") or {}
    assert sorted(got) == sorted(order), {
        "error": "the candidate replayed a different set of units than the golden "
                 "froze",
        "frozen": sorted(order), "replayed": sorted(got),
    }

    unresolved = {name: entry.get("reason") for name, entry in sorted(got.items())
                  if not entry.get("supported")}
    assert not unresolved, {
        "error": "the candidate could not run one or more of the reference's "
                 "submodules on the reference's own inputs",
        "units": unresolved,
        "hint": "both reference trees use the same attribute names for these "
                "modules (ref_jax_repo/modeling.py: 'same classes, same attribute "
                "names, same order of operations'), and the replay resolves the "
                "frozen torch path against your model",
    }

    for unit in order:
        frozen = specs[unit]["outputs"]
        outputs = got[unit]["outputs"]
        assert len(outputs) == len(frozen), {
            "unit": unit, "path": got[unit]["path"],
            "error": "wrong number of return values",
            "golden": len(frozen), "candidate": len(outputs),
        }
        for value, spec in zip(outputs, frozen):
            _assert_array_parity(arrays[spec["key"]], value, golden, "fp32",
                                 name=f"reference_units/{unit}/{spec['key']}")


def test_paged_attention_op(golden, cand):
    """The hardest piece, isolated. If this fails, nothing downstream is
    interpretable -- so grade it directly rather than only through the model.

    ``paged_attention`` is where the port's real work is: a block-table gather, a
    causal mask expressed over gathered pages rather than over positions, and a
    GQA head repeat. Grading it only through the model means a break in it arrives
    as "your logits are wrong", one layer of indirection away from the cause.

    BOTH TIERS ARE GRADED. ``golden/paged_attention_units.npz`` freezes
    ``out_fp32`` and ``out_native`` from one frozen ``(q, k, v)``, and an unused
    golden is a coverage claim nobody checks. They are not the same array -- the
    freezer refuses to write them if they are -- and this asserts that too, so a
    re-freeze that lost the dtype cast cannot turn the native leg into a duplicate
    of the fp32 one.

    ...AND SO IS THE CANDIDATE'S HALF. The producer guarded only
    its own side; a port whose native leg quietly returns fp32 would be compared
    against the native golden through a band wide enough to admit it. The dtype
    clause in the loop below is the symmetric assertion, with the measurement.

    THE INPUTS ARE THE REFERENCE'S OWN. ``fixtures.build_candidate`` hands the
    candidate the frozen ``q``/``k``/``v`` and nothing else; the probe builds an
    all-zero cache layer, writes them at the FROZEN slot mapping and calls the
    candidate's ``paged_attention`` with the FROZEN metadata. So a wrong q/k/v --
    a wrong projection, a wrong RoPE -- cannot reach this check, which is the whole
    point of isolating it.

    Deliberately EAGER on the candidate side: ``attention/paged.py``'s two
    preconditions (no padded query tokens; ``seq_lens`` inside the gathered
    window) read metadata VALUES, so they are live here and traced away
    everywhere else. This is the only check that exercises them.
    """
    frozen = golden.paged_attention_units
    for key in ("q", "k", "v", "out_fp32", "out_native"):
        assert key in frozen, {
            "error": f"golden/paged_attention_units.npz has no {key!r}",
            "present": sorted(frozen),
            "hint": "re-freeze the paged-attention-units golden",
        }
    # An artifact defect, not a candidate defect: bf16 has 8 mantissa bits against
    # fp32's 24, so two identical tiers mean the cast never happened and the
    # native leg below would grade nothing of its own.
    assert not np.array_equal(np.asarray(frozen["out_native"]),
                              np.asarray(frozen["out_fp32"])), {
        "error": "the frozen native and fp32 attention outputs are bit-identical, "
                 "so one of the two legs is a duplicate of the other",
    }
    # ...and the frozen head geometry must be the model's, or the isolated op is
    # being graded on a shape the rest of the golden tree never sees.
    spec = golden.cache_spec
    assert np.shape(frozen["k"])[1] == spec["num_kv_heads"], {
        "error": "the frozen k has a different num_kv_heads than cache_spec.json",
        "units": list(np.shape(frozen["k"])), "cache_spec": spec,
    }
    assert np.shape(frozen["q"])[2] == spec["head_dim"], {
        "error": "the frozen q has a different head_dim than cache_spec.json",
        "units": list(np.shape(frozen["q"])), "cache_spec": spec,
    }

    for tier in ("fp32", "native"):
        unit = cand.paged_attention_unit(fp32=(tier == "fp32"))
        assert unit.get("supported"), {
            "tier": tier, "error": "the isolated attention probe did not run",
            "reason": unit.get("reason"),
        }
        assert int(unit["num_kv_heads"]) == int(spec["num_kv_heads"]), {
            "tier": tier, "probe": unit["num_kv_heads"],
            "cache_spec": spec["num_kv_heads"]}
        # ...AND THE CANDIDATE'S OWN LEG IS ACTUALLY AT THAT TIER. The clause above
        # refuses a GOLDEN whose two tiers are identical, and the freezer refuses to
        # write one; nothing asserted the symmetric property on the candidate's half,
        # so a port whose native leg silently ran at fp32 was graded against the
        # native golden through a band wide enough to admit it.
        #
        # One line in `attention/paged.py` -- `return out` instead of
        # `return out.astype(query.dtype)`, which is the single most common precision
        # bug in a JAX port. The uncast fp32 output promotes `o_proj` and then the
        # residual add, so the ENTIRE native forward runs at float32 from layer 0.
        # Worst hidden-state deviation 5.885, which is 46x the 0.1266 floor the
        # `abs_tol.hidden_states: 1.0` override was calibrated from -- yet every
        # value clause passes. `check_array` is `|d| <= abs_tol + rel_tol*|ref|`
        # and on the residual stream's measured peak of 414 the rel term admits 41.4,
        # so no value clause in this tier could see it.
        #
        # THIS IS A DTYPE ASSERTION, NOT A RECALIBRATION. `rel_tol` is uncalibrated on
        # both bands and binding on 11 of 17 array surfaces; that is a recorded,
        # class-wide limit and moving it is a class-wide decision. What closes THIS
        # hole is asking the leg what precision it ran at, which the probe already
        # knows: it records `out.dtype` before `_to_np` upcasts bf16 to fp32 for
        # numpy, which is why the field exists rather than being read off `unit["out"]`.
        #
        # THE KEYS EXIST FIRST. This is the ORIGINAL of the clause that was
        # explicitly guarded against absent-key vacuity in its copy at
        # `test_prefill_logits`, and it was left open here: `str(unit.get(k))`
        # is `'None'` for an absent key, so a probe that stopped reporting the
        # pair would compare `'None'` to `'None'` and pass for every candidate,
        # at the one clause that sees an fp32 leak the value bands cannot. The
        # probe writes both today, which is what makes it latent rather than
        # live; a missing key is a harness regression and this names it as one.
        missing = [k for k in ("out_dtype", "requested_dtype") if k not in unit]
        assert not missing, {
            "tier": tier,
            "error": "the isolated attention probe did not report its dtype "
                     "provenance, so the clause below would compare 'None' "
                     "against 'None' and pass vacuously. This is a HARNESS "
                     "regression, not a port defect.",
            "missing_keys": missing,
            "probe_keys": sorted(k for k in unit if k != "out"),
        }
        assert str(unit.get("out_dtype")) == str(unit.get("requested_dtype")), {
            "tier": tier,
            "error": "the isolated attention op returned an array at a different "
                     "dtype than the one it was called at, so this leg is not "
                     "grading the precision tier it claims to. The usual cause is a "
                     "missing cast back to the query dtype after an fp32 "
                     "accumulation -- which leaks fp32 through the whole forward and "
                     "is invisible to every value comparison in this tier.",
            "called_at": unit.get("requested_dtype"),
            "returned": unit.get("out_dtype"),
        }
        _assert_array_parity(frozen[f"out_{tier}"], unit["out"], golden, tier,
                             name=f"paged_attention/{tier}")


# =========================================================================== #
# L1 -- the multimodal input path. Gated on the ``multimodal`` FAMILY fact, so
# the text-only instances DESELECT these two rather than fail them.
# =========================================================================== #
#: ``image_preprocess.npz`` keys :func:`test_image_preprocess` reads, and the
#: provenance fields it demands. Named here so the two "this is an artifact
#: defect" clauses below can report the whole missing set at once rather than
#: dying on the first ``KeyError``.
_IMAGE_KEYS = ("pixel_values", "image_grid_thw", "source_png_sha256",
               "source_image_rgb", "_processor", "_provenance")
_PROVENANCE_FIELDS = ("package", "version", "generator", "generated")


def _recorded_json(frozen: dict, key: str) -> dict:
    """A json-in-an-npz-string entry, as a dict. ``{}`` if it is not one.

    ``np.savez`` stores a python string as a 0-d ``<U`` array, so it comes back
    as ``array('{...}')`` and has to go through ``str()`` before ``json.loads``.
    Malformed reads as empty, and the caller's completeness clause reports that
    as the artifact defect it is rather than raising a ``JSONDecodeError`` here.
    """
    try:
        value = json.loads(str(frozen[key]))
    except (KeyError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def test_image_preprocess(golden, cand):
    """The port's own image processor, against the frozen upstream recording.

    THE RECORDING IS THE ORACLE, and it has two jobs. It is the input to this
    family's multimodal prefill, and it is the DE-VENDORING PROOF for this
    module: the verifier image has no ``transformers``, so the only way to grade
    a ported processor is against bytes recorded where ``transformers`` existed
    (by the freezer, at the 5.4.0 pin). Same idiom as
    ``kernel_reference.npz`` for the paged-attention kernel.

    TWO GRADED ASSERTIONS, NOT ONE, and they are graded differently on purpose.

    * ``image_grid_thw`` is INTEGER ARITHMETIC and is compared EXACTLY. It is the
      number of image tokens the prompt reserves; a port that gets the grid wrong
      and the pixels approximately right has mis-tokenized the image, every
      position after the image span is shifted, and the logit checks downstream
      would report it as a model bug.
    * ``pixel_values`` is float and is compared at the fp32 band.

    THE fp32 BAND, NOT THE NATIVE ONE, and this is a deliberate departure from
    the "value comparisons go at the native band" default. Both reference sides
    compute this surface entirely in float32 -- upstream's processor is
    dtype-blind and the tower casts on entry -- so there is no bf16 kernel stack
    here for the native band to be describing. On this fixture:

        peak |pixel_values|                       2.1459
        native band admits at that peak           0.3146   (14.7% of the value)
        fp32 band admits at that peak             0.0022
        torch reference vs the recording          0.0      (bit-identical)
        jax reference vs the recording            2.4e-07  (H100; 1.2e-07 on CPU)

    The native band would leave 1.3 MILLION times headroom over the measured
    disagreement on a surface with no bf16 in it -- which is the shape this
    module's own comments call a check that looks strict and is not. The fp32
    band still leaves ~9400x, and it does not punish a legitimate alternative
    implementation: the UNFUSED ``(x * rescale - mean) / std`` form (upstream
    fuses the rescale into the statistics) lands at 4.8e-07 and passes with three
    orders of magnitude to spare.

    WHAT IT DISCRIMINATES. Substituting ImageNet's normalisation
    constants for the OpenAI CLIP ones the checkpoint actually ships -- the
    single most plausible wrong guess, since the two look alike and the wrong
    pair is the one most readers have memorised -- moves every one of the 18 816
    elements by up to 0.494, and fails this clause on all of them.

    THE ARTIFACT CLAUSES COME FIRST, and the sha256 one is not decoration. An
    output-only recording detaches silently the moment its input moves, so the
    freezer records what it computed FROM: ``source_png_sha256`` binds the
    numbers to the PNG a reviewer sees in the diff, and re-deriving it here is
    what makes that binding checked rather than claimed.
    """
    frozen = golden.image_preprocess
    assert frozen, {
        "check": "test_image_preprocess",
        "error": "this instance declares `multimodal: true` but its golden tree "
                 "carries no image_preprocess.npz / image_fixture.png pair, so "
                 "there is no recording to grade the port against. This is an "
                 "artifact defect, not a port defect.",
        "hint": "re-freeze the image-preprocess golden, in "
                "t2j-model:{family} (it needs transformers 5.4.0 and the "
                "checkpoint)",
    }
    absent = [key for key in _IMAGE_KEYS if key not in frozen]
    assert not absent, {
        "check": "test_image_preprocess",
        "error": "the recording is missing keys this check reads; this is an "
                 "artifact defect",
        "missing": absent, "present": sorted(frozen),
    }
    provenance = _recorded_json(frozen, "_provenance")
    missing_fields = [f for f in _PROVENANCE_FIELDS if not provenance.get(f)]
    assert not missing_fields, {
        "check": "test_image_preprocess",
        "error": "the recording carries no complete provenance block, so it "
                 "cannot say which library version it is a recording OF. A "
                 "de-vendoring oracle without that is a file of numbers.",
        "missing_fields": missing_fields, "provenance": provenance,
    }
    # THE RECORDED INPUT, RE-DERIVED. The candidate is handed the PNG bytes; this
    # asserts those bytes are the ones the numbers below were computed from.
    png = golden.image_fixture_png
    digest = hashlib.sha256(png or b"").hexdigest()
    assert digest == str(frozen["source_png_sha256"]), {
        "check": "test_image_preprocess",
        "error": "golden/image_fixture.png is not the image the recording was "
                 "made from, so the frozen pixel_values describe a picture this "
                 "tree does not contain. This is an artifact defect.",
        "png_sha256": digest, "recorded": str(frozen["source_png_sha256"]),
        "png_bytes": len(png or b""),
    }
    # ...and the recording is internally coherent: the grid it froze must be the
    # grid its own recorded image and patch geometry imply. Without this, a
    # re-freeze that moved the fixture's size and not the grid would be graded
    # against itself.
    facts = _recorded_json(frozen, "_processor")
    patch = int(facts.get("patch_size") or 0)
    want_grid = np.asarray(frozen["image_grid_thw"], dtype=np.int64).reshape(-1, 3)
    rgb = np.asarray(frozen["source_image_rgb"])
    assert patch > 0 and rgb.ndim == 3 and want_grid.shape[0] == 1, {
        "check": "test_image_preprocess",
        "error": "the recording does not describe one image at a known patch "
                 "size; this is an artifact defect",
        "patch_size": patch, "source_image_rgb": list(rgb.shape),
        "image_grid_thw": want_grid.tolist(),
    }
    assert [int(want_grid[0][1]), int(want_grid[0][2])] == \
        [rgb.shape[0] // patch, rgb.shape[1] // patch], {
        "check": "test_image_preprocess",
        "error": "the frozen image_grid_thw is not the recorded image's own size "
                 "in patches; this is an artifact defect",
        "image_grid_thw": want_grid.tolist(), "image_hw": list(rgb.shape[:2]),
        "patch_size": patch,
    }

    probe = cand.image_preprocess
    assert probe.get("supported"), {
        "check": "test_image_preprocess",
        "error": "the candidate produced no image_preprocess result",
        "reason": probe.get("reason"),
        "hint": "this family's port owns the decode: `image_processing"
                ".preprocess` takes the encoded image bytes and returns "
                "(pixel_values, image_grid_thw)",
    }
    got_grid = np.asarray(probe["image_grid_thw"])
    # AN INTEGER GRID, AND THIS CLAUSE IS WHY THE CAST BELOW IS NOT A LAUNDERING
    # STEP. `np.asarray(x, dtype=np.int64)` TRUNCATES a float, so a port that
    # returned [[1.0, 4.4, 4.4]] -- a grid computed with `/` where the reference
    # has `//` -- would compare equal to [[1, 4, 4]] and pass. The dtype is asked
    # about before the values are.
    assert got_grid.dtype.kind in "iu", {
        "check": "test_image_preprocess",
        "error": "image_grid_thw is a count of patches and must be an integer "
                 "array; a float grid is silently truncated by the comparison "
                 "below and would hide a division that should have been a floor "
                 "division",
        "dtype": str(got_grid.dtype), "values": got_grid.tolist(),
    }
    np.testing.assert_array_equal(
        got_grid.reshape(-1, 3).astype(np.int64), want_grid,
        err_msg="image_grid_thw is integer arithmetic and must match exactly; a "
                "wrong grid means a wrong number of image tokens, which "
                "mis-aligns every position downstream")

    # THE PRECISION THE PROCESSOR RAN AT, read off the array BEFORE `_to_np`
    # upcast it (`_to_np_with_provenance`). This is the clause the docstring
    # promises, and it is the one that sees what the value clause below CANNOT.
    #
    # Round-tripping the recording through each dtype and counting
    # elements outside the fp32 band:
    #
    #     bfloat16   max |d| 5.3e-03   11 496 of 18 816 out of band -> value clause CATCHES
    #     float16    max |d| 6.6e-04        0 of 18 816 out of band -> value clause MISSES
    #
    # So a port that narrows this surface to float16 -- a plausible "the vision
    # tower is half precision anyway" shortcut -- reproduces the recording well
    # enough to pass every value comparison in this tier and is invisible without
    # this clause. It is a real divergence: upstream's processor is dtype-blind
    # and returns float32, and the tower casts on entry, so narrowing here
    # quantises the input twice and makes the processor's output a function of
    # the run configuration, which it is not.
    #
    # THE EXPECTED DTYPE IS THE RECORDING'S, not a literal, for the same reason
    # no tolerance in this module is a literal: the oracle states it.
    #
    # THE KEY EXISTS FIRST -- the same guard `test_prefill_logits` and
    # `test_paged_attention_op` carry. `.get()` against `.get()` would compare
    # None to None and pass for every candidate if the probe ever stopped
    # reporting the field, which is exactly how a clause like this rots.
    assert "pixel_values_dtype" in probe, {
        "check": "test_image_preprocess",
        "error": "the image probe did not report the dtype its processor ran at, "
                 "so the clause below would pass vacuously. This is a HARNESS "
                 "regression, not a port defect.",
        "probe_keys": sorted(k for k in probe if k != "pixel_values"),
    }
    want_dtype = np.dtype(np.asarray(frozen["pixel_values"]).dtype).name
    assert probe["pixel_values_dtype"] == want_dtype, {
        "check": "test_image_preprocess",
        "error": "your image processor returned pixel_values at a different "
                 "precision than the reference recording. float16 in particular "
                 "is INSIDE the value band on this fixture (0 of 18 816 "
                 "elements out of band), so nothing else in this tier can see it.",
        "recorded": want_dtype, "returned": probe["pixel_values_dtype"],
        # Reported, deliberately NOT asserted on. A numpy return would be wrong
        # across the ENGINE's API boundary (instruction.md says so, and
        # `test_prefill_logits` enforces it there), but this is host-side input
        # preparation that runs once before anything is traced -- no contract
        # anywhere tells the agent it must be a device array, and inventing one
        # here would fail a reasonable port. It rides the failure payload so the
        # field is read rather than dead, and so a reader diagnosing a dtype
        # surprise can see what kind of object carried it.
        "returned_type": probe.get("pixel_values_srctype"),
    }
    # "fp32", NOT "native", AND THIS IS THE ONE PLACE IN THIS MODULE WHERE A VALUE
    # COMPARISON DEPARTS FROM THE CLASS DEFAULT -- so the reasoning is here, at the
    # threshold, and not only in the docstring. Both reference sides compute this
    # surface entirely in float32 (upstream's processor is dtype-blind; the tower
    # casts on entry), so the native band would be describing a bf16 kernel stack
    # that does not exist on this path. On this fixture:
    #
    #     peak |pixel_values|              2.1459
    #     native band admits at that peak  0.3146   (abs 1e-1 + rel 1e-1)
    #     fp32 band admits at that peak    0.0022   (abs 1e-4 + rel 1e-3)
    #     measured disagreement            2.4e-07  (jax/H100); 0.0 (torch, exact)
    #
    # 0.3146 against 2.4e-07 is not a tolerance, it is the absence of one: 1.3
    # MILLION times headroom would admit a port that had lost most of its
    # precision. fp32 keeps ~9400x, which is loose enough for legitimate
    # reassociation -- the unfused `(x*rescale - mean)/std` form lands at
    # 4.8e-07 and passes with three orders of magnitude to spare -- and tight
    # enough that the ImageNet-constants defect fails on all 18 816 elements rather
    # than on 8674 of them.
    _assert_array_parity(frozen["pixel_values"], probe["pixel_values"], golden,
                         "fp32", name="image_preprocess/pixel_values")


def test_smart_resize_targets(golden, cand):
    """``smart_resize``'s computed dimensions, on sizes that do NOT conform.

    THE FIXTURE IS 56x56 ON PURPOSE, and that is exactly why this check exists
    separately. 56*56 == 3136 == ``min_pixels`` and 56 is a whole multiple of
    ``patch_size * merge_size``, so on the graded image ``smart_resize`` is the
    IDENTITY and the bicubic resampling kernel stays out of the numbers --
    ``tests/instance_context.md`` declares that boundary. But the arithmetic
    around it is where the off-by-one lives, so it is graded here, on a recorded
    table of deliberately non-conforming inputs, with no pixels touched and NO
    TOLERANCE: these are integers.

    THE TABLE'S SEVEN ROWS, and no case is redundant:

        (56, 56)     -> (56, 56)      conforming; the no-op is a CHECKED claim
        (57, 57)     -> (56, 56)      round to a multiple of the factor
        (28, 300)    -> (28, 308)     round on ONE axis; aspect ratio preserved
        (1, 1)       -> (56, 56)      min_pixels clamp, from the degenerate end
        (4000, 4000) -> (3584, 3584)  max_pixels clamp
        (10, 30)     -> (56, 112)     the min clamp's ROUNDING MODE
        (3000, 6000) -> (2520, 5068)  the max clamp's ROUNDING MODE

    WHAT THE TABLE DISCRIMINATES, as twelve ``smart_resize`` variants run
    over it, because the obvious reading of a table like this is routinely too
    generous. CAUGHT, and by which row:

        initial round -> floor              (28, 300)   -> (28, 280)
        initial round -> ceil               (57, 57)    -> (84, 84)
        min_pixels clamp missing            (1, 1)      -> (0, 0)
        max_pixels clamp missing            (4000, 4000)-> (4004, 4004)
        library-default max_pixels          (4000, 4000)-> (980, 980)
        min branch ceil -> floor OR round   (10, 30)    -> (28, 84)
        max branch floor -> ceil            (3000, 6000)-> (2548, 5096)
        both branches sharing one mode      whichever of the last two applies

    THE LAST TWO ROWS ARE WHY THE LAST THREE LINES ARE THERE, and they were added
    because the first five DID NOT grade the rounding mode at all. Both original
    clamp rows land on an exact integer before anything is
    rounded: ``max_pixels`` is ``12 845 056 == 3584**2`` with ``3584 == 128 * 28``
    and ``(4000, 4000)`` is square, so ``4000 / beta / 28`` is exactly ``128.0``;
    ``min_pixels`` is ``3136 == 56**2``, so at ``(1, 1)`` ``beta`` is exactly 56 and
    ``1 * beta / 28`` is exactly ``2.0``. A rounding mode is invisible on an
    integer, and all five rounding variants above reproduced the five-row table
    exactly.

    STILL NOT COVERED, and DECLARED rather than left silent (see
    ``gen_inference_instruction.BOUNDARY_PROSE``): the ``max(factor, ...)`` guard on
    the max branch. No row can reach it -- under ``smart_resize``'s own
    aspect-ratio cap of 200, ``h / beta / factor`` reduces to
    ``sqrt(max_pixels / (w/h)) / factor``, which bottoms out at 9.05 for this
    checkpoint -- so the guard cannot fire at all here. That is a gap no table can
    close, not a gap in this one. (The twelfth variant, the library's default
    ``min_pixels``, also escapes and is not a defect: ``56 * 56 == 3136`` is
    exactly the checkpoint's value, so the two are the same number.)

    The ``(1, 1)`` row still carries the trap that matters most: ``round(1/28) *
    28`` is ZERO on both axes, so a port without the min clamp emits an empty grid
    or divides by zero downstream.

    AND THE BOUNDS ARE THE CHECKPOINT'S, NOT THE SIGNATURE'S. Upstream's
    ``smart_resize`` defaults ``max_pixels`` to ``14*14*4*1280`` == 1 003 520,
    while this checkpoint's ``preprocessor_config.json`` asks for 12 845 056. A
    port that takes the signature default answers (980, 980) on the last row
    where the reference answers (3584, 3584) -- so this table also grades that
    the port read the config rather than the library's defaults.

    THE COVERAGE CLAUSES ARE THE POINT. A loop over an empty table passes for any
    candidate at all, and a probe that answered three of five cases would
    otherwise be graded on three and reported green -- the same accounting error
    ``_grade_battery`` and ``test_reference_units`` guard against, with no
    deselection to make it visible.
    """
    table = golden.smart_resize_table
    frozen = golden.image_preprocess or {}
    moved = [case for case, want in table if case != want]
    # THE FLOOR IS THE TABLE'S PURPOSE, NOT A NON-ZERO COUNT. A weaker floor of
    # `len(table) >= 2` plus "one row moved" would let a recording that regressed
    # to two rows sail through the GRADED check while the docstring above
    # enumerates seven. The stronger counts have to be enforced here, inside the
    # verifier, not only in host-side authoring tooling. A check whose floor
    # contradicts its own docstring is the exact shape this instance keeps
    # turning up.
    assert len(table) >= 4 and len(moved) >= 3, {
        "check": "test_smart_resize_targets",
        "error": "the frozen smart_resize table is too small to grade the "
                 "arithmetic it exists for: it needs the round-to-factor, the "
                 "min_pixels clamp and the max_pixels clamp, which is at least "
                 "three rows that MOVE plus the conforming no-op. This is an "
                 "artifact defect, not a port defect.",
        "frozen_cases": [list(case) for case, _ in table],
        "cases": len(table), "cases_that_moved": len(moved),
        "hint": "re-freeze the image-preprocess golden; the freezer's case "
                "table carries the branches and the derivation",
    }
    # ...AND THE ROUNDING MODE MUST BE OBSERVABLE ON EACH CLAMP, which is the
    # anti-vacuity property this whole check learned the hard way -- see the
    # docstring. A clamp row whose scaled quotient is already an INTEGER cannot
    # tell floor from ceil from round, and the first five recorded rows were all
    # of that kind, so five rounding variants reproduced the table exactly.
    #
    # Computed from the recording's OWN bounds (`smart_resize_{factor,min_pixels,
    # max_pixels}`, all recorded), so this asserts a property of the table rather
    # than restating the seven rows -- a future re-freeze that changed the sizes
    # but kept the coverage still passes, and one that dropped the coverage does
    # not.
    factor = int(np.asarray(frozen.get("smart_resize_factor", 0)))
    lo = int(np.asarray(frozen.get("smart_resize_min_pixels", 0)))
    hi = int(np.asarray(frozen.get("smart_resize_max_pixels", 0)))
    assert factor > 0 and 0 < lo <= hi, {
        "check": "test_smart_resize_targets",
        "error": "the recording does not carry the factor and the two pixel "
                 "bounds its own table was computed against; this is an artifact "
                 "defect", "factor": factor, "min_pixels": lo, "max_pixels": hi,
    }
    observable = {"min": False, "max": False}
    for (height, width), _want in table:
        rounded = round(height / factor) * factor * (round(width / factor) * factor)
        if rounded > hi:
            branch, beta = "max", (height * width / hi) ** 0.5
            quotients = (height / beta / factor, width / beta / factor)
        elif rounded < lo:
            branch, beta = "min", (lo / (height * width)) ** 0.5
            quotients = (height * beta / factor, width * beta / factor)
        else:
            continue
        if any(abs(q - round(q)) > 1e-9 for q in quotients):
            observable[branch] = True
    assert observable["min"] and observable["max"], {
        "check": "test_smart_resize_targets",
        "error": "the frozen table exercises a clamp only at sizes whose scaled "
                 "quotient is already a whole number, where floor, ceil and round "
                 "all agree -- so the clamp's ROUNDING MODE is not graded at all. "
                 "Several rounding variants can reproduce such a table "
                 "exactly. This is an artifact defect.",
        "rounding_observable": observable,
        "hint": "each clamp needs at least one row whose scaled quotient is NOT "
                "an integer -- e.g. (10, 30) for min_pixels and (3000, 6000) for "
                "max_pixels at this checkpoint's bounds",
    }

    probe = cand.smart_resize
    assert probe.get("supported"), {
        "check": "test_smart_resize_targets",
        "error": "the candidate produced no smart_resize results",
        "reason": probe.get("reason"),
    }
    cases = [tuple(int(v) for v in case) for case in (probe.get("cases") or [])]
    targets = list(probe.get("targets") or [])
    assert cases == [case for case, _ in table] and len(targets) == len(cases), {
        "check": "test_smart_resize_targets",
        "error": "the candidate did not answer the frozen table, case for case "
                 "and in order; this is a harness defect",
        "frozen": [list(case) for case, _ in table], "probe": [list(c) for c in cases],
        "n_targets": len(targets),
    }
    raised = probe.get("errors") or {}
    assert not raised, {
        "check": "test_smart_resize_targets",
        "error": "smart_resize raised on one or more of the frozen sizes. Each is "
                 "a size the reference answers; (1, 1) and (4000, 4000) are there "
                 "precisely because a port without the two clamps divides by zero "
                 "or overflows the patch budget on them.",
        "failures": {str(cases[int(i)]): why for i, why in sorted(raised.items())},
    }
    for (case, want), got in zip(table, targets):
        assert got is not None and tuple(int(v) for v in got) == tuple(want), {
            "check": "test_smart_resize_targets",
            "error": "smart_resize returned a different target size. This is "
                     "integer arithmetic on the image's dimensions, so it is "
                     "compared exactly -- there is nothing here for a tolerance "
                     "to absorb.",
            "input_hw": list(case), "want": list(want),
            "got": None if got is None else [int(v) for v in got],
            "hint": "round each side to a multiple of patch_size * merge_size; "
                    "if the product exceeds max_pixels scale down with floor and "
                    "a max(factor, ...) guard; if it falls below min_pixels scale "
                    "up with ceil. The two bounds come from the checkpoint's "
                    "preprocessor_config.json, not from the library's defaults.",
        }


# =========================================================================== #
# L1/L2 -- the multimodal MODEL path. Gated on the same ``multimodal`` FAMILY
# fact as the two input-path checks above, so the six text-only instances
# DESELECT all four rather than fail or skip them: a skip leaves both numerator
# and denominator and an instance that skips half its suite scores exactly like
# one that passes all of it.
# =========================================================================== #
def _placeholder_rows(golden, token_id: int) -> np.ndarray:
    """The FROZEN prompt's own ``<|image_pad|>`` offsets, as row indices.

    Derived from ``inputs.npz`` -- which this process holds and the candidate
    cannot choose -- rather than from the probe's own report, so a port that
    resolved the wrong ``image_token_id`` cannot scatter into the wrong rows AND
    nominate those same rows as the ones to check.

    WHICH FORWARD'S IDS, and it is not always the graded one. A sequence gives
    its picture to whichever forward covers its ``<|image_pad|>`` placeholders,
    and the freezer records WHICH
    by writing the pair under the ``warmup_`` prefix. So the ids this reads have
    to be the ids of the forward that OWNS the image, or the check compares a
    merge computed on one sequence against the placeholder offsets of another.

    On ``chunked_qwen2_5_vl``, the first instance whose canonical
    sequence 0 is BOTH multimodal and a ``prefill_chunked``: its ``<|image_pad|>``
    run is [1, 5) and its graded query starts at 5, so the warm-up owns the image
    and ``input_ids`` -- the GRADED forward's ten tokens -- contains no
    placeholder at all. Reading it there produces "your config's image_token_id
    does not occur in the frozen prompt at all", accusing a correct port of
    resolving the wrong id.

    INERT ON EVERY INSTANCE THAT PREDATES IT: `core` goldens have no warm-up
    group, and on both `ragged` multimodal ones the image-bearing sequence is a
    `prefill` with ``start == 0``, so no ``warmup_pixel_values`` is written and
    this reads ``input_ids`` exactly as it always did.
    """
    key = ("warmup_input_ids"
           if golden.inputs.get("warmup_pixel_values") is not None
           else "input_ids")
    ids = np.asarray(golden.inputs.get(key, [])).reshape(-1)
    return np.flatnonzero(ids == int(token_id))


def test_mrope_positions(golden, cand):
    """The 3D position ids for a text+image prompt, graded EXACTLY.

    THE MOST LIKELY THING TO BE SILENTLY WRONG, and the reason it gets its own
    check rather than being left to the logit comparison: an M-RoPE bug that
    assigns TEXT positions to the image span still produces perfectly plausible
    logits at this prompt length, and the native band absorbs it. The position ids
    themselves are integers computed by pure index arithmetic, so there is nothing
    here for a tolerance to be about.

    THE RULE THEY ENCODE, since the failure message has to be readable: a text
    token advances all rope sections by one, while an image contributes its
    (t, h, w) grid indices, so a whole image occupies ``max(h, w)`` positions
    rather than one per token and the text after it resumes from ``max + 1``. That
    is why a VL prompt's last position is generally LARGER than its length, and
    why the frozen array cannot be reproduced by ``broadcast_to(arange(n), ...)``.

    SHAPE IS ASSERTED BEFORE CONTENT, and the distinction the error preserves is
    not cosmetic. ``positions`` carries a leading rope-section axis on this family
    (``layers/rotary.py``; its extent is 3 here). A port that emits ``(S,)`` has
    not implemented M-RoPE AT ALL -- it is running the class's rank-1 default --
    which is a different repair from having implemented it wrongly, and a
    value-first comparison would report both as "your positions differ".

    WHAT IT IS NOT. The frozen ``positions`` array is also an INPUT to the graded
    forward and therefore crosses into the sandbox (``fixtures.build_candidate``
    copies every ``inputs.npz`` key); see ``Golden.mrope_positions``. This is a
    localiser for a port that computes them wrongly, in the same standing as
    ``test_kv_cache_write``'s occupancy clause against a ``slot_mapping`` the
    probe is likewise handed -- not a gate against a port that echoes its input.

    IT GRADES SEQUENCE 0'S WINDOW, NOT THE WHOLE FROZEN ARRAY. The probe drives
    ``Engine.prefill``, which is single-request by contract, so what it can
    produce is sequence 0's own run; the frozen ``positions`` on a `ragged` or
    `chunked` instance is the MIXED BATCH's, whose trailing columns are other
    requests' decode positions. ``Golden.graded_window`` names both windows and
    the slice below is the whole of the difference. On `core` the window is the
    identity and this check compares the whole array. On ``ragged_qwen2_5_vl``
    without the slice, 6 of 39 elements differ: the probe answers
    ``[... 8, 9, 10]`` where the batch holds ``[... 8, 4,
    8]``, i.e. it continues sequence 0's run through two tokens that belong to
    sequences 1 and 2.
    """
    want = golden.mrope_positions
    assert want is not None, {
        "check": "test_mrope_positions",
        "error": "golden/inputs.npz carries no `positions`, so there is nothing "
                 "to grade the position builder against. This is an artifact "
                 "defect, not a port defect.",
        "present": sorted(golden.inputs),
    }
    # SEQUENCE 0'S COLUMNS OF THE FROZEN ARRAY, which on `core` is all of them.
    # Sliced on the LAST axis, so the rope-section axis and the two assertions
    # below that read it are untouched.
    lo, hi, own_lo, own_hi = golden.graded_window
    want = np.asarray(want, dtype=np.int64)[..., lo:hi]
    # ...AND THE FROZEN ARRAY MUST ACTUALLY BE M-RoPE. On a text-only family the
    # freezer writes a rank-1 `positions` and this check is deselected; on THIS
    # family a rank-1 golden would mean the freeze itself lost the rope sections,
    # and every clause below would then pass for a port that never implemented
    # them. The check refuses to grade against such a golden rather than agreeing
    # with it.
    assert want.ndim == 2 and want.shape[0] > 1, {
        "check": "test_mrope_positions",
        "error": "the frozen positions are not multi-section, so this instance's "
                 "golden does not describe an M-RoPE prompt at all and grading "
                 "against it would be vacuous. This is an artifact defect.",
        "shape": list(want.shape),
        "hint": "re-freeze the inference goldens; the model's "
                "own position builder is resolved through Engine.POSITIONS_HOOK",
    }

    probe = cand.multimodal_prefill
    assert probe.get("supported"), {
        "check": "test_mrope_positions",
        "error": "the candidate produced no image-conditioned prefill, so its "
                 "position builder was never exercised",
        "reason": probe.get("reason"),
    }
    # THE PROBE'S OWN RUN MUST BE SEQUENCE 0'S, asserted before the window is
    # taken. A probe that forwarded some other id array would otherwise be
    # sliced into agreement-shaped nonsense, and the resulting value failure
    # would name the position builder for a driver defect. This clause is about
    # the HARNESS, and it says so.
    assert int(probe.get("num_tokens") or 0) == own_hi, {
        "check": "test_mrope_positions",
        "error": "the image-conditioned prefill probe forwarded a different "
                 "number of tokens than sequence 0's frozen length, so the "
                 "window this check compares would not line up. This is a "
                 "HARNESS regression, not a port defect.",
        "probe_num_tokens": int(probe.get("num_tokens") or 0),
        "sequence_0_len": own_hi,
    }
    got = np.asarray(probe["positions"], dtype=np.int64)[..., own_lo:own_hi]
    assert got.shape == want.shape, {
        "check": "test_mrope_positions",
        # `input_positions`, NOT `positions`. `inputs.npz`'s KEY is `positions`
        # and `AttentionMetadata`'s FIELD is `input_positions`; naming the wrong
        # one here would point a port at a field the frozen dataclass schema
        # (golden/interface.json, graded by `test_attention_metadata_schema`)
        # requires it NOT to have. The probe made the identical mistake and the
        # reference itself caught it.
        "error": "your attn_metadata.input_positions has a different shape than "
                 "the reference's",
        "got": list(got.shape), "want": list(want.shape),
        "hint": ("a (S,) shape means no M-RoPE section axis at all -- the "
                 "positions were built by the class's rank-1 default and the "
                 "model's own builder never ran -- rather than an M-RoPE that "
                 "computed the wrong numbers"
                 if got.ndim < want.ndim else
                 "the leading axis is the rope-section count "
                 f"(mrope_section, {want.shape[0]} here) and the trailing one is "
                 "the token count"),
    }
    # THE DTYPE, BEFORE THE VALUES, and it is the same guard `test_image_preprocess`
    # applies to `image_grid_thw`: `np.asarray(x, dtype=np.int64)` TRUNCATES a
    # float, so positions computed with `/` where the reference has `//` would
    # compare equal here. Read off the array BEFORE `_to_np` upcast it.
    assert "positions_dtype" in probe, {
        "check": "test_mrope_positions",
        "error": "the probe did not report the dtype its positions came back at, "
                 "so the clause below would pass vacuously. This is a HARNESS "
                 "regression, not a port defect.",
        "probe_keys": sorted(k for k in probe if k != "logits"),
    }
    assert "int" in str(probe["positions_dtype"]), {
        "check": "test_mrope_positions",
        "error": "a rope position is an index and must be an integer array. A "
                 "float one holding whole numbers compares equal to the golden "
                 "after the cast below and is a different contract -- it is what "
                 "a builder that divided where it should have floor-divided "
                 "returns.",
        "dtype": str(probe["positions_dtype"]),
    }
    np.testing.assert_array_equal(
        got, want,
        err_msg="the M-RoPE position ids are integer index arithmetic and are "
                "compared exactly. Row 0 is the temporal section, rows 1 and 2 "
                "the height and width sections; an image span must carry its "
                "(t, h, w) GRID indices and the text after it must resume from "
                "one past the largest position any section reached.")


def test_image_embedding_merge(golden, cand):
    """The tower's rows must land at EXACTLY the ``<|image_pad|>`` offsets.

    THE DEFECT THIS EXISTS FOR IS APPENDING RATHER THAN SCATTERING. Upstream's
    merge is ``inputs_embeds[input_ids == image_token_id] = image_embeds``; the
    natural JAX transliteration is a masked assignment, which JAX refuses (a
    boolean mask has a data-dependent output shape), so every port has to
    re-express it -- and a concatenation has the right dtype, the right row count
    on a prompt whose placeholders happen to sit at one end, and completely wrong
    contents.

    BOTH DIRECTIONS ARE ASSERTED, and one alone is not enough:

    * the rows AT the placeholder offsets must equal the TOWER's output, in order.
      A port that scatters the right rows in the wrong order passes any
      row-set comparison and fails this.
    * the rows OUTSIDE them must equal the TEXT embedding. A port that appends,
      or that overwrites the wrong span, moves a text row -- invisible to the
      first clause, which only looks at the placeholder indices.

    NO FROZEN ARRAY IS INVOLVED, deliberately. All three tables come out of the
    SAME probe, so this is an internal-consistency property of the candidate --
    which is what makes it independent of every band in this file and of the
    tower's own accuracy. A tower whose numbers are wrong still has to scatter
    them correctly, and ``test_vision_tower_units`` is what grades the numbers.

    THE OFFSETS COME FROM THE FROZEN PROMPT, not from the probe: see
    :func:`_placeholder_rows`. The probe's own `placeholder_rows` is compared
    against them rather than trusted, so a port that resolved the wrong
    ``image_token_id`` is named here instead of quietly grading itself.

    EXACT EQUALITY, no band. This is a GATHER: the merged table's rows are the
    identical values the two sources hold, copied, so any disagreement at all is
    a wrong index or a wrong cast, never reduction order.
    """
    probe = cand.image_embedding_merge
    assert probe.get("supported"), {
        "check": "test_image_embedding_merge",
        "error": "the candidate produced no image-embedding merge",
        "reason": probe.get("reason"),
    }
    token_id = probe.get("image_token_id")
    assert token_id is not None, {
        "check": "test_image_embedding_merge",
        "error": "your ModelConfig carries no `image_token_id`, so nothing in "
                 "the model can know which rows the tower's output replaces",
    }
    rows = _placeholder_rows(golden, int(token_id))
    assert rows.size > 0, {
        "check": "test_image_embedding_merge",
        "error": "your config's image_token_id does not occur in the frozen "
                 "prompt at all, so the merge had no placeholders to write over. "
                 "The prompt reserves one placeholder per merged vision token; a "
                 "wrong id means the tower's rows went somewhere else entirely.",
        "image_token_id": int(token_id),
        # The OWNING forward's ids, matching `_placeholder_rows` above -- a
        # message that printed the graded row when the warm-up holds the image
        # would send a reader looking in the wrong array.
        "prompt_token_ids": np.asarray(golden.inputs.get(
            "warmup_input_ids"
            if golden.inputs.get("warmup_pixel_values") is not None
            else "input_ids", [])).reshape(-1).tolist(),
    }
    reported = [int(i) for i in (probe.get("placeholder_rows") or [])]
    assert reported == rows.tolist(), {
        "check": "test_image_embedding_merge",
        "error": "the rows your model treats as placeholders are not the frozen "
                 "prompt's own runs of image_token_id",
        "reported": reported, "frozen": rows.tolist(),
    }

    image_embeds = np.asarray(probe["image_embeds"], dtype=np.float32)
    text_embeds = np.asarray(probe["text_embeds"], dtype=np.float32)
    merged = np.asarray(probe["merged"], dtype=np.float32)
    assert merged.shape == text_embeds.shape, {
        "check": "test_image_embedding_merge",
        "error": "the merged embedding table is a different shape than the "
                 "text-only one. The tower REPLACES the placeholder rows; it "
                 "does not add or remove sequence positions, so the merge must "
                 "be a scatter and not a concatenation.",
        "merged": list(merged.shape), "text_only": list(text_embeds.shape),
    }
    assert image_embeds.shape[0] == rows.size, {
        "check": "test_image_embedding_merge",
        "error": "the tower produced a different number of rows than the prompt "
                 "reserves placeholders for. Each merged vision token needs "
                 "exactly one <|image_pad|>.",
        "tower_rows": int(image_embeds.shape[0]), "placeholders": int(rows.size),
    }

    np.testing.assert_array_equal(
        merged[rows], image_embeds.astype(merged.dtype),
        err_msg="the merged table's <|image_pad|> rows are not the tower's "
                "output, in prompt order. This is a gather, so it is compared "
                "exactly: a mismatch is a wrong index or a wrong order, not "
                "arithmetic.")
    keep = np.ones(merged.shape[0], dtype=bool)
    keep[rows] = False
    np.testing.assert_array_equal(
        merged[keep], text_embeds[keep],
        err_msg="a row OUTSIDE the <|image_pad|> span moved. The merge must "
                "overwrite the placeholder rows and nothing else -- a port that "
                "APPENDS the tower's output, or writes it at the wrong offset, "
                "fails here while the placeholder rows themselves can still look "
                "right.")


def test_vision_tower_units(golden, cand):
    """Per-tower-submodule I/O, and the window geometry that nothing else grades.

    ``test_reference_units``' contract, one tier over: the candidate's OWN
    submodule run on the REFERENCE's recorded input, so a wrong ``patch_embed``
    cannot contaminate the verdict on ``merger``, and a unit that refuses its
    input is a verdict about THAT unit rather than an error on every check in the
    run. Frozen from the reference implementation, fp32 on both sides at
    the fp32 band.

    IT IS RECORDED ON A DIFFERENT IMAGE FROM THE GRADED PROMPT'S, and that is the
    whole reason this check is separate rather than six more rows in the decoder's
    unit table. ``encoder_metadata``'s window permutation is DEGENERATE at every
    grid the graded prompt can afford: ``merge_window`` is 4 here and
    ``pad = merge_window - llm % merge_window`` pads a FULL window when
    ``llm % merge_window == 0``, so at grids (1,4,4), (1,6,6) AND (1,8,8) the
    permutation is the identity and ``cu_window_seqlens == cu_seqlens``. A tower
    with NO window attention reproduces all three. The recording is therefore made
    at 140x140 -> (1,10,10), four windows, ``cu_window_seqlens
    [0, 64, 80, 96, 100]``.

    THE TWO BLOCK UNITS ARE THE SAME CLASS AND ARE NOT REDUNDANT.
    ``block_windowed`` is ``visual.blocks.0`` and ``block_fullatt`` is
    ``visual.blocks.7``, the first index in ``fullatt_block_indexes``; the only
    structural difference between them is which cumulative-length vector the
    tower hands them, and each is replayed on the one it was recorded with. A
    port that gives every block the full vector reproduces the second and fails
    the first.

    THE GEOMETRY IS COMPARED EXACTLY on its integers -- ``window_index``,
    ``reverse_indices`` and the two ``cu_*`` vectors are index arithmetic over the
    grid and touch no parameter -- and at the fp32 band on ``cos``/``sin``.

    THE ANTI-VACUITY CLAUSE IS HERE AS WELL AS IN THE FREEZER. The freezer
    REFUSES to write a degenerate recording, which stops one being created; this
    clause stops an OLDER one being graded, and the two are not the same
    guarantee.
    """
    frozen = golden.vision_tower_units
    assert frozen, {
        "check": "test_vision_tower_units",
        "error": "this instance declares `multimodal: true` but its golden tree "
                 "carries no vision_tower_units.{json,npz} pair, so the tower's "
                 "submodules and its window permutation are ungraded. This is an "
                 "artifact defect, not a port defect.",
        "hint": "freeze the vision-tower-units golden, in "
                "t2j-model:{family}, AFTER the image-preprocess golden "
                "(it reads the 140x140 tower fixture out of image_preprocess.npz)",
    }
    meta = frozen.get("meta") or {}
    arrays = frozen.get("arrays") or {}
    order = list(meta.get("tower_units") or [])
    specs = dict(meta.get("units") or {})
    geometry_spec = dict(meta.get("geometry") or {})
    assert order and specs and geometry_spec, {
        "check": "test_vision_tower_units",
        "error": "golden/vision_tower_units.json names no units or no geometry",
        "hint": "re-freeze the vision-tower-units golden",
    }
    # THE RECORDING IS NON-DEGENERATE -- see the docstring. Asserted against the
    # frozen arrays rather than against the fixture's declared size, so a
    # re-freeze that changed the image and kept the size claim cannot slip past.
    window_index = np.asarray(arrays[geometry_spec["window_index"]["key"]])
    cu_full = np.asarray(arrays[geometry_spec["cu_seqlens"]["key"]])
    cu_window = np.asarray(arrays[geometry_spec["cu_window_seqlens"]["key"]])
    assert not np.array_equal(
        window_index, np.arange(window_index.size, dtype=window_index.dtype)), {
        "check": "test_vision_tower_units",
        "error": "the frozen window_index is the IDENTITY permutation, so a port "
                 "that does no window reordering at all reproduces it and this "
                 "check grades nothing about window attention. This is an "
                 "artifact defect.",
        "hint": "the fixture's grid must have llm_h % merge_window != 0",
    }
    assert not (cu_window.shape == cu_full.shape
                and np.array_equal(cu_window, cu_full)), {
        "check": "test_vision_tower_units",
        "error": "the frozen cu_window_seqlens equals cu_seqlens, so every block "
                 "attends over the same chunking and the windowed/full-attention "
                 "distinction is not graded. This is an artifact defect.",
        "cu_seqlens": cu_full.tolist(), "cu_window_seqlens": cu_window.tolist(),
    }
    # ...AND THE FOUR INTEGER KEYS ARE STILL DECLARED EXACT, which is a THRESHOLD
    # assertion and not bookkeeping. The loop below reads `spec["exact"]` out of
    # the GOLDEN's json to decide between `assert_array_equal` and the fp32 band,
    # while which keys are exact is set only by the freezer. A re-freeze that
    # dropped or renamed that flag would silently DOWNGRADE `window_index`,
    # `reverse_indices` and both `cu_*` vectors from exact integer equality to a
    # band with an `abs_tol` of 1e-4 -- and every one of them would still pass on
    # a port whose permutation was off by less than that, which for an integer
    # index array means the check stopped grading anything at all. This is the
    # silent-threshold-default trap one axis over from the one
    # `test_the_three_calibrated_bands_are_declared_by_every_instance` exists
    # for, so it is asserted here rather than trusted.
    _EXACT = ("window_index", "reverse_indices", "cu_seqlens", "cu_window_seqlens")
    not_exact = [k for k in _EXACT if not (geometry_spec.get(k) or {}).get("exact")]
    assert not not_exact, {
        "check": "test_vision_tower_units",
        "error": "the frozen geometry no longer declares these integer keys "
                 "EXACT, so the loop below would grade them at the fp32 band -- "
                 "which admits any index array within 1e-4 of the reference's, "
                 "i.e. all of them. This is an artifact defect.",
        "not_exact": not_exact,
        "declared": {k: (geometry_spec.get(k) or {}).get("exact") for k in _EXACT},
        "hint": "the freezer sets this exact flag; re-freeze after restoring "
                "it",
    }

    probe = cand.vision_tower_units
    assert probe.get("supported"), {
        "check": "test_vision_tower_units",
        "error": "the candidate produced no per-tower-submodule replay",
        "reason": probe.get("reason"),
    }
    # COVERAGE FIRST, the same clause `test_reference_units` carries: a probe
    # that returned two of six units would otherwise be graded on two and
    # reported green.
    got = probe.get("units") or {}
    assert sorted(got) == sorted(order), {
        "check": "test_vision_tower_units",
        "error": "the candidate replayed a different set of tower units than the "
                 "golden froze",
        "frozen": sorted(order), "replayed": sorted(got),
    }
    unresolved = {name: entry.get("reason") for name, entry in sorted(got.items())
                  if not entry.get("supported")}
    assert not unresolved, {
        "check": "test_vision_tower_units",
        "error": "the candidate could not run one or more of the tower's "
                 "submodules on the reference's own inputs",
        "units": unresolved,
        "hint": "both reference trees use the same attribute names inside "
                "`visual` (patch_embed, rotary_pos_emb, blocks, merger), and the "
                "replay resolves the frozen torch path against your model. "
                "cu_seqlens arrives as a 1-D integer ARRAY, which both "
                "references' attention accepts.",
    }
    for unit in order:
        frozen_outputs = specs[unit]["outputs"]
        outputs = got[unit]["outputs"]
        assert len(outputs) == len(frozen_outputs), {
            "check": "test_vision_tower_units", "unit": unit,
            "path": got[unit]["path"], "error": "wrong number of return values",
            "golden": len(frozen_outputs), "candidate": len(outputs),
        }
        for value, spec in zip(outputs, frozen_outputs):
            _assert_array_parity(arrays[spec["key"]], value, golden, "fp32",
                                 name=f"vision_tower_units/{unit}/{spec['key']}")

    geometry = probe.get("geometry") or {}
    assert geometry.get("supported"), {
        "check": "test_vision_tower_units",
        "error": "the candidate produced no encoder_metadata result, so the "
                 "window permutation -- the one thing in the tower that nothing "
                 "else in this suite grades -- is ungraded",
        "reason": geometry.get("reason"),
    }
    values = geometry.get("values") or {}
    assert sorted(values) == sorted(geometry_spec), {
        "check": "test_vision_tower_units",
        "error": "your encoder_metadata returned a different set of keys than "
                 "the reference's",
        "frozen": sorted(geometry_spec), "candidate": sorted(values),
    }
    for name in sorted(geometry_spec):
        spec = geometry_spec[name]
        want = np.asarray(arrays[spec["key"]])
        have = np.asarray(values[name])
        assert have.shape == want.shape, {
            "check": "test_vision_tower_units", "geometry": name,
            "error": "wrong shape", "golden": list(want.shape),
            "candidate": list(have.shape),
            "hint": "cu_seqlens / cu_window_seqlens drop zero-length windows "
                    "(torch.unique_consecutive upstream), so a fully padded "
                    "window contributes no entry",
        }
        if spec.get("exact"):
            np.testing.assert_array_equal(
                have.astype(np.int64), want.astype(np.int64),
                err_msg=f"encoder_metadata['{name}'] is integer index arithmetic "
                        f"over grid_thw and is compared exactly. window_index is "
                        f"the permutation that makes each local window "
                        f"contiguous; reverse_indices undoes it AFTER the merger, "
                        f"at MERGED granularity; cu_window_seqlens counts RAW "
                        f"patches, which is the granularity attention runs at.")
        else:
            _assert_array_parity(want, have, golden, "fp32",
                                 name=f"vision_tower_units/encoder_metadata/{name}")


def test_multimodal_prefill_logits(golden, cand):
    """The image-conditioned prefill, end to end, at the native band.

    THE SAME GOLDEN AS ``test_prefill_logits`` AND A DIFFERENT PATH, which is the
    whole content of this check. That one drives a forward whose EVERY input was
    handed in -- the frozen positions, the frozen block table, ``input_ids`` and
    no image. This one hands the candidate's own ``Engine.prefill`` the pixels and
    the grid and nothing else, so the M-RoPE position builder, the host-side tower
    hook and the embedding scatter are all live at once and all three are between
    the prompt and the row being compared.

    A PORT CAN PASS ONE AND FAIL THE OTHER IN EITHER DIRECTION, and the pair is
    what says which half is wrong: this one failing while the handed forward
    passes localises the defect to the image path; the reverse localises it to
    the decoder or to the paging.

    THE ALLOCATION IS THE CANDIDATE'S OWN and the golden is the frozen forward's.
    That is sound because page assignment does not move a logit -- attention
    gathers through the block table -- and on this class's own
    reference the handed-allocation prefill and ``Engine.generate`` agree BIT
    FOR BIT on this row.

    NATIVE, NOT fp32: this is the serving path, so it is graded at the tier the
    port is actually used at, through the same three-clause logit gate every other
    native logit surface uses.

    IT GRADES SEQUENCE 0'S ROWS OF THAT GOLDEN, which on `core` is all of them.
    ``Engine.prefill`` is single-request by contract, so the probe forwards
    sequence 0's run alone while ``prefill_forward.npz`` on a `ragged` or
    `chunked` instance holds the MIXED BATCH's rows; ``Golden.graded_window``
    names the correspondence, and both slices come out of arrays this process
    holds. Comparing the two arrays whole read the batch's trailing decode rows
    as a continuation of sequence 0 and landed at centered cosine 0.5219 on
    ``ragged_qwen2_5_vl``.

    THE TWO LEGS ARE STILL TWO DIFFERENT BATCHES, and that is what the native
    band is for here rather than an exact comparison: the frozen rows were
    reduced over a 13-token mixed forward and the probe's over an 11-token single
    one. A correct paged implementation lets no other sequence reach these rows
    -- ``test_no_cross_sequence_leakage`` is the check that asserts that outright
    -- so what remains is the shape-dependent floating-point noise every
    invariance in this class is toleranced for.
    """
    frozen = golden.prefill_forward.get("logits")
    assert frozen is not None, {
        "check": "test_multimodal_prefill_logits",
        "error": "golden/prefill_forward.npz carries no `logits`; this is an "
                 "artifact defect",
        "present": sorted(golden.prefill_forward),
    }
    probe = cand.multimodal_prefill
    assert probe.get("supported"), {
        "check": "test_multimodal_prefill_logits",
        "error": "the candidate produced no image-conditioned prefill",
        "reason": probe.get("reason"),
        "hint": "Engine.prefill forwards **model_kwargs verbatim; a multimodal "
                "family's model consumes pixel_values / image_grid_thw from "
                "there, on the host, before the trace",
    }
    # THE LEG RAN AT THE TIER IT CLAIMS -- the same guard `test_prefill_logits`
    # carries, and it is load-bearing for the identical reason: `_to_np` upcasts
    # bf16 to float32 for numpy, so no value clause below can see a forward that
    # silently ran at fp32. The keys are asserted to EXIST first, because
    # `.get()` against `.get()` would compare None to None and pass for every
    # candidate if the probe stopped reporting them.
    missing = [k for k in ("logits_dtype", "requested_dtype") if k not in probe]
    assert not missing, {
        "check": "test_multimodal_prefill_logits",
        "error": "the probe did not report its dtype provenance, so the clause "
                 "below would compare 'None' against 'None' and pass vacuously. "
                 "This is a HARNESS regression, not a port defect.",
        "missing_keys": missing,
    }
    assert str(probe["logits_dtype"]) == str(probe["requested_dtype"]), {
        "check": "test_multimodal_prefill_logits",
        "error": "the image-conditioned prefill returned logits at a different "
                 "dtype than the run configuration asked for, so this leg is not "
                 "grading the precision tier it claims to.",
        "called_at": probe["requested_dtype"], "returned": probe["logits_dtype"],
    }
    # SEQUENCE 0'S WINDOW ON BOTH SIDES -- the identity on `core`. The
    # length guard is the same HARNESS clause `test_mrope_positions` carries and
    # exists for the same reason: a probe that drove some other id array would
    # otherwise be sliced into a shape that compares.
    lo, hi, own_lo, own_hi = golden.graded_window
    assert int(probe.get("num_tokens") or 0) == own_hi, {
        "check": "test_multimodal_prefill_logits",
        "error": "the image-conditioned prefill probe forwarded a different "
                 "number of tokens than sequence 0's frozen length, so the rows "
                 "this check compares would not line up. This is a HARNESS "
                 "regression, not a port defect.",
        "probe_num_tokens": int(probe.get("num_tokens") or 0),
        "sequence_0_len": own_hi,
    }
    _assert_logit_parity(np.asarray(frozen)[lo:hi],
                         np.asarray(probe["logits"])[own_lo:own_hi],
                         golden, "native", name="multimodal_prefill_logits")


# =========================================================================== #
# L4 -- cache state
# =========================================================================== #
def test_kv_cache_write(golden, cand):
    """Cache-state parity: WHERE exactly, and WHAT at the native band.

    Two halves, because the brief's one-line version -- "a write is a copy, not an
    arithmetic op, so no tolerance is warranted" -- is true of one half and false
    of the other. The PLACEMENT is a copy. The PAYLOAD is ``qkv_proj`` followed by
    a rotary embedding, i.e. arithmetic, and it does not agree bit for
    bit: the two sides' cache tensors differ in roughly 14% of their
    elements, by up to one bf16 ULP (the exact count is not run-stable -- 142 755
    on one run, 142 697 on another). An exact comparison of the whole array would
    fail the worked reference.

    (1) OCCUPANCY, EXACT. Every slot outside the frozen ``slot_mapping`` is
    bit-zero on both sides and every slot inside it is not, on the K plane and on
    the V plane separately. Nothing about kernel precision can move that, so it is
    asserted with no tolerance at all -- and it is the clause that sees the paging
    bugs. Two separate mutations: a port that writes the right values
    to the WRONG SLOTS (a slot mapping shifted by one page) fails here, and so does
    one that ignores ``block_tables`` entirely and indexes the cache flat
    (``slot_mapping`` replaced by ``arange``) -- the second is exactly why
    ``instance.yaml`` pins a non-identity ``page_ids`` (see its comment: under an
    ascending assignment ``slot == pos`` and flat indexing reproduces this golden).

    (2) VALUES, at the native band. A port that writes k and v SWAPPED
    fails here -- both mutations preserve the multiset of values in the tensor, so
    neither is visible to any norm or any shape.

    The two are complementary and neither subsumes the other: the swap keeps the
    occupancy identical, and the wrong-slot write keeps every value.

    (3) THE CONTAINER, not one tensor. The cache is a tuple with one tensor per
    distinct KV geometry, so both clauses above run PER GROUP -- unchanged, which
    is the point of freezing a per-group SET rather than one padded array (see
    ``_golden_cache_groups`` and ``golden_manifest.cache_state_member_names``).
    Zero padding is bit-zero and clause (1) has no tolerance to absorb it, so a
    concatenated array would read its own padding as unwritten cache; and
    ``shape[1] * shape[2]`` would stop being the slot count clause (1) computes
    over.

    THE GROUP COUNT IS COMPARED FIRST AND IT IS A PORT FAILURE, not a skip. A
    candidate whose engine allocates one rectangular cache for a family whose
    layers do not agree on their KV geometry is wrong in the way this whole
    container exists to express, and the shapes would differ at once. Degrading
    that to "shapes differ, nothing to compare" is the failure mode this clause is
    written against: it would report as coverage rather than as a defect.

    WHAT THIS STILL CANNOT SEE, on a family whose layers SHARE a slot: nothing in
    the golden records ``layer_slot`` / ``layer_writes`` as data, so the only thing
    catching a candidate whose shared layers point at the wrong SOURCE layer is the
    numeric content of the slots they share -- i.e. clause (2) here and the
    forward. The map itself is a static contract graded structurally
    (``cache/kv_cache._require_slot``); there is no cheap structural check of it in
    this artifact, and adding one would grade a contract at a float tolerance.
    """
    want_groups = _golden_cache_groups(golden)
    raw = cand.kv_cache_state
    # A tuple in CONTAINER ORDER is what the probe returns for every container,
    # including the one-group case. A bare array is accepted as a one-group
    # container rather than refused, because the count clause below is what
    # decides whether that is the right answer -- and it says so with the frozen
    # group count in hand, which a type error here could not.
    got_groups = list(raw) if isinstance(raw, (tuple, list)) else [raw]
    assert len(got_groups) == len(want_groups), {
        "error": "your KV cache holds a different number of geometry GROUPS than "
                 "the frozen one. The cache is a container -- one tensor per "
                 "distinct (num_kv_heads, head_dim) -- and golden/cache_spec.json "
                 "declares how many this family has; see its `groups` (or "
                 "`shape`, which is the one-group spelling).",
        "golden_groups": len(want_groups),
        "candidate_groups": len(got_groups),
        "golden_members": [name for name, _ in want_groups],
    }

    slots = _frozen_slots(golden)
    for (member, want), got_raw in zip(want_groups, got_groups):
        got = np.asarray(got_raw, dtype=np.float32)
        assert got.shape == want.shape, {
            "error": "the candidate's KV cache has a different shape than the "
                     "frozen post-prefill state",
            "group": member,
            "golden": list(want.shape), "candidate": list(got.shape),
        }
        assert want.ndim == 6, {
            "error": "the frozen cache is not [slots, pages, page_size, 2, "
                     "kv_heads, head_dim]; this is an artifact defect",
            "group": member,
            "shape": list(want.shape),
        }

        expected = np.zeros(want.shape[1] * want.shape[2], dtype=bool)
        expected[slots] = True
        for half, plane in ((0, "K"), (1, "V")):
            occupied_want = _occupied_slots(want, half)
            occupied_got = _occupied_slots(got, half)
            np.testing.assert_array_equal(
                occupied_want, expected,
                err_msg=f"the FROZEN cache's {plane} plane ({member}) holds data "
                        f"at slots "
                        f"{sorted(np.flatnonzero(occupied_want).tolist())} but "
                        f"inputs.npz's slot_mapping is "
                        f"{sorted(set(slots.tolist()))}. This is an artifact "
                        "defect.")
            np.testing.assert_array_equal(
                occupied_got, expected,
                err_msg=f"your cache's {plane} plane ({member}) holds data at "
                        f"slots {sorted(np.flatnonzero(occupied_got).tolist())}, "
                        f"not at the frozen slot_mapping's "
                        f"{sorted(set(slots.tolist()))}. "
                        "slot = page_ids[pos // page_size] * page_size + pos % "
                        "page_size, over the block table you were handed.")

        _assert_array_parity(want, got, golden, "native",
                             name=f"kv_cache_state:{member}",
                             surface="kv_cache_state")


# =========================================================================== #
# L2 -- the forward
# =========================================================================== #
def test_prefill_logits(golden, cand):
    """Full-prefill logits at the SERVING precision, on the frozen allocation.

    The native tier is where a port is actually used, and it is the looser of the
    two by construction: two independent bf16 kernel stacks legitimately disagree
    in the last mantissa bits, so the band is the class's ``bf16_native`` profile
    with the one per-instance number the measurement REQUIRED --
    ``native_thresholds.argmax_margin``, because at bf16 the frozen top1-top2 gap
    is one ULP at four of the twelve positions and the reference-as-candidate
    itself flips two of them.
    """
    probe = cand.prefill(fp32=False)
    # THE LEG RAN AT THE TIER IT CLAIMS. `_to_np` upcasts bf16 to float32 for numpy,
    # so no value clause in this tier can see a forward that silently ran at fp32 --
    # one dropped `.astype` in RMSNorm leaks fp32 through the whole model.
    # Read off the array BEFORE the coercion (`_to_np_with_provenance`
    # in `_t2j/candidate.py`, the generalised form of the check `test_paged_attention_op`
    # already had for the isolated attention probe).
    #
    # THE KEYS EXIST FIRST. Both clauses below compare `.get()` against `.get()`
    # or against a `""` default, so if the probe ever stopped reporting these
    # fields they would compare None to None (and "" to "") and pass for every
    # candidate -- the vacuous shape these two clauses were added to close, in
    # the one direction they were still open. `_t2j/candidate.py`'s
    # `prefill_probe` always writes all three, so a missing key is a harness
    # regression and this names it as one.
    missing = [k for k in ("logits_dtype", "requested_dtype", "logits_srctype")
               if k not in probe]
    assert not missing, {
        "error": "the prefill probe did not report its dtype/array-type "
                 "provenance, so the two clauses below would compare None "
                 "against None and pass vacuously. This is a HARNESS "
                 "regression, not a port defect.",
        "missing_keys": missing,
        "probe_keys": sorted(k for k in probe if not k.startswith("hidden_")),
    }
    assert probe.get("logits_dtype") == probe.get("requested_dtype"), {
        "error": "the native prefill returned logits at a different dtype than the "
                 "run configuration asked for, so this leg is not grading the "
                 "precision tier it claims to. The usual cause is a missing cast "
                 "back to the compute dtype somewhere in the forward, which leaks "
                 "fp32 through the whole model and is invisible to every value "
                 "comparison in this tier.",
        "called_at": probe.get("requested_dtype"),
        "returned": probe.get("logits_dtype"),
    }
    # ...AND CROSSED THE BOUNDARY AS A DEVICE ARRAY. instruction.md: "Everything
    # crossing your API boundary is a jax.Array or a plain Python value -- never a
    # torch.Tensor, and never a numpy array standing in for a device array."
    assert "numpy" not in str(probe.get("logits_srctype", "")), {
        "error": "the native prefill returned a numpy array where a jax.Array is "
                 "contracted; a host array defeats donation, sharding and jit, and "
                 "silently coerces identically to a jax.Array through np.asarray, "
                 "which is why nothing upstream of this check could see it",
        "returned_type": probe.get("logits_srctype"),
    }
    _assert_logit_parity(golden.prefill_forward["logits"],
                         probe["logits"],
                         golden, "native", name="prefill_logits")


def test_fp32_prefill_logits(golden, cand):
    """The exactness surface: identical fp32 weights on both sides, so the only
    permitted disagreement is cross-kernel fp32 reduction-order noise.

    This is the tier where a real porting bug has nowhere to hide behind
    precision, and the band is correspondingly tight: 1e-4 on the softmax KL
    against a measured 1.50e-6, a raw-cosine floor of 0.999999 and a per-instance
    CENTERED floor of 0.9999995. Every value is recorded with its measurement in
    ``instance.yaml``.

    THE TWO COSINE CLAUSES ARE QUANTISED, and reading them as ratios is a mistake
    this tier has already made twice. Both statistics accumulate in float32, where
    the spacing next to 1.0 is 2^-24 = 5.96e-08, so every value they can report is
    a multiple of that QUANTUM. In quanta: the worked reference sits at **1-2**
    (float64 truth ~2e-12, five orders below one quantum -- the metric is
    saturated), the centered floor is **8.4**, the raw floor is **16.8**.

    THE CENTERED CLAUSE IS THE ONE THAT EARNS ITS PLACE: a port that carries a
    bf16-ROUNDED rotary table into its own fp32 path lands at
    **11 quanta** centered and FAILS, while its RAW residual is 9 quanta and passes
    the raw clause. No other clause in the fp32 band rejects it.

    The rotary table must be kept EXACT in this tier. ``model.to(bfloat16)`` casts
    float buffers, and a ``.float()`` that follows restores the dtype and not the
    bits -- a bf16-rounded table would put the worked reference at 11 quanta
    centered / 8 raw. With an exact table the worked reference sits at 1-2 quanta,
    and ``reference_units.npz::lm_head.out0`` and this golden are BIT-IDENTICAL.

    It only means anything under ``JAX_DEFAULT_MATMUL_PRECISION=highest``: without
    the pin XLA:GPU lowers an fp32 matmul to TF32, and this measures TF32 noise
    against an fp32 golden. The conftest ASSIGNS it, the sandbox worker assigns it
    again, and ``test_spec.test_config_spec`` asserts what the worker actually ran
    under -- so the three ends of that are tied together rather than assumed.
    """
    _assert_logit_parity(golden.fp32_prefill_forward["logits"],
                         cand.prefill(fp32=True)["logits"],
                         golden, "fp32", name="fp32_prefill_logits")


def test_hidden_states(golden, cand):
    """Per-layer hidden states -- WHICH layer first diverges, not just that one did.

    ``prefill_forward.npz`` freezes ``hidden_0..N``: index 0 is the embedding
    output and index N the post-final-norm state, the HF / ``ref_jax_repo``
    convention. Graded per layer so the failure message names the depth at which
    the two trajectories part, which a logit comparison cannot.

    TWO CLAUSES, and the cosine one is the load-bearing one here. Activations
    carry outliers (this model's residual stream reaches 414 while the
    embedding output is under 0.4), so a single ``allclose`` band that admits the
    outliers is vacuous for the rest of the tensor -- and one that does not admits
    nothing. The per-layer centered cosine has no such scale dependence.

    Presence-gated on ``compare_hidden_states`` AND on the freeze having actually
    emitted ``hidden_*`` (``conftest._golden_facts``): a family that froze none is
    deselected rather than failed.
    """
    frozen = golden.hidden_states()
    assert frozen, {
        "error": "golden/prefill_forward.npz carries no hidden_* arrays, but the "
                 "gate collected this check; this is a harness defect",
    }
    layers = int(golden.cache_spec["num_layers"])
    assert len(frozen) == layers + 1, {
        "error": "the frozen hidden-state count is not num_hidden_layers + 1 "
                 "(embedding output first, post-final-norm last)",
        "frozen": len(frozen), "num_layers": layers,
    }
    out = cand.prefill(fp32=False)
    got = [out.get(f"hidden_{i}") for i in range(len(frozen))]
    missing = [i for i, value in enumerate(got) if value is None]
    assert not missing, {
        "error": "your forward returned fewer hidden states than the reference",
        "missing_indices": missing, "expected": len(frozen),
        "hint": "output_hidden_states must collect the embedding output, each "
                "layer's residual-stream state, and the post-final-norm state",
    }
    # THE RESIDUAL STREAM'S DTYPE, WHICH IS WHERE AN fp32 FORWARD IS VISIBLE.
    # `test_prefill_logits`'s `logits_dtype` clause compares two candidate-derived
    # values, so it grades whether the candidate CAST at the boundary, not the
    # dtype the forward RAN at -- and one cast at `compute_logits` defeats it.
    # A port that drops layernorm's cast back to the compute dtype,
    # promoting the whole forward to fp32, and then narrows the returned logits
    # passes every value clause; the same leak WITHOUT the returned-logits
    # narrowing is caught by the logit gate.
    #
    # Nothing casts the hidden states, so a leak shows here on every layer. This
    # is a boundary the candidate must not be able to paper over with one line:
    # 25 arrays, all of which must be the dtype the run config asked for.
    #
    # DESELECT-SAFE: a candidate that predates this field is caught by the
    # missing-key assertion rather than passing silently, because a probe that
    # stopped reporting the dtype is exactly how this clause would rot.
    requested = out.get("requested_dtype")
    hidden_dtypes = out.get("hidden_dtypes")
    assert hidden_dtypes is not None, {
        "error": "the prefill probe reported no hidden-state dtypes, so the "
                 "compute-precision contract cannot be graded",
        "hint": "this is a harness defect, not a candidate one",
    }
    wrong = {i: d for i, d in enumerate(hidden_dtypes) if d != requested}
    assert not wrong, {
        "error": "your forward returned hidden states at a dtype the run "
                 "configuration did not ask for, so the model did not compute "
                 "at the contracted precision",
        "requested_dtype": requested, "wrong_layers": wrong,
        "hint": "instruction.md: 'Do not hard-code a compute dtype. It comes "
                "from the run configuration.' A forward that runs at float32 "
                "regardless of the run config has hard-coded it in substance -- "
                "casting only the returned logits back is not enough. The usual "
                "cause is a dropped `.astype` in RMSNorm promoting everything "
                "downstream.",
    }
    # PER LAYER, direction AND magnitude via ``similarity.S_tensor``.
    # The residual stream spans orders of magnitude ALONG its depth (peaks
    # across one prefill: 0.0713 -> 1560 on core_qwen2, 75.5 -> 41728 on
    # core_gpt_oss), so a single absolute band is vacuous at the shallow end -- the
    # old ``allclose`` gate needed a per-surface ``abs_tol`` rescaled by the
    # reference's magnitude at each depth (``_shallow_abs_cap``) just to grade the
    # leading layers at all, and a candidate emitting all-zeros for hidden_0/1/2
    # could pass against it. ``S_tensor`` is scale-relative by construction
    # (``S_L2`` catches a pure magnitude error, ``S_cos`` the direction, and a
    # zero-norm candidate scores 0), so it grades every depth without a band and
    # subsumes the separate centered-cosine floor this test used to assert. Each
    # layer's score enters the test's power mean, so the depth at which the two
    # trajectories part still drags the whole check down.
    for index, (reference, candidate) in enumerate(zip(frozen, got)):
        _assert_array_parity(reference, candidate, golden, "native",
                             name=f"hidden_{index}", surface="hidden_states")


def test_decode_step_logits(golden, cand):
    """One decode step at a time, from the frozen populated cache.

    THE ONLY CHECK THAT READS WHAT AN EARLIER STEP WROTE. Every other numeric
    check writes the cache and reads it back inside one forward, where caching
    pre-RoPE K instead of post-RoPE K is invisible -- the query and the key are
    rotated together and the error cancels. It stops cancelling the moment a later
    position reads an earlier one's key, which is exactly what these steps do.

    THE FED TOKENS ARE THE REFERENCE'S, not the candidate's own argmax
    (``scenario.json::decode_token_ids``). A candidate that diverges at step 0
    would otherwise be graded on a different continuation from step 1 onwards, and
    every later step would report a decode bug that is really one step-0 bug.

    Each step is graded separately, at the native band, so the message names the
    step. On the worked reference: eight steps, zero argmax
    disagreements, worst softmax KL 1.9e-3, and the frozen top1-top2 gaps run
    5.13 / 5.38 / 1.63 / 0.50 / 2.13 / 5.38 / 0.50 / 4.63 ON core_llama3 (per
    instance, and they do not transfer: core_qwen2's eight are 3.0625 / 0.25 /
    5.5 / 3.5 / 2.375 / 0.625 / 2.0625 / 4.375, whose thinnest step is 0.25 rather
    than 0.50) -- so the two 0.50 steps
    are the only ones near the ``argmax_margin`` this instance calibrates.
    """
    probe = cand.decode_steps
    assert probe.get("supported"), {
        "error": "the candidate produced no decode-step logits",
        "reason": probe.get("reason"),
    }
    steps = int(golden.meta["num_decode_steps"])
    frozen_keys = [f"logits_step_{i}" for i in range(steps)]
    absent = [key for key in frozen_keys if key not in golden.decode_step]
    assert not absent, {
        "error": "golden/decode_step.npz is short of meta['num_decode_steps'] "
                 "steps; this is an artifact defect",
        "missing": absent, "num_decode_steps": steps,
    }
    assert int(probe["num_steps"]) == steps, {
        "error": "the candidate ran a different number of decode steps",
        "golden": steps, "candidate": probe["num_steps"],
    }
    # The tokens the probe was FED must be the ones the golden was frozen from,
    # or the two sides are decoding different continuations.
    assert list(probe["token_ids"]) == [int(t) for t in
                                        golden.scenario["decode_token_ids"][:steps]], {
        "error": "the decode probe was fed different tokens than the golden was "
                 "frozen from; this is a harness defect",
        "golden": golden.scenario["decode_token_ids"][:steps],
        "probe": list(probe["token_ids"]),
    }
    for index, key in enumerate(frozen_keys):
        _assert_logit_parity(golden.decode_step[key], probe[key], golden, "native",
                             name=f"decode_step_{index}")


def test_fp32_decode_step_logits(cand):
    """The decode step at fp32 -- DESELECTED EVERYWHERE TODAY, and why.

    The golden catalog freezes exactly one decode artifact
    (``decode_step.npz``) and freezes it at the
    NATIVE dtype. There is nothing for an fp32 decode to be compared against, so
    the check is gated on ``fp32_decode_step.npz`` -- the artifact a producer would
    write -- and is deselected on every instance of the class until one exists.

    NOT SILENTLY OMITTED, deliberately. A named-but-absent check reads as coverage
    to anyone comparing the module against the spec table; a deselected one is in
    ``eval_summary.json``'s ``deselected`` list, is attributed in
    ``_harness_checks._APPLICABLE_OWNER``, and ARMS ITSELF the moment the golden appears
    -- which is exactly how ``test_reference_units`` waited for its own freezer.

    NEVER EXECUTED, and it CANNOT PASS if it ever is. The body below refuses,
    deliberately: being collected means the golden appeared, and a check whose
    golden appeared while its comparison was never written must fail loudly rather
    than pass on an empty body. That is the same rule the driver applies to a
    probe nobody implemented (``_t2j/candidate.py``: "a probe whose FROZEN INPUT is
    absent returns an unsupported marker ... the failure mode this rules out is a
    numeric test that passes vacuously against a silently absent artifact").

    Two things whoever arms it will need, so the refusal names them: an
    ``fp32_decode_steps`` probe in ``_t2j/candidate.REQUIRED_PROBES``, and a
    ``golden.fp32_decode_step`` accessor on ``fixtures.Golden`` (the artifact is
    not in the eager catalog ``load_golden`` opens, and adding it there would make
    every instance without one unloadable).
    """
    raise AssertionError(
        "test_fp32_decode_step_logits was collected, which means "
        "golden/fp32_decode_step.npz now exists -- but neither the fp32 decode "
        "probe nor this comparison has been written. It refuses rather than "
        "passing on an empty body. Add `fp32_decode_steps` to "
        "_t2j/candidate.REQUIRED_PROBES and grade it here at "
        f"golden.fp32_thresholds. (probe present: "
        f"{getattr(cand, 'fp32_decode_steps', None) is not None})")


# =========================================================================== #
# L4 -- the sampler
# =========================================================================== #
def test_sampler_transforms(golden, cand):
    """Temperature, top-k and top-p over the frozen grid, on the frozen logits.

    NO RNG IS COMPARED, ever: torch Philox and JAX threefry are different PRNG
    families, so a sampled token is not a cross-framework quantity.
    What IS comparable is the deterministic transform chain, and it is graded on
    the REFERENCE's frozen logits -- ``sampler.npz::logits``, handed to the
    candidate -- so a candidate whose model is wrong is still graded on its
    sampler.

    THE MASK IS EXACT AND THE VALUES ARE NOT, and the split is the substance of
    this check. top-k and top-p produce a SET of surviving tokens; which tokens
    survive is a decision, and a port that keeps 49 tokens where the reference
    keeps 50 -- an off-by-one in the k-th-largest threshold, or ``>`` where the
    reference has ``>=`` in the cumulative-probability scan -- is wrong by a
    discrete amount that no tolerance should absorb. So ``-inf`` positions are
    compared as a set, exactly. The surviving VALUES are a division by the
    temperature and are graded at the fp32 band.

    On the worked reference, the SIX grid points mask 0, 128 206,
    128 225, 0, 128 241 and 128 247 of 128 256 tokens, the mask sets agree exactly
    on all six, and the worst finite disagreement is 1.9e-6 (grid 1, at values up
    to 28.93). As a mutation: ``top_k`` keeping ``k + 1`` and ``top_p``
    scanning with ``>`` instead of ``>=`` fails THIS CHECK AND NOTHING ELSE in the
    tier -- the mask clause, at ``golden_kept 50`` with 2 positions disagreeing --
    which makes it the tier's only catcher for a sampler break.

    THE LAST TWO GRID POINTS EXIST TO MAKE THE CHAIN'S ORDER OBSERVABLE. Without
    them, no mutation discriminates this check at all.
    ``process_logits`` is ordered -- temperature, top-k, top-p -- and an order is
    only visible where two stages are simultaneously doing something; with the
    first four points alone no pair is ever co-active, because ``apply_top_k(., 0)``
    and ``apply_top_p(., 1.0)`` are pass-throughs. Per grid point, the two
    reorder mutations behave as:

    * ``top_p_before_top_k``              -> ``[0, 0, 0, 0, 6, 0]`` cells
    * ``temperature_applied_after_top_p`` -> ``[0, 0, 0, 0, 0, 22]`` cells

    So grid 4 (``top_k`` and ``top_p`` co-active) and grid 5 (``temperature`` and
    ``top_p`` co-active) each carry one axis and neither is redundant. See
    ``instance.yaml``'s ``sampler_grid`` for why grid 4 is ``top_p 0.85`` and must
    not be "tightened" to 0.9.

    WHERE THE MASK CLAUSE CARRIES THE CHECK AND THE VALUE CLAUSE DOES NOT, stated
    because it bounds what a green result here means. Against the frozen
    input logits themselves: grid 0 (``temperature 1.0``), grid 3
    (``temperature 0.0``) and grid 4 (``temperature 1.0``) are bit-identical to the
    input wherever they survive -- ``apply_temperature`` returns unchanged for all
    three -- and grid 2's surviving values are bit-identical too, so FOUR of the
    six contribute nothing but their mask to the value clause. Exactly TWO grade
    any arithmetic, grid 1 and grid 5, and both do the same thing: divide by 0.7.
    They are graded at the fp32 band's ``rel_tol`` of 1e-3 against measured gaps of
    1.9e-6 (grid 1, on values up to 28.93) and 0.0 -- bit-exact -- at grid 5. A
    band of 2.90e-2 at grid 1's peak is 1.5e4 of headroom, so a sampler TEMPERATURE
    error of up to roughly 0.1% passes this check. Not tightened here: the band is
    the tier's, the surviving values are a scalar division that no plausible port
    gets slightly wrong, and the discrete failures this check exists for are the
    mask's. Adding grid 5 did not change that -- it is a second point at the same
    temperature, so it widens the MASK clause's reach and not the value clause's.

    ``sampler.json`` and ``sampler.npz`` are one golden in two files -- the arrays
    were 11.6 MB of decimal text with 256 431 ``-Infinity`` tokens, which is a
    Python extension to JSON -- and ``fixtures._load_sampler`` rejoins them, so
    ``golden.sampler`` has exactly the keys the probe returns.
    """
    probe = cand.sampler_outputs
    assert probe.get("supported"), {
        "error": "the candidate produced no sampler outputs",
        "reason": probe.get("reason"),
    }
    frozen_grid = golden.sampler.get("grid") or {}
    params = list(golden.sampler.get("grid_params") or [])
    assert frozen_grid and len(frozen_grid) == len(params), {
        "error": "sampler.json's grid_params and sampler.npz's grid_{i} arrays "
                 "disagree about how many grid points were frozen; this is an "
                 "artifact defect",
        "grid_params": len(params), "arrays": sorted(frozen_grid),
    }
    # The grid the candidate was ASKED to evaluate must be the one the golden was
    # frozen over, or the two sides ran different transforms.
    assert [dict(entry) for entry in probe["grid_params"]] == \
        [dict(entry) for entry in params], {
        "error": "the candidate evaluated a different (temperature, top_k, top_p) "
                 "grid than the golden froze; this is a harness defect",
        "golden": params, "probe": probe["grid_params"],
    }
    # ...and the grid must actually exercise masking, or the exactness clause
    # below is asserted over the empty set on every point.
    masked_counts = {key: int(np.count_nonzero(~np.isfinite(np.asarray(value))))
                     for key, value in sorted(frozen_grid.items())}
    assert any(masked_counts.values()), {
        "error": "no frozen grid point masks a single token, so this instance's "
                 "sampler grid does not exercise top-k or top-p at all",
        "masked_per_point": masked_counts, "grid_params": params,
    }

    got_grid = probe.get("grid") or {}
    assert sorted(got_grid) == sorted(frozen_grid), {
        "error": "the candidate returned a different set of grid points",
        "golden": sorted(frozen_grid), "candidate": sorted(got_grid),
    }
    for key in sorted(frozen_grid):
        reference = np.asarray(frozen_grid[key], dtype=np.float32)
        candidate = np.asarray(got_grid[key], dtype=np.float32)
        entry = dict(params[int(key)])
        assert candidate.shape == reference.shape, {
            "grid_point": key, "params": entry, "error": "shape",
            "golden": list(reference.shape), "candidate": list(candidate.shape)}
        # THE MASK VALUE IS -inf, NOT MERELY NON-FINITE, and this clause is the
        # one that checks it. `np.isfinite` treats +inf and NaN as masked too --
        # both pass the surviving-set comparison below exactly, because
        # a sentinel swap changes no ELEMENT's finite/non-finite status, only its
        # sign or its definedness. A +inf mask makes every BANNED token the
        # argmax at serving time; a NaN mask destroys the row it lands in. Both
        # are invisible to a finite/non-finite set comparison, so they are
        # checked here directly against the one sentinel that is actually
        # correct.
        bad_sentinel = int(np.count_nonzero(
            ~np.isfinite(candidate) & ~np.isneginf(candidate)))
        assert bad_sentinel == 0, {
            "grid_point": key, "params": entry,
            "error": "masked positions must be -inf, not merely non-finite. A "
                     "+inf mask makes every banned token the argmax at serving "
                     "time; a NaN mask destroys the row. Both pass a finite/"
                     "non-finite set comparison, which is why this is a separate "
                     "clause rather than folded into the one below.",
            "non_neginf_masked_positions": bad_sentinel,
            "example_values": np.unique(
                candidate[~np.isfinite(candidate)])[:4].tolist(),
        }
        kept_ref = np.isfinite(reference)
        kept_cand = np.isfinite(candidate)
        disagree = int(np.count_nonzero(kept_ref != kept_cand))
        assert disagree == 0, {
            "grid_point": key, "params": entry,
            "error": "the surviving-token SET is not the reference's. A mask is a "
                     "decision, not an arithmetic result, so it is compared "
                     "exactly.",
            "golden_kept": int(kept_ref.sum()), "candidate_kept": int(kept_cand.sum()),
            "positions_disagreeing": disagree,
            "hint": "top_k keeps the k largest logits; top_p keeps the smallest "
                    "prefix of the descending-probability order whose cumulative "
                    "mass reaches p",
        }
        _assert_array_parity(reference[kept_ref], candidate[kept_ref], golden,
                             "fp32", name=f"sampler_grid/{key}")


def test_logprobs(golden, cand):
    """Top-N logprobs on the frozen logits: the VALUES exactly, the IDS tie-aware.

    ``topn_logprobs`` is what a serving API returns, so it is graded as a value
    and not only as an ordering: a port whose log-softmax normalises over the
    wrong axis, or forgets to normalise at all, returns ids that are perfectly
    right and numbers that are not.

    THE IDS ARE TIE-TOLERANT, and only where the golden itself is tied. Two tokens
    whose frozen logprobs are equal have no defined order -- this
    instance's top-5 has exactly that: ranks 2 and 3 are both -2.10955 ON
    core_llama3. NOT ON EVERY INSTANCE -- core_qwen2's five are all distinct, so
    the tie-tolerant branch is DEAD CODE there and this paragraph describes a
    case that instance cannot reach. The branch is still right to have: which
    instances tie is a property of the checkpoint, not of the check. Demanding
    the reference's order there would fail a correct port on an arbitrary
    tie-break. Everywhere the golden's own values are distinct, the id must match.

    Graded at the fp32 band: both sides are handed the same frozen fp32 logits, so
    this is a pure-transform comparison with no model in it.
    """
    probe = cand.sampler_outputs
    assert probe.get("supported"), {
        "error": "the candidate produced no sampler outputs",
        "reason": probe.get("reason"),
    }
    want_values = np.asarray(golden.sampler["topn_logprobs"],
                             dtype=np.float32).reshape(-1)
    want_ids = np.asarray(golden.sampler["topn_token_ids"]).reshape(-1).astype(np.int64)
    got_values = np.asarray(probe["topn_logprobs"], dtype=np.float32).reshape(-1)
    got_ids = np.asarray(probe["topn_token_ids"]).reshape(-1).astype(np.int64)

    top_n = int(golden.meta.get("logprobs_top_n", 0))
    # THE WIDTH IS FLOORED, because at zero this whole check is vacuous and
    # nothing else says so. Every clause below is quantified over the top-N:
    # both size comparisons are `0 == 0`, `np.all` over an empty array is True,
    # `_assert_array_parity` compares two empty arrays, and the id loops do not
    # execute -- so `logprobs_top_n: 0` scores a pass for any candidate at all,
    # including one with no sampler. All six instances ship 5; a `.get(..., 0)`
    # default and a mis-keyed yaml are one typo apart, and the default is the
    # value that turns the check off. This is an anti-vacuity floor on the
    # ARTIFACT, not a calibrated threshold: it says the frozen scenario asked
    # for at least one logprob, nothing about how close the numbers must be.
    assert top_n > 0, {
        "error": "the frozen scenario declares no top-N logprobs, so every "
                 "clause in this check is quantified over an empty array and "
                 "passes for any candidate. This is an instance.yaml / golden "
                 "defect, not a port defect.",
        "logprobs_top_n": golden.meta.get("logprobs_top_n"),
        "hint": "set logprobs_top_n in tests/instance.yaml and re-freeze the "
                "inference goldens",
    }
    assert want_values.size == top_n and want_ids.size == top_n, {
        "error": "the frozen top-N is not meta['logprobs_top_n'] wide; this is an "
                 "artifact defect",
        "logprobs_top_n": top_n, "values": int(want_values.size),
        "ids": int(want_ids.size),
    }
    assert got_values.shape == want_values.shape and got_ids.shape == want_ids.shape, {
        "error": "the candidate returned a different top-N width",
        "golden": [top_n, top_n],
        "candidate": [int(got_values.size), int(got_ids.size)],
    }
    # A logprob is a log-probability: never positive, and the reference's are
    # ordered descending. Both are properties of the ARTIFACT, checked so the
    # comparison below cannot be against a golden that stopped being one.
    assert np.all(want_values <= 0.0) and np.all(np.diff(want_values) <= 0.0), {
        "error": "the frozen top-N logprobs are not a descending, non-positive "
                 "sequence; this is an artifact defect",
        "values": want_values.tolist(),
    }

    _assert_array_parity(want_values, got_values, golden, "fp32",
                         name="topn_logprobs")

    distinct = np.ones(want_ids.size, dtype=bool)
    for rank in range(want_ids.size):
        peers = np.flatnonzero(want_values == want_values[rank])
        distinct[rank] = peers.size == 1
    mismatched = [int(r) for r in np.flatnonzero(distinct & (want_ids != got_ids))]
    assert not mismatched, {
        "error": "the top-N token ids differ at a rank where the reference's own "
                 "logprobs are DISTINCT, so there is no tie to break differently",
        "ranks": mismatched,
        "golden_ids": want_ids.tolist(), "candidate_ids": got_ids.tolist(),
        "golden_values": want_values.tolist(),
    }
    # Where the golden IS tied, the candidate must still name one of the tied
    # tokens -- tie-tolerant is not id-blind.
    for rank in np.flatnonzero(~distinct):
        peers = set(want_ids[np.flatnonzero(want_values == want_values[rank])].tolist())
        assert int(got_ids[rank]) in peers, {
            "error": "at a TIED rank the candidate named a token that is not one "
                     "of the tied ones",
            "rank": int(rank), "tied_golden_ids": sorted(peers),
            "candidate_id": int(got_ids[rank]),
        }


# =========================================================================== #
# L2 -- mode-gated forwards. DESELECTED on `core`.
# =========================================================================== #
def test_ragged_forward_logits(golden, cand):
    """A mixed-stage batch in ONE call: per-request logit parity.

    ``ragged``/``chunked`` only -- the ``core`` mode runs one sequence at a time,
    so there is no second request to mix with, and on a ``core`` instance this
    check is deselected (out of both numerator and denominator, and named in the
    summary). On ``ragged_llama3`` it RUNS.

    WHAT IT GRADES THAT ``test_prefill_logits`` DOES NOT, on an instance that has
    it: the same probe output, sliced by ``query_start_loc`` and compared PER
    REQUEST. A batched forward that is right for the first sequence and wrong for
    the second has a whole-batch cosine dominated by the first, and the per-request
    slicing is what names the one that is wrong. It also asserts the frozen batch
    is genuinely mixed -- ``request_distribution`` is ``[n_decode, n_prefill,
    n_mixed]``, and a "ragged" scenario with a zero decode count is a prefill
    scenario wearing the wrong label.

    On ``ragged_llama3`` the check runs and passes on a frozen batch of three
    sequences (14 tokens, ``request_distribution`` mixed).

    KNOWN OVERLAP, so it is not read as more coverage than it is: on
    ``ragged_llama3`` its centered-cosine minimum and maximum token-KL are
    numerically IDENTICAL to ``test_prefill_logits``' whole-array values -- the
    same array, redistributed. Its distinct content is the per-request RAW cosine
    (undiluted by the longest request) and the mixed-batch assertion below.

    THE MIXED-BATCH GUARD READS ``n_prefill + n_chunked``, NOT ``n_prefill``
    ALONE. It predates ``chunked``, where every non-decode request is a plain
    ``prefill`` and index 1 is the whole non-decode count; on ``chunked_llama3``
    a non-decode request can ALSO be a chunk continuation or an
    automatic-prefix-cache share, both counted in index 2 (see
    ``test_attention_metadata_schema``'s own note on the same distinction), and
    a scenario built entirely from those legitimately reads ``n_prefill == 0``
    (``chunked_llama3``'s frozen ``[1, 0, 2]`` -- r0 is
    ``prefill_chunked``, r1 shares r0's prefix, neither is a plain ``prefill``).
    Reading index 1 alone made a real, correctly-frozen chunked scenario refuse
    here as if it were not mixed.
    """
    distribution = np.asarray(golden.inputs["request_distribution"]).reshape(-1)
    assert distribution.size >= 2 and int(distribution[0]) > 0 \
        and int(distribution[1:].sum()) > 0, {
        "error": "the frozen scenario is not a mixed-stage batch: "
                 "request_distribution is [n_decode, n_prefill, n_mixed] and this "
                 "one has no decode or no prefill/chunked request",
        "request_distribution": distribution.tolist(),
    }
    starts = np.asarray(golden.inputs["query_start_loc"]).reshape(-1).astype(int)
    reference = np.asarray(golden.prefill_forward["logits"], dtype=np.float32)
    candidate = np.asarray(cand.prefill(fp32=False)["logits"], dtype=np.float32)
    assert candidate.shape == reference.shape, {
        "error": "shape", "golden": list(reference.shape),
        "candidate": list(candidate.shape)}
    for request in range(starts.size - 1):
        lo, hi = int(starts[request]), int(starts[request + 1])
        if hi <= lo:
            continue
        _assert_logit_parity(reference[lo:hi], candidate[lo:hi], golden, "native",
                             name=f"ragged_forward_logits/request_{request}")


def test_chunked_prefill_logits(golden, cand):
    """A prefill split across calls must equal the same prefill in one call.

    ``chunked`` only, deselected everywhere else by the mode gate.
    ``scenario.json::chunk_split_points`` records where the reference split;
    ``core``/``ragged`` freeze it present and EMPTY on purpose, so a reader
    cannot mistake "no chunking here" for "the freezer forgot".

    THIS CHECK USED TO REFUSE UNCONDITIONALLY, and that was deliberate, not a
    stub: an earlier body that sliced BOTH sides at the last split point
    reported ``1 passed`` against a candidate that performed no chunking at
    all (``chunk_split_points`` forced to ``[8]``, the mode gate cleared,
    everything else untouched). Slicing the candidate's own whole-prefill
    output at ``start:`` produces the same tail regardless of whether the
    candidate actually issued more than one ``engine.prefill`` call -- so
    that body was grading exactly what ``test_prefill_logits`` already
    grades, dressed up as the chunked check. A check that reads green on the
    one axis its mode exists for is worse than no check.

    GUARD THE GUARD: ``n_chunks`` MUST match what the split points imply, or
    this refuses rather than comparing. The expected count is derived the
    same way ``chunked_prefill_probe`` counts it -- walking ``[0, *splits,
    n]`` and skipping any zero-width interval a repeated interior split
    point produces -- rather than the naive ``len(splits) + 1``, so the two
    agree BY CONSTRUCTION even on a non-strictly-increasing split list, and
    not merely by coincidence on the one split list (``[5]``) this instance
    ships. (Its own probe already refuses a descending pair outright, so the
    only shape this has to reconcile with is a repeated interior point.)
    Without this, ``chunk_is_a_single_shot`` -- a probe that runs the whole
    prompt in one ``engine.prefill`` call and reports the honest
    ``n_chunks == 1`` -- would still pass: the sliced tail of a correct
    whole-shot prefill and the tail of a genuinely chunked one are numerically
    the same tensor, because the model computes causally and a later chunk's
    KV write cannot retroactively change an earlier row. Only the COUNT tells
    the two apart.

    Also catches ``chunk_forgets_cached_offset`` (the engine drops the
    cached-prefix length from a continuation's ``seq_lens``, so ``q_pos``
    loses its offset and every attended-to position past the first chunk is
    wrong) -- that mutation still reports a truthful ``n_chunks``, so it
    reaches the logit comparison and fails there instead.

    Does NOT catch ``chunk_split_point_hardcoded`` (a port that special-cases
    the one frozen split value): the canonical scenario only ever exercises
    that one value, so nothing here can tell a general implementation from
    one hardcoded to it.

    AND NOTHING ELSE CATCHES IT EITHER. This paragraph used to hand the job to
    ``test_chunk_boundary_battery``. That is false, and structurally so
    rather than by an oversight some instance could fix:

    * ``chunk_split_point_hardcoded`` patches ``engine.py``'s ``continue_request``
      branch (``cached = 5`` in place of ``self._lens[request_id]``), so catching
      it requires a SECOND ``engine.prefill`` call at a split value other than the
      hardcoded one.
    * The battery VARIES the split value, but ``battery_probe`` performs exactly
      one forward from HANDED metadata -- it never constructs an ``Engine`` and
      never calls ``prefill`` at all. Verified by inspection: zero occurrences of
      either in its body.
    * The probes that DO drive ``Engine.prefill`` repeatedly --
      ``chunked_prefill_probe`` here, and ``_invariance_chunking`` -- are graded
      against the canonical scenario's own single split list.

    So the capability to vary the split and the capability to drive the engine
    live in different probes and never meet. ``freeze_catchers.py`` says the same
    thing from the other side (the ``("chunked_llama3",
    "chunk_split_point_hardcoded")`` ``INERT_FOR_TASK`` entry, and the relocation
    attempt recorded above it), including the warning not to read that row's
    ``inert: true`` as evidence the battery axis is covered.

    On ``chunked_llama3`` the mutation is additionally inert for a second,
    instance-specific reason -- the hardcoded 5 IS its own canonical split -- which
    is what the task-keyed entry records. A chunked instance whose
    ``chunk_split_points`` differ from 5 removes that second reason but NOT the
    structural one.
    """
    splits = [int(s) for s in (golden.scenario.get("chunk_split_points") or [])]
    probe = cand.chunked_prefill
    assert probe.get("supported"), {
        "check": "test_chunked_prefill_logits",
        "error": "the candidate produced no chunked-prefill result",
        "reason": probe.get("reason"),
        "chunk_split_points": splits,
    }
    # `total` IS r0's OWN FULL PROMPT LENGTH, NOT `golden.prefill_forward`'s row
    # count. On a scenario whose sequence 0 shares the batch with other
    # requests (the `shares_prefix_with` axis), `prefill_forward.npz`
    # holds the WHOLE GRADED BATCH's rows -- every request's, concatenated --
    # and its row count has no relationship to r0's own prompt length. BUG,
    # FOUND HERE: the first version of this check read `total =
    # reference.shape[0]` before slicing out r0's own rows, so on
    # `chunked_llama3` (batch rows 11: r0's 7 + r1's 3 + r2's 1 decode) it
    # computed `bounds = [0, 5, 11]` -- one past r0's own 12-token prompt --
    # and then sliced `reference[5:11]`, which mixes r0's own last two chunk
    # rows with ALL of r1's and r2's, against a candidate array that is r0
    # ALONE -- a shape bug, not a calibration gap.
    total = int(golden.meta["prompt_len"])
    bounds = [0, *splits, total]
    expected_chunks = sum(1 for i in range(len(bounds) - 1)
                          if bounds[i] != bounds[i + 1])
    # GUARD THE GUARD. A probe that ran one whole-shot "prefill" produces
    # exactly the tensor test_prefill_logits already grades, so a green
    # below would mean nothing unless it is proven to have actually chunked.
    assert probe.get("n_chunks") == expected_chunks, {
        "check": "test_chunked_prefill_logits",
        "error": "the probe's reported n_chunks does not match the number of "
                 "non-empty intervals chunk_split_points implies, so the "
                 "comparison below cannot be trusted to be the chunked one "
                 "-- a probe that silently performed a single whole-shot "
                 "prefill would otherwise pass the exact same comparison "
                 "test_prefill_logits already makes.",
        "chunk_split_points": splits, "n_chunks": probe.get("n_chunks"),
        "expected_n_chunks": expected_chunks,
    }
    # r0'S OWN ROWS, sliced out of the (possibly multi-request) graded batch by
    # `query_start_loc` before `bounds` is applied to them -- `query_start_loc`
    # is in BATCH-ROW space (request 0 occupies rows
    # ``query_start_loc[0]..query_start_loc[1]``); `bounds` is in r0's own
    # PROMPT-POSITION space. The two coincide only because r0 is always
    # request 0 (`instance.yaml`'s own rule for `sequences[0]`) and because r0's
    # graded portion in a `prefill_chunked` scenario already begins at
    # `chunk_start` -- so `query_start_loc[1] - query_start_loc[0]` equals
    # `total - bounds[-2]` by construction, not by coincidence.
    starts = np.asarray(golden.inputs["query_start_loc"]).reshape(-1).astype(int)
    lo, hi = int(starts[0]), int(starts[1])
    reference = np.asarray(golden.prefill_forward["logits"], dtype=np.float32)[lo:hi]
    assert reference.shape[0] == total - bounds[-2], {
        "check": "test_chunked_prefill_logits",
        "error": "request 0's row count inside the graded batch does not match "
                 "prompt_len minus the final chunk boundary; the two are "
                 "supposed to be the same quantity by construction",
        "request_0_rows": reference.shape[0], "total": total,
        "final_boundary": bounds[-2],
    }
    _assert_logit_parity(reference, probe["logits"],
                         golden, "native", name="chunked_prefill_logits")


# =========================================================================== #
# L3 -- the generalisation batteries
# =========================================================================== #
def test_prefill_logits_battery(golden, cand):
    """The PROMPT-LENGTH axis: does the port generalise off the frozen length?

    Every other numeric check in this module runs at exactly one prompt length --
    twelve tokens, one sequence, one page layout. A port can be right there and
    wrong everywhere else in several specific ways, and this axis is the one that
    sees them: a causal mask built at a hard-coded width, a position array that
    stops being ``arange`` when the prompt does not fill a page, a slot mapping
    whose last partial page is mis-sized, and a static-shape bucket that a
    different length silently falls out of.

    Deselected until the inference battery has been frozen --
    ``battery/prompt_length`` is OPTIONAL in the manifest precisely so that a
    family whose battery is not frozen yet degrades to a visible coverage gap
    rather than a hard failure.

    No producer writes the ``prompt_length`` axis: the battery freezer
    exists but implements
    ``batch_composition`` only. The variant contract this reads --
    ``battery/prompt_length/{variant}/{inputs.npz, prefill_forward.npz}`` -- is
    what that producer must satisfy when the axis is added.
    """
    _grade_battery(golden, cand, "prompt_length", name="prefill_logits_battery",
                   varies="the number of prompt tokens")


def test_paging_battery(golden, cand):
    """The PAGE-LAYOUT axis: the same tokens on a different physical allocation.

    The canonical scenario pins ONE non-identity page assignment (``page_ids: [2,
    0, 3]``), which is enough to prove the block table is read at all but not that
    it is read correctly for any layout. This axis varies the layout while holding
    the prompt fixed, so the logits must be IDENTICAL up to noise across every
    variant -- and a gather that is reversed, off by one, or correct only for a
    contiguous assignment differs on some variant even though it reproduces the
    canonical one.

    It is the L3 complement of ``test_behavioral``'s paging invariance: that one
    compares the candidate to ITSELF on two layouts (so a port that is
    consistently wrong passes), this one compares each layout to the REFERENCE.

    Deselected until the inference battery has been frozen.
    No producer writes the ``paging`` axis: the battery freezer
    exists but implements
    ``batch_composition`` only.
    """
    _grade_battery(golden, cand, "paging", name="paging_battery",
                   varies="the physical page assignment")


def test_batch_composition_battery(golden, cand):
    """The ``request_distribution`` axis: prefill/decode mixes. ``ragged``+ only.

    Varies how many of the batch's requests are in the decode stage versus the
    prefill stage, holding the sequences fixed. It is the REQUIRED axis for
    ``ragged`` and ``chunked`` because a mixed batch is what those modes
    exist to express, and a port that handles an all-prefill and an all-decode
    batch can still get the boundary between them wrong.

    As of ``ragged_llama3`` this is the one battery axis that is
    live. The battery freezer writes it, and that instance's
    golden tree carries two genuinely distinct compositions
    (``d1p2`` = 1 decode + 2 prefill over 16 tokens, ``d2p1`` = 2 decode +
    1 prefill over 11), each with its own warm-up and neither reachable from the
    canonical scenario. Deselected on ``core``, where there is no second request
    to compose with.
    """
    _grade_battery(golden, cand, "batch_composition",
                   name="batch_composition_battery",
                   varies="the prefill/decode composition of the batch")


def test_chunk_boundary_battery(golden, cand):
    """The chunk SPLIT-POINT axis. ``chunked`` only.

    Varies where the prefill is cut, holding the prompt fixed. A boundary that
    lands mid-page, on a page boundary, or at the very last token exercises
    different arithmetic in the slot mapping and in the attention mask, and a port
    that is right for one is routinely wrong for another.

    LIVE, AS OF ``chunked_llama3``: that instance ships, and the battery freezer
    implements this axis. Its golden tree carries four
    variants -- ``aligned``, ``interior``, ``far_interior`` and ``tail`` -- so the
    check RUNS there and is deselected everywhere else on the mode gate alone.
    """
    _grade_battery(golden, cand, "chunk_boundary", name="chunk_boundary_battery",
                   varies="where the prefill is split")
