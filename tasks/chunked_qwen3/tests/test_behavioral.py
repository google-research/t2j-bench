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

"""Behavioral stage: does the port BEHAVE the same, end to end?

The other two tiers grade pieces against frozen artifacts. This one grades the
engine as a running system: it drives the candidate's own decode loop, its own
pager and its own allocator, and it asks two different kinds of question.

CONTINUOUS, LIKE THE NUMERIC TIER. EIGHT of the fourteen
checks now grade FIDELITY on a [0,1] similarity rather than a pass/fail band, funnel
it through :func:`_score`, and the conftest aggregates the per-check scores with
``similarity.power_mean`` (p<1, low-tail sensitive) into ``behavior_score`` -- so a
port that BEHAVES nearly right but not exactly no longer banks full behavioral credit
off a green pass rate, and a single wholly-broken behavior bites super-linearly while
a uniform faithful set (~0.999) is unchanged. The scored eight are the four
golden-fidelity checks named below and the four toleranced self-parity checks
(``test_decode_prefill_consistency``, ``test_batching_invariance``,
``test_chunking_invariance``, ``test_prefix_cache_invariance``). The other SIX --
``test_paging_invariance``, ``test_cache_reuse_safety``, ``test_determinism``,
``test_no_cross_sequence_leakage``, ``test_generate_runs``, ``test_serve_end_to_end``
-- are bit-exact / structural / wiring invariants with no continuous signal; they stay
HARD GATES (conftest ``_BEHAVIORAL_GATE_TESTS``): a PASS banks no credit, a FAIL docks
the reward. Every scored check KEEPS its structural asserts (shape, dtype, argmax
agreement, page identity, probe support) as hard gates and grades only the fidelity.

FOUR CHECKS COMPARE AGAINST A FROZEN ARTIFACT: ``test_greedy_sequence_fp32``,
``test_greedy_sequence``, ``test_decode_trajectory`` and ``test_held_out_logprobs``
(the last reads ``golden.held_out["ref_logprobs"]``; ``test_serve_end_to_end``, by
contrast, reads only ``golden.meta`` and compares the candidate against itself, so
it does not belong on this list). ``test_greedy_sequence_fp32`` is the benchmark's
**named** equivalence claim -- same prompt, same sampled sequence, fixed seed --
and it is the only check in the suite that grades the decode loop, the pager and
the allocator as a SINGLE end-to-end verdict over N steps.

WHICH CHECK ACTUALLY CARRIES THE TIER, AND IT IS NOT THE NAMED ONE.
Over the seven mutations that score **Spec 17/17 and Numeric 9/9** -- the exact
set that justifies this tier existing at all:

| check | catches |
|---|---|
| ``test_decode_trajectory`` | **6 of 7** (all but M5) |
| ``test_decode_prefill_consistency`` | **4 of 7** |
| ``test_greedy_sequence_fp32`` | 4 of 7, three of them only at generated step 7 of 8 |
| ``test_greedy_sequence`` | 3 of 7 -- the same steps as its fp32 sibling |
| ``test_generate_runs`` | 1 of 7 |
| ``test_paging_invariance`` / ``test_cache_reuse_safety`` / ``test_determinism`` | 0 of 7 |
| the tier as a whole | **7 of 7** |

So the LIKELIHOOD check is the load-bearing one and the sequence check is the
published claim. A greedy continuation this strongly determined is nearly
insensitive to the distribution it is drawn from: a decode RoPE position
off-by-one is wrong from the FIRST decode step, drives the chosen-token logprob
291x outside the fp32 band, and still emits all eight reference tokens at both
precisions. **``num_decode_steps`` is a sensitivity parameter of the sequence
check**, not just a cost knob -- at 8 it is already marginal here, and at 7 it
would catch only the one bug that does not depend on the eighth step.

THE REST COMPARE THE CANDIDATE AGAINST ITSELF. A self-consistency property cannot
be overfitted to a frozen artifact and needs no freeze to arm, which is why the
design leans on them. Their yield is uneven and is
recorded per check rather than claimed as a family:
``test_decode_prefill_consistency`` drives the candidate's own ``Engine`` and
catches four decode-path bugs that both other tiers miss;
``test_paging_invariance`` runs a prefill only, never reaches the page boundary
at position 16, and catches **none of the** realistic paging/decode bugs put to
it, though it does catch a block table ignored outright, so it is thin rather
than inert; ``test_cache_reuse_safety`` catches an allocator that re-hands a
live page and a gather that reads ``bt + 1``, on two separate clauses;
``test_determinism`` passes for any implementation that is a pure function of
its inputs. Four of them read a
TOLERANCE out of ``meta`` (no frozen array enters the comparison), which is the
class rule about where tolerances live rather than a dependency on the
reference's answer.

TWO PRECISIONS, AND THE SPLIT IS EMPIRICAL, NOT STYLISTIC
---------------------------------------------------------
The equality clause runs at **fp32** and the likelihood clause is bounded at
fp32, because at native precision the quantities they compare are not separable
from noise. Reference-as-candidate:

* per-token chosen-token logprob gap: **3.755e-05** at fp32, **7.1e-02 ...
  1.2e-01** at native -- while ONE correct implementation's own cross-process
  spread is **4.96e-02**. A native logprob bound would need >= 2e-1, i.e. would
  grade nothing.
* the tokens themselves agree at both precisions on every run, so the equality
  clause is not carried by the precision choice -- but the fp32 leg is where a
  correct port cannot miss for
  ARITHMETIC reasons: the smallest fp32 margin on this scenario is 0.344 logits
  against a cross-port fp32 gap of 3.76e-05, a factor of ~9000.

NO ``xfail`` ANYWHERE IN THIS MODULE, and no ``skip``. Both are excluded from the
grade -- the binary tally the gates feed and the ``power_mean`` the scored checks
feed -- so a check conditioned on the CANDIDATE's own output that resolves to
``xfail`` deletes itself from the denominator because of how the candidate behaved.
A near-tie divergence is graded against a **frozen** margin instead
(:func:`test_greedy_sequence`), so the check always scores. Inapplicable checks are
DESELECTED by ``conftest`` before collection and named in ``eval_summary.json``.

WHAT IS DESELECTED ON A ``core`` INSTANCE, all four named in the
summary:

* ``test_batching_invariance``, ``test_no_cross_sequence_leakage`` -- need a
  second concurrent sequence; ``core`` runs one at a time.
* ``test_chunking_invariance``, ``test_prefix_cache_invariance`` -- need a split
  prefill, likewise.

A fifth check, ``test_held_out_nll``, is retired rather than repaired: it was
presence-gated on BOTH a frozen band and an out-of-band result -- neither of
which any producer or any caller ever wrote, in either task class. See the L6
banner at the foot of this module.

THRESHOLDS ARE NEVER HARD-CODED HERE, exactly as in ``test_numeric.py``. The five
scalars this module reads -- ``fp32_thresholds.logprob_atol``,
``fp32_thresholds.logprob_cumulative_atol``, ``fp32_thresholds.held_out_logprob_atol``,
``fp32_thresholds.held_out_seq_atol`` and ``native_thresholds.decisive_margin`` --
come from ``instance.yaml`` via ``golden/meta.json``, each recorded there with
its measured floor, its chosen value and the ratio. They are served through
``Golden.fp32_extras`` / ``Golden.native_extras`` (they are not
``NumericThresholds`` fields) and read with a KeyError-loud ``[...]``: a missing
calibrated threshold must kill the check, not silently grade at a default
nobody measured.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

# THE LOGIT GATE, IMPORTED RATHER THAN RESTATED. Four checks here compare
# two logit rows the CANDIDATE produced, and a logit row is defined only up to a
# per-token additive constant -- so an elementwise `allclose` is not an
# equivalence measure on it and a literal tolerance in this file
# would be the one thing the class forbids everywhere else.
#
# This is why it is an import and not a copy. An elementwise
# `assert_allclose(rtol=2e-3, atol=2e-3)` for `test_decode_prefill_consistency`
# FAILS on the WORKED REFERENCE -- 104 737 of 128 256 elements outside the band,
# max |d| 0.1328, which is one bf16 ULP at that magnitude. Graded through the
# calibrated native band
# instead, the same two rows sit at centered cosine 0.9999332 (floor 0.998, 30x
# headroom on the residual) and token KL 1.04e-03 (budget 0.1, 97x), with the
# argmax agreeing. The gate is not looser -- it is the right measure.
from test_numeric import _assert_logit_parity  # noqa: E402

# THE INTERIOR-BOUNDARY PREDICATE, imported for the reason `conftest` imports
# `second_prompt_length`: the probe decides whether the leg CAN run and this
# module decides whether to REQUIRE it, and one predicate written twice is a
# predicate that stops agreeing. See `test_decode_prefill_consistency`.
from _t2j.candidate import interior_boundary_position  # noqa: E402

# THE SCORING TOOLKIT, the same one test_numeric grades on. The banded-scalar and
# power-mean scorers turn this module's logprob bands into continuous [0,1] fidelity
# (see the scoring machinery below), so behavioral grades like numeric does.
from _t2j import similarity  # noqa: E402

# --------------------------------------------------------------------------- #
# Continuous behavioral scoring, MIRRORING test_numeric's mechanism.
# EIGHT checks grade FIDELITY the way the numeric stage does: each funnels a [0,1]
# similarity through :func:`_score`, and the conftest aggregates the per-check scores
# with ``similarity.power_mean`` (p<1, low-tail sensitive) into ``behavior_score``.
# The other SIX behavioral checks are pure GATES -- bit-exact / structural / wiring
# invariants named in the conftest's ``_BEHAVIORAL_GATE_TESTS`` -- so a gate PASS
# banks no continuous credit (it is kept OUT of the power mean) and a gate
# FAIL files 0.0. This mirrors the training grader so behavioral has the same
# anti-inflation shape: a uniform faithful set (~0.999) is unchanged, a single
# wholly-broken check bites super-linearly.
#
# THIS MODULE OWNS ITS OWN RECORDER, deliberately separate from test_numeric's. The
# four self-parity checks call :func:`test_numeric._assert_logit_parity`, whose body
# records into test_numeric's module-global recorder -- but that recorder is bound
# only by test_numeric's autouse fixture, which does NOT run for a test in THIS
# module, so that internal record is a no-op here. What this module grades on is the
# RAW per-call score the helper RETURNS (``{"score": ...}``), fed to the recorder
# bound below. Two independent accumulators, no cross-contamination.
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

    Mirrors :func:`test_numeric._score`. ``similarity.power_mean`` sanitises
    non-finite -> 0.0 and clips to [0,1], so a degenerate ``s`` cannot inflate the
    recorded score. Re-recording every call (rather than once at the end) keeps the
    value in the CALL-phase report the conftest reads, without a teardown hook.
    """
    _SCORE_ACC.append(float(s))
    agg = similarity.power_mean(_SCORE_ACC)
    if _RECORDER is not None:
        _RECORDER("score", agg)
    return agg


def _sequence_agreement(got: np.ndarray, want: np.ndarray) -> float:
    """Continuation-token agreement as a [0,1] fraction: leading matching tokens / N.

    A free-running sequence is graded on how far it tracks the reference before the
    FIRST divergence: once a token differs the continuation conditions on a different
    prefix, so tokens past it no longer measure the same run. 1.0 iff every token
    matches; 0.0 if they diverge at step 0; i/N if the first divergence is at step i.
    Assumes equal shape -- the caller asserts it as a hard gate.
    """
    if want.size == 0:
        return 1.0
    matches = np.asarray(got) == np.asarray(want)
    if bool(np.all(matches)):
        return 1.0
    return int(np.argmax(~matches)) / float(want.size)


# --------------------------------------------------------------------------- #
# Threshold plumbing. Read from the golden, never restated here.
# --------------------------------------------------------------------------- #
#: Per-token absolute bound on the fp32 chosen-token logprob. ``fp32`` band only
#: -- it is a bound on fp32 arithmetic noise and means nothing at bf16, where the
#: model's own cross-process spread is three orders larger.
#: ``_harness_checks._KNOWN_THRESHOLD_EXTRAS`` enforces the band.
_LOGPROB_ATOL_KEY = "logprob_atol"

#: Absolute bound on the CUMULATIVE sequence logprob. Separately calibrated, and
#: that is the point -- see :func:`test_decode_trajectory`. ``fp32`` band only.
_LOGPROB_CUMULATIVE_ATOL_KEY = "logprob_cumulative_atol"

#: Sequence-level top1-top2 decisiveness margin. ``native`` band only -- it exists
#: because bf16 quantises the frozen gaps to ULP multiples, which is not a
#: condition the fp32 tier has.
_DECISIVE_MARGIN_KEY = "decisive_margin"

#: Per-token absolute bound on the held-out set's TEACHER-FORCED chosen-token
#: logprob (L6). ``fp32`` band only, on exactly the argument ``logprob_atol``
#: carries -- see :func:`test_held_out_logprobs`.
_HELD_OUT_ATOL_KEY = "held_out_logprob_atol"

#: Absolute bound on the held-out set's PER-SEQUENCE summed logprob. Separately
#: and DIRECTLY measured, never derived from ``held_out_logprob_atol`` -- see
#: :func:`test_held_out_logprobs` for why the derived form could not fail.
_HELD_OUT_SEQ_ATOL_KEY = "held_out_seq_atol"


def _logprob_atol(golden) -> float:
    """``fp32_thresholds.logprob_atol``, or ``KeyError``.

    NO ``.get(...)`` DEFAULT. The alternative -- fall back to some constant here
    -- is how a calibrated gate silently becomes an uncalibrated one: the check
    keeps passing, at a tolerance nobody measured, and nothing in the summary
    says so. ``_harness_checks.check_threshold_extras_name_something_a_check_reads``
    already refuses this key on the wrong band, so a misfiled value fails Spec
    rather than arriving here.
    """
    return float(golden.fp32_extras[_LOGPROB_ATOL_KEY])


def _logprob_cumulative_atol(golden) -> float:
    """``fp32_thresholds.logprob_cumulative_atol``, or ``KeyError``.

    NO DERIVED FALLBACK, and specifically not ``logprob_atol * N``: a derived
    bound like that could not fail (see :func:`test_decode_trajectory`). A
    fallback here would silently restore an unfalsifiable clause.
    """
    return float(golden.fp32_extras[_LOGPROB_CUMULATIVE_ATOL_KEY])


def _decisive_margin(golden) -> float:
    """``native_thresholds.decisive_margin``, or ``KeyError``. See :func:`_logprob_atol`."""
    return float(golden.native_extras[_DECISIVE_MARGIN_KEY])


def _held_out_atol(golden) -> float:
    """``fp32_thresholds.held_out_logprob_atol``, or ``KeyError``. See :func:`_logprob_atol`."""
    return float(golden.fp32_extras[_HELD_OUT_ATOL_KEY])


def _held_out_seq_atol(golden) -> float:
    """``fp32_thresholds.held_out_seq_atol``, or ``KeyError``. See :func:`_logprob_atol`."""
    return float(golden.fp32_extras[_HELD_OUT_SEQ_ATOL_KEY])


def _tokens(value) -> np.ndarray:
    """A token sequence as a flat int64 array, whatever integer width it arrived in."""
    return np.asarray(value).reshape(-1).astype(np.int64)


def _unsupported(probe: dict, *, check: str, what: str) -> dict:
    """The failure payload for a probe that answered ``{"supported": False}``.

    An APPLICABLE check whose probe is unsupported FAILS -- it does not skip and
    it does not deselect. Deselection is decided before collection, from the
    frozen golden alone (``conftest._golden_facts``); by the time a probe runs,
    "the candidate cannot do this" is a verdict about the candidate.
    """
    return {"check": check, "error": f"the candidate produced no {what}",
            "reason": probe.get("reason")}


def _legs(probe: dict, first: str, second: str, *, check: str, what: str):
    """The two arrays an invariance probe must return, or a legible failure.

    ``probe[key]`` would raise ``KeyError`` from inside the check and report a
    harness-shaped traceback for a candidate-shaped fact. The four mode-gated
    invariants below are the ones that need this: their probes are unimplemented
    until I2, so the KEY NAMES here are the contract those probes must satisfy,
    not a description of code that exists. Stating the contract in the check is
    the point -- I2 adds a mode, not a test suite.
    """
    assert probe.get("supported"), _unsupported(probe, check=check, what=what)
    missing = [k for k in (first, second) if k not in probe]
    assert not missing, {
        "check": check, "error": "the probe did not return both legs of the "
                                 "invariant, so there is nothing to compare",
        "missing": missing, "returned": sorted(probe),
    }
    return probe[first], probe[second]


# =========================================================================== #
# The golden-free invariance family.
#
# Every one of these compares the CANDIDATE against ITSELF under a transformation
# that must not change the answer. None reads a frozen artifact, so none can be
# overfitted to one, and each arms itself on an instance that has no goldens at
# all.
# =========================================================================== #
def test_paging_invariance(cand):
    """Same logical KV, different physical pages -> identical logits.

    Golden-free. The canonical scenario hands out one page list and grades the
    result against a frozen array, so it cannot distinguish "the block table was
    read" from "the block table was ignored and the cache indexed flat" whenever
    the two agree. This probe runs the same prompt twice under two DIFFERENT page
    assignments the candidate's own allocator produced, so the logical sequence
    is identical and only the physical layout moved.

    BIT-EXACT, not a tolerance: both legs run in one process, on one device, with
    one compiled program, so the only thing that differs is which pages the
    gather reads. Anything but equality is a real indexing difference.

    LIMITED PAGING DISCRIMINATION ON THIS INSTANCE, and that is a limitation
    of the geometry rather than of the property. Both legs are a PREFILL ONLY over
    a 12-token prompt, so neither reaches position 16 where this instance's page
    boundary is: a clamped page-index gather and a block table truncated
    to two pages BOTH PASS this check. It is falsifiable -- an engine with
    in-process nondeterminism fails it -- but it catches none of the
    realistic paging/decode bugs put to it. ``test_decode_prefill_consistency``'s
    far leg is what covers the boundary; see its docstring.
    """
    a = cand.invariance("paging_default")
    b = cand.invariance("paging_permuted")
    for probe, leg in ((a, "paging_default"), (b, "paging_permuted")):
        assert probe.get("supported"), _unsupported(
            probe, check="test_paging_invariance", what=f"{leg} result")
    np.testing.assert_array_equal(a["logits"], b["logits"])


def test_decode_prefill_consistency(golden, cand):
    """Token t's decode logits == its logits from a prefill over the same prefix.

    Golden-free in the sense that matters -- no frozen ARRAY enters the
    comparison, both sides are the candidate's own output, and the check arms
    itself on an instance with no goldens at all.

    IT GRADES THE CANDIDATE'S OWN DECODE PATH, and that is the whole point. Both
    legs run through the candidate's own ``Engine.prefill`` / ``Engine.decode_step``
    / ``Engine.reset``, so the candidate computes the positions, the ``seq_lens``,
    the block table and the slot itself. That is what lets it catch "a wrong
    position offset, a stale ``seq_lens``, a slot computed from the wrong page": a
    probe that built its own ``AttentionMetadata`` and its own slot out of the
    FROZEN page ids -- so the GRADER supplied those four quantities -- would pass
    mutations of exactly those three.

    WHAT IT CATCHES (each on a copy of the reference via ``T2J_JAX_REPO``; Spec
    17/17 and Numeric 9/9 are blind to all of these):

    * decode RoPE position off-by-one (``_metadata`` gets ``pos-1``) -- FAILS on
      the NEAR leg, centered cosine 0.9758 against the 0.998 floor
    * stale ``seq_lens`` (``new_len - 1``) -- FAILS on the NEAR leg, 0.9738
    * decode KV written at slot ``t-1`` -- FAILS
    * a slot gather that clamps the page index -- FAILS at 0.9786, and **only via
      the FAR comparison**; the near leg never leaves the prompt's own pages, which
      is precisely why the far leg exists
    * rotary position taken from the query index -- FAILS; it is the one
      model-side mechanism in this list.

    TWO COMPARISON POINTS. NEAR is position ``prompt_len-1``. FAR is the first
    position on the page after the prompt's last -- 16 here -- reached by decoding
    the frozen continuation tokens, and compared against a single prefill over the
    same prefix. The far leg is required whenever the geometry admits one, and
    this check asserts that it happened: a probe that silently stopped crossing
    the boundary would otherwise read as full coverage.

    AND A THIRD WHERE THE SECOND CANNOT REACH, which is a coverage hole this class
    shipped rather than a refinement. The far position is OUTSIDE the prompt, so it
    needs continuation tokens and ``max_model_len`` headroom. On a geometry with
    prompt 135, ``page_size`` 32, 8 decode steps and ``max_model_len``
    160, the next boundary is 160, twenty-six tokens away and one past the limit,
    so ``far`` is ``None`` and the boundary leg did not run at all. With it absent,
    NOTHING in that instance's graded run makes ``Engine._grow`` extend an existing
    allocation -- prefill takes 5 pages, the deepest position any check reaches is
    143, still 5 pages -- so a port whose ``_grow`` allocates once and never grows
    again is bit-identical to the reference everywhere the suite looks.

    So when the far leg cannot run, the probe crosses the boundary from INSIDE the
    prompt: prefill ``ids[:b]`` where ``b`` is the last page boundary strictly
    inside it (128 on core_gpt_oss), decode ``ids[b]``, compare against row ``b``
    of the full prefill the near leg already ran. The decode step is the call that
    must allocate page ``b // page_size``. Same self-comparison and same band as
    the near leg, and a one-step decode like it rather than the far leg's chain.
    ONE new prompt shape; no golden, no re-freeze.

    ITS DECIDING CLAUSE IS THE ALLOCATION, NOT THE LOGITS, and that is a
    measurement rather than a preference. Without this leg, an engine with
    ``Engine._grow`` mutated to allocate once and never grow passes everywhere --
    prefill takes 5 pages in one call and the deepest
    position the suite reaches is 143, still 5. With the leg, the decode at 128
    clamps (``slot_mapping`` indexes a jnp array; JAX clamps out-of-bounds rather
    than raising), writing its KV over position 96's slot and reading position 0's
    key as its own -- and the LOGIT comparison still passes, because one wrong key
    in 129 is inside a native tier this family widened to
    ``centered_cosine_floor: 0.95`` for its expert-routing flips. The page
    identity is what rejects it: the boundary page must be one the prefill did not
    already hold, and the slot must land in it. Exact, and no tolerance involved.

    WHICH LEG IS REQUIRED IS DECIDED HERE, from ``meta``'s geometry, and the probe
    is not trusted about either. ``interior_boundary_position`` is imported from
    the module the probe calls, for the reason ``second_prompt_length`` is: the two
    ends of a gate written as two expressions have already disagreed once in this
    class. A geometry with no far position AND no interior boundary (a prompt
    spanning less than two pages) reports both as ``None`` and is graded on the
    near leg alone, which is the honest answer rather than a skip.

    WHAT IT STILL CANNOT SEE, and this is not hypothetical: a decode path and a
    prefill path that are wrong *identically*, since the property is an agreement
    between the two. A block table truncated to two pages makes both
    legs gather page 0 past position 16, so they agree and this check PASSES while
    ``test_greedy_sequence_fp32`` and ``test_decode_trajectory`` both fail it.

    TOLERANCED, AND AT THE CALIBRATED NATIVE BAND rather than a literal. The two
    legs run DIFFERENT PROGRAMS -- a 1-token query against a populated cache
    versus a full ragged prefill -- so their reductions have different shapes and
    bf16 accumulates them in a different order; exact equality would be asserting
    a property of rounding. The band comes from ``meta``, like every other
    tolerance in this class.
    """
    r = cand.invariance("decode_prefill")
    assert r.get("supported"), _unsupported(
        r, check="test_decode_prefill_consistency", what="decode/prefill result")
    # SELF-COMPARISON. Both rows are the candidate's, so the argmax clause must not
    # take its decisive set from either of them -- see _assert_logit_parity. SCORED:
    # the parity is graded continuously (score_logits), and each leg that runs feeds
    # its [0,1] score into this test's running power mean; the top-1 argmax agreement
    # and the structural allocation clauses below stay HARD gates.
    _score(_assert_logit_parity(
        np.asarray(r["prefill_last_row"]).reshape(1, -1),
        np.asarray(r["decode_logits"]).reshape(1, -1),
        golden, "native", name="test_decode_prefill_consistency[near]",
        self_comparison=True)["score"])

    # THE FAR LEG IS NOT OPTIONAL WHERE THE GEOMETRY ADMITS ONE. Computed here from
    # the frozen meta rather than taken on trust from the probe: "the probe says it
    # could not reach a second page" and "the probe stopped trying" must not be the
    # same answer, and only the far leg sees a page-boundary defect.
    page_size = int(golden.meta["page_size"])
    prompt_len = int(golden.meta["prompt_len"])
    far = page_size * ((prompt_len + page_size - 1) // page_size)
    reachable = (far - prompt_len + 1 <= int(golden.meta["num_decode_steps"])
                 and far + 1 <= int(golden.meta["max_model_len"]))
    assert r.get("far_position") == (far if reachable else None), {
        "check": "test_decode_prefill_consistency",
        "error": "the probe did not compare at the page boundary the frozen "
                 "geometry can reach. The near comparison never leaves the pages "
                 "the prompt already occupies, so it cannot see a page-boundary "
                 "defect -- a clamped page gather passed this check for exactly "
                 "that reason before the far leg existed",
        "expected_far_position": far if reachable else None,
        "probe_far_position": r.get("far_position"),
        "page_size": page_size, "prompt_len": prompt_len,
    }
    if reachable:
        _score(_assert_logit_parity(
            np.asarray(r["far_prefill_row"]).reshape(1, -1),
            np.asarray(r["far_decode_logits"]).reshape(1, -1),
            golden, "native", name="test_decode_prefill_consistency[far]",
            self_comparison=True)["score"])

    # THE INTERIOR LEG, ON THE SAME TERMS: required exactly when the far leg could
    # not run and the prompt spans two pages, and required from THIS side so a
    # probe that quietly stopped crossing the boundary reads as a failure rather
    # than as coverage. `interior_boundary_position` is the probe's own predicate.
    interior = None if reachable else interior_boundary_position(prompt_len, page_size)
    assert r.get("interior_position") == interior, {
        "check": "test_decode_prefill_consistency",
        "error": "the probe did not compare at the page boundary inside the "
                 "prompt. Where the frozen scenario cannot decode PAST its last "
                 "page, this is the only comparison in which a decode step has to "
                 "allocate a page at all -- without it Engine._grow's growth "
                 "branch is never exercised by the graded run",
        "expected_interior_position": interior,
        "probe_interior_position": r.get("interior_position"),
        "far_position": r.get("far_position"),
        "page_size": page_size, "prompt_len": prompt_len,
    }
    if interior is not None:
        # THE ALLOCATION, ASSERTED DIRECTLY, because on some geometries its EFFECT
        # is inside the band. With `Engine._grow` mutated to allocate once and
        # never grow, `slot_mapping` indexes a jnp array and
        # JAX clamps an out-of-bounds index rather than raising, so the decode at
        # 128 writes over position 96's slot and reads position 0's key as its
        # own -- one wrong key in 129, which this family's deliberately wide
        # native tier (centered_cosine_floor 0.95, for its expert-routing flips)
        # does NOT reject. The page identity does, exactly and with no tolerance.
        index = interior // page_size
        # DESELECT-SAFE, the same discipline the dtype clause in this module
        # applies -- and SPLIT BY KIND, because the three keys are not the same
        # kind of value and one predicate cannot be right for all of them.
        #
        # THE TWO SEQUENCE KEYS MUST BE PRESENT AND NON-EMPTY. `set(r.get(...) or
        # [])` is the empty set for an ABSENT field AND for an EMPTY one, and
        # `boundary_page not in set()` passes for every candidate either way, so
        # `is None` alone left exactly half the hazard its own comment named
        # open: `interior_prefill_pages` is built from the CANDIDATE's own
        # `PrefillResult.page_ids` and no other graded clause requires it
        # non-empty, so a probe reporting `[]` sailed through and the clause
        # below passed vacuously. This is the only thing standing between a
        # `_grow`-allocates-once port and passing the whole suite.
        #
        # `interior_slot` IS A SCALAR AND ZERO IS LEGAL. It is
        # `int(np.asarray(_to_np(step.slot_mapping)).reshape(-1)[0])`, and slot 0
        # is a perfectly ordinary slot, so it is checked for PRESENCE only. The
        # one-token repair -- `if not r.get(k)` over all three -- would report a
        # correct candidate's zero slot as a harness defect and fail this check
        # for a port that is right. That is the same wrong grade with the sign
        # flipped, which is why the two conditions are computed and REPORTED
        # separately rather than folded into one list.
        absent = [k for k in ("interior_prefill_pages", "interior_block_table",
                              "interior_slot") if r.get(k) is None]
        empty = [k for k in ("interior_prefill_pages", "interior_block_table")
                 if k not in absent and len(r[k]) == 0]
        assert not absent and not empty, {
            "check": "test_decode_prefill_consistency",
            "error": "the probe reported no usable allocation for the interior "
                     "boundary leg, so the clauses that grade it would pass "
                     "vacuously -- an EMPTY page list makes `boundary_page not "
                     "in held` true for every candidate exactly as an absent one "
                     "does. This is a HARNESS defect, not a port defect.",
            "absent_keys": absent,
            "empty_sequence_keys": empty,
            "interior_position": interior,
        }
        table = list(r["interior_block_table"])
        held = set(r["interior_prefill_pages"])
        assert len(table) > index, {
            "check": "test_decode_prefill_consistency",
            "error": "the block table at the boundary decode is too short to "
                     "address the page this position lives on",
            "block_table": table, "needed_index": index,
        }
        boundary_page = table[index]
        assert boundary_page not in held, {
            "check": "test_decode_prefill_consistency",
            "error": "the decode that crossed a page boundary did not allocate a "
                     "new page -- the block table's entry for this position is a "
                     "page the prefill already holds, so this token's KV was "
                     "written over an earlier token's",
            "position": interior, "block_table_index": index,
            "page_at_that_index": boundary_page,
            "pages_the_prefill_held": sorted(held),
            "hint": "Engine._grow must extend an existing allocation, not only "
                    "create the first one. An out-of-range page index is CLAMPED "
                    "by jnp indexing rather than raising, so this is silent.",
        }
        assert int(r["interior_slot"]) // page_size == boundary_page, {
            "check": "test_decode_prefill_consistency",
            "error": "the boundary decode's slot does not lie in the page its "
                     "own block table names for that position",
            "slot": r.get("interior_slot"), "page_size": page_size,
            "block_table": table, "position": interior,
        }
        _score(_assert_logit_parity(
            np.asarray(r["interior_prefill_row"]).reshape(1, -1),
            np.asarray(r["interior_decode_logits"]).reshape(1, -1),
            golden, "native", name="test_decode_prefill_consistency[interior]",
            self_comparison=True)["score"])


def test_cache_reuse_safety(cand):
    """Freeing and reallocating pages must not corrupt a live sequence.

    Golden-free, bit-exact, and graded in TWO clauses that see different faults.

    (1) THE LIVE SEQUENCE'S OWN KV, compared directly. The probe snapshots
    ``cache[:, live_pages]`` after the prompt is written and again after the pool
    is churned around it. A pool that hands a live request's page to a second
    request moves those bytes; nothing else in this suite looks at them.

    (2) THE LOGITS, before and after that churn. This clause catches a different
    thing from clause (1): a GATHER that reads the wrong
    pages pulls the scribbled noise into the output even though the live pages
    themselves were never touched.

    WHY (1) EXISTS, because "add a second clause" is not a reason. Clause (2)
    ALONE CANNOT DETECT AN ALLOCATOR DEFECT AT ALL, and that is structural rather
    than a property of this geometry: ``modeling.py`` writes the KV cache before it
    reads it, within the same forward, so the probe's closing forward rewrites
    every slot it then gathers and its output is a pure function of
    ``(model, prompt_ids, meta, slot)`` -- independent of the incoming cache.
    Clause (2) alone is ``test_determinism`` on a different page set.

    Both directions, as reference-mutation controls:

    * ``allocator_reuses_live_page_silently`` -- ``BlockPool.free`` forgets which
      pages are live, so ``allocate`` hands pages 0 and 1 back out and random KV
      is scribbled over all twelve live slots. Without clause (1) it is caught by
      none of the other graded checks. It fails here, on clause (1) only.
    * ``block_table_gather_off_by_one`` -- ``bt + 1`` moves the read window onto a
      scratch page. Fails here on clause (2) only; clause (1) passes, because the
      live pages really were left alone.

    So neither clause subsumes the other. The check is not "falsifiable via
    in-process nondeterminism only": clause (2) is falsified by the gather
    mutation, and no allocator defect could falsify the check as a whole.
    """
    r = cand.invariance("cache_reuse")
    assert r.get("supported"), _unsupported(
        r, check="test_cache_reuse_safety", what="cache-reuse result")
    missing = [k for k in ("live_kv_before", "live_kv_after") if k not in r]
    assert not missing, {
        "check": "test_cache_reuse_safety",
        "error": "the probe did not report the live sequence's own KV, so the "
                 "only clause that can see an allocator defect has nothing to "
                 "compare. A probe that silently stopped snapshotting would "
                 "otherwise read as full coverage.",
        "missing": missing, "returned": sorted(r),
    }
    # PER GROUP, and iterated rather than `np.asarray`-ed into one block. The KV
    # cache is a container -- one tensor per distinct KV geometry -- and the probe
    # reports a tuple in container order for every family, one group or several.
    # `np.asarray` of that tuple stacks it while the groups happen to agree on
    # their shape and raises "inhomogeneous shape" the moment they do not, so a
    # heterogeneous family would fail this check on its container's arithmetic
    # rather than on its allocator. The comparison itself is unchanged: exact,
    # bit-for-bit, on each group.
    before = list(r["live_kv_before"])
    after = list(r["live_kv_after"])
    assert len(before) == len(after), {
        "check": "test_cache_reuse_safety",
        "error": "the probe reported a different number of KV cache groups "
                 "before and after the churn. This is a HARNESS regression, not a "
                 "port defect: the two snapshots are taken off the same container.",
        "before_groups": len(before), "after_groups": len(after),
    }
    for index, (was, now) in enumerate(zip(before, after)):
        np.testing.assert_array_equal(
            np.asarray(was), np.asarray(now),
            err_msg="the KV of the pages your live request owns changed while the "
                    "pool was churned around it. Its pages were never freed, so no "
                    "allocation the pool made in between may reach them: "
                    f"kv_cache group {index} of {len(before)}, "
                    f"live_pages={r.get('live_pages')} "
                    f"reallocated={r.get('reused_pages')}.")
    np.testing.assert_array_equal(r["before_reuse"], r["after_reuse"])


def test_determinism(cand):
    """Two identical calls, bit-identical results. No RNG is on this path.

    Golden-free, and deliberately the weakest-looking check here: it is the one
    that says the OTHER seven mean something. Every invariance above compares two
    runs, so if the candidate were nondeterministic within a process those
    comparisons would be measuring noise and their failures would be unreadable.

    Both legs run in the SAME worker for that reason. bf16 on these cards IS
    irreproducible ACROSS processes (two correct runs' prefill logits
    differ by up to 1.9e-1, driven by XLA's GEMM autotuner), and that is a
    property of the environment rather than of the port -- so grading it would
    fail correct candidates at random.

    WHAT IT COSTS TO PASS, stated plainly so the +1 in the numerator is understood
    for what it is: both legs are the same jitted program on the same device in
    the same process, so **any implementation that is a pure function of its
    inputs passes** -- which is every JAX port that is not deliberately stateful.
    It is not unfalsifiable (a ``compute_logits`` that reads a Python
    call counter fails it, and fails the two bit-exact invariants above with it),
    but it is close to free for an honest port. That is the intended shape.
    """
    r = cand.invariance("determinism")
    assert r.get("supported"), _unsupported(
        r, check="test_determinism", what="determinism result")
    np.testing.assert_array_equal(r["first"], r["second"])


# --------------------------------------------------------------------------- #
# The four mode-gated invariants. Written now, DESELECTED on `core`
# (conftest._APPLICABLE), so I2 adds a mode rather than a test suite. Each names
# a probe that is registered in `candidate.INVARIANCE_KINDS` and answers
# `{"supported": False}` on a `core` engine -- so if the gate ever opens on an
# instance whose engine has no such path, the check FAILS with the reason instead
# of quietly passing.
# --------------------------------------------------------------------------- #
def test_batching_invariance(golden, cand):
    """Sequence i alone == sequence i inside a ragged batch. ``ragged``/``chunked``.

    Toleranced at the calibrated native band, not exact and not at a literal: the
    two legs run different shapes, so their reductions associate differently.

    On an instance with a ragged batch the probe is implemented and this check
    runs and passes; on a ``core`` instance, which runs one sequence at a time,
    it is deselected. ``test_chunking_invariance`` and
    ``test_prefix_cache_invariance`` below defer to this docstring and are still
    unverified; they say so themselves.
    """
    alone, batched = _legs(cand.invariance("batching"), "alone", "batched",
                           check="test_batching_invariance", what="batching result")
    _score(_assert_logit_parity(np.asarray(alone).reshape(1, -1),
                                np.asarray(batched).reshape(1, -1),
                                golden, "native", name="test_batching_invariance",
                                self_comparison=True)["score"])


def test_no_cross_sequence_leakage(cand):
    """Perturb sequence j; every OTHER sequence's logits must be BIT-unchanged.

    ``ragged``/``chunked``. The classic paged-attention bug, hit directly: a
    block table read one row too wide, a mask that spans requests, a page handed
    to two live sequences. EXACT -- the other sequences' inputs did not move and
    their shapes did not change, so neither may their output.

    GUARD THE GUARD, and this clause is not decoration. Bit-equality is
    satisfied by a probe that perturbed NOTHING -- an unchanged batch trivially
    produces an unchanged result -- so the probe reports whether its own
    perturbation moved the request it perturbed, and a False there fails here
    rather than reading as the strongest possible pass. This class has already
    shipped one check that no defect could move (`test_sampler_transforms`
    before its grid grew), and an exact self-comparison is the shape most
    exposed to it.
    """
    result = cand.invariance("cross_sequence_leakage")
    before, after = _legs(
        result, "before_perturbation", "after_perturbation",
        check="test_no_cross_sequence_leakage", what="leakage result")
    assert result.get("perturbation_moved_its_own_request") is not False, {
        "check": "test_no_cross_sequence_leakage",
        "error": "the probe's perturbation left the perturbed request's own "
                 "logits bit-identical, so the comparison below is between two "
                 "runs of the same batch and cannot detect leakage",
        "perturbed_request": result.get("perturbed_request"),
    }
    np.testing.assert_array_equal(before, after)


def test_chunking_invariance(golden, cand):
    """Chunked prefill == single-shot prefill over the same prompt. ``chunked``.

    Golden-free: see ``_invariance_chunking``'s docstring for the 0/7 caveat
    this whole family carries, and ``test_chunked_prefill_logits`` for the
    frozen comparison this one is not a substitute for.
    """
    chunked, single = _legs(cand.invariance("chunking"), "chunked", "single_shot",
                            check="test_chunking_invariance", what="chunking result")
    _score(_assert_logit_parity(np.asarray(single).reshape(1, -1),
                                np.asarray(chunked).reshape(1, -1),
                                golden, "native", name="test_chunking_invariance",
                                self_comparison=True)["score"])


def test_prefix_cache_invariance(golden, cand):
    """A served cached prefix == the same prefix recomputed. ``chunked``.

    GUARD THE GUARD, and this clause is not decoration -- it is the one this
    whole probe family exists to demonstrate the need for. An
    engine that finds a hash hit (so ``PrefixCache.stats()["hits"]`` moves)
    and then discards it, recomputing the shared tokens anyway, produces
    IDENTICAL cached/recomputed logits by construction -- it ran the same
    computation both times -- so the comparison below would pass it at its
    strongest possible margin for the weakest possible reason. The probe
    reports ``cache_hit`` from what the forward's OWN
    ``attn_metadata.input_positions`` show (fewer positions, starting at
    ``shared_prefix_len``), not from the hit counter alone, precisely
    because the counter cannot tell a hit that was USED from one that was
    merely found. See ``_invariance_prefix_cache``'s docstring for the
    details.
    """
    result = cand.invariance("prefix_cache")
    cached, recomputed = _legs(
        result, "cached", "recomputed",
        check="test_prefix_cache_invariance", what="prefix-cache result")
    assert result.get("cache_hit") is not False, {
        "check": "test_prefix_cache_invariance",
        "error": "the probe's own forward metadata shows the 'cached' leg did "
                 "not actually serve a prefix-cache hit (either the hit "
                 "counter never moved, or the query was not reduced to the "
                 "uncached tail), so the comparison below is between two "
                 "runs of the SAME cold computation and proves nothing about "
                 "the cache",
        "hits_delta": result.get("hits_delta"),
        "input_positions": result.get("input_positions"),
    }
    _score(_assert_logit_parity(np.asarray(recomputed).reshape(1, -1),
                                np.asarray(cached).reshape(1, -1),
                                golden, "native", name="test_prefix_cache_invariance",
                                self_comparison=True)["score"])


# =========================================================================== #
# The golden-consuming behavioral checks.
# =========================================================================== #
def test_generate_runs(golden, cand):
    """T1 structural: the engine ran the whole loop and its own outputs cohere.

    Nothing here compares a VALUE to the reference -- that is the next three
    checks. This one asserts the shape of what came back, which is what makes
    their failures readable: "your third token is wrong" and "your engine emitted
    five tokens" are different repairs.

    Six clauses. Clause 0 and clause 5 are described at the end of this docstring
    rather than inline:

    0. the serving KV cache is the dtype ``RunConfig`` asked for;
    1. the probe is supported and produced the requested number of tokens;
    2. ``step_logprobs`` is parallel to ``tokens`` and finite -- a ``-inf`` here
       is the signature of a logprob taken from a masked (post-top-k) row;
    3. the engine emitted one pre-sampling logit row per generated token;
    4. **row i's argmax IS token i.** Under the default ``SamplingParams``
       (temperature 1.0, top_k 0, top_p 1.0) the sampler's transform is the
       identity and greedy is exactly that argmax, so this is integer equality
       with no tolerance -- and it is golden-free.
    5. every per-step row crossed the API boundary as a ``jax.Array``, not a
       numpy array standing in for one.

    WHAT CLAUSE 4 CATCHES. An engine that appends its per-step row
    AFTER the decode step instead of before -- the natural off-by-one, and one
    that leaves the tokens, the logprobs and every golden comparison in this
    module completely untouched. Spec and Numeric are blind to it and so are the
    other 7 behavioral checks; only THIS one fails, at ``first_disagreement``
    0 with the shifted ``row_argmax`` in the message. It is the only check in the
    suite that sees it.

    WHAT IT DOES NOT SEE. Clause 4 is an INTERNAL consistency property: a port
    whose rows and tokens are consistently wrong together satisfies it. That is
    the golden-consuming checks' job, and one of them (``test_greedy_sequence``)
    catches exactly that -- a port that emits one wrong token fails both this
    check and the sequence check.

    CLAUSE 0, FIRST IN THE BODY: **the SERVING cache is the dtype
    the run configuration asked for.** ``instruction.md`` states it in as many
    words -- "one array of shape [layers, pages, page_size, 2, kv_heads, head_dim],
    allocated at the dtype the run configuration asks for -- not at a dtype you
    pick" -- and nothing else on the native leg reads it. Only the FP32 leg was
    asserted (``test_greedy_sequence_fp32``), so an ``Engine`` that hardcodes
    ``jnp.float32`` for its serving cache and ignores ``RunConfig.dtype`` would
    satisfy that clause by accident. Without this clause such an engine passes
    everywhere; with it, the same engine fails here and nowhere else. A serving
    engine that silently doubles its KV footprint is a real defect at this
    class's 20B end.

    IT REUSES THE FP32 LEG'S LOOKUP AND ITS FAIL-CLOSED BEHAVIOUR, unchanged and
    fail-closed in both directions: ``kv_cache_dtype_observed`` is read off the
    engine AFTER ``generate()``, is ``None`` when nothing of the contracted shape was
    reachable, and NEVER falls back to what this grader asked for. So a port that
    renames its cache still passes, a lazily-allocating port still passes, and a
    port that hides it fails -- exactly as on the fp32 leg. The dtype it is compared
    to is ``repo_config['dtype']``, which is the value the grader hands the engine,
    so this is a comparison between what was asked and what was built and not
    between the pin and itself.

    CLAUSE 5, THE ONE THIS CHECK CAN REACH THAT ``test_prefill_logits`` CANNOT.
    ``instruction.md``'s boundary contract ("never a numpy array standing in for a
    device array") is general, not scoped to one probe, but ``test_prefill_logits``
    grades ``prefill_probe``, which drives the model directly through
    ``_run_frozen_prefill`` and never constructs an ``Engine`` -- so a numpy leak at
    the ``Engine.prefill`` return is invisible there no matter what it asserts.
    ``Engine.generate`` DOES call ``Engine.prefill`` (it seeds its loop with
    ``self.prefill(input_ids, request_id).logits[-1]``) and appends that value to
    ``GenerateResult.logits_per_step`` unconverted, before any cast -- so THIS
    check's per-step rows are the reachable surface for that specific leak, at step
    0 only (every later step comes from ``decode_step``, a different call). A
    candidate whose ``Engine.prefill`` hands back ``PrefillResult(..., np.asarray(
    logits), ...)`` is caught by nothing else -- ``tokens`` and
    ``step_logprobs`` are both rebuilt with the engine's own ``jnp.array(...)`` call
    and launder the leak on the way in, and no check anywhere asserts the array
    TYPE of a per-step row. This clause is what closes it.
    """
    want_steps = int(golden.meta["num_decode_steps"])
    probe = cand.generation()
    assert probe.get("supported"), _unsupported(
        probe, check="test_generate_runs", what="generation result")

    want_dtype = str(golden.meta["repo_config"]["dtype"])
    observed = probe.get("kv_cache_dtype_observed")
    assert observed == want_dtype, {
        "check": "test_generate_runs",
        "error": (
            "the serving generation leg did not demonstrate a KV cache at all. It "
            "was looked for as `Engine.kv_cache` and then as any array of the "
            "contracted cache shape held by the engine -- both AFTER generate(), so "
            "a lazily-allocated cache counts. An engine that will not say what it "
            "allocated has not shown the property"
            if observed is None else
            "the serving generation leg allocated its KV cache at a dtype the run "
            "configuration did not ask for. The cache dtype is contract, not a "
            "choice: a cache wider than the serving dtype doubles the pool's "
            "footprint and every token still agrees, so nothing else here warns"),
        "run_config_dtype": want_dtype,
        "requested": probe.get("kv_cache_dtype_requested"),
        "observed": observed,
        "found_via": probe.get("kv_cache_dtype_source"),
        "cache_shape_looked_for": probe.get("kv_cache_shape_looked_for"),
    }

    tokens = _tokens(probe["tokens"])
    logprobs = np.asarray(probe["step_logprobs"], dtype=np.float64).reshape(-1)
    assert tokens.shape == (want_steps,), {
        "check": "test_generate_runs",
        "error": "the engine did not run the whole decode loop",
        "num_decode_steps": want_steps, "tokens_returned": int(tokens.size),
    }
    assert logprobs.shape == tokens.shape and np.all(np.isfinite(logprobs)), {
        "check": "test_generate_runs",
        "error": "step_logprobs must be finite and parallel to tokens. A -inf is "
                 "the signature of a logprob read off the PROCESSED (masked) "
                 "distribution rather than the raw one",
        "tokens": int(tokens.size), "step_logprobs": int(logprobs.size),
        "non_finite": int(np.count_nonzero(~np.isfinite(logprobs))),
    }

    rows = cand.generation_step_logits()
    assert len(rows) == want_steps, {
        "check": "test_generate_runs",
        "error": "the engine emitted a different number of pre-sampling logit "
                 "rows than tokens; each row is the distribution its token was "
                 "sampled from",
        "rows": len(rows), "tokens": int(tokens.size),
    }
    # CLAUSE 5 -- see the docstring. `logits_per_step_srctypes` is recorded per
    # step because only step 0 (seeded from `Engine.prefill`) can carry this leak;
    # every later step comes from `decode_step` instead.
    srctypes = probe.get("logits_per_step_srctypes") or []
    bad_steps = [i for i, t in enumerate(srctypes) if "numpy" in str(t)]
    assert not bad_steps, {
        "check": "test_generate_runs",
        "error": "engine.generate()'s per-step logits crossed the API boundary as "
                 "a numpy array at at least one generated step, where a jax.Array "
                 "is contracted (instruction.md: 'never a numpy array standing in "
                 "for a device array'). The usual cause is Engine.prefill handing "
                 "back a numpy array that GenerateResult.logits_per_step then "
                 "carries through unconverted",
        "bad_steps": bad_steps, "srctypes": srctypes,
    }
    argmax = np.asarray([int(np.argmax(np.asarray(row))) for row in rows],
                        dtype=np.int64)
    assert np.array_equal(argmax, tokens), {
        "check": "test_generate_runs",
        "error": "an emitted token is not the argmax of the row the engine says "
                 "it sampled it from. Greedy over the default SamplingParams "
                 "(temperature 1.0, top_k 0, top_p 1.0) is exactly that argmax, "
                 "so this is an internal inconsistency: the per-step rows are off "
                 "by one, or the token came from a different distribution",
        "first_disagreement": int(np.argmax(argmax != tokens)),
        "row_argmax": argmax.tolist(), "tokens": tokens.tolist(),
    }


def test_greedy_sequence_fp32(golden, cand):
    """Same prompt, same completion -- THE HEADLINE EQUIVALENCE CHECK.

    Free-running: the candidate is handed the prompt and nothing else, and drives
    its own decode loop, its own pager and its own allocator for N steps. SCORED:
    graded on continuation-token agreement against ``fp32_generation.npz``
    (matching prefix / N, 1.0 iff every token matches) rather than an all-or-nothing
    exact-equality assert -- the argmax margins below still make a correct port land
    at 1.0, but a port that tracks the reference for most of the sequence banks that.
    The fp32-KV-cache and length clauses remain HARD gates.

    It is not the only probe that runs without the frozen allocation -- ``block_table``,
    ``slot_mapping``, ``compilation`` and FOUR invariance legs are also
    ``USE_FROZEN_ALLOCATION[...] = False`` (``paging_default``, ``paging_permuted``,
    ``cache_reuse``, and ``decode_prefill``). What is
    unique is the SCOPE of one
    verdict: this is the only check that grades the decode loop, the pager and the
    allocator together, over N steps, against the reference's own answer. With
    ``prompt_len`` 12 and ``page_size`` 8 the N decode steps write positions
    12..19, so a single verdict spans pages 0 -> 1 -> 2 and crosses a page
    boundary at generated step 4.

    READ ``test_decode_trajectory`` FIRST -- IT IS THE STRONGER OF THE PAIR.
    Over the seven mutations that clear Spec 17/17 and Numeric 9/9, this
    check catches **4**, the likelihood check catches **6**. Three of this check's
    four are token divergences and all three land at generated step **7 of 8**, so
    that part of its discrimination rests on the eighth step existing: at
    ``num_decode_steps=7`` only the KV-dtype clause would still fire. This check is
    the benchmark's published claim; it is not the tier's discriminator.

    WHY EXACT EQUALITY IS FAIR HERE. The fp32 top1-top2
    margins at the eight generated steps are 0.344 / 5.18 / 5.46 / 1.72 / 0.402 /
    2.09 / 5.33 / 0.480, and the cross-port fp32 logprob gap is 3.76e-05 -- a
    factor of ~9000 at the tightest step. A correct port cannot miss it, so the
    absence of a tolerance costs nothing.

    THE fp32 KV CACHE CLAUSE, AND WHAT IT DOES AND DOES NOT ESTABLISH.
    ``fp32_generation.npz`` was frozen with the cache at ``fp32``, and the same
    reference generate on the
    same fp32 model with the bf16 SERVING cache sits 6.765e-03 from that artifact
    against 3.755e-05 -- 180x -- while every token still agrees.

    IT ESTABLISHES that the engine is holding a cache of the pinned dtype, read off
    the engine's own object AFTER ``generate()``. It looks for ``Engine.kv_cache``
    first and then for any array of the contracted cache shape the engine holds, so
    a port that names its cache something else -- which ``interface.json`` permits,
    since it pins no ``Engine`` attribute -- is not failed for a naming choice, and
    neither is one that allocates lazily on the first prefill.

    IT STILL FAILS CLOSED, which is the point. Nothing found, or shape-matched
    arrays that disagree about dtype, reports absence -- never the dtype this
    grader asked for. Both directions hold: a port whose ONLY deviation is
    calling its cache ``_kvstore`` passes (it is a legal port, so failing it would
    be a false negative), a lazily
    allocating port passes, and an engine that hides its cache AND allocates bf16
    regardless of ``RunConfig.dtype`` still FAILS, because the array it does hold
    reports ``bfloat16``.

    It does NOT establish that every read and write went through that cache;
    ``test_decode_trajectory`` is what actually measures the consequence.

    WHAT IT CATCHES. Each mutation below is applied to a copy of the
    reference via ``T2J_JAX_REPO``.

    * **q/v/k fusion swap** -- concatenate q, v, k while ``STACKED_PARAMS`` still
      declares q(0), k(1), v(2). Spec scores a CLEAN SWEEP, 17/17. This check
      failed at generated step 0. (Numeric also failed 6 of 9, so here the
      headline check CONFIRMS Numeric rather than finding a hole -- which is what
      it had to do: a mutation that drops the fp32 centered cosine to 0.055 and
      passes this check would mean this check was not working.)
    * **decode position off-by-one** -- ``decode_step`` writes its KV at position
      ``t-1``. **Spec 17/17 AND NUMERIC 9/9 PASS**; this check failed at generated
      step 7, along with ``test_greedy_sequence`` and ``test_decode_trajectory``.
      This tier is the only one that sees it, because the Numeric decode probe is
      driven from the frozen slot arithmetic rather than through the candidate's
      ``Engine.decode_step``.

    AND WHAT IT DOES NOT CATCH -- three blind spots, recorded rather than
    left to be discovered. In each, all eight tokens still agree at BOTH
    precisions, so this check is green:

    * **a page-boundary bug past position 16** (the slot gather clamps its page
      index, right for positions 0..15 and wrong from generated step 5). Spec
      17/17, Numeric 9/9, this check green; ``test_decode_trajectory`` failed at
      1.479e-01 against the 4.0e-04 band.
    * **a decode RoPE position off-by-one**, which is worse: it is wrong from the
      FIRST decode step, not late. Spec 17/17, Numeric 9/9, this check green;
      ``test_decode_trajectory`` failed at 1.164e-01, **291x** outside the band.
      Lateness is not the cause -- a strongly-determined greedy continuation is
      simply insensitive to the distribution it is drawn from.
    * **an engine that hides its KV cache and allocates bf16 regardless of
      ``RunConfig.dtype``** -- the 180x trap. This check's own
      dtype clause fails it on an unobservable cache, and
      ``test_decode_trajectory`` catches the consequence at 6.765e-03.

    So the equality clause and the likelihood clause miss disjoint things in both
    directions (a 1.05 logit rescale is invisible to the likelihood-preserving
    argmax and caught only by the trajectory check), which is the argument for
    having both -- with the trajectory check as the stronger half.
    """
    probe = cand.generation(fp32=True)
    assert probe.get("supported"), _unsupported(
        probe, check="test_greedy_sequence_fp32", what="fp32 generation result")

    want_dtype = str(golden.meta["fp32_kv_cache_dtype"])
    observed = probe.get("kv_cache_dtype_observed")
    assert observed == want_dtype, {
        "check": "test_greedy_sequence_fp32",
        "error": (
            "the fp32 generation leg did not demonstrate an fp32 KV cache. It was "
            "looked for as `Engine.kv_cache` and then, failing that, as any array "
            "of the contracted cache shape held by the engine -- both AFTER "
            "generate(), so a lazily-allocated cache counts. An engine that will "
            "not say what it allocated has not shown the property"
            if observed is None else
            "the fp32 generation leg allocated its KV cache at the wrong dtype. "
            "The fp32 golden was frozen with an fp32 cache; the bf16 serving "
            "cache moves the per-token logprob 180x further from it with every "
            "token still agreeing, so nothing else here would warn"),
        "meta_fp32_kv_cache_dtype": want_dtype,
        "requested": probe.get("kv_cache_dtype_requested"),
        "observed": observed,
        "found_via": probe.get("kv_cache_dtype_source"),
        "cache_shape_looked_for": probe.get("kv_cache_shape_looked_for"),
        "serving_dtype": golden.meta["repo_config"]["dtype"],
    }

    want = _tokens(golden.fp32_generation["tokens"])
    got = _tokens(probe["tokens"])
    assert got.shape == want.shape, {
        "check": "test_greedy_sequence_fp32", "error": "sequence length mismatch",
        "golden": int(want.size), "candidate": int(got.size),
    }
    # SCORED: the exact-equality verdict becomes a graded continuation-token
    # agreement -- leading matching tokens / N, 1.0 iff the whole completion matches. The
    # dtype and length clauses above stay HARD gates (a wrong cache dtype or a length
    # mismatch is structural, not a fidelity gradient); the sequence itself is graded, so a
    # port that tracks the reference for six of eight steps banks that instead of the
    # all-or-nothing the exact assert charged. Still ONE verdict over the decode loop, the
    # pager and the allocator; the Numeric tier localises which of them moved.
    _score(_sequence_agreement(got, want))


def test_greedy_sequence(golden, cand):
    """The same free-running run at NATIVE precision, graded on the FIRST divergence.

    ON ``core_llama3`` THIS IS AN EXACT-EQUALITY CHECK WITH A DEAD MARGIN BRANCH,
    and calling it "margin-graded rather than ``xfail``-on-tie" would be wrong
    here. ``decisive_margin`` is 0.125 and the smallest generation-aligned golden
    margin is 0.375, so **no step on this instance is excusable** and the check
    reduces to "the native tokens must match the golden bit for bit". The branch
    is retained for other families in the class, where thinner margins are
    expected; `instance.yaml` records the same thing beside the value.

    IT ADDS NO DISCRIMINATION HERE: across twelve mutations it failed
    exactly when ``test_greedy_sequence_fp32`` failed, at the same generated step,
    and never otherwise. It is kept as redundant-but-falsifiable -- it is the only
    consumer of ``decisive_margin``, and the branch is proven falsifiable below --
    not because it caught anything the fp32 leg did not.

    Passes if the sequence matches, or if the first divergence sits at a step the
    GOLDEN records as a near-tie. Steps after the first divergence are not
    evaluated: once a token differs the continuation is conditioned on a different
    token, so the golden's margins stop describing what the candidate is doing.

    THE MARGIN COMES FROM THE GOLDEN, NEVER FROM THE CANDIDATE. A port with flat
    logits would otherwise manufacture its own exemption, and it is the cheapest
    exemption in the suite to manufacture. ``golden.decode_margins`` is derived
    from the frozen prefill row and ``decode_step.npz`` and is GENERATION-aligned
    -- ``generate`` emits token 0 from the last prefill row and token i>0 from
    decode step i-1, so reading ``decode_step.npz`` positionally would compare
    step i against the margin of step i+1.

    NO ``xfail``. An ``xfail`` conditioned on the candidate's own output leaves
    ``passed + failed + error`` and so deletes a graded check from the denominator
    because of how the candidate behaved. This form always scores.

    THIS BRANCH DOES NOT FIRE ON THE REFERENCE, AND THAT IS RECORDED RATHER THAN
    HIDDEN: the native free-running sequence matches the golden exactly,
    so a correct port takes the equality path.

    THE HEADROOM IS PER INSTANCE, AND ON ONE OF THEM IT IS INVERTED. On
    core_llama3 the smallest golden margin, 0.375, is 3x the largest
    cross-port deviation (0.125). On core_qwen2 the smallest generation-aligned
    margin is
    **0.25** and the deviation at that step's own top-2 columns is also
    **0.25** -- 1.0x, not 3x -- so `decisive_margin: 0.125` sits 2x BELOW the
    mechanism bound there rather than comfortably under it, and the exposure is a
    FALSE FAILURE at that step which no value below 0.25 removes. Each instance's
    own numbers are in its `instance.yaml` beside the key; read those, not this.
    An unexercised branch is a vacuity risk, so it was proven falsifiable in BOTH
    directions. END TO END, on a copy of the reference whose engine emits one
    wrong token at generated step 1: this check FAILED, ``assert 5.125 < 0.125``,
    with the step and both numbers in the message. IN PROCESS, calling this very
    function with a stand-in for ``decode_margins`` -- the only way to reach the
    excused path, since no real golden here has a margin under 0.125 -- the same
    divergence at a golden margin of 0.0625 PASSES, at 0.125 FAILS (the clause is
    a strict ``<``), a later step's near-tie does NOT excuse an earlier real
    divergence, a margins array shorter than the sequence FAILS on the parallelism
    guard rather than by IndexError, and an absent ``decisive_margin`` raises
    ``KeyError`` instead of grading at a default.
    """
    probe = cand.generation()
    assert probe.get("supported"), _unsupported(
        probe, check="test_greedy_sequence", what="generation result")

    want = _tokens(golden.generation["tokens"])
    got = _tokens(probe["tokens"])
    assert got.shape == want.shape, {
        "check": "test_greedy_sequence", "error": "sequence length mismatch",
        "golden": int(want.size), "candidate": int(got.size),
    }
    if np.array_equal(got, want):
        _score(1.0)
        return

    margins = np.asarray(golden.decode_margins, dtype=np.float64)
    # The margins and the sequence must be the same length, or "the first
    # divergence has no margin" would be answered by an IndexError three frames
    # down -- or, worse, by a wrap-around index that excuses the wrong step. HARD
    # gate: a golden/harness integrity fault, not a candidate fidelity gradient.
    assert margins.shape == want.shape, {
        "check": "test_greedy_sequence",
        "error": "golden.decode_margins is not parallel to the generated "
                 "sequence, so the divergent step cannot be graded",
        "margins": int(margins.size), "tokens": int(want.size),
    }
    i = int(np.argmax(got != want))
    margin = float(margins[i])
    limit = _decisive_margin(golden)
    # SCORED: a divergence at a step the GOLDEN records as a near-tie
    # (margin < limit) is excused -> 1.0; a
    # divergence at a DECISIVE step is graded on continuation-token agreement (matching
    # prefix / N) rather than failing outright. The margin still comes from the golden,
    # never the candidate, so a flat-logit port cannot manufacture its own exemption.
    _score(1.0 if margin < limit else _sequence_agreement(got, want))


def test_decode_trajectory(golden, cand):
    """Per-token and cumulative sequence logprob, fp32, at a calibrated tolerance.

    The sequence check says the tokens agree; this says the model was equally
    CONFIDENT in them. A port that gets every argmax right while distorting the
    distribution -- a temperature applied twice, a logit scale, a missing final
    norm, a softmax on the wrong axis -- passes ``test_greedy_sequence_fp32`` and
    fails here. On a copy of the reference whose ``compute_logits``
    multiplies by 1.05, ``test_greedy_sequence_fp32`` and ``test_greedy_sequence``
    both PASS (the argmax is invariant to a positive rescale) and this check
    FAILS. That pair is the whole argument for having both -- and it runs in the
    other direction too: a page-boundary bug that only bites past position 16 leaves
    every token agreeing, so the equality clauses pass and only this one fails,
    at 1.48e-01 against the 4.0e-04 band.

    TWO CLAUSES, EACH WITH ITS OWN MEASURED FLOOR. The second one must not be
    bounded at ``logprob_atol * N``: that is exactly the
    bound ``|Σd| ≤ Σ|d| ≤ N·max|d| ≤ N·tol`` already guarantees once the per-token
    clause -- asserted first -- has passed, so no input could fail it. A systematic
    per-step bias ``b`` only exceeds ``N·tol`` once ``b > tol``, i.e. once clause 1
    has already fired.

    It has its own directly measured floor (1.6825e-04), and the bound it carries
    is **1.9× tighter** than the derived form -- ``3.2e-03 / 1.7e-03 = 1.88``.

    IT STILL BINDS: a
    systematic bias is caught from ``1.7e-03 / 8 = 2.13e-04`` upward while the
    per-token clause tolerates up to 4.0e-04, so the window this clause covers on
    its own is ``b ∈ [2.13e-04, 4.0e-04]``. On a reference copy
    that adds a constant 3.0e-04 to every step logprob, the per-token clause PASSES
    at 3.2e-04 and this clause FAILS at 2.4e-03 -- arithmetic consistent with the
    1.9× ratio. Real, falsifiable and marginal. Total sequence likelihood is
    half of what this benchmark claims to prove, so it gets a real bound rather
    than none.

    fp32 ONLY, and the native leg deliberately has no logprob bound at all. See
    the module docstring: at bf16 the cross-port gap and one correct
    implementation's own cross-process spread are the same size.
    """
    probe = cand.generation(fp32=True)
    assert probe.get("supported"), _unsupported(
        probe, check="test_decode_trajectory", what="fp32 generation result")

    want = np.asarray(golden.fp32_generation["step_logprobs"], dtype=np.float64).reshape(-1)
    got = np.asarray(probe["step_logprobs"], dtype=np.float64).reshape(-1)
    tol = _logprob_atol(golden)
    assert got.shape == want.shape, {
        "check": "test_decode_trajectory", "error": "step_logprobs length mismatch",
        "golden": int(want.size), "candidate": int(got.size),
    }
    # SCORED: the two band clauses become two graded votes instead of two hard
    # asserts. The per-token clause is the low-tail power mean of each step's banded logprob
    # parity (score_scalar_banded at the calibrated fp32 logprob_atol -> 1.0 in-band,
    # ramping to 0 a few tolerances out); the cumulative clause is one banded score at its
    # own directly-measured floor. Both feed this test's running power mean, so a
    # distribution distortion the argmax hides (a 1.05 logit rescale, a doubled temperature)
    # docks the score exactly where a hard band would fail, and a systematic per-step
    # bias too small for the per-token clause still bites the cumulative one.
    per_token = [similarity.score_scalar_banded(float(w), float(g), tol)
                 for w, g in zip(want, got)]
    _score(similarity.power_mean(per_token) if per_token else 0.0)
    cumulative_tol = _logprob_cumulative_atol(golden)
    _score(similarity.score_scalar_banded(
        float(want.sum()), float(got.sum()), cumulative_tol))


# =========================================================================== #
# L5 -- the entry point. The other L5 checks grade the decode LOOP; this one
# grades whether anything composes from a config at all.
# =========================================================================== #
#: What the serve probe must report on a ``multimodal`` instance, and the only
#: two keys :func:`_assert_serve_carried_the_image` reads. Named once so the
#: probe contract and the failure message cannot drift apart.
#:
#: ``serve_image_path`` is the string the probe put in ``ServeConfig.images``;
#: ``serve_image_sha256`` is the sha256 of the bytes it wrote there.
#:
#: BOTH ARE THE PROBE'S SELF-REPORT AND NEITHER IS A MEASUREMENT, which is worth
#: stating because the obvious reading of the pair is wrong. This grader never
#: opens the path -- it cannot: the probe's ``TemporaryDirectory`` is gone by the
#: time a check runs -- so it hashes ``golden.image_fixture_png`` and compares
#: that against a STRING the probe supplied. A probe reporting
#: ``serve_image_path="not-a-real-path"`` with the correct sha PASSES both
#: clauses. What they establish is that the probe *claims* to have served the
#: frozen fixture from a file, and that a probe which quietly stopped doing so
#: reports as a failure rather than as coverage -- the same thing
#: ``far_position`` establishes about the page-boundary leg, and no more. The
#: end-to-end evidence that ``serve.py`` really decoded a PNG lives in the
#: reference serve driver, which drives ``run_serve`` against a real file.
_SERVE_IMAGE_PATH_KEY = "serve_image_path"
_SERVE_IMAGE_SHA_KEY = "serve_image_sha256"


def _assert_serve_carried_the_image(golden, probe) -> None:
    """The multimodal leg's INPUT contract. See :func:`test_serve_end_to_end`.

    THREE clauses, and they do NOT all name the same owner. The first is a
    GOLDEN gap -- an instance declaring ``multimodal: true`` whose frozen tree
    carries no ``image_fixture.png`` -- and the second and third are HARNESS
    defects. Each message says which; none of them is ever a port defect,
    because nothing a candidate writes changes what the probe fed in.

    They are asserted from this side anyway, for the reason the two
    page-boundary legs are: a probe that silently stopped supplying the image
    would otherwise read as full coverage of a tower that never ran.
    """
    fixture = golden.image_fixture_png
    assert fixture is not None, {
        "check": "test_serve_end_to_end", "clause": "multimodal",
        "error": "this instance declares `multimodal: true` but its frozen "
                 "golden carries no image_fixture.png, so there is no image to "
                 "serve and nothing to bind the served one to. This is a GOLDEN "
                 "gap (the image-preprocess freezer writes the pair), not a "
                 "port defect.",
    }
    path = probe.get(_SERVE_IMAGE_PATH_KEY)
    assert isinstance(path, str) and path, {
        "check": "test_serve_end_to_end", "clause": "multimodal",
        "error": "the probe did not drive serve.py with an image PATH, so the "
                 "comparison above graded a text run on a family whose whole "
                 "addition is the vision tower. serve.py must be handed a FILE "
                 "to open -- `ServeConfig.images` -- exactly as it is handed raw "
                 "prompt strings; a config carrying pre-computed pixel_values "
                 "would move the decode and the patchification out of the port. "
                 "This is a HARNESS defect, not a port defect.",
        "serve_image_path": path,
        "returned": sorted(probe),
    }
    want = hashlib.sha256(bytes(fixture)).hexdigest()
    assert probe.get(_SERVE_IMAGE_SHA_KEY) == want, {
        "check": "test_serve_end_to_end", "clause": "multimodal",
        "error": "the bytes serve.py was pointed at are not golden/"
                 "image_fixture.png. Both legs of the comparison above would "
                 "still agree with each other on any image at all, so without "
                 "this clause the check would read as coverage of a picture no "
                 "other check in the suite has ever seen. This is a HARNESS "
                 "defect, not a port defect.",
        "expected_sha256": want,
        "served_sha256": probe.get(_SERVE_IMAGE_SHA_KEY),
        "serve_image_path": path,
    }


def test_serve_end_to_end(golden, cand):
    """serve.py, driven only by a config, reproduces the hand-wired engine.

    THE MIS-WIRING GATE, and it is the reason this check exists rather than
    being folded into test_greedy_sequence_fp32. The harness wires the
    candidate itself everywhere else -- RunConfig, ModelConfig.from_hf, build,
    convert_checkpoint, apply_params, Engine -- so a repo in which NOTHING
    composes from a config passes every other check in the suite. What it puts
    on the graded path: resolve_model_dir, load_tokenizer's
    DECODE (elsewhere graded only as "returns non-None"), the RESOLVED dtype
    contract (narrower than it sounds -- see below), and result serialization.
    NOT its ENCODE: both probes hand `prompt_token_ids` in, so the text half of
    serve.py's input builder (`_prompt_ids`, spelled `_prompt_inputs` on a
    multimodal family) never reaches its `tokenizer.encode` branch and no check
    in this suite executes it.

    ON A MULTIMODAL INSTANCE THIS CHECK ALSO REQUIRES THAT AN IMAGE WAS SERVED,
    and the gated block near the top of this function is what makes that a
    requirement rather than a hope. Two things are asserted, both exact:

    1. **the probe reports having handed serve.py a PATH.** The image is the
       analogue of `prompts` carrying raw strings -- `ServeConfig.images` names a
       FILE and serve.py's own `_prompt_inputs` opens it, decodes it through
       `PIL` and runs this repo's `image_processing.preprocess` over it. A probe
       that handed pre-computed `pixel_values` in would move that whole
       computation out of the port, and the token clause below would then be
       comparing two legs that were both given the answer. So the path is
       required HERE, from the grader's side, exactly as `far_position` and
       `interior_position` are: "the probe could not" and "the probe stopped
       trying" must not be the same answer.
    2. **the probe reports that the path held the frozen fixture.** Its sha256 is
       compared against `golden/image_fixture.png`'s own -- the image every other
       multimodal check is frozen against. Without it a probe could satisfy
       clause 1 with any file at all, both legs would still agree with each
       other, and the check would read as coverage of an image nothing else in
       the suite has ever seen.

    BOTH ARE SELF-REPORTS AND THE WORD "reports" IS DOING WORK IN EACH. This
    grader cannot open the path -- the probe's `TemporaryDirectory` is gone by
    check time -- so a probe naming a path that never existed, with the right
    sha, PASSES. Read the pair as "the probe declares it did this, and
    a probe that stops declaring it fails", not as a filesystem measurement. See
    `_SERVE_IMAGE_PATH_KEY`.

    WHY THERE IS NO TOLERANCE AND NO EXTRA COMPARISON HERE. The token clause
    below already grades the image end to end on this family: `serve.py` decoded
    and patchified the PNG itself, the by-hand leg used the runtime's own
    `image_processing.preprocess` over the same bytes, and the two feed the same
    tower -- so a port whose processor is wrong by one ULP-scale step diverges in
    the tokens and fails there, with the right message. What the two clauses
    above add is that the comparison was about an IMAGE at all.

    AND IT IS A WIRING CHECK, WHICH BOUNDS WHAT THAT SENTENCE MEANS. Both legs
    run the same model in the same process, so a defect they SHARE is invisible
    to it. Scattering the merged image rows at
    wrong offsets leaves the tokens unchanged. The tower's scatter arithmetic is
    graded by `test_image_embedding_merge`, not here.

    AND WHAT THE CANONICAL PROMPT ALREADY FORCES, on the class's first multimodal
    instance: its frozen `prompt_token_ids` carry the
    `<|vision_start|> <|image_pad|>*N <|vision_end|>` run, so a serve.py handed
    that prompt WITHOUT an image cannot quietly degrade to text --
    `modeling.get_mrope_input_positions` refuses a placeholder run with no grid,
    by name. The "text leg" is therefore unchanged CODE that happens to be driven
    by a multimodal prompt on this one instance, and it stays literally identical
    on the four text-only families, which is what keeps the check comparable.

    WHAT THE DECODE CLAUSE ESTABLISHES, AND WHAT IT DOES NOT. It establishes
    PROVENANCE: that `text` is this checkpoint's tokenizer decoding the very
    ids serve.py returned, so a port that fabricates the field -- stringified
    ids, a truncated decode, a constant -- fails. It does NOT establish that
    `load_tokenizer` is correct. Both sides of the comparison call the
    candidate's own loader, so a systematically wrong tokenizer agrees with
    itself and passes, and the only other tokenizer grading in the suite is
    test_loader_spec's `has_tokenizer` ("returned non-None"). That residual is
    recorded rather than papered over: THERE IS NO CANNED-ENCODE CHECK ANYWHERE
    IN THIS SUITE. The comparison is deliberately a decode against a decode and
    never a decode->encode round trip -- BPE normalization makes
    `encode(decode(ids)) == ids` false for legitimate inputs, so a round trip
    would false-fail correct ports.

    WHAT THE DTYPE CLAUSES DO AND DO NOT CATCH.
    They grade the CONFIG LAYER: a serve.py that ECHOES a dtype other than the
    one it was handed, or that defaults `kv_cache_dtype` to something other than
    the compute dtype. They do NOT grade the BUILD.
    `results['config']['dtype']` is whatever `get_serve_args` settled on -- what
    was ASKED FOR -- and it is fixed before `_build` ever runs, so a `_build`
    that constructs the model at a dtype of its own choosing is invisible to an
    echo of the config. With the `serve_hardcoded_dtype` control
    (`_build` hardcodes bfloat16 and ignores cfg.dtype), THIS CHECK PASSES, and
    `test_held_out_logprobs` catches it alone, at 137x its band. Two independent
    reasons, both of which would have to change: the echo is upstream of the
    substitution, and this probe requests the NATIVE dtype, which on a bf16
    instance makes that particular substitution a literal no-op here. So L6 is
    the only thing in this suite standing between the class and a serve.py that
    builds at the wrong precision. Do not read the two clauses below as covering
    it, and do not weaken or gate L6 on the grounds that L5 "already grades the
    dtype contract".

    SELF-CONSISTENCY, NOT GOLDEN PARITY, so it duplicates nothing: token
    equality against the frozen artifact stays in test_greedy_sequence_fp32,
    and this check cannot be overfitted to a frozen array because there is no
    array to overfit to.

    EXACT, WITH NO MARGIN. Both legs run the same code in the same framework
    in the same process at the same dtype on the same device, so the
    comparison is deterministic. native_thresholds.decisive_margin does NOT
    apply here: it excuses a cross-framework argmax flip, and there is no
    cross-framework gap in this comparison. Importing it would let a
    genuinely mis-wired serve.py through.
    """
    probe = cand.serve()
    assert probe.get("supported"), _unsupported(
        probe, check="test_serve_end_to_end", what="serve.py result")

    # THE MULTIMODAL INPUT CONTRACT, ASSERTED BEFORE THE OUTPUT IS READ. A probe
    # that RETURNED a payload built without an image produced a verdict about a
    # text run, and reporting "your tokens differ" first would send the reader to
    # the port for a defect that is the harness's.
    #
    # THE ORDERING BUYS NOTHING AGAINST A PROBE THAT RAISED, and the earlier
    # wording of this comment claimed otherwise. `run_all` captures a raising
    # probe as `{"__probe_error__": ...}` and `Candidate._probe` re-raises it as
    # `CandidateError` at READ time -- which is `cand.serve()`, the first line of
    # this function -- so on an instance whose probe cannot even build its config
    # nothing below here executes. That is the state of `core_qwen2_5_vl` today:
    # its canonical prompt carries `<|image_pad|>` and the probe passes no image,
    # so `run_serve` raises inside the worker. These clauses become reachable
    # when the probe is fixed, and their standing value is as a permanent guard
    # against a LATER probe quietly dropping the image again.
    #
    # Gated on the frozen `multimodal` fact -- the same fact
    # `conftest._APPLICABLE` gates the six image checks on -- and not on whether
    # the probe happened to return the keys, because "the probe returned nothing"
    # is precisely the condition this block exists to fail on.
    if bool(golden.meta.get("multimodal")):
        _assert_serve_carried_the_image(golden, probe)

    results = probe["results"]
    assert "generations" in results, {
        "check": "test_serve_end_to_end",
        "error": "run_serve returned no `generations` under do_generate=True",
        "keys": sorted(results),
    }
    gens = results["generations"]
    assert len(gens) == 1, {
        "check": "test_serve_end_to_end",
        "error": "one prompt in, one generation record out",
        "n_records": len(gens),
    }
    gen = gens[0]

    want = np.asarray(probe["wired_tokens"], dtype=np.int64).reshape(-1)
    got = np.asarray(gen["tokens"], dtype=np.int64).reshape(-1)
    assert got.shape == want.shape and np.array_equal(got, want), {
        "check": "test_serve_end_to_end",
        "error": "serve.py's tokens differ from the same engine wired by hand. "
                 "Both legs are the same code in the same process at the same "
                 "dtype, so this is a WIRING difference -- a checkpoint path "
                 "never resolved, apply_params never called, or a dtype "
                 "substituted -- and not a precision one",
        "serve_tokens": got.tolist(), "hand_wired_tokens": want.tolist(),
    }

    logprobs = np.asarray(gen["step_logprobs"], dtype=np.float64).reshape(-1)
    assert logprobs.shape == got.shape, {
        "check": "test_serve_end_to_end",
        "error": "step_logprobs must be parallel to tokens",
        "n_tokens": int(got.size), "n_logprobs": int(logprobs.size),
    }
    assert np.isfinite(logprobs).all(), {
        "check": "test_serve_end_to_end",
        "error": "a step logprob is not finite",
        "step_logprobs": logprobs.tolist(),
    }

    # The resolved config, echoed back. A serve.py that quietly substitutes a
    # dtype IN THE CONFIG LAYER fails HERE rather than producing numbers nothing
    # can explain -- but only there: see the dtype paragraph in the docstring for
    # the measurement showing that a substitution inside `_build` sails past both
    # clauses below and is caught by `test_held_out_logprobs` alone.
    meta = golden.meta
    assert probe["resolved_dtype"] == str(meta["dtype"]), {
        "check": "test_serve_end_to_end",
        "error": "run_serve resolved a different compute dtype than the config "
                 "asked for",
        "requested": str(meta["dtype"]), "resolved": probe["resolved_dtype"],
    }
    assert probe["resolved_kv_cache_dtype"] == str(meta["dtype"]), {
        "check": "test_serve_end_to_end",
        "error": "kv_cache_dtype must default to the compute dtype when the "
                 "config does not name one",
        "requested": str(meta["dtype"]),
        "resolved": probe["resolved_kv_cache_dtype"],
    }

    # The tokenizer clauses. The ONLY place in the suite that grades what the
    # tokenizer DID; the sole other mention of it is test_loader_spec's
    # `has_tokenizer`, i.e. "load_tokenizer returned non-None".
    #
    # EACH MESSAGE DESCRIBES ITS OWN CLAUSE. A message must not claim more than
    # its assertion tests: "load_tokenizer must be used to decode, not merely
    # loaded" would overstate an assertion that tests only
    # `isinstance(text, str) and text`. A serve.py that stringifies the ids
    # (`" ".join(str(t) for t in tokens)`) and never calls decode at all passes
    # this clause, so the message stays within what the clause can support.
    # Do not widen a message past its assertion.
    text = gen.get("text")
    assert isinstance(text, str) and text, {
        "check": "test_serve_end_to_end",
        "error": "generations[0]['text'] must be a non-empty string",
        "text": text,
    }

    assert probe.get("expected_text") is not None, {
        "check": "test_serve_end_to_end",
        "error": "the runtime could not decode serve.py's own tokens, so there "
                 "is nothing to compare `text` against: load_tokenizer returned "
                 "None for a checkpoint that ships tokenizer.json. "
                 "test_loader_spec's has_tokenizer clause grades that directly",
        "has_tokenizer": probe.get("has_tokenizer"),
    }
    assert text == probe["expected_text"], {
        "check": "test_serve_end_to_end",
        "error": "generations[0]['text'] is not this checkpoint's tokenizer "
                 "decoding generations[0]['tokens']. Both sides are the SAME "
                 "decode call on the SAME ids -- serve.py's own tokens, put "
                 "through load_tokenizer(resolve_model_dir(...)) -- so this is "
                 "not a normalization difference and not a precision one: the "
                 "text did not come from the checkpoint's tokenizer",
        "serve_text": text,
        "expected_text": probe["expected_text"],
    }


# =========================================================================== #
# L6 -- held-out logprob parity. The check that lands here reads
# `Candidate.held_out()` against `golden/held_out.npz` and is ALWAYS ON: no
# presence gate, no env var, no out-of-band runner.
#
# A RETIRED CHECK, and why it is gone rather than fixed: `test_held_out_nll`
# read a band from a `golden/held_out_nll.json` that no producer ever
# wrote, against an NLL supplied at `$T2J_INFERENCE_RESULT` that nothing ever set.
# Both halves had to be present for `conftest._APPLICABLE` to collect it, so it
# executed on ZERO instances in BOTH task classes for the whole life of the
# benchmark -- while reading, in the module docstring and in the applicability
# table, as coverage. Its replacement grades a REQUIRED artifact with no gate at
# all, which is the only shape that cannot fail this way.
# =========================================================================== #
def test_held_out_logprobs(golden, cand):
    """Teacher-forced logprob parity over the held-out set, fp32 (L6).

    WHAT DISTINGUISHES THIS FROM L3, stated honestly because "held out" alone
    does not: every golden is agent-invisible -- tests/ reaches only the
    separate verifier -- so the L3 battery inputs are equally unseen. What is
    unique here is that it grades the candidate's OWN top-level entry point,
    composed from a config, over a set that moves every serving axis at once.
    L3 grades harness-driven probes against frozen arrays, one axis at a time.

    TEACHER-FORCED, NOT FREE-RUNNING, and that is load-bearing. Scoring the
    reference's continuation is well-defined everywhere; once a candidate
    emits a different token the two sequences condition on different prefixes
    and a per-step difference stops measuring precision at all.

    TWO-SIDED, AND NO ONE-SIDED FLOOR. The training class's L6 carries
    `V_jax <= V_torch + tol` because a trained model can generalize WORSE.
    Nothing is trained here, and the dangerous direction is a logprob that is
    too HIGH: a candidate assigning too much probability to the continuation
    is what future-token leakage looks like -- a dropped causal mask,
    attention reading past seq_lens, cross-page bleed. Those are this class's
    headline failure modes and a "no worse" bound passes them unconditionally.

    FP32 ONLY. instance.yaml records the measurement on the SIBLING quantity --
    the free-running generation's chosen-token logprob (logprob_atol's block),
    not this teacher-forced one -- at bf16, against one CORRECT
    implementation's own cross-process spread; see that block for the numbers
    rather than a restatement here. At native the effect is not separable from
    noise and a native band would grade nothing.

    CLAUSE 2 IS EMPIRICAL, NEVER DERIVED. Bounding it at
    held_out_logprob_atol * cont_len is exactly what clause 1 already
    guarantees (|sum d| <= sum|d| <= N*max|d| <= N*tol, and clause 1 is
    asserted first), so no input could fail it. instance.yaml carries the same
    lesson at length for logprob_cumulative_atol.
    """
    probe = cand.held_out()
    assert probe.get("supported"), _unsupported(
        probe, check="test_held_out_logprobs", what="held-out scoring result")

    want = np.asarray(golden.held_out["ref_logprobs"], dtype=np.float64)
    got = np.asarray(probe["logprobs"], dtype=np.float64)
    cont_lens = np.asarray(golden.held_out["cont_lens"], dtype=np.int64)
    assert got.shape == want.shape, {
        "check": "test_held_out_logprobs",
        "error": "the scored logprob grid does not match the frozen one",
        "golden_shape": list(want.shape), "candidate_shape": list(got.shape),
    }

    # Mask the right-padding. Reading past cont_lens compares zeros to zeros
    # and would dilute every statistic below with free passes.
    valid = np.arange(want.shape[1])[None, :] < cont_lens[:, None]
    tol = _held_out_atol(golden)
    # SCORED: the masked per-token band and the per-sequence band become two
    # graded votes instead of two hard asserts. Per-token grades ONLY the valid
    # (non-padding) positions -- a banded logprob parity per kept entry, low-tail-pooled.
    # Per-sequence grades each row's summed logprob at its own directly-measured floor.
    # Teacher-forced, fp32, so this measures the distribution over prompts the port was
    # never tuned against; future-token leakage (a dropped causal mask, cross-page bleed)
    # pushes the continuation logprob too high and docks both.
    keep = valid.reshape(-1)
    per_token = [similarity.score_scalar_banded(float(w), float(g), tol)
                 for w, g, k in zip(want.reshape(-1), got.reshape(-1), keep) if k]
    _score(similarity.power_mean(per_token) if per_token else 0.0)

    seq_tol = _held_out_seq_atol(golden)
    want_seq = (want * valid).sum(axis=1)
    got_seq = (got * valid).sum(axis=1)
    per_seq = [similarity.score_scalar_banded(float(w), float(g), seq_tol)
               for w, g in zip(want_seq, got_seq)]
    _score(similarity.power_mean(per_seq) if per_seq else 0.0)
