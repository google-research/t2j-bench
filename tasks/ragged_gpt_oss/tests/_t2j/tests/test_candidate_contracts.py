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

"""The driver's contracts, checked WITHOUT spawning a sandbox.

Everything here is about the parent side: the probe registry, the "no probe may
be absent or None" guard, the unknown-invariant guard, and the frozen-allocation
policy TABLE (the behaviour that table produces is checked live in
``test_candidate.py``). Needing no checkpoint, no GPU and no golden, these run in
CI, which is where the tables most need a tripwire.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[3] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from _t2j import candidate as C  # noqa: E402

# Only ``test_the_worker_cannot_import_the_graders_own_modules`` below needs
# this -- every other test in this file drives ``_unrun()`` / a faked
# ``mp.get_context`` and needs neither a GPU nor the checkpoint. That one uses
# the real, session-scoped ``cand`` fixture (see ``conftest.py``), which spawns
# an actual worker, so it inherits the same requirement ``test_candidate.py``
# gates its whole module on.
_REPO_ID = "meta-llama/Llama-3.2-1B-Instruct"
_DATA_ROOTS = [Path("/data")]
if os.environ.get("T2J_INFERENCE_DATA_ROOT"):
    _DATA_ROOTS.append(Path(os.environ["T2J_INFERENCE_DATA_ROOT"]))
_NEEDS_CHECKPOINT = pytest.mark.skipif(
    not any((root / _REPO_ID / "config.json").exists() for root in _DATA_ROOTS),
    reason="needs the baked checkpoint at /data/{repo_id} "
           "(or $T2J_INFERENCE_DATA_ROOT/{repo_id} on an authoring host)",
)


def _unrun() -> C.Candidate:
    """A handle that has never spawned anything -- ``__init__`` runs nothing."""
    return C.Candidate(
        str(TESTS / ".." / "solution" / "ref_jax_repo"),
        scenario={"page_ids": [0], "inputs": {}},
        repo_config={"model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct"},
    )


def test_constructing_a_candidate_runs_nothing():
    cand = _unrun()
    assert cand._results is None and cand._error is None


def test_an_unknown_invariant_raises_before_the_sandbox_is_even_spawned():
    """A typo in a behavioral module must not read as "not supported".

    Checked on an UNRUN handle on purpose: the name is validated against the
    static registry, so the error arrives in milliseconds and cannot be confused
    with a candidate failure. If this ever needed a sandbox run, a typo in a
    test would cost a checkpoint conversion to discover.
    """
    cand = _unrun()
    with pytest.raises(RuntimeError, match="unknown probe"):
        cand.invariance("not_a_real_invariant")
    assert cand._results is None, "the guard must not have spawned the worker"


def test_every_registered_invariant_is_answerable():
    """The registry and the implementation must not drift apart.

    Every non-multi-sequence kind needs a ``_invariance_<kind>`` method on the
    runtime; the multi-sequence ones are answered by the mode gate. A kind added
    to the registry with no producer would otherwise surface as a
    ``CandidateError`` in the middle of a graded run.
    """
    for kind in C.INVARIANCE_KINDS:
        assert hasattr(C._CandidateRuntime, f"_invariance_{kind}"), kind
    assert C._MULTI_SEQUENCE_KINDS < C.INVARIANCE_KINDS


def test_the_allocation_policy_is_per_probe_and_names_the_allocator_probes():
    """The allocation policy, as a table.

    The three checks that GRADE the allocator must derive their own allocation,
    and the artifacts that are allocation-DEPENDENT must be handed the frozen
    one. A single global flag in either direction shows up here as a table that
    is all-True or all-False.
    """
    policy = C.USE_FROZEN_ALLOCATION
    derived = {name for name, frozen in policy.items() if not frozen}
    handed = {name for name, frozen in policy.items() if frozen}

    # test_block_table_contract / test_paging_invariance / test_cache_reuse_safety.
    assert {"block_table", "slot_mapping"} <= derived
    assert {"invariance.paging_default", "invariance.paging_permuted"} <= derived
    assert "invariance.cache_reuse" in derived

    # The allocation-dependent artifacts: a different-but-correct allocator must
    # not fail these.
    assert {"prefill", "fp32_prefill", "kv_cache_state", "decode_steps",
            "paged_attention_unit"} <= handed

    assert derived and handed, "a global flag would empty one of these sets"


# Public accessors that are not themselves probe keys, each with the reason.
# Anything NOT listed here must name a key in REQUIRED_PROBES.
#
# Empty because `generation` and `generation_step_logits` are METHODS (they
# take `fp32`), which takes both out of the property scan below. The dict stays
# because it is the documented exemption seam and the clause that consults it must
# keep working; the two methods are covered by the explicit assertions at the end
# of `test_the_required_probe_set_covers_every_public_probe_accessor`, alongside
# `prefill` / `paged_attention_unit` / `invariance`.
_ACCESSOR_IS_A_VIEW_OF: dict[str, str] = {}
_ACCESSORS_THAT_ARE_NOT_PROBES = {
    # The sandbox's leak list, carried beside the results rather than inside
    # them: it is evidence about the RUN, not an artifact the candidate produced.
    "banned_leaks",
}


def test_the_required_probe_set_covers_every_public_probe_accessor():
    """A probe you can READ must be a probe the worker is required to PRODUCE.

    DERIVED from ``vars(C.Candidate)`` rather than transcribed. A hand-written
    list is a snapshot of the class on the day it was written: it cannot fail
    when an accessor is ADDED, which is exactly the drift that let
    ``loader_facts`` -- a public accessor -- sit outside ``REQUIRED_PROBES``,
    where a worker that silently stopped producing it would have surfaced as a
    ``KeyError`` mid-grade instead of the named ``CandidateError`` the guard
    exists to give.
    """
    required = set(C.REQUIRED_PROBES)
    properties = {name for name, value in vars(C.Candidate).items()
                  if isinstance(value, property) and not name.startswith("_")}
    assert properties, "no public properties found -- this test is checking nothing"
    for name in sorted(properties - _ACCESSORS_THAT_ARE_NOT_PROBES):
        key = _ACCESSOR_IS_A_VIEW_OF.get(name, name)
        assert key in required, (
            f"Candidate.{name} reads probe {key!r}, which no probe is required to "
            "produce; add it to REQUIRED_PROBES (or to the documented exemptions)"
        )

    # The accessors that take an argument, and so are not properties. Every
    # `fp32=`-switched pair is named on BOTH sides here: the property scan above
    # cannot see them, so this is the only place a probe key dropped from one of
    # these pairs would surface.
    assert {"prefill", "fp32_prefill"} <= required
    assert {"paged_attention_unit", "fp32_paged_attention_unit"} <= required
    assert {"generation", "fp32_generation"} <= required
    assert "invariance" in required
    # ...and they really are callables on the class, or the three lines above are
    # asserting the probe table against itself while the accessor they name has
    # been renamed or deleted.
    for name in ("prefill", "paged_attention_unit", "generation",
                 "generation_step_logits", "invariance"):
        assert callable(vars(C.Candidate).get(name)), (
            f"Candidate.{name} is not a callable accessor any more; the "
            "argument-taking clause above no longer describes the class"
        )


def _full_results() -> dict:
    return {name: {"ok": True} for name in C.REQUIRED_PROBES} | {
        "invariance": {kind: {"supported": True} for kind in C.INVARIANCE_KINDS}
    }


def test_a_complete_result_set_validates():
    assert C._validate_results(_full_results()) is not None


def test_a_probe_that_came_back_None_is_an_error_not_a_pass():
    """The mutation this guard exists for: ``except Exception: return None``.

    A ``None`` that reaches a numeric check is a comparison against nothing --
    depending on the comparator it can even PASS. It must be a hard error naming
    the probe instead.
    """
    results = _full_results()
    results["prefill"] = None
    with pytest.raises(C.CandidateError, match="None: \\['prefill'\\]"):
        C._validate_results(results)


def test_a_probe_that_was_never_produced_is_an_error_not_a_pass():
    results = _full_results()
    del results["kv_cache_state"]
    with pytest.raises(C.CandidateError, match="absent: \\['kv_cache_state'\\]"):
        C._validate_results(results)


def test_a_missing_invariance_kind_is_an_error():
    results = _full_results()
    results["invariance"].pop("determinism")
    with pytest.raises(C.CandidateError, match="determinism"):
        C._validate_results(results)


def test_an_invariance_probe_error_marker_is_not_an_incomplete_result():
    """The completeness check must be SKIPPED when ``invariance`` is itself a
    captured probe error, not read as if its one key (``__probe_error__``) were
    the kind registry.

    ``run_all`` runs ``invariance_probes()`` as a single probe, so a raise
    inside it makes ``results["invariance"]`` the marker
    ``{"__probe_error__": ...}`` rather than a partial kind dict. Reading
    through that shape as before would report EVERY registered kind "missing"
    and raise a global ``CandidateError`` for what is actually a probe-level
    fault ``Candidate.invariance`` already isolates on read -- costing every
    check in the run, not just the invariance ones (the exact step-function
    failure ``test_one_failed_probe_fails_only_the_checks_that_read_it`` exists
    to close, for this one probe in particular). That live behaviour needs a
    real sandboxed run to exercise; this pins the same contract at the dict
    level, so a revert is caught on a commit with no sandbox and no GPU.
    """
    results = _full_results()
    results["invariance"] = {"__probe_error__": "boom: synthetic"}
    C._validate_results(results)      # must NOT raise

    # Contrast, so the marker check is not just vacuously permissive: a
    # GENUINELY incomplete (non-error) invariance dict still raises, exactly
    # as `test_a_missing_invariance_kind_is_an_error` above pins.
    incomplete = _full_results()
    incomplete["invariance"].pop("determinism")
    with pytest.raises(C.CandidateError, match="determinism"):
        C._validate_results(incomplete)


def test_generation_step_logits_refuses_to_degrade_to_an_empty_list():
    """The second present-or-fails hole: an accessor that answers ``[]``.

    ``test_generate_runs``'s argmax clause walks the per-step logits and requires
    row i's argmax to be token i. Hand it an empty list and it walks nothing --
    indistinguishable from "every row agreed" -- so a candidate whose engine drops
    ``logits_per_step`` would score a pass on a clause that examined nothing.

    BOTH LEGS, because the fp32 one is a separate probe key and an accessor that
    guarded only the native leg would leave the same hole one argument over.
    """
    for fp32, key in ((False, "generation"), (True, "fp32_generation")):
        cand = _unrun()
        cand._results = _full_results() | {
            key: {"tokens": [1, 2], "logits_per_step": []}}
        with pytest.raises(C.CandidateError, match="per-step logits"):
            cand.generation_step_logits(fp32=fp32)


class _FakeArray:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype


class _Engine:
    """The three shapes a candidate's Engine can take, without a GPU."""

    def __init__(self, named=None, hidden=None, extra=None):
        if named is not None:
            self.kv_cache = named
        if hidden is not None:
            self._kvstore = hidden
        if extra is not None:
            self.pool = extra


def _runtime():
    """A ``_CandidateRuntime`` shell carrying only what the dtype lookup reads."""
    runtime = C._CandidateRuntime.__new__(C._CandidateRuntime)
    runtime.num_pages, runtime.page_size = 8, 8
    runtime.model_config = types.SimpleNamespace(
        num_hidden_layers=16, num_key_value_heads=8, head_dim=64)
    return runtime


_SHAPE = (16, 8, 8, 2, 8, 64)


def test_the_kv_cache_dtype_lookup_finds_a_renamed_cache_but_never_invents_one():
    """F7: fail-closed, and NOT at the cost of a legal port.

    ``interface.json`` pins no ``Engine`` attribute, so a port that calls its cache
    something other than ``kv_cache`` is legal. Without this lookup, such a port --
    numerics bit-identical -- is a false negative against a correct implementation.

    The fallback must not become the tautology it replaced, so the two halves are
    pinned together: a renamed cache of the contracted shape IS found, and a
    hidden bf16 cache is found and reported as bf16 rather than as the dtype the
    grader asked for. Nothing here can answer with the requested value, because
    the requested value is not passed in.
    """
    runtime = _runtime()
    assert runtime._expected_cache_shape() == _SHAPE

    # 1. The contracted name wins, and says so.
    assert runtime._observed_cache_dtype(
        _Engine(named=_FakeArray(_SHAPE, "float32"))) == ("float32", "attribute")
    # 2. A renamed cache of the right shape is found by scan -- the legal port.
    assert runtime._observed_cache_dtype(
        _Engine(hidden=_FakeArray(_SHAPE, "float32"))) == ("float32", "shape_scan")
    # 3. ...and the scan reports what it FOUND, not what anyone wanted: the
    #    hidden-and-bf16 engine still reads bf16, which is what fails the check.
    assert runtime._observed_cache_dtype(
        _Engine(hidden=_FakeArray(_SHAPE, "bfloat16"))) == ("bfloat16", "shape_scan")
    # 4. Nothing of that shape -> absence, never a default.
    assert runtime._observed_cache_dtype(_Engine()) == (None, None)
    assert runtime._observed_cache_dtype(
        _Engine(extra=_FakeArray((4, 4), "float32"))) == (None, None)
    # 5. Nested one level (a list/dict of caches) is reached.
    assert runtime._observed_cache_dtype(
        _Engine(extra=[_FakeArray(_SHAPE, "float32")])) == ("float32", "shape_scan")
    assert runtime._observed_cache_dtype(
        _Engine(extra={"c": _FakeArray(_SHAPE, "float32")})) == ("float32", "shape_scan")
    # 6. Two cache-shaped arrays that DISAGREE is not something to pick from.
    ambiguous = _Engine(hidden=_FakeArray(_SHAPE, "float32"),
                        extra=_FakeArray(_SHAPE, "bfloat16"))
    assert runtime._observed_cache_dtype(ambiguous) == (None, None)
    # 7. A property that raises anything at all is absence, not a candidate-wide
    #    error -- and the scan still gets its chance.
    class _Raises(_Engine):
        @property
        def kv_cache(self):
            raise RuntimeError("boom")

    engine = _Raises(hidden=_FakeArray(_SHAPE, "float32"))
    assert runtime._observed_cache_dtype(engine) == ("float32", "shape_scan")

    # 8. THE CACHE IS A CONTAINER. `engine.kv_cache` holds a tuple of arrays --
    #    one for every family shipped today -- and the tuple has no `.dtype` of
    #    its own. Without the container clause this fell through to the scan and
    #    reported `shape_scan`, which is a true dtype under a false provenance:
    #    `test_the_fp32_generation_leg_allocates_an_fp32_kv_cache` asserts
    #    `attribute`, so the reference's own engine would have failed it.
    assert runtime._observed_cache_dtype(
        _Engine(named=(_FakeArray(_SHAPE, "float32"),))) == ("float32", "attribute")
    assert runtime._observed_cache_dtype(
        _Engine(named=[_FakeArray(_SHAPE, "bfloat16")])) == ("bfloat16", "attribute")
    # 9. ...and a container whose groups DISAGREE about dtype is not something to
    #    pick from either. It falls through to the scan, which sees two
    #    cache-shaped arrays of different dtypes and refuses -- the same
    #    fail-closed answer clause 6 gives, reached through the container.
    assert runtime._observed_cache_dtype(
        _Engine(named=(_FakeArray(_SHAPE, "float32"),
                       _FakeArray(_SHAPE, "bfloat16")))) == (None, None)
    # An empty container is absence, not a dtype.
    assert runtime._observed_cache_dtype(_Engine(named=())) == (None, None)


def test_an_empty_allocation_source_is_an_error_not_a_pass():
    """The frozen-allocation control has to have been EXERCISED, not merely declared."""
    results = _full_results()
    results["allocation_source"] = {}
    with pytest.raises(C.CandidateError, match="allocation_source"):
        C._validate_results(results)


# --------------------------------------------------------------------------- #
# A superset RunConfig must still construct (S1).
# --------------------------------------------------------------------------- #
def test_a_run_config_with_an_extra_field_is_still_constructible():
    """test_spec.test_interface says 'Superset policy: extras are fine.'

    The driver contradicted it: it iterated the CANDIDATE's __dataclass_fields__ and
    indexed the harness's fixed 9-key repo_config, so one extra field was a bare
    KeyError that killed the worker and scored the floor.
    """
    import dataclasses

    from _t2j.candidate import _run_config_kwargs

    @dataclasses.dataclass
    class RunConfig:
        page_size: int
        num_pages: int
        enable_prefix_caching: bool = False

    supplied = {"page_size": 8, "num_pages": 8, "max_model_len": 32}
    kwargs = _run_config_kwargs(RunConfig, supplied)
    assert kwargs == {"page_size": 8, "num_pages": 8}
    RunConfig(**kwargs)      # must not raise


def test_a_run_config_that_requires_a_field_outside_the_contract_names_it():
    """The other half: a field the CONTRACT declares but the grader never supplies
    is still an error -- just a legible one, naming the field, rather than a bare
    ``TypeError`` from the constructor."""
    import dataclasses

    from _t2j.candidate import _run_config_kwargs

    @dataclasses.dataclass
    class RunConfig:
        page_size: int
        not_in_repo_config: int

    with pytest.raises(RuntimeError, match="not_in_repo_config"):
        _run_config_kwargs(RunConfig, {"page_size": 8})


def test_a_run_config_classvar_is_not_mistaken_for_a_field_to_supply():
    """``__dataclass_fields__`` includes ``ClassVar`` pseudo-fields;
    ``dataclasses.fields()`` does not. A module constant declared inside the
    dataclass must not be demanded of ``repo_config``."""
    import dataclasses
    from typing import ClassVar

    from _t2j.candidate import _run_config_kwargs

    @dataclasses.dataclass
    class RunConfig:
        page_size: int
        SOME_CONSTANT: ClassVar[int] = 7

    kwargs = _run_config_kwargs(RunConfig, {"page_size": 8})
    assert kwargs == {"page_size": 8}
    RunConfig(**kwargs)      # must not raise


# --------------------------------------------------------------------------- #
# The frozen scenario's two pins must agree (B6).
# --------------------------------------------------------------------------- #
def test_the_two_frozen_pins_of_one_allocation_must_agree():
    """``scenario.json:page_ids`` vs ``inputs.npz:block_tables``.

    The decode slot arithmetic reads the first and the attention gather reads
    the second. A freezer that updates one and not the other produces a decode
    that reads pages the prefill never wrote -- and the failure lands on the
    CANDIDATE, in a probe both of whose inputs came from the golden.
    """
    import numpy as np

    # Agreeing, with the row padded out to pages_per_seq: fine.
    C.assert_frozen_allocation_agrees(np.array([4, 5, 0, 0], dtype=np.int32), [4, 5])
    # 2-D table, one row per sequence: `core` is row 0.
    C.assert_frozen_allocation_agrees(np.array([[4, 5, 0, 0]], dtype=np.int32), [4, 5])
    # Nothing to compare against is not a contradiction.
    C.assert_frozen_allocation_agrees(None, [4, 5])
    C.assert_frozen_allocation_agrees(np.array([4, 5], dtype=np.int32), [])

    with pytest.raises(ValueError, match="contradicts itself"):
        C.assert_frozen_allocation_agrees(np.array([0, 1, 0, 0], dtype=np.int32), [4, 5])
    # Right pages, wrong ORDER -- the block table is a gather, so order is the
    # allocation and not a presentation detail.
    with pytest.raises(ValueError, match="contradicts itself"):
        C.assert_frozen_allocation_agrees(np.array([5, 4, 0, 0], dtype=np.int32), [4, 5])
    # A table too short to hold the pinned pages.
    with pytest.raises(ValueError, match="contradicts itself"):
        C.assert_frozen_allocation_agrees(np.array([4], dtype=np.int32), [4, 5])


# --------------------------------------------------------------------------- #
# The bounded receive (B1).
# --------------------------------------------------------------------------- #
class _StallingConn:
    """A pipe that says "ready" and then takes far too long to deliver."""

    def __init__(self, delay: float, payload):
        self.delay = delay
        self.payload = payload

    def poll(self, timeout=None):
        return True

    def recv(self):
        time.sleep(self.delay)
        return self.payload

    def close(self):
        pass


class _FakeProcess:
    def __init__(self, *_a, **_k):
        self.alive = True

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def join(self, timeout=None):
        self.alive = False


class _FakeContext:
    def __init__(self, conn):
        self.conn = conn

    def Pipe(self, duplex=False):        # noqa: N802 - mirrors multiprocessing
        return self.conn, _StallingConn(0.0, None)

    def Process(self, *a, **k):          # noqa: N802 - mirrors multiprocessing
        return _FakeProcess()


def test_a_worker_that_stalls_MID_SEND_cannot_outlive_the_timeout(monkeypatch):
    """``poll()`` says a message started; it does not say it will finish.

    An unbounded ``recv()`` after a satisfied ``poll()`` hands the candidate an
    unlimited extension of its own ``generate_timeout_s``: a child that writes
    half a pickle and wedges (an XLA hang mid-serialisation, an OOM kill between
    chunks) blocks the grader forever, and no later timeout can reclaim it.
    """
    cand = _unrun()
    cand._timeout = 0.2
    stalled = _StallingConn(30.0, {"ok": True, "results": _full_results()})
    monkeypatch.setattr(C.mp, "get_context", lambda _name: _FakeContext(stalled))

    started = time.monotonic()
    with pytest.raises(C.CandidateError, match="generate_timeout_s"):
        cand._ensure_run()
    elapsed = time.monotonic() - started
    assert elapsed < 10.0, (
        f"the receive was not bounded: it took {elapsed:.1f}s to give up on a "
        "worker that stalled part-way through sending"
    )


def test_a_bounded_receive_still_delivers_a_message_that_arrives(monkeypatch):
    """...and the bound must not turn a slow-but-fine worker into a failure."""
    cand = _unrun()
    cand._timeout = 30.0
    prompt = _StallingConn(0.0, {"ok": True, "results": _full_results()})
    monkeypatch.setattr(C.mp, "get_context", lambda _name: _FakeContext(prompt))

    results = cand._ensure_run()
    assert set(C.REQUIRED_PROBES) <= set(results)


class _RaisingConn(_StallingConn):
    """A pipe that says "ready" and then blows up on the receive side -- the
    shape of a child killed part-way through the pickle, not a clean EOF."""

    def __init__(self, exc):
        super().__init__(0.0, None)
        self._exc = exc

    def recv(self):
        raise self._exc


class _CountingContext(_FakeContext):
    """Like ``_FakeContext``, but remembers every ``_FakeProcess`` it handed
    out, so a test can tell a memoised failure from a respawn."""

    def __init__(self, conn):
        super().__init__(conn)
        self.processes: list[_FakeProcess] = []

    def Process(self, *a, **k):          # noqa: N802 - mirrors multiprocessing
        proc = super().Process(*a, **k)
        self.processes.append(proc)
        return proc


def test_a_receive_side_failure_is_memoised_and_the_child_is_reaped(monkeypatch):
    """Without this fix, `_ensure_run` sets `self._error` only on a well-formed
    `{"ok": False}` payload and catches only `EOFError` around the receive.
    CPython raises `EOFError` for a zero-byte read but `OSError("got end of
    file during message")` for a child killed part-way through the pickle --
    precisely the case `_recv_within`'s own docstring exists for -- and that
    escapes BEFORE `self._hard_kill(process)` runs. Every one of the 31
    session-scoped checks that share one `Candidate` would then respawn an
    unreaped worker still holding the converted weights: 31 x ~68s against the
    1800s verifier budget. A void grade, not a low one.
    """
    cand = _unrun()
    cand._timeout = 5.0
    raising = _RaisingConn(OSError("got end of file during message"))
    ctx = _CountingContext(raising)
    monkeypatch.setattr(C.mp, "get_context", lambda _name: ctx)

    with pytest.raises(C.CandidateError):
        cand._ensure_run()
    assert cand._error is not None, "the failure was not memoised; the next check respawns"
    assert len(ctx.processes) == 1
    assert ctx.processes[0].alive is False, "the child was not reaped on the failure path"

    with pytest.raises(C.CandidateError):
        cand._ensure_run()          # must not spawn a second worker
    assert len(ctx.processes) == 1, "a memoised failure must not respawn a worker"


# --------------------------------------------------------------------------- #
# H3: `Golden.fp32_thresholds` must MERGE over the class floor, like
# `native_thresholds` does -- not construct from the instance value alone.
# --------------------------------------------------------------------------- #
def test_a_partial_fp32_band_falls_back_to_the_class_floor_not_a_TypeError():
    """``NumericThresholds`` gives a dataclass default to ``argmax_margin``
    only; ``abs_tol``, ``rel_tol``, ``cosine_floor`` and ``max_kl`` are all
    required. Without this merge, ``fp32_thresholds`` constructs straight from
    ``meta["fp32_thresholds"]`` alone, so a PARTIAL fp32 band -- missing any
    of those four -- raises ``TypeError`` from the constructor, inside every
    numeric and behavioral check that reads this property: reward 0, charged
    to the candidate, for an instance-authoring omission that has nothing to
    do with the port.

    ``TESTS`` is already on ``sys.path`` (see the top of this module), so
    ``fixtures`` -- a sibling of ``_t2j``, one level up -- is importable
    without any extra path surgery.
    """
    from fixtures import Golden

    g = Golden.__new__(Golden)
    g.meta = {
        "thresholds": {"abs_tol": 0.1, "rel_tol": 0.1, "cosine_floor": 0.98,
                        "max_kl": 0.1},
        # Partial: only `abs_tol` overridden. A pre-fix `fp32_thresholds`
        # would raise `TypeError: missing 2 required positional arguments:
        # 'rel_tol' and 'cosine_floor'` (`max_kl` too) right here.
        "fp32_thresholds": {"abs_tol": 1e-4},
    }
    band = g.fp32_thresholds
    assert band.abs_tol == 1e-4                # the instance override wins
    assert band.cosine_floor == 0.999999, (     # from FP32_EXACT_THRESHOLDS,
        "the merge base must be the fp32 class floor, not meta['thresholds'] "
        "(whose cosine_floor is 0.98) and not the dataclass's own defaults"
    )


def _overlay_module():
    """This package's own ``conftest.py``, imported as a plain module.

    Loaded by PATH rather than by name: ``conftest`` as an import name resolves to
    the GRADED tree's conftest (``tests/`` is on ``sys.path``; ``_t2j/tests`` is
    not), and pytest's own copy of this file is registered under an internal name
    that is not part of any contract. Re-executing it is safe -- it defines
    fixtures and one ``sys.path`` insert, and no import-time side effect beyond
    that.
    """
    import importlib.util

    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("_t2j_tests_overlay", path)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module absent from there makes the
    # decorator raise `AttributeError: 'NoneType' object has no attribute
    # '__dict__'`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_synthetic_golden_overlay_is_a_contract_complete_golden(tmp_path):
    """The overlay's SYNTHETIC half satisfies the same meta contract a freeze does.

    THIS TEST EXISTS BECAUSE THE CODE IT COVERS RUNS NOWHERE ELSE. ``conftest.
    golden_dir`` returns the real ``tests/golden`` untouched the moment that tree
    is complete -- which it has been on every authoring host and in every container
    since the canonical freeze -- so ``_synthesize`` executes only in the two states
    a developer hits and CI does not: a bare clone, and a host whose git-ignored
    ``golden/meta.json`` has not been fetched yet. A transcribed meta drifts behind
    the contract with nothing red, erroring the driver tests in this package at
    setup (``KeyError: 'fp32_kv_cache_dtype'``
    out of ``build_candidate``, via the session-scoped ``cand`` /
    ``misallocated_cand`` / ``desynced_cand``), plus checks in
    ``_harness_checks`` failing on the fully-synthetic tree.

    So it is exercised UNCONDITIONALLY here, against a tmp dir, in the state that
    rots first (nothing real to overlay). No checkpoint, no GPU, no sandbox: the
    three things asserted are the three that broke.

    ``build_candidate`` is CONSTRUCTION ONLY -- ``Candidate.__init__`` runs nothing
    (``test_constructing_a_candidate_runs_nothing``) -- and it is the exact call
    site that raised, so it belongs in a no-sandbox contract test rather than
    behind the checkpoint gate.
    """
    import _harness_checks
    from fixtures import build_candidate, load_golden

    overlay = _overlay_module()
    overlay._synthesize(tmp_path)

    golden = load_golden(tmp_path)
    # 1. Every artifact `fixtures.load_golden` opens is there and parses -- that is
    #    `load_golden` returning at all -- and every key `build_candidate`
    #    hard-indexes out of `meta` is present. The regression was here.
    build_candidate(TESTS, golden, allow_reference_default=True)

    # 2. Every instance.yaml key the freezer mirrors is mirrored. Compared as a KEY
    #    SET, not by value: `check_meta_mirrors_instance_yaml` below already
    #    compares values, and this names the failure as "the synthesis is behind
    #    the yaml" when a new key is added, which is the way this drifts.
    import yaml

    inst = yaml.safe_load((TESTS / "instance.yaml").read_text(encoding="utf-8"))
    unmirrored = sorted(
        set(inst) - _harness_checks._INSTANCE_KEYS_NOT_MIRRORED - set(golden.meta))
    assert not unmirrored, (
        f"the synthetic meta is missing instance.yaml key(s) {unmirrored}. "
        "_synthesize mirrors the yaml wholesale for exactly this reason -- if this "
        "fires, something started transcribing keys again.")

    # 3. The benchmark's own self-checks pass on it. A synthetic golden that the
    #    precheck would refuse is not a stand-in for a real one, and this is the
    #    clause that would have caught the failures described above.
    findings = _harness_checks.run(golden)
    assert not findings, {
        "error": "the synthetic golden overlay fails the benchmark's own "
                 "self-checks, so the driver's unit tests would be running "
                 "against a tree the grader refuses",
        "failed": [name for name, _ in findings],
        "detail": {name: message[:400] for name, message in findings},
    }


def test_the_parent_side_of_the_driver_imports_no_jax_and_no_torch():
    """The candidate is never imported into the pytest process, and neither is
    the framework it runs on: jax belongs to the sandbox child, so importing the
    driver must not initialise a second CUDA context in the parent."""
    before = set(sys.modules)
    import importlib

    importlib.reload(C)
    assert "torch" not in sys.modules
    # jax may already be resident for unrelated reasons; what matters is that
    # importing the driver did not pull it in.
    assert "jax" not in (set(sys.modules) - before)


@_NEEDS_CHECKPOINT
def test_the_worker_cannot_import_the_graders_own_modules(cand):
    """The scratch cwd (already fixed) only removed the RELATIVE glob. This is
    about a SEPARATE hole: a ``spawn`` child inherits the parent's ``sys.path``,
    so before this fix the grader's own ``tests/`` directory was importable
    from inside the worker -- ``import fixtures; fixtures.load_golden(...)``
    returned every frozen answer with no model involved at all.

    NOT CLAIMED HERE: that the goldens are unreachable. ``_t2j.candidate`` --
    this very module -- was already imported before the worker could even
    start (multiprocessing had to import it to find ``_worker_entry``), so it
    stays in ``sys.modules`` no matter what happens to ``sys.path``, and
    ``sys.modules['_t2j.candidate'].__file__`` still names this file, from
    which the golden directory is a couple of ``.parent``s away. That is the
    ABSOLUTE-PATH route and no in-process ``sys.path``/``sys.modules`` edit can
    close it -- the goldens are still on disk, still readable by this
    process's own uid. This test asserts only that the MODULE route is closed:
    the tests directory is off the path, and no grader module -- including the
    one already loaded -- can be reached by a FRESH import through it.
    """
    report = cand.sandbox_report
    assert report.get("relative_golden_hits") == [], report.get("relative_golden_hits")
    assert report.get("tests_dir_on_sys_path") == [], {
        "error": "the grader's own tests directory is still importable inside "
                 "the sandbox, so `import fixtures; fixtures.load_golden(...)` "
                 "returns every frozen answer",
        "entries": report.get("tests_dir_on_sys_path"),
    }
    assert report.get("grader_modules_importable") == [], {
        "error": "a grader module is still reachable by a FRESH import from "
                 "inside the sandbox after the tests-directory prune",
        "importable": report.get("grader_modules_importable"),
    }


@_NEEDS_CHECKPOINT
def test_the_torch_blocker_survives_into_a_subprocess(cand):
    """The marker env var ``T2J_TORCH_SANDBOX`` is not, on its own, enough to
    block torch in a child: nothing READS it -- only the constant's own
    definition and env-strip lists elsewhere. A candidate
    that did its work in ``subprocess.run([sys.executable, ...])`` -- or its
    own spawn child -- runs with no meta-path blocker at all:
    ``install_torch_blocker`` only ever touches the CALLING process's
    ``sys.meta_path``, and ``banned_modules_in_sys`` only ever inspects the
    calling process's ``sys.modules``.

    This asserts the fix from the CONSUMER side: a real subprocess, spawned by
    the worker and inheriting only its environment, still gets
    ``TorchBlockedError`` rather than importing torch (or, in a torch-free
    image, rather than a plain ``ModuleNotFoundError`` -- the type is the only
    thing that tells "refused" apart from "absent", exactly as it does for the
    in-process probe beside it in ``sandbox_report``).
    """
    report = cand.sandbox_report
    assert report.get("subprocess_torch_import_error_type") == "TorchBlockedError", {
        "error": "a subprocess launched from inside the sandbox was not "
                 "torch-blocked -- the blocker did not survive the spawn",
        "raised": report.get("subprocess_torch_import_error_type"),
    }


# --------------------------------------------------------------------------- #
# Per-probe isolation (F1): one probe's failure must fail only its own readers.
# --------------------------------------------------------------------------- #
@_NEEDS_CHECKPOINT
def test_one_failed_probe_fails_only_the_checks_that_read_it(cand):
    """`reward = passed/(passed+failed+error)` promises partial credit; without
    per-probe isolation the driver delivers a step function.

    All probes run in one worker with no per-probe catch, so any raise costs
    nearly every check -- the 2/31 floor, identical to `import torch`. A port
    that is 90% right must not score what a torch import scores.

    A REAL run (the ``cand`` fixture, not a hand-built dict), because the point
    is that ``Candidate._probe`` raises on a value that ``run_all`` actually
    produced this way -- a hand-rolled results dict would not prove the two
    ends of the contract (the marker shape ``run_all`` writes and the shape
    ``_probe`` reads) agree.

    ``cand`` is SESSION-scoped, so its cached results are mutated in place and
    restored in ``finally`` -- this is the last test in the file specifically so
    that ordering accident is harmless, but there is no reason to leave the
    session handle poisoned for whoever runs this file next.
    """
    from _t2j.candidate import CandidateError

    original = cand._ensure_run()
    results = dict(original)
    results["sampler_outputs"] = {"__probe_error__": "boom: synthetic"}
    cand._results = results
    try:
        with pytest.raises(CandidateError) as exc:
            _ = cand.sampler_outputs
        assert "sampler_outputs" in str(exc.value)
        # ...and an unrelated probe is untouched.
        assert cand.prefill(fp32=False)["logits"] is not None
    finally:
        cand._results = original
