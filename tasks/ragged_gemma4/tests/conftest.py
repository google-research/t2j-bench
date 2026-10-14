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

"""Pytest wiring for the standalone, torch-free inference grader.

The candidate defaults to ../solution/ref_jax_repo; Harbor overrides it with
T2J_JAX_REPO. Golden/configuration health, collection completeness, and the pinned
device are grading preconditions. Their failure raises pytest.UsageError, so the
entrypoint reports an invalid run rather than a candidate reward.

Spec outcomes gate the continuous Numeric and Behavioral scores emitted in
eval_summary.json. Inapplicable checks are deselected. Helper unit tests are
excluded from graded collection, and sessions with no graded outcomes cannot
report overall PASS. Performance measurement helpers remain available but are
not part of the three-stage reward.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

# Never write bytecode caches next to the sources: a bare `pytest
# tasks/*/tests` imports these modules in place, so __pycache__ would
# land under tasks/. Set before the fixtures/_t2j imports below.
sys.dont_write_bytecode = True

# Hermetic + offline grading: the candidate must not hit the network, and the
# torch-blocked sandbox subprocess inherits this environment.
#
# ASSIGNED, NOT `setdefault`, for the reason spelled out at
# JAX_DEFAULT_MATMUL_PRECISION below: `setdefault` is the wrong verb for a
# MANDATORY pin. These two used to be defaults, and THIS FILE IS THE ONLY THING
# THAT SETS THEM under Harbor -- `tests/test.sh` exports neither and neither
# Dockerfile carries an ENV line -- so an inherited value from the calling shell
# was the whole decision. With setdefault, `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=true
# python -c 'import conftest'` left them at '0' and 'true', i.e. a grade with a
# live hub, while `JAX_DEFAULT_MATMUL_PRECISION` seventeen lines down was pinned
# by assignment in the same preamble.
#
# WHAT THIS COSTS, STATED RATHER THAN LEFT TO BE FOUND. `test_spec.test_loader_spec`
# used to assert `os.environ.get(name) == "1"` on these two, in THIS process --
# and once they are assigned here, that clause cannot fail: nothing between this
# line and the assertion can change them, and the candidate runs in a `spawn`
# child that cannot write back. A graded clause that cannot fail is this repo's
# cardinal sin, so it was not left in place beside a pin that made it vacuous:
# it now reads the value out of the SANDBOX CHILD's own environment
# (`sandbox_report['offline_env']`), which is where the claim has content -- the
# child is the process that would do the downloading -- and where a candidate
# that clears the variable before its probes run is observable. See that check
# for the measurement.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Not redundant with PREALLOCATE: that only stops the up-front grab, while the
# BFC pool is still capped at MEM_FRACTION, which defaults to 0.75 of the card.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
# fp32 MEANS fp32. Without this XLA:GPU lowers an fp32 matmul to TF32 (10-bit
# mantissa) and the fp32-exactness tier -- the one surface where both sides run
# identical fp32 weights and the gate is 1e-6-tight -- measures TF32 noise
# instead of a porting bug. The freezer runs under the same pin, so the two
# sides agree by construction rather than by coincidence.
#
# Assigned, NOT `setdefault`: this is a Global Constraint of the class, mandatory
# wherever fp32 goldens are produced or graded. `setdefault` lets an inherited
# `JAX_DEFAULT_MATMUL_PRECISION=default` from the calling shell win silently, and
# the result is a green fp32 tier measured at TF32 -- the failure this pin exists
# to prevent, restored by the one line that looks like it prevents it.
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

# XLA GEMM AUTOTUNING OFF, unconditionally and everywhere the graded tree is entered.
# The autotuner times candidate kernels at compile time and keeps the fastest, so the
# same graph can lower to different kernels in two runs of the same code -- one of the
# drivers of the cross-process bf16 spread instance.yaml calibrates against (up to
# 1.9e-01 on the prefill logits). The grading SKU is pinned, so autotuning buys nothing
# and costs reproducibility.
#
# APPENDED AND IDEMPOTENT, not assigned. `XLA_FLAGS` is one flat string: an assignment
# would drop whatever the caller set -- and Harbor's `tests/test.sh` DOES set this same
# flag before pytest starts, so the two would fight. The idempotence guard is what makes
# setting it in both places safe rather than duplicative.
#
# THE FLAG REACHES THE CANDIDATE THROUGH THE ENVIRONMENT, NOT THROUGH THIS FILE. The
# candidate runs in a `mp.get_context("spawn")` child which never imports this module --
# it is not on the candidate's import path -- but does inherit `os.environ`, and that is
# the whole mechanism. The worker process's own
# XLA_FLAGS carries `--xla_gpu_autotune_level=0` exactly once.
#
# THE CANDIDATE CAN READ THIS FILE, and nothing here assumes otherwise. The worker shares
# this process's uid and has unrestricted file I/O, so it can READ anything under the
# tests tree by absolute path. Nothing here depends on it not doing so -- reading this
# file would tell a candidate the autotuner is off, which is not a secret -- but a "never
# sees" reachability claim would be wrong, so this comment does not make one.
_XLA_AUTOTUNE_OFF = "--xla_gpu_autotune_level=0"
if _XLA_AUTOTUNE_OFF not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_XLA_AUTOTUNE_OFF}".strip()

# Make the vendored `_t2j` package and sibling `fixtures` importable regardless
# of pytest's rootdir / import mode.
_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from fixtures import build_candidate, load_golden  # noqa: E402
from _t2j import golden_manifest, similarity  # noqa: E402
from _t2j.candidate import CandidateError, CandidateTimeout, second_prompt_length  # noqa: E402

# --------------------------------------------------------------------------- #
# The graded invocation is `pytest tasks/{mode}_{family}/tests`, and
# `_t2j/tests/` is a normal pytest package sitting underneath it. Without this
# the GRADER'S OWN unit tests are collected by the grade: in the verifier image
# both /data and tests/golden exist, so they RUN -- spawning a second candidate
# worker and burning GPU wall clock -- and several of them legitimately fail for
# a correct port (they pin `STACKED_PARAMS` row order and both halves of the
# embedding tie, which are properties of the reference, not of the contract).
# The reward SCALAR is safe (`summary_to_reward` reads only the three
# `eval_summary` stages, and these stems are not in `_STAGE_OF`), but pytest's
# exit code is not. Defence belongs at the tree, not at one caller, so it is here
# rather than in the verifier's `test.sh` module list.
#
# It blocks RECURSION, not access: pytest does not apply collect_ignore_glob to a
# path named directly on the command line, so the driver's own suite still runs
# as `pytest tests/_t2j/tests/test_*.py`. That asymmetry is the point -- the
# graded invocation names the DIRECTORY.
collect_ignore_glob = ["_t2j/tests/*"]


@pytest.fixture(scope="session")
def golden():
    """Frozen reference artifacts + materialized thresholds."""
    return load_golden(_TEST_DIR / "golden")


@pytest.fixture(scope="session")
def cand(golden):
    """Candidate repo (default ../solution/ref_jax_repo; T2J_JAX_REPO overrides),
    driven in a torch-blocked sandbox against the frozen scenario.

    Built lazily: the sandbox subprocess only spawns on the first probe access,
    so a missing/broken repo surfaces as a clear failure in the test that uses it
    rather than at collection time.
    """
    return build_candidate(_TEST_DIR, golden)


# --------------------------------------------------------------------------- #
# The pinned-device gate.
#
# Every golden in this class is frozen on one H100 SKU, and the fp32 tier is
# graded at 1e-6. Running the suite anywhere else does not produce a slightly
# different number -- it produces a number that means nothing, and on CPU it
# produces one silently. So the device is a PRECONDITION, checked once per
# session and raised on, never skipped and never warned about.
# --------------------------------------------------------------------------- #
def _active_device_kind() -> str:
    """The JAX device this session will actually run on."""
    import jax

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reports no devices at all; the grade cannot run here")
    return str(devices[0].device_kind)


def _normalize_device(name: str) -> str:
    """Fold the cosmetic differences between two spellings of the same SKU.

    ``jax``'s ``device_kind`` and ``nvidia-smi``'s model string agree on the part
    that matters (``h100 80gb hbm3``) and disagree on vendor prefix, separators
    and case. Matching on the folded form keeps the gate from failing a correct
    H100 over a hyphen, while still refusing an A100 or a CPU.
    """
    folded = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    folded = re.sub(r"\bnvidia\b", " ", folded)
    return " ".join(folded.split())


def _device_matches(want: str, got: str) -> bool:
    """True iff the reported device satisfies the pin.

    Containment in ONE direction only -- the pin must be contained in what the
    runtime reports (``want`` ⊆ ``got``). The spec pins a single SKU, ``NVIDIA
    H100 80GB HBM3``, and containment the other way would let a device that
    reports the bare family name ``H100`` satisfy it: an H100 PCIe 80GB, or an
    H100 NVL, neither of which the goldens were calibrated on. On the
    grading host, ``jax``'s ``device_kind`` is the FULL string ``NVIDIA H100 80GB
    HBM3``, so the one-directional test costs nothing here and refuses the
    under-specified spellings. An empty side never matches -- an unknown device
    is not a match.

    A MIG SLICE IS REFUSED, and it is the one thing one-directional containment
    would otherwise wave through: ``NVIDIA H100 80GB HBM3 MIG 1g.10gb`` CONTAINS
    the pin while being a seventh of the card, with a fraction of the SMs, its
    own L2 partition and no NVLink. It is a materially different device and the
    goldens are calibrated on the whole one, so the qualifier disqualifies rather
    than decorates. Only an unqualified pin refuses it -- a pin that names a MIG
    profile itself is asking for that profile and is matched normally.
    """
    a, b = _normalize_device(want), _normalize_device(got)
    if not a or not b:
        return False
    if "mig" in b.split() and "mig" not in a.split():
        return False
    return a in b


def _assert_pinned_device(golden_dir, resolve=_active_device_kind) -> str:
    """Raise unless the active device is the one ``meta.provenance`` pins.

    ``provenance.gpu_model`` names the device this grade MUST RUN ON. It is not
    the device the golden was produced on -- the freeze is torch on CPU -- so the
    wording matters: a reader who believes the golden was frozen here would
    conclude that a matching device makes the comparison exact, when what the pin
    actually buys is that every tier is calibrated against one grading SKU.

    ``resolve`` is injected so the gate is testable without a GPU. An absent or
    empty ``gpu_model`` raises too: an unverifiable pin is not a satisfied pin,
    and defaulting to "assume it is right" is exactly the silent CPU fallback
    this exists to prevent.
    """
    meta = json.loads((Path(golden_dir) / "meta.json").read_text())
    want = str((meta.get("provenance") or {}).get("gpu_model") or "").strip()
    if not want:
        raise RuntimeError(
            f"{golden_dir}/meta.json carries no provenance.gpu_model, so the "
            "pinned-device precondition cannot be verified. Re-freeze the golden "
            "with a freezer that stamps provenance."
        )
    got = resolve()
    if not _device_matches(want, got):
        raise RuntimeError(
            f"this grade must run on the pinned device. golden/meta.json says it "
            f"must be graded on {want!r}; JAX reports {got!r}. Every tier -- Spec "
            "included -- is calibrated against that SKU, so running elsewhere "
            "produces numbers that cannot be compared."
        )
    return got


def _require_pinned_device(golden_dir) -> None:
    """:func:`_assert_pinned_device` at a CALL SITE, as a pytest usage error.

    Two things this wrapper does that the raise inside cannot.

    IT NARROWS THE SCOPE. The precondition used to run in ``pytest_configure``,
    which fires for every session that touches this tree -- and by this file's
    own case-2 rule that includes ``pytest tests/_t2j/tests/test_*.py``, the
    driver's own unit tests, which grade nothing and would then have required an
    H100 on a developer box the moment a golden landed. It now runs at the end of
    collection, gated on at least one GRADED item having survived deselection, so
    it still fires before any graded probe runs and covers every path that can
    reach one.

    IT CHANGES THE EXIT SHAPE. A ``RuntimeError`` escaping a pytest hook is an
    INTERNALERROR: exit code 3 and a traceback through pytest's own frames, which
    reads as a broken harness rather than as a wrong machine. ``UsageError`` is
    what pytest has for "you invoked this wrong": it prints ``ERROR: <message>``
    and exits 4. The message is the same one, unchanged.

    ``_assert_pinned_device`` keeps raising ``RuntimeError`` -- its unit tests key
    off that type, and the distinction is real: the function states a fact about
    the device, this call site decides what pytest should do about it.
    """
    try:
        _assert_pinned_device(golden_dir)
    except RuntimeError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _require_healthy_harness(golden_dir) -> bool:
    """The benchmark self-checks, as a pytest usage error -- THE DEVELOPER PATH.

    Returns whether the golden set LOADED, which is the scope every gate that needs a
    real instance shares (see the call site): a tree whose golden will not load is not
    grading anything, and the ``golden`` fixture says so louder than any gate could.

    An earlier change moved ``check_golden_manifest``,
    ``check_applicability_table_names_real_checks`` and
    ``check_threshold_extras_name_something_a_check_reads`` out of the graded
    denominator and into ``_grade_precheck.py``, which only ``tests/test.sh`` runs.
    That was right for the score and wrong for the feedback loop: it left them
    running in NO developer path, so a broken golden or a typo'd gate stayed
    invisible until somebody did a full container grade. This is the second half of
    that move, and it costs the denominator nothing -- a UsageError is not a
    collected item, so a healthy harness contributes zero tests and an unhealthy one
    aborts the session instead of scoring it.

    THE SAME SHAPE AS THE OTHER THREE GATES, deliberately: raised at the end of
    collection, gated on something graded having survived it, and reported as a
    ``UsageError`` (``ERROR: <message>``, exit 4) rather than as a failure. All four
    gates answer "this session cannot produce a meaningful number", which is not a
    statement about the candidate.

    A GOLDEN THAT WILL NOT LOAD IS NOT REPORTED HERE. ``load_golden`` raises
    ``FileNotFoundError`` naming the artifact and the freezer, and the ``golden``
    fixture will surface exactly that in every check -- louder and more accurate than
    anything this gate could say about it. Same rule the device gate already applies
    to a missing ``meta.json``.

    IT FIRES BEFORE THE DEVICE GATE. A golden with a hole in it is worth reporting on
    the right machine too, and the two messages name different repairs.
    """
    from fixtures import load_golden

    try:
        golden = load_golden(golden_dir)
    except Exception:            # noqa: BLE001 - ANY unloadable golden
        # (OSError, ValueError, KeyError) was too narrow. `fixtures._guard`
        # raises RuntimeError on a Git LFS pointer and `np.load` raises
        # zipfile.BadZipFile on a truncated .npz -- neither subclasses those
        # three, so both escaped this hook as an INTERNALERROR (exit 3), which
        # `tests/test.sh` does not intercept (it traps only exit 4).
        # Execution then reached `summary_to_reward.py` with no
        # eval_summary.json and wrote reward 0.0 -- a harness fault scored as
        # a failed port, which is the one thing the gates in this module
        # exist to prevent. `BaseException` is deliberately NOT caught here:
        # KeyboardInterrupt/SystemExit must still propagate.
        return False                # the `golden` fixture says it better; let it
    import _harness_checks

    findings = _harness_checks.run(golden)
    if not findings:
        return True
    detail = "\n\n".join(f"--- check_{name} ---\n{message}" for name, message in findings)
    raise pytest.UsageError(
        f"{len(findings)} of this benchmark's own self-checks failed, so this suite "
        "cannot grade anything meaningfully. These describe the TASK, not the "
        "candidate: a golden set with a hole in it, an applicability gate that "
        "deselects nothing, or a threshold key no check reads. Nothing was graded and "
        "no score is implied. Re-run them alone with "
        "`python tests/_grade_precheck.py --harness-only`, and prove they can fail "
        f"with `--prove-guards`.\n\n{detail}")


def _require_complete_geometry(facts, golden_dir) -> None:
    """A HARNESS DEFECT IS NOT A DESELECTION (a pytest usage error, like the device).

    An applicability predicate that reads a golden key which is not there answers
    from a default, and the default silently removes a graded check -- the same
    accounting error as an unfalsifiable pass, with the sign flipped and no trace in
    the summary. ``int(meta.get("prompt_len") or 0)`` was exactly that: one missing
    key and a Spec check disappears, scoring the candidate 1.0 over a smaller suite.

    Scoped the same way the device gate is: only for a golden that IS one (a
    ``mode`` was declared) and only when something graded survived collection. With
    no ``meta.json`` at all the ``golden`` fixture already raises
    ``FileNotFoundError`` naming the freezer in every check, which is louder and more
    accurate, and a non-graded session (the driver's own unit tests) has no
    business demanding a complete golden.

    It fires BEFORE the device gate, which is deliberate: a partial golden is worth
    reporting even on the right machine, and the two messages name different repairs.
    """
    if facts.get("mode") is None or not facts.get("geometry_missing"):
        return
    raise pytest.UsageError(
        f"{golden_dir}/meta.json declares mode {facts['mode']!r} but omits "
        f"{facts['geometry_missing']}, which applicability predicates read. A missing "
        "geometry key would silently deselect a graded check rather than fail "
        "anything, so this refuses to grade against a partial golden. Re-freeze it "
        "with the golden freezer."
    )


# --------------------------------------------------------------------------- #
# Applicability deselection: the reward is a continuous pass rate,
# so NO check may skip at grade time. A check that cannot apply to THIS instance
# -- because its mode has no such concept, or because the reference never froze
# the artifact it compares against -- is DESELECTED here (never collected),
# keeping it out of both numerator and denominator. Everything that survives
# collection must resolve to pass or fail: a candidate that cannot run an
# APPLICABLE check fails it (the check's own body asserts ``supported``), it does
# not skip. Applicability is read purely from the frozen golden (torch-free, no
# candidate run), so it self-adjusts if the goldens change.
# --------------------------------------------------------------------------- #
def _npz_keys(path: Path) -> set[str]:
    """Array names inside an .npz WITHOUT loading any array (npz is a zip of .npy)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    except (OSError, zipfile.BadZipFile):
        return set()


def _has_battery(golden_dir: Path, axis: str) -> bool:
    """A frozen battery AXIS is a directory of variants, not a file.

    ``.exists()`` would answer True for a stray file at that path, which is the
    silent-coverage-loss shape ``golden_manifest`` already documents for the same
    token grammar.
    """
    root = golden_dir / "battery" / axis
    return root.is_dir() and any(root.glob("*/inputs.npz"))


#: The mode ladder, in ladder order: `core` (subset of) `ragged` (subset of)
#: `chunked`. THE SINGLE SOURCE OF TRUTH for "every mode this class admits" --
#: the instruction generator's own test reads this constant rather
#: than hardcoding a parallel list, so a fourth rung cannot ship with no
#: `MODE_PROSE` entry the way `chunked` briefly did (both places would have to
#: independently agree to forget it, rather than one silently drifting from
#: the other).
MODE_LADDER: tuple[str, ...] = ("core", "ragged", "chunked")


def _golden_facts(golden_dir) -> dict:
    golden_dir = Path(golden_dir)
    meta: dict = {}
    try:
        meta = json.loads((golden_dir / "meta.json").read_text())
    except (OSError, ValueError):
        pass
    prefill_keys = _npz_keys(golden_dir / "prefill_forward.npz")
    mode = meta.get("mode")
    # Is there a SECOND prompt length to prefill at? Answered by
    # `_t2j.candidate.second_prompt_length` -- THE SAME FUNCTION the probe calls to
    # decide whether its new-shape leg can run, imported rather than restated,
    # because a gate and the thing it gates that are two expressions will stop
    # agreeing (they did, in both directions). Only that one leg rides this; the
    # counter-liveness and same-shape assertions need no second length and are in a
    # check that always runs. Read off the frozen golden -- no candidate run.
    # Read STRICTLY. `int(meta.get(k) or 0)` fails OPEN: a golden missing one key
    # silently removed a Spec check, which is a harness defect being forgiven on the
    # candidate's behalf. Absence is recorded and `_collect` raises on it; it is not
    # turned into a deselection.
    #
    # UNUSABLE counts the same as ABSENT, and that is not pedantry: reading only for
    # `is None` failed closed on absence but OPEN on garbage. `page_size: 0` is not
    # None, so it cleared the missing-key check, made `second_prompt_length` return
    # None, and silently deselected a graded check with exit 0 -- the exact hole the
    # strict read was written to close, one value type over. A geometry key that is
    # not a positive int cannot describe a geometry, so it is reported, not tolerated.
    def _unusable(key: str) -> bool:
        value = meta.get(key)
        return not (isinstance(value, int) and not isinstance(value, bool) and value > 0)

    geometry_missing = sorted(k for k in ("prompt_len", "page_size") if _unusable(k))
    has_second_length = bool(not geometry_missing and second_prompt_length(
        meta.get("prompt_len"), meta.get("page_size")) is not None)
    # THE MODE LADDER. `has_chunked` is "mode is at or above the chunked
    # rung" and NOT `mode == MODE_LADDER[-1]` -- the latter hardcodes "last
    # rung" rather than "chunked", so it would silently invert the moment a
    # fifth mode is appended after `chunked`: existing `chunked_llama3`
    # goldens would then read `has_chunked=False` and quietly deselect
    # test_chunked_prefill_logits / test_chunking_invariance /
    # test_chunk_boundary_battery -- the exact silent-deselection shape this
    # class's own comments call the worst kind of defect, recreated one
    # level down from where it was closed for MODE_PROSE. And NOT
    # `mode != "core"` either: read it that loose and a `ragged` instance
    # collects both chunked-only numeric checks and both chunked-only
    # invariants, all four of which then fail against an engine that has no
    # chunked path -- four failures with no obvious cause.
    #
    # AND A MODE OFF THE LADDER IS REFUSED, HERE, rather than answered. Every
    # membership test below is `mode in MODE_LADDER[n:]`, so an UNRECOGNISED mode
    # answers False to all of them -- which is not "unknown", it is exactly the
    # `core` gate set. A fourth rung added to `_t2j.golden_manifest
    # .REQUIRED_GOLDENS` (the allow-list the grade actually enforces) but not to
    # `MODE_LADDER` would therefore collect 33 of 44 against a full chunked
    # engine and score 1.0 -- the silent-deselection shape the paragraph above
    # closes one level up, reappearing as a typo in one string.
    #
    # RAISED FROM THE FACT READER, not from a `_require_*` gate beside the other
    # four, and that is the one thing about it worth arguing. The gates run AFTER
    # `_collect` has already applied these facts; a gate would abort the session
    # (which is enough for the SCORE), but it would leave this function still
    # handing wrong facts to anything else that asks. There is no gap between
    # deciding and using if the decision refuses.
    #
    # `mode is None` IS NOT AN UNKNOWN MODE. A tree with no `meta.json` at all --
    # the driver's own unit tests, every synthetic tree the harness uses -- reads
    # None here, and `_require_complete_geometry` and the device gate both
    # already treat that as "not a golden this speaks for". Only a golden that
    # DECLARES something is held to the ladder.
    if mode is not None and mode not in MODE_LADDER:
        raise pytest.UsageError(
            f"{golden_dir}/meta.json declares mode {mode!r}, which is not on the "
            f"mode ladder {list(MODE_LADDER)}. Every applicability predicate reads "
            "the ladder by membership, so an unrecognised mode does not deselect "
            "loudly -- it answers the CORE gate set and quietly removes every "
            "ragged- and chunked-only check while the run reports a clean pass "
            "rate over what is left. Add the rung to conftest.MODE_LADDER (and to "
            "_t2j.golden_manifest.REQUIRED_GOLDENS, which is what the grade "
            "enforces) or fix the spelling in instance.yaml and re-freeze."
        )
    has_ragged = mode in MODE_LADDER[1:]
    has_chunked = mode in MODE_LADDER[2:]
    return {
        "mode": mode,
        "has_ragged": has_ragged,
        "has_chunked": has_chunked,
        # Prefix caching is what `chunked` adds on top of partial prefill; it has
        # no separate golden, so it rides the same gate.
        "has_prefix_cache": has_chunked,
        # Family properties, from meta -- not sniffed from the inputs. Both are
        # facts about the MODEL (gpt_oss is MoE, qwen2_5_vl is
        # multimodal), and the freezer knows them while a key-name heuristic
        # would only guess.
        "is_moe": bool(meta.get("is_moe")),
        "multimodal": bool(meta.get("multimodal")),
        # Geometry, not a feature: whether the frozen prompt admits a second,
        # shorter length for the recompilation probe to prefill at.
        "has_second_prompt_length": has_second_length,
        # Geometry keys a real golden must declare. Non-empty -> `_collect` raises;
        # never silently a deselection.
        "geometry_missing": geometry_missing,
        # Optional features, presence-gated on what the freeze actually emitted.
        "has_hidden": bool(meta.get("compare_hidden_states"))
        and any(k.startswith("hidden_") for k in prefill_keys),
        "has_battery_prompt_length": _has_battery(golden_dir, "prompt_length"),
        "has_battery_paging": _has_battery(golden_dir, "paging"),
        "has_battery_batch_composition": _has_battery(golden_dir, "batch_composition"),
        "has_battery_chunk_boundary": _has_battery(golden_dir, "chunk_boundary"),
        "refunits": (golden_dir / "reference_units.npz").exists()
        and (golden_dir / "reference_units.json").exists(),
        # THE FP32 DECODE TIER HAS NO ARTIFACT IN THE GOLDEN CATALOG, and this is a
        # spec/catalog gap rather than a missing freeze. The spec lists
        # `test_fp32_decode_step_logits` beside `test_decode_step_logits`, but the catalog
        # freezes exactly one decode golden -- `decode_step.npz`, at the NATIVE
        # dtype -- so there is nothing for an fp32 decode probe to be compared
        # against. Presence-gated on the artifact a future producer would write,
        # so the check ARMS ITSELF the day one lands (exactly how
        # `test_reference_units` waited for its own freezer) and is meanwhile
        # named in `eval_summary.json`'s `deselected` list rather than being
        # silently absent from the module. Recorded as a spec/catalog gap so the catalog
        # is corrected rather than quietly diverged from.
        "has_fp32_decode": (golden_dir / "fp32_decode_step.npz").exists(),
        # NO L6 FACT, DELIBERATELY. `test_held_out_logprobs` is always-on: its
        # golden (`held_out.npz`) is REQUIRED by `golden_manifest`, so absence is
        # a hard load failure in `fixtures.load_golden` rather than a fact that
        # quietly removes the check. The pair of facts that used to live here --
        # `has_validation` (a `held_out_nll.json` nothing ever wrote) and
        # `has_validation_result` (a `$T2J_INFERENCE_RESULT` nothing ever set) --
        # gated the predecessor check dark on every instance in both task
        # classes, which is the failure mode this whole table has to be read
        # against: a gate that can never open reads as coverage.
    }


# name -> predicate(facts) that must be True for the check to be COLLECTED for
# this instance. A name absent here always runs.
_APPLICABLE = {
    # Mode-gated. `core` runs one sequence at a time, so nothing that needs a
    # second concurrent sequence can apply to it.
    "test_ragged_forward_logits": lambda f: f["has_ragged"],
    "test_batching_invariance": lambda f: f["has_ragged"],
    "test_no_cross_sequence_leakage": lambda f: f["has_ragged"],
    "test_batch_composition_battery": lambda f: f["has_ragged"] and f["has_battery_batch_composition"],
    "test_chunked_prefill_logits": lambda f: f["has_chunked"],
    "test_chunking_invariance": lambda f: f["has_chunked"],
    "test_prefix_cache_invariance": lambda f: f["has_prefix_cache"],
    "test_chunk_boundary_battery": lambda f: f["has_chunked"] and f["has_battery_chunk_boundary"],
    # Geometry-gated. The only SPEC-tier entry in this table, and it gates ONLY the
    # assertion that genuinely needs a second prompt length -- the model-body
    # recompilation bound. Counter liveness and the threshold-free same-shape check
    # live in `test_compilation_is_cached`, which is absent from this table and
    # therefore always runs: the same-shape leg is the only detector of a port that
    # rebuilds its jit wrapper per call, and an instance with a short prompt must not
    # lose that while reading as fully covered.
    "test_static_shape_recompilation": lambda f: f["has_second_prompt_length"],
    # FAMILY-gated, and this is the first pair keyed on a MODEL property rather
    # than on a mode or an artifact. Six of the seven instances are text-only:
    # they have no image, their port ships no `image_processing` module, and both
    # probes answer the unsupported marker there -- so these two DESELECT rather
    # than fail, and appear in `eval_summary.json`'s `deselected` list.
    #
    # NOT PRESENCE-GATED on `golden/image_preprocess.npz`, deliberately. A missing artifact
    # on an instance that DECLARES `multimodal: true` is a broken freeze, and
    # gating on the file would turn it into a silent deselection -- the failure
    # shape this whole table is written against. Gated on the family fact, a
    # multimodal instance with no recording fails `test_image_preprocess` by name
    # and says which file to re-freeze.
    #
    # THE MODE LADDER IS UNAFFECTED: `multimodal` is False on every `core`,
    # `ragged` and `chunked` instance that is not this family, so
    # `test_the_ragged_tier_nests_between_core_and_chunked`'s nesting sweep sees
    # a predicate that is False at every rung, which nests trivially.
    "test_image_preprocess": lambda f: f["multimodal"],
    "test_smart_resize_targets": lambda f: f["multimodal"],
    # ...AND THE FOUR THAT GRADE THE MODEL RATHER THAN THE INPUT PATH. Same fact,
    # same reasoning, one axis further in: the pair above stops at the processor,
    # these reach the tower, the M-RoPE position builder, the embedding scatter
    # and the image-conditioned prefill. Gated on the FAMILY fact for the reason
    # spelled out above -- an instance that declares `multimodal: true` and
    # cannot answer one of them must FAIL by name, not deselect.
    #
    # NOT PRESENCE-GATED ON `vision_tower_units.{json,npz}` EITHER, which is the
    # one place the argument could plausibly have gone the other way, since that
    # artifact is OPTIONAL in `_t2j.golden_manifest` exactly as
    # `reference_units.*` is and `test_reference_units` IS presence-gated. The
    # difference is what absence MEANS. `reference_units.*` is an optional
    # localiser for a tier that is graded anyway (`test_prefill_logits` sees the
    # same defect, one indirection later), so an instance without it has less
    # diagnosis and no less coverage. The tower has no such second gate: its
    # VALUES are graded here and in `test_param_norms` and nowhere else, and the
    # window permutation is graded here and NOWHERE else at all. A missing
    # recording on a multimodal instance is therefore a coverage hole, and
    # presence-gating it would report that hole as inapplicability.
    "test_vision_tower_units": lambda f: f["multimodal"],
    "test_mrope_positions": lambda f: f["multimodal"],
    "test_image_embedding_merge": lambda f: f["multimodal"],
    "test_multimodal_prefill_logits": lambda f: f["multimodal"],
    # Presence-gated optional features.
    "test_hidden_states": lambda f: f["has_hidden"],
    # Gated on an artifact the golden catalog does not yet name -- see
    # `has_fp32_decode` above. Deselected on EVERY instance today, and visible as
    # such in the summary.
    "test_fp32_decode_step_logits": lambda f: f["has_fp32_decode"],
    "test_prefill_logits_battery": lambda f: f["has_battery_prompt_length"],
    "test_paging_battery": lambda f: f["has_battery_paging"],
    "test_reference_units": lambda f: f["refunits"],
    # Everything else -- paged attention, cache write, prefill, decode, sampler,
    # generation, serve.py end to end, the held-out logprob band, and the
    # golden-free invariants -- applies to every mode and is absent from this
    # table, so it always runs.
}


#: WHY each gated check can be deselected, as a machine-readable kind. Reported in
#: ``eval_summary.json`` beside the flat ``deselected`` list.
#:
#: The flat list alone cannot answer the question a reader actually has. An
#: instance deselects somewhere between seven and eleven checks depending on its
#: mode (11 on the two `core` instances, 7 on
#: `ragged_llama3`), and they are not the same kind of thing. A reader scanning
#: the names cannot tell which will run on some OTHER instance from which cannot
#: run anywhere. Those need different responses: the first is coverage the class
#: already has and this instance does not exercise, the second is coverage the
#: class does not have at all.
#:
#: NO COUNT IS HARD-CODED HERE, and one used to be ("this instance deselects
#: twelve checks"). This file is a SHARED TEMPLATE rendered byte-identically into
#: every instance, so any per-instance number in it is false for all but one of
#: them -- and that one stopped being true as soon as a second instance shipped.
#:
#: FIVE KINDS, and the split between the two "no golden" ones is the point:
#:
#: * ``mode``             -- inapplicable to this instance's mode; runs on `ragged`
#:                           or `chunked`. Not a gap, and never shrinks.
#: * ``geometry`` /
#:   ``optional_feature`` -- inapplicable to this instance's frozen scenario.
#:                           Likewise not a gap.
#: * ``no_golden_here``   -- a PRODUCER EXISTS and this instance's golden simply
#:                           does not carry the artifact. One re-freeze away.
#: * ``no_producer``      -- NO PRODUCER EXISTS ANYWHERE IN THE FREEZER TOOLING, so the check
#:                           is dark on EVERY instance in the class and always has
#:                           been. This is the only kind that is missing coverage
#:                           rather than inapplicability, and the only one whose
#:                           count should ever fall.
#:
#: Combined with ``+`` where either condition alone would deselect.
#:
#: Keys are kept in lockstep with :data:`_APPLICABLE` by a clause in
#: ``_harness_checks.check_applicability_table_names_real_checks``.
_DESELECTION_KIND: dict[str, str] = {
    "test_ragged_forward_logits": "mode",
    "test_batching_invariance": "mode",
    "test_no_cross_sequence_leakage": "mode",
    "test_chunked_prefill_logits": "mode",
    "test_chunking_invariance": "mode",
    "test_prefix_cache_invariance": "mode",
    # THE BATTERY PRODUCER NOW EXISTS, and these four rows are no longer the same
    # answer. The battery freezer implements exactly ONE
    # of the four axes -- `batch_composition` -- so that
    # axis is mode-gated only, and it RUNS on `ragged_llama3`, whose golden tree
    # carries `battery/batch_composition/{d1p2,d2p1}`. The other three axes still
    # have no producer anywhere in the freezer tooling.
    #
    # This mattered beyond the comment: `no_producer` is defined below as "no
    # producer exists ANYWHERE in the freezer tooling, so the check is dark on EVERY instance",
    # and it is "the only one whose count should ever fall". Leaving
    # `batch_composition` marked `no_producer` published a false claim of missing
    # class-wide coverage into every `eval_summary.json` -- including the one
    # instance where the check actually runs.
    # `mode+no_golden_here`, not `mode`, and the `+` is the taxonomy's own rule
    # three paragraphs up: "Combined with `+` where EITHER condition alone would
    # deselect." The predicate is `has_ragged AND has_battery_batch_composition`,
    # and on the two `core` instances BOTH conjuncts are false -- they
    # are not ragged, and their golden trees carry no `battery/` at all. The
    # second half is `no_golden_here` ("a PRODUCER EXISTS and this instance's
    # golden simply does not carry the artifact"), which is exactly true now that
    # the producer is written. It was `mode+no_producer` when no producer
    # existed; the correction was to the second term, not to its presence.
    "test_batch_composition_battery": "mode+no_golden_here",
    # THE SAME CORRECTION: the battery freezer now
    # implements `chunk_boundary` too (`build_chunk_boundary`), so this is no
    # longer dark on every instance -- it is mode-gated (`chunked` only) plus
    # `no_golden_here` on every instance whose golden tree lacks the axis.
    # `mode+no_producer` would now be the false claim this table's own history
    # (the paragraph above) warns about.
    #
    # AND IT IS NO LONGER DESELECTED EVERYWHERE. This row used to add "which
    # today is every shipped instance (none is `chunked`)"; `chunked_llama3`
    # ships, its golden tree carries `battery/chunk_boundary/{aligned,interior,
    # far_interior,tail}`, and the check RUNS there. The five other instances
    # deselect it on BOTH conjuncts, which is what keeps the `+` correct.
    "test_chunk_boundary_battery": "mode+no_golden_here",
    # THE LAST TWO AXES NOW HAVE A PRODUCER TOO, and unlike the pair above these
    # carry NO `mode+` term. The battery freezer implements
    # `prompt_length` (`build_prompt_length`) and `paging` (`build_paging`), the
    # two SINGLE-SEQUENCE axes -- one prefill whose LENGTH moves, one whose PAGE
    # LAYOUT moves. Neither needs a ragged or a chunked scenario to exist, so
    # `_APPLICABLE` gates them on `has_battery_{prompt_length,paging}` ALONE, with
    # no mode conjunct: they are the discriminating checks that apply to every
    # instance, `core` included, where the other two axes never run. The kind is
    # therefore `no_golden_here` on its own -- a producer exists, and an instance
    # whose golden tree does not yet carry `battery/{prompt_length,paging}/` is one
    # re-freeze away, not dark class-wide. `no_producer` would publish the same
    # false class-wide coverage gap the two rows above were corrected for; a
    # `mode+` prefix would be a mode gate these predicates do not have.
    "test_prefill_logits_battery": "no_golden_here",
    "test_paging_battery": "no_golden_here",
    # DARK CLASS-WIDE, and the last of what used to be a PAIR here: the golden catalog
    # still freezes no fp32 decode artifact and nothing in the freezer tooling writes one. Its
    # former partner, `test_held_out_nll`, was retired -- the held-out freezer
    # now writes `held_out.npz` and `test_held_out_logprobs` grades it with no gate
    # at all, so it is absent from both tables rather than listed here as dark.
    "test_fp32_decode_step_logits": "no_producer",
    # A PRODUCER EXISTS (the units freezer) and has run on this
    # instance, so this check RUNS here. It would be deselected only on an instance
    # whose golden lacks the artifact -- a re-freeze away, not a gap in the class.
    # `optional_feature`, and NOT `no_producer`. Both halves of that matter.
    # The image-preprocess freezer EXISTS and has run, so claiming
    # `no_producer` would publish the false class-wide coverage gap the two
    # battery rows above were corrected for. And it is not `no_golden_here`
    # either: the six text-only instances are not MISSING a recording they ought
    # to have -- there is no image in their scenario to record. That is
    # inapplicability, which is what `optional_feature` means, one axis over from
    # `test_hidden_states` below (a scenario property there, a MODEL property
    # here; the taxonomy has no separate kind for that and does not need one).
    "test_image_preprocess": "optional_feature",
    "test_smart_resize_targets": "optional_feature",
    # The same kind, and for the same two reasons the pair above gives:
    # the image-preprocess freezer and
    # the vision-tower freezer both EXIST and have run, so
    # `no_producer` would publish a false class-wide coverage gap; and the six
    # text-only instances are not MISSING a tower recording they ought to have --
    # they have no tower. That is inapplicability, which is what
    # `optional_feature` means.
    "test_vision_tower_units": "optional_feature",
    "test_mrope_positions": "optional_feature",
    "test_image_embedding_merge": "optional_feature",
    "test_multimodal_prefill_logits": "optional_feature",
    "test_reference_units": "no_golden_here",
    "test_hidden_states": "optional_feature",
    "test_static_shape_recompilation": "geometry",
}


def _collect(config, items, golden_dir):
    """Deselect the inapplicable items in place. Extracted so it is testable.

    DESELECT, never skip. ``reward = passed / (passed + failed + error)``, so a
    skipped check leaves the numerator AND the denominator and an instance that
    skips half its suite scores exactly like one that passes all of it.
    """
    facts = _golden_facts(golden_dir)
    kept, deselected = [], []
    for item in items:
        pred = _APPLICABLE.get(item.name.split("[", 1)[0])
        (kept if (pred is None or pred(facts)) else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
    # WHAT WAS DESELECTED, recorded for the summary. Deselection keeps a check out of
    # both numerator and denominator, so an instance that deselects half its suite
    # scores like one that passes all of it -- correct for the reward, invisible to
    # anyone reading the result. `eval_summary.json` names them (see
    # `pytest_sessionfinish`), which is the difference between "this instance has no
    # chunked path" and "this check quietly stopped running".
    _DESELECTED[:] = sorted(item.name.split("[", 1)[0] for item in deselected)
    return facts


#: Check names deselected this session, in name order. Written by :func:`_collect`,
#: read by :func:`pytest_sessionfinish`.
_DESELECTED: list[str] = []


#: What survived collection. ``kept`` counts every item, ``graded`` only those
#: whose module stem is in :data:`_STAGE_OF`. Both are read POST-deselection --
#: that is the whole content of "whole-session emptiness, not per-stage": a
#: pre-deselection count would call a session non-empty on the strength of checks
#: the mode gate had just removed. ``None`` means the collection hook never ran
#: at all, which is also not a state in which anything was graded.
_COLLECTED: dict[str, int | None] = {"kept": None, "graded": None}

#: The graded check NAMES that survived collection (parametrisation stripped, like
#: :data:`_DESELECTED`). Read by :func:`_require_complete_collection`, which needs the
#: names rather than the count: ``collected + deselected`` must reconstruct the whole
#: declared suite, and a count alone cannot say WHICH check went missing.
_COLLECTED_NAMES: list[str] = []


#: FLOOR on how many graded checks this class's four modules declare, collected and
#: deselected together. A CLASS constant, not a per-instance one, and that is what makes
#: it checkable: deselection moves a check between the two buckets and never out of the
#: sum, so the total is the same on every mode. All four graded modules are byte copies
#: of a shared template, so every instance ships the same 45.
#:
#: WHY A CONSTANT AS WELL AS THE DERIVED SET BELOW. The derived set is read out of the
#: modules themselves, so it moves with them: gut a module and both sides shrink
#: together. The constant is the one statement that does not, and it is deliberately a
#: SECOND edit -- a narrowing upstream now has to come here and lower a number a reviewer
#: reads, rather than regenerating cleanly. It does not make a narrowing impossible; it
#: makes it visible.
#:
#: IT IS A FLOOR, SO IT LAGS SILENTLY UPWARD AND FAILS LOUDLY DOWNWARD. Adding a 45th
#: check does NOT fire this gate: `declared` becomes 45, `accounted` becomes 45, the sets
#: still agree and 45 >= 44. Nothing will remind you to raise the number, and until you
#: do, the floor protects 44 of the 45 -- a real but smaller guarantee. REMOVING one
#: fires immediately (43 < 44), which is the direction that matters, because that is the
#: direction the measured defect went. So: raise it in the same commit that adds a check;
#: lowering it is a decision someone should have to argue for.
#:
#: LOWERED 43 -> 42, and here is the argument, because the paragraph above
#: says one is owed. ``test_held_out_nll`` was RETIRED, not narrowed: it was gated on a
#: ``held_out_nll.json`` no producer ever wrote AND a ``$T2J_INFERENCE_RESULT`` nothing
#: ever set, so it executed on zero instances in both task classes -- it was a declared
#: check that graded nothing, which is precisely the coverage this floor cannot see (the
#: floor counts DECLARATIONS). Removing it costs no measurement. Its replacements,
#: ``test_serve_end_to_end`` and ``test_held_out_logprobs``, are ungated and land in the
#: two tasks after this one, which take this number back to 44 -- and unlike their
#: predecessor they will actually run.
#:
#: RAISED 42 -> 43, in the SAME commit that adds ``test_serve_end_to_end``
#: (L5 -- the entry point) -- the paragraph above's own rule ("raise it in
#: the same commit that adds a check") applied to itself rather than left for a later
#: reviewer to notice. The check is ungated (absent from ``_APPLICABLE``, so it always
#: runs), which is exactly the shape this floor cannot see fail to run. One more raise,
#: to 44, is still owed when ``test_held_out_logprobs`` (L6) lands.
#:
#: RAISED 43 -> 44, the raise the paragraph above said was owed, in the
#: SAME commit that adds ``test_held_out_logprobs`` (spec's L6 -- held-out logprob
#: parity, replacing the never-executed ``test_held_out_nll``). Also ungated (absent
#: from ``_APPLICABLE``), so it too always runs and this floor would otherwise be
#: blind to it disappearing.
#:
#: RAISED 44 -> 45, in the SAME commit that adds ``test_decode_speed``
#: (the perf stage). GATED (present in ``_APPLICABLE``, keyed on
#: ``perf_graded``), which is exactly the shape this floor was raised for in the L5/L6
#: rows above: a gated check still counts toward ``declared`` and this constant must
#: track it, or a check that starts deselecting on every instance -- which is
#: `test_decode_speed`'s own state through all of phase 1 -- could also silently stop
#: being DECLARED and this floor would not notice either.
#:
#: RAISED 45 -> 47, in the SAME commit that adds
#: ``test_image_preprocess`` and ``test_smart_resize_targets`` (the multimodal
#: input path). GATED, on the ``multimodal`` family fact, so on the six text-only
#: instances they are deselected -- which is exactly the shape the two rows above
#: raise this for: a gated check still counts toward ``declared``, and a pair that
#: is deselected everywhere but one instance is the easiest kind to lose without
#: anything noticing.
#:
#: RAISED 47 -> 51, in the SAME commit that adds the four remaining
#: multimodal checks -- ``test_vision_tower_units``, ``test_mrope_positions``,
#: ``test_image_embedding_merge`` and ``test_multimodal_prefill_logits``. All four
#: are GATED on the same ``multimodal`` family fact as the pair above, so all four
#: are deselected on the six text-only instances and this floor is the only thing
#: that would notice if they stopped being DECLARED.
#:
#: LOWERED 51 -> 50, and here is the argument the paragraph at the top requires.
#: ``test_decode_speed`` was RETIRED with ``test_perf.py`` -- the perf stage was
#: removed from this class (see :data:`_STAGE_OF`). It was GATED on ``perf_graded``,
#: which is ``False`` on every instance, so like ``test_held_out_nll`` before it it
#: was a declared check that graded nothing on any instance, and removing it costs no
#: measurement. Its one non-timing assertion -- that a timed run reproduces the
#: golden decode tokens -- is already graded, ungated, by ``test_greedy_sequence`` in
#: ``test_behavioral.py``, so no coverage leaves with it; only the never-calibrated
#: timing gate does. The perf MEASUREMENT tooling under ``_t2j/perf`` stays on disk
#: for a future calibration but no longer declares a graded check here.
#:
#: IT IS RAISED IN THE TEMPLATE, WHICH IS RENDERED WITH THE CHECKS, and that
#: ordering is what makes it safe: the floor is compared against the modules in
#: the SAME ``tests/`` directory this file was rendered into, so an instance
#: carrying the old conftest and the old ``test_numeric.py`` is self-consistent at
#: 47 and one carrying both new copies is self-consistent at 51. The failure mode
#: to avoid is the reverse -- raising the number here in one commit and adding the
#: checks in another, which makes every instance red in between.
_MIN_DECLARED_GRADED_CHECKS = 50


def _declared_graded_checks() -> dict[str, set[str]]:
    """``{module stem: {test function names it defines}}`` for the graded stems in
    :data:`_STAGE_OF` (three today: spec, numeric, behavioral).

    AST, never imported -- same rule as ``_harness_checks._module_functions``, whose
    parser this reuses rather than keeping a second copy. A stem whose file is absent
    maps to the empty set, which is the condition :func:`_require_complete_collection`
    reports first.
    """
    import _harness_checks

    out: dict[str, set[str]] = {}
    for stem in sorted(_STAGE_OF):
        module = _TEST_DIR / f"{stem}.py"
        out[stem] = ({n for n in _harness_checks._module_functions(module)
                      if n.startswith("test_")} if module.exists() else set())
    return out


#: ``tests/instance.yaml``'s per-instance graded denominator. See
#: :func:`_instance_graded_check_count`.
_GRADED_CHECKS_KEY = "graded_checks"


def _instance_graded_check_count() -> tuple[int | None, str | None]:
    """How many checks THIS instance must GRADE -- ``(count, None)`` or ``(None, why)``.

    THE ONE NUMBER THE OTHER TWO CLAUSES CANNOT SEE. Both of them are stated over
    ``collected | deselected``, which is invariant under deselection by
    construction: moving a name from one bucket to the other changes neither the
    union nor the module's declaration count. So a gate that widened until it
    deselected 42 of the 44 declared checks passed both: keeping 2 of 44 yields
    ``reward 0.5, n_skipped 0``, and had both passed,
    ``reward 1.0, overall PASS`` over 2 of 44, traceable only through a
    ``deselected`` list no consumer reads.

    ``tests/instance.yaml``, AND THE SOURCE IS THE CHOICE HERE, so the reasons are
    written down rather than left implicit:

    * IT IS IN THE IMAGE. ``tests/Dockerfile`` is ``COPY tasks/{task}/tests
      /tests`` -- the whole directory, ``instance.yaml`` included -- and
      ``.dockerignore`` excludes only ``tasks/*/solution``. Not assumed:
      ``_harness_checks.check_meta_mirrors_instance_yaml`` already opens this exact
      path inside the verifier and treats its absence as a failure, and it runs
      BEFORE this gate (``_require_healthy_harness`` at the call site), so by the
      time this reads the file the image has already been proven to carry it. The
      ``yaml`` import is that check's too, pinned via flax.
    * IT IS AN EXACT GATE, NOT A FLOOR, which a per-mode floor derived from the
      ladder could not be. A floor catches over-deselection only; equality also
      catches UNDER-deselection -- a mode-gated check that starts running on an
      instance whose engine has no such path, which fails four checks with no
      obvious cause and is the other half of the same accounting.
    * IT IS ALREADY THIS CLASS'S PER-INSTANCE SOURCE OF TRUTH for exactly this
      number: the host grading path asserts ``n_passed == graded_checks`` against
      the same key. That enforcement needs docker and an H100, i.e. it does NOT
      run inside the container that produces the grade -- which is the gap this
      closes, with the same number so the two can never disagree.
    * ``meta.json`` WOULD BE THE WRONG PLACE. ``graded_checks`` is deliberately not
      mirrored there (``_harness_checks._INSTANCE_KEYS_NOT_MIRRORED``): mirroring
      it would put it inside the stale-golden comparison and make ADDING a graded
      check demand a re-freeze of every golden.

    AN UNUSABLE VALUE IS A FINDING, NOT A PASS. Absent, non-int, or non-positive
    all return a reason and the caller raises: ``check_meta_mirrors_instance_yaml``
    exempts this key from its comparison, so nothing else in the tree would notice
    a yaml that stopped declaring it, and "the number I gate against is missing"
    must not read as "the collection is fine". Same rule ``_golden_facts`` applies
    to a geometry key that is not a positive int.
    """
    import yaml                                  # pinned via flax

    path = _TEST_DIR / "instance.yaml"
    if not path.is_file():
        return None, f"there is no {path} to read the denominator from"
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, f"{path} could not be parsed: {exc}"
    value = data.get(_GRADED_CHECKS_KEY)
    if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
        return None, (f"{path} declares no usable {_GRADED_CHECKS_KEY!r} "
                      f"(got {value!r}); nothing else in the tree reads that key "
                      f"at grade time, so its absence is silent everywhere else")
    return int(value), None


#: Set when the invocation asked for LESS than the whole tests directory, so
#: :func:`pytest_sessionfinish` can say so in the artifact. Read-only elsewhere.
_PARTIAL_SELECTION: list[str] = []


def _selection_narrowed(config) -> str | None:
    """Why this invocation is a SUBSET of the suite, or ``None`` if it is the suite.

    Every graded entry point names the tests DIRECTORY -- ``tests/test.sh`` runs
    ``pytest /tests`` and the reference-mutation controls name ``tests``. An
    author debugging one check does not, and
    :func:`_require_complete_collection` must not tell them their tree is broken when
    what is narrow is their command line. This is the one predicate that separates the
    two, and it is deliberately conservative: anything that could be a subset counts as
    one, because the cost of being wrong in that direction is a gate that does not fire
    on a developer box, while the other direction is a false refusal of a real grade.

    ALSO covers ``--deselect``/``-x``/``--maxfail``, added for the same reason as
    ``-k``/``-m``: they narrow the SESSION, not the declared tree, so
    ``_require_complete_collection`` (which reads the tree) cannot see them and
    pytest applies ``--deselect`` in its own ``modifyitems`` hook, which runs
    AFTER this file's -- so by the time that gate runs the declared set has
    already reconstructed. This function runs at COLLECTION time though, where
    ``config.option`` is fully parsed, so it sees the flag before a single item
    executes.

    ONE CHECK COVERS BOTH ``-x`` AND ``--maxfail``, not two: verified against the
    installed pytest (9.1.0), ``-x``/``--exitfirst`` is ``action="store_const",
    dest="maxfail", const=1`` -- there is no ``config.option.exitfirst``
    attribute at all, so a check for one would be dead code that never fires.
    Both flags are indistinguishable once parsed; ``config.option.maxfail``
    alone reports the effective state of either.
    """
    if getattr(config.option, "keyword", ""):
        return f"-k {config.option.keyword!r}"
    if getattr(config.option, "markexpr", ""):
        return f"-m {config.option.markexpr!r}"
    if getattr(config.option, "deselect", None):
        return f"--deselect {list(config.option.deselect)}"
    if getattr(config.option, "maxfail", None):
        return f"--maxfail={config.option.maxfail}"
    for arg in (config.args or []):
        if "::" in str(arg):
            return f"a node id was named on the command line ({arg})"
        if Path(str(arg)).is_file():
            return f"a single file was named on the command line ({arg})"
    return None


def _require_complete_collection() -> None:
    """THE WHOLE SUITE RAN, OR NOTHING DID -- a pytest usage error, like the gates above.

    ``reward = passed / (passed + failed + error)`` is a pass rate over what was
    COLLECTED, so a suite that lost checks scores over the survivors and reads as a
    plausible 1.0. Without this gate, deleting ``test_behavioral.py``
    from the tests tree and grading the reference gives ``23 passed, 7 deselected``, reward
    **1.0**, ``overall PASS``, ``n_failed 0``, and every harness self-check
    green.
    Seventeen of thirty-one checks -- the entire Behavioral tier -- were gone, and the
    only trace was ``deselected`` dropping 12 -> 7, which nothing compared against
    anything.

    THREE CLAUSES, because the three faults have different repairs.

    * THE DECLARED SET IS ACCOUNTED FOR. Every ``test_*`` the four graded modules
      define must be either collected or deselected, and nothing may be collected that
      no module declares. This is what sees a module present-but-not-collected -- a
      renamed check, a stale ``python_files``, a ``collect_ignore`` that grew a glob --
      none of which ``_harness_checks`` can see, because its subject is the tables and
      this one's subject is the collection.
    * THE DECLARED SET IS BIG ENOUGH. See :data:`_MIN_DECLARED_GRADED_CHECKS` for why a
      constant is here beside a derived set, and for what it does and does not buy.
    * THE COLLECTED SET IS THE ONE THIS INSTANCE OWES. Neither clause above floors
      what actually RAN: both are stated over ``collected | deselected``, which
      deselection cannot move. So over-deselection was unguarded at grade time --
      keeping 2 of 44 scored ``reward 0.5`` with ``n_skipped 0``, and
      would have scored ``1.0, overall PASS`` over two checks had both passed. The
      count comes from ``tests/instance.yaml``; see
      :func:`_instance_graded_check_count` for why that source and why EQUALITY
      rather than a floor.

    An ABSENT graded module fails ``_harness_checks
    .check_applicability_table_names_real_checks`` first, which runs before this and
    names the repair precisely; both clauses below still name it, so this gate is not
    silent if that one is ever narrowed.

    ``UsageError`` (exit 4), not a failure: a tests tree that is missing checks is a
    statement about the IMAGE, not about the port, and ``test.sh`` turns exit 4 into
    "no grade happened, no reward.json". Writing 23/23 would say the port scored
    perfectly, which is exactly the sentence this exists to prevent.

    IT DOES NOT FIRE ON A NARROWED INVOCATION -- ``pytest tests/test_spec.py``, a node
    id, ``-k``, ``-m``. There the tree is whole and the COMMAND LINE is narrow, which
    is a different fact and a legitimate one; refusing it would tell an author
    debugging one check that their checkout is broken. Every graded entry point names
    the directory (see :func:`_selection_narrowed`), so nothing that produces a real
    grade takes that branch, and ``eval_summary.json`` carries ``partial_selection``
    when it does -- so a subset score can never be read as a grade.
    """
    declared = _declared_graded_checks()
    absent_modules = sorted(stem for stem, names in declared.items() if not names)
    expected: set[str] = set().union(*declared.values()) if declared else set()
    accounted = set(_COLLECTED_NAMES) | set(_DESELECTED)
    owed, why_not = _instance_graded_check_count()
    collected = len(_COLLECTED_NAMES)
    if (absent_modules or expected != accounted
            or len(expected) < _MIN_DECLARED_GRADED_CHECKS
            or owed is None or collected != owed):
        raise pytest.UsageError(
            "this tests tree does not hold the whole graded suite, so a pass rate over "
            "it would be a pass rate over whatever survived. Nothing was graded and no "
            "score is implied.\n"
            f"  graded modules with no test functions: {absent_modules or 'none'}\n"
            f"  declared by the modules:               {len(expected)} "
            f"(floor {_MIN_DECLARED_GRADED_CHECKS})\n"
            f"  collected + deselected:                {len(accounted)}\n"
            f"  declared but never collected:          {sorted(expected - accounted)}\n"
            f"  collected but declared by no module:   {sorted(accounted - expected)}\n"
            f"  this instance owes (instance.yaml):    "
            f"{owed if owed is not None else why_not}\n"
            f"  collected after deselection:           {collected}\n"
            f"  deselected:                            {len(_DESELECTED)} "
            f"{list(_DESELECTED)}\n"
            f"  tests dir: {_TEST_DIR}"
        )


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    facts = _collect(config, items, _TEST_DIR / "golden")
    _COLLECTED["kept"] = len(items)
    graded = [item for item in items if _stage_of(item.nodeid) is not None]
    _COLLECTED["graded"] = len(graded)
    _COLLECTED_NAMES[:] = sorted({item.name.split("[", 1)[0] for item in graded})

    # A golden too partial to gate against, before the device precondition -- see
    # _require_complete_geometry. Gated on something graded surviving, exactly like
    # the device gate below and for the same reason.
    if _COLLECTED["graded"]:
        _require_complete_geometry(facts, _TEST_DIR / "golden")
        # ...and the benchmark's own health, the developer-path half of an earlier
        # fix. Same gating, same UsageError shape; see _require_healthy_harness.
        loaded = _require_healthy_harness(_TEST_DIR / "golden")
        # ...and that the suite this session is about to grade is the WHOLE suite.
        #
        # AFTER the harness checks and gated on the SAME condition they are -- the
        # golden actually loaded. Two reasons, and both are about not firing on
        # something that is not a grade. An ABSENT graded module is the harness
        # check's finding and its message names the repair, so it should speak
        # first; and a tree whose golden will not load is not grading anything, so
        # demanding the whole suite of it would turn every synthetic and
        # partially-provisioned tree into a usage error while the `golden` fixture
        # is already raising a better message in each check. What is left for this
        # gate, uniquely, is a real instance whose module is present and has stopped
        # being collected.
        narrowed = _selection_narrowed(config)
        _PARTIAL_SELECTION[:] = [narrowed] if narrowed else []
        if loaded and not narrowed:
            _require_complete_collection()

    # THE DEVICE PRECONDITION, here rather than in
    # pytest_configure. Two gates, and both are needed:
    #
    #   * at least one GRADED item survived. A session that grades nothing cannot
    #     produce a number, so the SKU it would have produced it on is not a
    #     precondition of anything -- and the driver's own unit tests come in
    #     under this conftest (see pytest_sessionfinish's case 2). Read POST
    #     deselection, so mode gating that removes every graded check also
    #     removes the requirement.
    #   * the golden EXISTS. With no golden the `golden` fixture already raises
    #     FileNotFoundError naming the freezer in every check (see
    #     fixtures._MISSING_GOLDEN_HINT), which is the louder and more accurate
    #     answer; pre-empting it with a device error would report the wrong cause.
    #
    # It still runs before a single probe: collection finishes before setup.
    if _COLLECTED["graded"] and (_TEST_DIR / "golden" / "meta.json").exists():
        _require_pinned_device(_TEST_DIR / "golden")


# --------------------------------------------------------------------------- #
# Per-instance JSON summary: map test outcomes to the three stages +
# overall, so the attempt harness can aggregate standalone grade runs.
#
# The REWARD SEMANTICS are the training class's, unchanged: same
# `reward = passed / (passed + failed + error)` excluding skips and xfails. The
# STAGE MAPPING is the training class's three stems again
# (spec/numeric/behavioral). It briefly gained a fourth, `perf`, which
# the training class has no equivalent of; that stage was RETIRED when
# `test_perf.py` was removed -- the measurement tooling under `_t2j/perf` stays on
# disk, dormant, for a future calibration. Two further things here are not the
# training class's either, both of them guards against a session that graded
# nothing reporting that it did -- see `pytest_sessionfinish` for the four-case
# write rule and for why an empty tally is not a passing stage. The training class
# needs neither: `tasks/*/tests` holds its three graded modules and nothing else,
# so it has no way to be asked for a grade it did not perform.
# --------------------------------------------------------------------------- #
_STAGE_OF = {"test_spec": "spec", "test_numeric": "numeric", "test_behavioral": "behavioral"}
_TALLY: dict[str, dict[str, int]] = {}

# CONTINUOUS SCORING, ADDITIVE (deep fix). The binary `_TALLY` above stays
# byte-identical -- `graded_outcomes`, the six-case write rule and the completeness
# gates all read it -- and this is a PARALLEL accumulator that the reward's SCORED
# dimensions consume instead of their pass rates. Each scored check funnels a [0,1]
# fidelity through `record_property('score', s)` (see `test_numeric._score` and
# `test_behavioral._score`); `pytest_sessionfinish` aggregates it with
# `similarity.power_mean` (p<1). BOTH numeric and behavioral are scored this way now,
# mirroring the training grader; spec stays binary-only.
_SCORES: dict[str, dict[str, float]] = {}

# The dimensions graded on a continuous power mean (spec is binary-only).
_SCORED_STAGES = ("numeric", "behavioral")

# Pure-GATE tests on a scored stage: exact grids / positions / placement / bit-exact
# self-consistency / wiring, no continuous fidelity signal. A gate PASS records no
# score and is kept OUT of the power mean (it banks no unearned credit); a
# gate FAIL files 0.0. Everything else on a scored stage reaches a scoring path
# (`test_numeric`'s two logit/array gates; `test_behavioral._score`) on its pass path,
# so a pass with no recorded score would be a harness regression and scores 0.0.
_NUMERIC_GATE_TESTS = frozenset({
    "test_smart_resize_targets", "test_mrope_positions", "test_image_embedding_merge",
})
# Behavioral hard gates: bit-exact / structural / wiring invariants that
# carry no continuous signal -- they PASS at a constant on any correct port and can
# only FAIL. The eight scored behavioral checks are everything else on the stage.
_BEHAVIORAL_GATE_TESTS = frozenset({
    "test_paging_invariance", "test_cache_reuse_safety", "test_determinism",
    "test_no_cross_sequence_leakage", "test_generate_runs", "test_serve_end_to_end",
})
_STAGE_GATE_TESTS = {"numeric": _NUMERIC_GATE_TESTS, "behavioral": _BEHAVIORAL_GATE_TESTS}

# Lever B -- the DISCRIMINATORS of a scored dimension: checks a port that reproduces
# the single frozen scenario but does NOT generalise still fails. For inference these are
# the L3 generalisation batteries -- each varies ONE axis (prompt length, page layout,
# prefill/decode composition, chunk split point) off the canonical scenario and compares
# to the REFERENCE, so a port that memorised the one frozen shot cannot pass a broken axis
# by looking similar to itself. In :func:`_dimension_score` each enters as its own hard MIN
# vote while every OTHER scored check collapses into ONE soft power-mean vote, so a correct
# single-shot forward can no longer average a failing off-scenario axis upward behind many
# correlated parity checks (count-dominance, the numeric inflation vector).
#
# EVERY BATTERY GRADES LOGIT PARITY (``score_logits``), so a FAITHFUL port sits at the
# ~0.999 ceiling on it REGARDLESS OF MODALITY -- unlike the training class's ``fp32_loss``
# veto (OOB through a multimodal vision tower), a battery is never legitimately OOB here,
# because it compares final logits head-on to the oracle golden. So the set needs NO
# modality/mode gate: a battery a mode or instance does not run simply never lands in
# ``_SCORES`` and never votes, and an instance whose batteries are not yet frozen reduces
# EXACTLY to the flat power mean. Membership is by exact test-function base name.
#
# NUMERIC ONLY. Behavioral stays a flat power mean (empty veto set): its bit-exact/wiring
# invariants are already hard gates routed out of the mean (see ``_BEHAVIORAL_GATE_TESTS``),
# and its remaining scored checks are cross-run fidelities with no dodge-proof MIN veto --
# mirrors the training class's empty ``_BEHAVIORAL_DISCRIMINATORS``.
_NUMERIC_DISCRIMINATORS = frozenset({
    "test_prefill_logits_battery",      # prompt-length generalisation axis
    "test_paging_battery",              # page-layout generalisation axis
    "test_batch_composition_battery",   # prefill/decode-mix axis (ragged/chunked)
    "test_chunk_boundary_battery",      # chunk split-point axis (chunked)
})
_BEHAVIORAL_DISCRIMINATORS: frozenset[str] = frozenset()
_DISCRIMINATORS = {"numeric": _NUMERIC_DISCRIMINATORS, "behavioral": _BEHAVIORAL_DISCRIMINATORS}


def _stage_of(nodeid: str) -> str | None:
    return _STAGE_OF.get(Path(nodeid.split("::", 1)[0]).stem)


def _test_func_name(nodeid: str) -> str:
    """``path::TestCls::test_foo[param]`` / bare ``path::test_foo`` -> ``test_foo``."""
    return nodeid.rsplit("::", 1)[-1].split("[", 1)[0]


def _read_score(report) -> float:  # noqa: ANN001
    """The [0,1] similarity a scored numeric test recorded via ``record_property``.

    Reads the single ``score`` user_property (``dict`` keeps the last write, i.e. the
    running power-mean the test accumulated over all its layers/steps/variants).
    Missing / non-numeric / non-finite -> 0.0 (a missing record earns no credit);
    clamped to [0,1] defensively.
    """
    props = dict(getattr(report, "user_properties", None) or [])
    try:
        s = float(props.get("score"))
    except (TypeError, ValueError):
        return 0.0
    if not (s == s) or s in (float("inf"), float("-inf")):
        return 0.0
    return min(1.0, max(0.0, s))


def pytest_exception_interact(node, call, report):  # noqa: ANN001 - pytest hook
    """A SANDBOX TIMEOUT ENDS THE SESSION UNSCORED. It must not become a reward.

    This is the fifth member of the family the module docstring describes as
    "the machine is wrong, not the candidate" -- the golden-health gate, the
    geometry gate, the pinned-device gate and the harness-health gate. All of
    them refuse to produce a number rather than produce a misleading one, and
    all of them land on exit 4, which ``test.sh`` turns into "abort with NO
    reward.json".

    WHY IT BELONGS WITH THEM. The budget bounds the
    single worker spawn that runs EVERY probe, and the honest reference needs a
    large fraction of it -- enough that it can exceed the budget on a loaded host
    under contention. When it does, its failures produce the SAME low reward as a
    port that cheats by importing torch (the ``torch_only`` control), and nothing
    in the artifact says which had happened.

    SCORING IT WAS THE BUG, not the threshold. A grade whose harness could not
    finish is not a low grade, it is an absent one -- and unlike a tolerance,
    no choice of budget makes this safe, because the failure mode is unbounded
    host load. A genuinely non-terminating port also ends here rather than
    scoring 0, and that is the right trade: a real hang reproduces on a quiet
    box and contention does not, so one re-run separates them, whereas nothing
    separates two identical rewards after the fact.

    Keyed on ``CandidateTimeout``, which ``Candidate._raise_error`` memoises, so
    every one of the session-scoped checks replaying the same failure agrees.
    """
    exc = call.excinfo.value if call.excinfo is not None else None
    if exc is None or type(exc).__name__ != "CandidateTimeout":
        return
    # THE MARKER IS FOR THE CONTROLS HARNESS, not for a human. `mutate.py` sees
    # only pytest's exit code, and exit 4 already means "one of conftest's
    # end-of-collection gates refused" -- a statement about the goldens, the
    # geometry or the device, which the controls harness correctly reports as a
    # REGRESSION. A sandbox timeout is a statement about host LOAD and must not
    # be reported that way, so it carries a token the harness can match on.
    pytest.exit(
        f"[T2J_SANDBOX_TIMEOUT] "
        f"the candidate sandbox exceeded its whole-sandbox budget, so this "
        f"session is ending UNSCORED rather than reporting the resulting "
        f"failures as a grade. This is a machine verdict, not a candidate "
        f"verdict: the reference itself exceeds this budget on a loaded host, "
        f"and the reward it would have produced is the one a torch-importing "
        f"candidate gets. Re-run on a quiet box; if it reproduces there, the "
        f"port really does hang.\n{exc}",
        returncode=4,
    )


def pytest_runtest_logreport(report):  # noqa: ANN001 - pytest hook
    stage = _stage_of(report.nodeid)
    if stage is None:
        return
    rec = _TALLY.setdefault(stage, {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0})
    if report.when == "call":
        if getattr(report, "wasxfail", None) is not None:
            rec["xfailed"] += 1
        elif report.passed:
            rec["passed"] += 1
        elif report.failed:
            rec["failed"] += 1
        elif report.skipped:
            rec["skipped"] += 1
    elif report.failed:  # setup/teardown error
        rec["error"] += 1
    elif report.skipped and report.when == "setup":
        rec["skipped"] += 1

    # ADDITIVE: file the SCORED dimension's continuous similarity alongside the binary
    # tally above (which is intentionally left untouched). Both numeric and behavioral
    # are scored; spec stays binary-only.
    if stage not in _SCORED_STAGES:
        return
    scores = _SCORES.setdefault(stage, {})
    is_gate = _test_func_name(report.nodeid) in _STAGE_GATE_TESTS[stage]
    if report.when == "call":
        if getattr(report, "wasxfail", None) is not None:
            pass  # no xfails on the scored stages by design; keep out of the mean
        elif report.passed:
            if not is_gate:  # a gate pass banks no credit
                scores[report.nodeid] = _read_score(report)
        elif report.failed:  # assertion / crash / torch-leak in the body -> 0.0
            scores[report.nodeid] = 0.0
        elif report.skipped:
            pass  # no grade-time skips by design; keep out of the mean
    elif report.failed:  # setup/teardown failure -> 0.0 for this test
        scores[report.nodeid] = 0.0


def _effective_discriminators(stage: str, facts: dict) -> frozenset[str]:
    """Per-instance discriminator set for a scored dimension (Lever B).

    Inference's discriminators are the generalisation batteries, which grade logit parity
    and so sit at the ~0.999 ceiling on any faithful port -- none is ever legitimately OOB,
    so (unlike the training class's modality-gated ``fp32_loss`` veto) the set is not
    narrowed by any fact. The hook stays for structural parity with the training grader and
    as the seam a future OOB-prone discriminator would gate on.
    """
    return _DISCRIMINATORS.get(stage, frozenset())


def _dimension_score(stage: str, scores: dict, facts: dict) -> float:
    """Lever B two-tier aggregation for a scored dimension.

    The forward-correlated checks (everything NOT a discriminator) collapse into ONE soft
    vote via the low-tail-sensitive power mean (p<1): a single wholly-broken forward
    check still bites super-linearly WITHIN the cluster, and a uniform faithful set (~0.999)
    is unchanged (p-invariance). Each generalisation battery then enters as its OWN hard MIN
    vote, so a port that memorised the frozen scenario can no longer dilute a failing
    off-scenario axis upward behind many correlated single-shot parity checks. With no
    discriminator present in ``scores`` -- an instance whose batteries are mode-gated out or
    not yet frozen -- ``votes`` is just the one collapsed cluster vote, so this reduces
    EXACTLY to the flat power mean the deep fix shipped.
    """
    if not scores:
        return 0.0
    disc_names = _effective_discriminators(stage, facts)
    disc = [v for nid, v in scores.items() if _test_func_name(nid) in disc_names]
    fwd = [v for nid, v in scores.items() if _test_func_name(nid) not in disc_names]
    votes = list(disc)
    if fwd:
        votes.append(similarity.power_mean(fwd))  # collapsed soft forward vote
    return round(min(votes), 6) if votes else 0.0


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001 - pytest hook
    # WHEN THIS FILE WRITES AN ARTIFACT AT ALL. `eval_summary.json` is a GRADING
    # artifact, and this hook fires for every session that touches this tree --
    # including a developer running the driver's own unit tests, which come in
    # under this conftest and grade nothing. Six cases, and each of the five
    # abnormal ones was reachable:
    #
    #   1. nothing collected            -> write, overall FAIL. A misconfigured
    #      grade run: `collect_ignore_glob` is this tree's only defence against
    #      an empty collection and the graded modules are named by a `test.sh`
    #      outside this file, so a stale path there collects nothing at all.
    #   2. items collected, none graded -> WRITE NOTHING. Not a grade run. This is
    #      `pytest tests/_t2j/tests/test_*.py`, the documented driver-suite
    #      invocation, which used to leave `overall: PASS` at the default path
    #      where nothing marked it as a developer artifact. A MISSING file is the
    #      louder failure: if a real `test.sh` ever collected only non-graded
    #      items, Harbor finds no summary and says so, rather than reading green.
    #   3. graded items, no outcomes    -> write, overall FAIL. `--collect-only`,
    #      or `-k` matching nothing: pytest's own keyword deselection is invisible
    #      to the collection count, so this cannot be caught at collection time.
    #   5. a PARTIAL selection that actually graded something -> WRITE NOTHING,
    #      for exactly case 2's reason. See below.
    #   6. graded items, PARTIAL outcomes -> WRITE NOTHING. Fewer outcomes were
    #      tallied than checks survived collection: `--deselect`, `-x`,
    #      `--maxfail`, or any exception escaping a hook once outcomes had begun
    #      accumulating. `_selection_narrowed` (case 5) now recognises
    #      `--deselect`/`-x`/`--maxfail` AT COLLECTION TIME, but this is the
    #      general net underneath it -- it catches a shortfall by its EFFECT
    #      (fewer outcomes than collected checks), so it also catches a cause
    #      collection time cannot see. See below.
    #   4. otherwise                    -> the tally.
    #
    # Cases 1 and 3 are distinguished in the artifact by WHICH key carries the
    # reason, because they have different causes and different repairs.
    collected, graded_items = _COLLECTED["kept"], _COLLECTED["graded"]
    graded_outcomes = sum(sum(rec.values()) for rec in _TALLY.values())

    # CASE 6: FEWER OUTCOMES THAN COLLECTED+GRADED CHECKS -> WRITE NOTHING.
    #
    # `graded_items` is what THIS hook's `pytest_collection_modifyitems` saw
    # survive collection; `graded_outcomes` is what `pytest_runtest_logreport`
    # actually tallied. They can disagree even when `graded_items` is correct and
    # nonzero. `pytest /tests --deselect test_numeric.py --deselect
    # test_behavioral.py` gives 14 passed, 29 deselected, reward 1.0, overall PASS,
    # no `partial_selection` key -- 17 of 31 graded checks, including every
    # parity and equivalence claim, never ran. `--deselect` is applied in
    # pytest's OWN `modifyitems` hook, which runs AFTER this file's, so
    # `graded_items` still reconstructs the whole declared set and
    # `_require_complete_collection` cannot see the shortfall. Same shape for
    # `-x`, `--maxfail`, and any exception escaping a hook once outcomes have
    # started accumulating (exit 2/3 -- `tests/test.sh` traps only exit 4).
    # Withholding the artifact is the same remedy cases 2 and 5 already apply: a
    # missing file is louder than a plausible number.
    #
    # `0 <` ON THE LEFT IS DELIBERATE, not redundant with `graded_items` being
    # truthy: `graded_outcomes == 0` with `graded_items > 0` is case 3
    # (`--collect-only`, or `-k` matching nothing), which WRITES an artifact
    # (`overall: FAIL`, `no_graded_outcomes`) rather than withholding one -- see
    # `test_graded_checks_collected_but_never_run_is_not_a_PASS`. This guard must
    # compose with that case, not swallow it, so it only fires on a PARTIAL,
    # NONZERO tally: strictly between empty and complete.
    #
    # THE DIAGNOSTIC NAMES THE ACTUAL CAUSE WHEN ONE IS KNOWN, not just the three
    # flags this guard was first written for. `-k`/`-m` land HERE, not in case 5
    # below, for the same reason `--deselect`/`-x`/`--maxfail` do: this file's
    # `pytest_collection_modifyitems` runs BEFORE pytest's own keyword/mark
    # filtering, so `graded_items` still counts the whole declared set and a
    # `-k`-narrowed session that passes its matched subset shows up here as a
    # partial, nonzero tally -- it never reaches case 5's
    # `_PARTIAL_SELECTION and graded_outcomes` check because this guard returns
    # first. `_selection_narrowed` already computed the real reason at collection
    # time (`_PARTIAL_SELECTION`), so use it when one is on record instead of
    # blaming `--deselect`/`-x`/`--maxfail` for a shortfall one of them did not
    # cause; the generic wording is kept only for the residual case that reason
    # can't see -- outcomes stopping mid-session with no narrowing flag on record
    # at all (an exception escaping a hook once outcomes had begun accumulating).
    if graded_items and 0 < graded_outcomes < graded_items:
        reason = _PARTIAL_SELECTION[0] if _PARTIAL_SELECTION else None
        guidance = (
            f"this session was narrowed by {reason}. Run the tests DIRECTORY "
            "with no such narrowing for a gradable run."
        ) if reason else (
            "Run the tests DIRECTORY with no --deselect, -x or --maxfail for a "
            "gradable run."
        )
        print(f"\n[t2j] NO eval_summary.json was written: {graded_outcomes} of "
              f"{graded_items} collected graded checks produced an outcome. A "
              f"reward derived from this session would be a pass rate over "
              f"whichever checks happened to run. {guidance}")
        return

    if collected and not graded_items:
        return

    # CASE 5: A SUBSET RUN THAT GRADED SOMETHING PRODUCES NO ARTIFACT AT ALL.
    #
    # `_require_complete_collection` stands down for a narrowed invocation because
    # there the TREE is whole and the command line is not -- but the tally such a
    # session produces is a pass rate over a fraction of the suite, and
    # `summary_to_reward.py` reads the three stage dicts and nothing else:
    # a `-k`-narrowed session that passes its subset yields `reward 1.0`, with the
    # only warning a `partial_selection` string in a file the reward reduction does
    # not read. That is the same "a partial run scores high" family as a graded
    # module going missing, one invocation over.
    #
    # WITHHOLDING THE INPUT IS A CHOICE, NOT THE ONLY OPTION LEFT.
    # `summary_to_reward.py` is authored once, in the shared template, so it could be
    # taught to refuse a summary carrying `partial_selection`.
    #
    # It is still not the place to make this ruling. A refusal there covers only the
    # shapes the artifact can DESCRIBE, and the general case here is a session that
    # stopped early with no key on record at all; and it converts a withheld grade
    # into a nonzero exit that `test.sh` would have to classify anyway. Withholding
    # the INPUT covers both, and is the same ruling case 2 already makes: a missing
    # file is louder than a plausible number.
    #
    # SCOPED TO SESSIONS THAT GRADED SOMETHING, deliberately. A narrowed session with
    # ZERO outcomes (`-k` matching nothing) is case 3 and still writes: it reports
    # `overall: FAIL` with an all-zero tally, which reduces to reward 0.0 and cannot
    # be mistaken for a grade. Only a subset that PASSED can lie.
    #
    # No graded entry point takes this branch: `tests/test.sh` runs `pytest /tests`,
    # and the reference-mutation controls name the directory.
    if _PARTIAL_SELECTION and graded_outcomes:
        print(f"\n[t2j] NO eval_summary.json was written: this session graded a "
              f"SUBSET of the suite ({_PARTIAL_SELECTION[0]}), so any reward derived "
              "from it would be a pass rate over a fraction of the benchmark. Run "
              "the tests DIRECTORY for a gradable run.")
        return

    # Instance facts (mode / multimodal / battery presence): drive Lever B's discriminator
    # gating below AND the coverage vector at the end of this hook. Computed once here,
    # after every early return, so both readers see the same dict.
    facts = _golden_facts(_TEST_DIR / "golden")

    def stage(name: str) -> dict:
        # A STAGE THAT GRADED NOTHING DID NOT PASS. "No failures and no errors"
        # is vacuously true of an empty tally, which put three green stage flags
        # next to a red `overall` in the artifact and three ticks against a FAIL
        # row in the dashboard collector. This is the same rule
        # `summary_to_reward._stage_pass` already applies downstream (`failed ==
        # 0 and error == 0 and passed > 0`), so the summary now agrees with the
        # reward instead of contradicting it.
        #
        # This flag only. `overall` is NOT derived from it: a stage that is
        # legitimately empty because mode gating deselected its checks must not
        # fail the session, which is the whole-session-only scoping the
        # empty-collection guard is held to as well.
        #
        # Note the key collision this rides on, inherited from the training class
        # and deliberately left alone: `{"passed": flag, **rec}` lets `rec`'s
        # integer `passed` COUNT overwrite the boolean whenever the stage ran. So
        # this computation is only ever observable on an EMPTY stage -- which is
        # exactly the case being fixed, and is why fixing it changes nothing about
        # a real grade's artifact.
        rec = _TALLY.get(name, {})
        ran = rec.get("passed", 0) + rec.get("failed", 0) + rec.get("error", 0)
        passed = ran > 0 and rec.get("failed", 0) == 0 and rec.get("error", 0) == 0
        return {"passed": passed, **rec}

    # SCORED DIMENSION SCORES (deep fix): the low-tail-sensitive power mean over the
    # per-check similarities filed above, plus the raw per-check scores and their count.
    # The binary stage block (stage(name)) is preserved for diagnostics/integrity; the
    # reward mapper consumes each dimension's `*_score`/`n`. `mean == 0.0` on an empty
    # set (no check on that stage scored). Numeric and behavioral are both scored now.
    def scored(name: str) -> tuple[dict, float]:
        s = dict(sorted(_SCORES.get(name, {}).items()))
        # Lever B two-tier aggregation (numeric); flat power mean where the discriminator
        # set is empty (behavioral) or unpopulated (no battery frozen/applicable).
        mean = _dimension_score(name, s, facts)
        return {**stage(name), "scores": s, "mean": mean, "n": len(s)}, mean

    numeric_block, numeric_score = scored("numeric")
    behavioral_block, behavior_score = scored("behavioral")

    summary = {
        "spec": stage("spec"),
        "numeric": numeric_block,
        "behavioral": behavioral_block,
    }
    summary["overall"] = "PASS" if all(
        s.get("failed", 0) == 0 and s.get("error", 0) == 0 for s in summary.values()
    ) else "FAIL"
    # Top-level scalars the reward mapper reads for the scored dimensions. Added AFTER
    # `overall`, like `coverage`/`deselected`, so they never enter the reduction above
    # (which iterates the stage dicts and would choke on a bare float).
    summary["numeric_score"] = numeric_score
    summary["behavior_score"] = behavior_score

    # AN EMPTY SESSION IS NOT A PASS. The reduction above is over failures, so a
    # session that graded nothing satisfies it vacuously and reports PASS on zero
    # evidence. The reward SCALAR was already safe (`summary_to_reward._rate` is
    # 0.0 on an empty tally), but `overall` is a published binary gate.
    #
    # DELIBERATE DIVERGENCE from the training class's conftest, which has no such
    # guard: there, `tasks/*/tests` always contains the three graded modules, so
    # an empty collection means a broken checkout rather than a mis-scoped
    # invocation.
    if not collected:
        summary["overall"] = "FAIL"
        summary["empty_collection"] = (
            "this session collected "
            f"{'no items' if collected == 0 else 'nothing (collection never ran)'}"
            ", so nothing was graded. An empty run cannot be a PASS: check that "
            "the graded modules exist under this tests/ directory and that the "
            "invocation names them."
        )
    elif not graded_outcomes:
        summary["overall"] = "FAIL"
        summary["no_graded_outcomes"] = (
            f"this session collected {graded_items} graded check(s) and ran none "
            "of them, so nothing was graded. `--collect-only` and a `-k` "
            "expression that matches nothing both land here: pytest filters by "
            "keyword after the collection hook, so an empty run of a non-empty "
            "collection cannot be caught any earlier than this."
        )
    # WHICH CHECKS WERE DESELECTED. Deselection keeps a check out of both numerator
    # and denominator, which is right for the reward and invisible in it: an instance
    # that deselects half its suite reports the same 1.0 as one that ran all of it.
    # Naming them is the difference between "this mode has no chunked path" and "a
    # gate typo quietly stopped grading something". Added AFTER ``overall``, like
    # ``coverage``, so it cannot enter the reduction, and ignored by
    # ``summary_to_reward.py``, which reads the three stages and ``overall`` only.
    # A SUBSET RUN SAYS SO -- and only reaches this line when it graded NOTHING, since
    # case 5 above returns without writing for any subset that produced outcomes. What
    # is left here is the `-k`-matched-nothing shape, which is already `overall: FAIL`
    # with an all-zero tally; this names the cause beside `no_graded_outcomes`. Added
    # after `overall`, like `coverage`, so it cannot enter the reduction, and ignored
    # by `summary_to_reward.py`.
    if _PARTIAL_SELECTION:
        summary["partial_selection"] = (
            f"this session asked for a SUBSET of the suite: {_PARTIAL_SELECTION[0]}. "
            "It graded nothing, so this artifact carries an all-zero tally rather than "
            "a partial score; a subset that DID grade something writes no artifact at "
            "all. A real grade names the tests DIRECTORY.")
    summary["deselected"] = list(_DESELECTED)
    # ...AND WHY, by kind. One of this instance's deselections
    # (`test_fp32_decode_step_logits`) is dark on EVERY instance in the class
    # because no producer for its golden exists, and a reader of the flat list
    # above cannot tell it from the mode-gated ones that will run as soon as an
    # instance has the mode. (It had a partner, `test_held_out_nll`, until L6
    # got a producer and became always-on.) Added after `overall`,
    # like `coverage`, so it cannot enter the reduction, and ignored by
    # `summary_to_reward.py`.
    summary["deselected_kinds"] = {
        name: _DESELECTION_KIND.get(name, "unclassified") for name in _DESELECTED}

    # Coverage vector: which required goldens the freeze emitted for this MODE,
    # so a coverage regression is visible in the summary itself and not only in a
    # failing test. Added AFTER ``overall`` so it never enters the reduction. ``facts``
    # was computed once above (Lever B and this reader share it).
    summary["coverage"] = golden_manifest.coverage_vector(_TEST_DIR / "golden", facts.get("mode"))

    # Never write under tasks/: real grading pins T2J_SUMMARY_DIR to a
    # location outside the (often bind-mounted) source tree -- Harbor test.sh
    # points it at /run/t2j-grade. The fallback is a temp
    # dir so an ad-hoc `pytest tasks/*/tests` still emits a summary.
    #
    # /run/t2j-grade, NOT /logs/verifier, and that is a grading-integrity boundary
    # rather than a tidiness one: under Harbor this hook runs as the unprivileged
    # `t2jgrade` uid (test.sh drops privileges for the pytest phase), and
    # /logs/verifier holds `reward.json` and must stay root-owned. test.sh copies
    # this file up to /logs/verifier/eval_summary.json, as root, the moment pytest
    # returns -- so the artifact still appears at its documented path and every
    # guard downstream reads a copy no unprivileged process can rewrite.
    out_dir = Path(os.environ.get("T2J_SUMMARY_DIR") or Path(tempfile.gettempdir()) / "t2j-grade")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if not os.environ.get("T2J_SUMMARY_DIR"):
        print(f"\n[t2j] eval_summary.json written to {out_path} (set T2J_SUMMARY_DIR to relocate)")
