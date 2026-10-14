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

"""Standalone grader wiring (generated; no torch, no engine).

`pytest tasks/{method}_{model}/tests` decides pass/fail with only jax + numpy +
pytest, given a candidate repo and this folder's golden/ + artifact/. The candidate
defaults to the committed golden ../solution/ref_jax_repo; under Harbor the verifier's test.sh
overrides it with T2J_JAX_REPO (the agent's produced /workspace/jax_repo).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

# Never write bytecode caches next to the sources: a bare ``pytest tasks/*/tests``
# imports these test modules in place, so __pycache__ would land
# under tasks/. Set before the fixtures/_t2j imports below so those stay uncached too.
# (Entry points also export PYTHONDONTWRITEBYTECODE=1, which additionally covers this
# conftest's own bytecode, written by pytest before this line runs.)
sys.dont_write_bytecode = True

# Hermetic + offline grading: the candidate must not hit the network, and the
# torch-blocked sandbox subprocess inherits this environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Not redundant with PREALLOCATE: that only stops the up-front grab, while the BFC pool
# is still capped at MEM_FRACTION, which defaults to 0.75 of the card. The 7B instances
# need more than 0.75 to backpropagate a float32 upcast of a bf16 checkpoint, so raise
# the ceiling here too -- tests/test.sh exports it, but a bare `pytest tasks/*/tests`
# does not go through test.sh.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

# Make the vendored `_t2j` package and sibling `fixtures` importable regardless of
# pytest's rootdir / import mode.
_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from fixtures import build_candidate, load_golden  # noqa: E402
from _t2j import golden_manifest, similarity  # noqa: E402


@pytest.fixture(scope="session")
def golden():
    """Frozen reference artifacts + materialized thresholds."""
    return load_golden(_TEST_DIR / "golden")


@pytest.fixture(scope="session")
def cand(golden):
    """Candidate repo (default ../solution/ref_jax_repo; T2J_JAX_REPO overrides), driven in a
    torch-blocked sandbox against frozen inputs.

    Built lazily: the sandbox subprocess only spawns on the first artifact access,
    so a missing/broken repo surfaces as a clear failure in the test that uses it
    rather than at collection time.
    """
    return build_candidate(_TEST_DIR, golden)


# --------------------------------------------------------------------------- #
# Applicability deselection: the reward is a continuous pass rate, so
# NO check may skip at grade time. A check that cannot apply to THIS instance --
# because the reference never froze the artifact it compares against, or the method
# has no such objective -- is deselected here (never collected), keeping it out of
# both numerator and denominator. Everything that survives collection must resolve
# to pass or fail: a candidate that cannot run an APPLICABLE check fails it (the
# check's own body asserts ``supported``), it does not skip. Applicability is read
# purely from the frozen golden (torch-free, no candidate run), so it self-adjusts
# if the goldens change.
# --------------------------------------------------------------------------- #
def _npz_keys(path: Path) -> set[str]:
    """Array names inside an .npz WITHOUT loading any array (npz is a zip of .npy)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    except (OSError, zipfile.BadZipFile):
        return set()


def _golden_facts(golden_dir: Path) -> dict:
    meta: dict = {}
    try:
        meta = json.loads((golden_dir / "meta.json").read_text())
    except (OSError, ValueError):
        pass
    grads: dict = {}
    try:
        grads = json.loads((golden_dir / "gradients.json").read_text())
    except (OSError, ValueError):
        pass
    forward_keys = _npz_keys(golden_dir / "forward.npz")
    fp32_keys = _npz_keys(golden_dir / "fp32_forward.npz")
    input_keys = _npz_keys(golden_dir / "inputs.npz")
    mm = ("pixel", "image", "audio", "video", "grid", "feature", "mel")
    return {
        "method": meta.get("training_method"),
        # LoRA freezes the base and trains only the adapters (all 45 grade under
        # finetuning_type=lora). Gates the base-freeze / optimizer-identity probes,
        # which read the candidate's own reconstructed training final_params.
        "is_lora": str((meta.get("repo_config") or {}).get("finetuning_type", "full")).lower() == "lora",
        "has_train_traj": (golden_dir / "train_trajectory.json").exists(),
        "has_generation": (golden_dir / "generation.npz").exists(),
        "has_fp32_forward": bool(fp32_keys),
        "has_fp32_loss": "loss" in fp32_keys,
        "has_forward_loss": "loss" in forward_keys,
        "has_hidden": bool(meta.get("compare_hidden_states"))
        and any(k.startswith("hidden_") for k in forward_keys),
        "has_per_param_grads": bool(grads.get("grads")),
        "multimodal": any(any(m in k.lower() for m in mm) for k in input_keys),
        # input-space battery: present when the freezer froze at
        # least one variant (text tasks only). Presence-gated like has_hidden/has_generation
        # -- an optional feature, not a manifest-required golden.
        "has_battery": (golden_dir / "battery").is_dir()
        and any((golden_dir / "battery").glob("*/inputs.npz")),
        # Marker-gated per-function/end-to-end checks (bloom instances) apply only when
        # their reference artifact was frozen; keyed by marker name below.
        "refunits": (golden_dir / "reference_units.npz").exists()
        and (golden_dir / "reference_units.json").exists(),
        "train_run": (golden_dir / "train_run.json").exists(),
        # PPO update-mechanics reference units: the torch-frozen I/O of
        # ppo/ppo_utils.py (GAE / clipped losses / masked-whiten / token-F1). Present only
        # where the freezer ran (the 15 PPO tasks). Gates test_ppo_update_units,
        # which executes the CANDIDATE's own ppo_utils against these oracle outputs -- the one
        # PPO gate that grades the real update math instead of self-reported train JSONs.
        "has_ppo_units": (golden_dir / "ppo_units.json").exists()
        and (golden_dir / "ppo_units.npz").exists(),
        # spec goldens (presence-gated). Each is produced by a future freezer that needs
        # the H100 checkpoint / L6 sweep, NOT the torch-free toolbox; until one exists the
        # matching spec check is deselected (kept out of numerator AND denominator).
        "has_config_spec": (golden_dir / "config_spec.json").exists(),
        "has_loader_spec": (golden_dir / "loader_spec.json").exists(),
        # L6 gate: present when the L6 freezer froze validation_loss.json (V_torch +
        # tol). The candidate's V_jax is produced IN-BAND -- the grader runs the real
        # training entrypoint from config.py inside the sandbox (candidate.validation_loss)
        # -- so no env var / precomputed result is needed; the frozen band is the only gate.
        "has_validation": (golden_dir / "validation_loss.json").exists(),
        # vision-tower sensitivity gate: present only where the FAITHFUL reference's
        # graded per-step objective is genuinely vision-dependent -- i.e. a paired probe
        # confirmed that perturbing pixel_values moves the numeric loss by >> floor
        # (the freezer froze tower_sensitivity.json). It is ABSENT
        # where the reference objective is vision-blind BY CONSTRUCTION: PaliGemma's
        # text-only LM-path forward (no pixel_values input), PPO's cached-vision rollout
        # surrogate, and the weights-only omni tower all return a bitwise-identical loss
        # under pixel perturbation. There a {0,1} loss-perturbation discriminator cannot
        # separate a faithful port from a tower stub (both score delta==0) and would only
        # deflate the honest reward, so it is deselected -- vision stays covered on those
        # tasks by the in-band L6/T3 real-training gates and numeric test_pixel_values.
        "has_tower_sensitivity": (golden_dir / "tower_sensitivity.json").exists(),
    }


# Marker name -> golden fact that must be True for a check carrying that marker to be
# COLLECTED. Used for the bloom per-function ("refunits") and end-to-end ("train_run")
# suites, which compare against an artifact frozen for only some instances. A check
# without either marker is unaffected.
_MARKER_FACTS = ("refunits", "train_run")


# name -> predicate(facts) that must be True for the check to be COLLECTED for this
# instance. A name absent here always runs.
def _requires(token: str):
    """Predicate: this instance's METHOD requires ``token`` per the golden manifest.

    Golden-PRESENCE deselection hides a freeze bug that drops a required golden -- the
    check silently leaves the pass-rate denominator, reading as "not applicable". Gating
    instead on what the METHOD must emit means a required-but-missing golden is COLLECTED
    and FAILS (its body asserts the artifact), and ``test_golden_manifest`` names it: a
    loud coverage regression, not a silent skip. Optional/feature checks stay
    presence-gated (e.g. test_hidden_states)."""
    def pred(f):
        method = str(f.get("method")).lower()
        return token in golden_manifest.REQUIRED_GOLDENS.get(method, frozenset())

    return pred


_APPLICABLE = {
    "test_hidden_states": lambda f: f["has_hidden"],
    "test_fp32_forward": _requires("fp32_forward:logits"),
    "test_fp32_loss": _requires("fp32_forward:loss"),
    "test_forward_loss": _requires("forward:loss"),
    "test_gradient_per_param": _requires("gradients:grads"),
    # Greedy-decode gates supersede the old xfail ``test_generation``. Both are
    # SFT-only (only SFT freezes generation) and TEXT-only ("start on text tasks":
    # multimodal generate would need vision inputs threaded through the grown sequence).
    "test_generation_logits": lambda f: _requires("generation.npz")(f) and not f["multimodal"],
    "test_generation_shape": lambda f: _requires("generation.npz")(f) and not f["multimodal"],
    "test_train_trajectory": _requires("train_trajectory.json"),
    # LoRA base-freeze + optimizer-identity read the 1-step training result, which is
    # produced only where the train loop is exercised (SFT+DPO freeze train_trajectory; PPO
    # does not). Gate on is_lora AND has_train_traj -> collects on the 30 SFT+DPO instances.
    "test_lora_base_frozen": lambda f: f["is_lora"] and f["has_train_traj"],
    "test_optimizer_identity": lambda f: f["is_lora"] and f["has_train_traj"],
    "test_dpo_objective": lambda f: f["method"] == "dpo",
    "test_ppo_loss_arrays": lambda f: f["method"] == "ppo",
    "test_ppo_old_policy_ratio": lambda f: f["method"] == "ppo",
    # PPO update-mechanics gate: PPO-only + presence-gated on the torch-frozen
    # ppo_units golden (the candidate's own ppo/ppo_utils.py graded vs oracle I/O).
    "test_ppo_update_units": lambda f: f["method"] == "ppo" and f["has_ppo_units"],
    # Input-space battery: forward-logit parity re-run over shape/length/batch/mask
    # variants. Text-only (varying length would break the vision-tower contract) and
    # presence-gated (deselected until the battery is frozen for this instance).
    "test_forward_logits_battery": lambda f: f["has_battery"] and not f["multimodal"],
    "test_fp32_forward_battery": lambda f: f["has_battery"] and not f["multimodal"],
    "test_causal_masking": lambda f: not f["multimodal"],
    "test_padding_invariance": lambda f: not f["multimodal"],
    "test_softmax_normalization": lambda f: not f["multimodal"],
    "test_batch_independence": lambda f: not f["multimodal"],
    # Numeric pixel_values similarity (test_pixel_values): the only float input
    # surface, present exactly on the multimodal instances (all carry pixel_values); the
    # integer/token keys stay the exact Spec gate in test_spec.test_data_pipeline.
    "test_pixel_values": lambda f: f["multimodal"],
    # Vision-tower sensitivity gate (test_mm_tower_sensitivity): golden-free {0,1}
    # self-consistency probe that the graded loss actually DEPENDS on the image content
    # (perturbing pixel_values must move the loss). Collected only where the FAITHFUL
    # reference is itself vision-dependent under this objective -- tower_sensitivity.json
    # frozen (SFT/DPO VL). Deselected where the reference is vision-blind by construction
    # (PaliGemma text-only forward, PPO cached-vision rollout, weights-only omni): there
    # the probe cannot discriminate a stub from the faithful port, and vision is covered
    # by the in-band L6/T3 gates + numeric test_pixel_values instead. See _golden_facts.
    "test_mm_tower_sensitivity": lambda f: f["has_tower_sensitivity"],
    # Config-DEFAULT parity: REQUIRED for every method (config_spec.json is frozen for
    # all 45), so a missing freeze FAILS loudly via test_golden_manifest instead of silently
    # deselecting. loader/validation remain presence-gated on their (future) goldens.
    "test_config_spec": _requires("config_spec.json"),
    "test_loader_spec": lambda f: f["has_loader_spec"],
    "test_validation_split": lambda f: f["has_validation"],
    # L6 behavioral gate: collected whenever the L6 freezer froze validation_loss.json
    # (V_torch + tol). The candidate's V_jax comes from the IN-BAND real training run
    # (candidate.validation_loss -> _ensure_real_train_run), so the frozen band is the sole
    # gate -- no env var, no out-of-band sweep, no re-materialization. A port that cannot
    # reproduce the reference within band (or within train_timeout_s) FAILS it (not skips).
    "test_validation_loss": lambda f: f["has_validation"],
    # T3 end-to-end gate: marker-gated on the ``train_run`` golden fact (deselected until
    # the freezer emits train_run.json). Its V_jax/train_loss come from the SAME in-band real
    # run as L6 (candidate.train_run). Kept explicit here too so the predicate and the marker
    # agree; grade-neutral -- collected iff the golden exists, then pass or fail.
    "test_train_run_end_to_end": lambda f: f["train_run"],
}


def pytest_configure(config):  # noqa: ANN001 - pytest hook
    config.addinivalue_line(
        "markers", "refunits: per-function parity vs golden/reference_units (deselected if absent)"
    )
    config.addinivalue_line(
        "markers", "train_run: end-to-end run vs golden/train_run.json (deselected if absent)"
    )


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    facts = _golden_facts(_TEST_DIR / "golden")
    kept, deselected = [], []
    for item in items:
        pred = _APPLICABLE.get(item.name.split("[", 1)[0])
        applicable = pred is None or pred(facts)
        # A check tagged with a marker-gated fact is deselected when its artifact is absent.
        if any(item.get_closest_marker(m) and not facts[m] for m in _MARKER_FACTS):
            applicable = False
        (kept if applicable else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


# --------------------------------------------------------------------------- #
# Per-instance JSON summary: Spec stays a binary tally; Numeric
# and Behavioral collect a per-check CONTINUOUS SIMILARITY in [0,1], each
# test recording it via ``record_property("score", s)``. The reward is the spec-gated,
# dimension-weighted mean of those similarities, not a pass rate.
# --------------------------------------------------------------------------- #
_STAGE_OF = {"test_spec": "spec", "test_numeric": "numeric", "test_behavioral": "behavioral"}
_SCORED_STAGES = ("numeric", "behavioral")
# De-count fillers: scored-stage tests that assert a structural / anti-hack
# invariant but carry NO continuous fidelity signal -- they record a constant ~1.0 on any
# structurally-correct port. Left in the power mean they bank unearned credit that dilutes
# the genuine similarity checks; routed to the binary tally instead, a PASS banks no
# continuous credit while a hard-assert FAIL (non-causal attention, swap-invariant "DPO"
# objective, severed backward, ...) still gates the reward multiplicatively via
# spec_pass_rate. Keyed by test-function base name (suite-wide, byte-identical across the 45
# tasks); a name absent from a given task's suite is simply never matched. This changes NO
# golden and needs no re-freeze. Genuine continuous scorers that share a prefix
# (test_generation_logits, test_gradient_loss_and_norm, test_gradient_authenticity,
# test_padding_invariance, test_determinism_static[already spec]) are intentionally absent --
# membership is exact, never prefix.
_GATE_TEST_NAMES = frozenset({
    # Vacuous / self-consistency invariants (no golden comparison):
    "test_softmax_normalization",
    "test_train_loss_trend",
    "test_determinism",
    "test_generation_shape",
    # {0,1} structural anti-hack discriminators (constant 1.0 on a correct port):
    "test_gradient_per_param",
    "test_optimizer_identity",
    "test_lora_base_frozen",
    "test_dpo_objective",
    "test_causal_masking",
    "test_mm_tower_sensitivity",
})
# Lever B (anti-inflation numeric aggregation). The numeric dimension is dominated by
# ~17 forward-correlated checks (logits/hidden/loss/battery/reference-units/param-norms/...)
# that CO-VARY: they are not independent evidence, so a flat power mean lets them dilute a
# lone failing objective discriminator UPWARD (count-dominance -- 20 passing forward checks
# pull one 0.05 objective failure to 0.05**(1/21)~=0.87). Fix: collapse the forward-correlated
# cluster into ONE soft vote (its own power mean, partial credit + p-invariance preserved) and
# combine it by MIN with each genuine objective/run DISCRIMINATOR below, so a wrong training
# objective can no longer be masked by a correct forward. The discriminators are fp32
# candidate-vs-oracle checks, ~0.999 on any faithful port so the min never punishes honesty:
# PPO ppo_utils and DPO get_batch_logps import the port's OWN objective on frozen
# inputs, while the other DPO trainer checks are grader-reconstructed (a faithful port need not
# expose dpo.trainer.DPOTrainer) and kept as MIN votes -- not in the cluster -- so their near-
# constant values cannot re-inflate it. SFT has no standalone torch-grounded objective symbol
# (torch_orig computes the loss inside the model via ForCausalLMLoss; the grader recomputes it
# from logits for a logits-only port), so its objective rides test_fp32_loss, promoted to a MIN
# veto for TEXT SFT only (per-task, via the multimodal fact) where the fp32 forward closes to
# ~0.999. A
# cross-framework check that is legitimately OOB (the forward cluster on a multimodal port,
# or the banded L6/T3 behavioral gates) is NEVER a hard veto, so a faithful OOB port is not
# zeroed. Membership is by exact test-function base name (suite-wide, byte-identical across
# the 45 tasks); a name absent from a task's suite is simply never matched. Changes NO golden.
_NUMERIC_DISCRIMINATORS = frozenset({
    "test_ppo_update_units",                # PPO: candidate ppo.ppo_utils update math
    "test_trainer_get_batch_loss_metrics",  # DPO: candidate dpo.trainer end-to-end objective
    "test_trainer_orpo_loss",               # DPO: isolated orpo objective
    "test_trainer_simpo_loss",              # DPO: isolated simpo objective
    "test_trainer_get_batch_logps",         # DPO: label-mask + per-example log-prob math
    "test_trainer_concatenated_forward",    # DPO: chosen/rejected split + log-probs
})
# Behavioral has NO fp32-immune discriminator: its only sharp gates (L6 test_validation_loss,
# T3 test_train_run_end_to_end) are CROSS-FRAMEWORK banded comparisons that are legitimately
# OOB for faithful multimodal ports, so they cannot be hard-min vetoes without zeroing a
# faithful port. Behavioral therefore stays a flat power mean (empty veto set); a non-dodgeable
# behavioral discriminator would require the self-referential recompute keystone (deferred).
_BEHAVIORAL_DISCRIMINATORS: frozenset[str] = frozenset()
_DISCRIMINATORS = {"numeric": _NUMERIC_DISCRIMINATORS, "behavioral": _BEHAVIORAL_DISCRIMINATORS}
# Spec binary tally: {stage: {passed,failed,skipped,xfailed,error}}.
_TALLY: dict[str, dict[str, int]] = {}
# Scored stages: {stage: {nodeid: score in [0,1]}}. A collected numeric/behavioral test
# ALWAYS lands here (pass -> its recorded similarity; fail/error/no-record -> 0.0); a
# deselected test never reaches logreport, so it stays out of numerator AND denominator.
_SCORES: dict[str, dict[str, float]] = {}
# Coarse per-stage counts for the retired-but-kept ``overall`` diagnostic only.
_SCORE_COUNTS: dict[str, dict[str, int]] = {}
# ``overall`` is a diagnostic flag ONLY (the reward no longer depends on it): a
# correct fp32 port lands at the ≈0.999 noise ceiling, so "all scores at ceiling"
# uses a lenient diagnostic bound. Nothing in the reward compares against this.
_NOISE_CEILING_DIAG = 0.99


def _stage_of(nodeid: str) -> str | None:
    return _STAGE_OF.get(Path(nodeid.split("::", 1)[0]).stem)


def _test_func_name(nodeid: str) -> str:
    """``path::TestCls::test_foo[param]`` / bare ``path::test_foo`` -> ``test_foo``."""
    return nodeid.rsplit("::", 1)[-1].split("[", 1)[0]


def _read_score(report) -> float:  # noqa: ANN001
    """The [0,1] similarity a scored test recorded via ``record_property('score', s)``.

    Missing / non-numeric / non-finite -> 0.0 (a missing record defaults to 0.0);
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


def pytest_runtest_logreport(report):  # noqa: ANN001 - pytest hook
    stage = _stage_of(report.nodeid)
    if stage is None:
        return
    # A scored-stage test that is a pure gate (no continuous fidelity signal) is
    # re-routed to the binary tally so a PASS banks no unearned credit in the power mean;
    # a hard-assert FAIL still docks the reward via spec_pass_rate.
    if stage in _SCORED_STAGES and _test_func_name(report.nodeid) in _GATE_TEST_NAMES:
        stage = "spec"
    if stage not in _SCORED_STAGES:  # Spec: unchanged binary tally.
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
        return

    # Numeric / Behavioral: file a continuous similarity for this nodeid.
    scores = _SCORES.setdefault(stage, {})
    cnt = _SCORE_COUNTS.setdefault(stage, {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0})
    if report.when == "call":
        if getattr(report, "wasxfail", None) is not None:
            # No xfails by design in the scored stages (uses deselect, not xfail);
            # if one appears, keep it out of the mean (record nothing).
            cnt["xfailed"] += 1
        elif report.passed:
            cnt["passed"] += 1
            scores[report.nodeid] = _read_score(report)
        elif report.failed:  # assertion / crash / torch-leak in the test body -> 0.0
            cnt["failed"] += 1
            scores[report.nodeid] = 0.0
        elif report.skipped:
            # No grade-time skips by design; keep out of the mean if one occurs.
            cnt["skipped"] += 1
    elif report.failed:  # setup/teardown failure -> 0.0 for this test
        cnt["error"] += 1
        scores[report.nodeid] = 0.0
    elif report.skipped and report.when == "setup":
        cnt["skipped"] += 1


def _effective_discriminators(stage: str, facts: dict) -> frozenset[str]:
    """Per-task discriminator set for a scored dimension (Lever B).

    Base = the recon/unit objective checks that sit at ~0.999 on ANY faithful port. SFT owns no
    standalone torch-grounded objective symbol, so on a TEXT SFT port -- where the fp32 cross-
    framework forward closes to the ~0.999 ceiling -- the fp32 forward loss (test_fp32_loss) is
    added as the objective MIN veto. On a MULTIMODAL SFT port the real fp32 forward runs through
    the vision tower and is legitimately OOB, so it stays a soft cluster member (never a hard
    veto), exactly like the banded L6/T3 behavioral gates."""
    base = _DISCRIMINATORS.get(stage, frozenset())
    if stage == "numeric" and str(facts.get("method")).lower() == "sft" and not facts.get("multimodal"):
        return base | {"test_fp32_loss"}
    return base


def _dimension_score(stage: str, scores: dict, facts: dict) -> float:
    # Lever B two-tier aggregation. The forward-correlated checks (everything NOT in the
    # stage's discriminator set) are collapsed into ONE soft vote via the low-tail-sensitive
    # power mean (p<1): a single wholly-broken forward check still bites super-linearly
    # WITHIN the cluster, a uniform faithful set (~0.999) is unchanged (p-invariance), and a
    # legitimately-OOB cross-framework cluster keeps partial credit instead of being zeroed.
    # Each genuine objective/run DISCRIMINATOR then enters as its own hard MIN vote, so a
    # correct forward can no longer dilute a failing objective upward (count-dominance). With
    # an empty discriminator set (behavioral) this reduces EXACTLY to the flat power mean.
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
    # Spec: binary counts + pass_rate (excludes skipped/xfailed).
    st = _TALLY.get("spec", {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0})
    spec_ran = st["passed"] + st["failed"] + st["error"]
    spec_pass_rate = round(st["passed"] / spec_ran, 6) if spec_ran else 0.0
    spec_block = {**st, "pass_rate": spec_pass_rate}

    facts = _golden_facts(_TEST_DIR / "golden")

    def scored(name: str) -> tuple[dict, float]:
        scores = _SCORES.get(name, {})
        mean = _dimension_score(name, scores, facts)
        return {"scores": dict(sorted(scores.items())), "mean": mean, "n": len(scores)}, mean

    numeric_block, numeric_score = scored("numeric")
    behavioral_block, behavior_score = scored("behavioral")

    # ``overall`` is a coarse diagnostic ONLY (reward is independent of it):
    # Spec clean AND every recorded numeric/behavioral score at the noise ceiling.
    all_scores = list(_SCORES.get("numeric", {}).values()) + list(_SCORES.get("behavioral", {}).values())
    spec_clean = st["failed"] == 0 and st["error"] == 0
    overall = "PASS" if (spec_clean and all(s >= _NOISE_CEILING_DIAG for s in all_scores)) else "FAIL"

    summary = {
        "spec": spec_block,
        "numeric": numeric_block,
        "behavioral": behavioral_block,
        "spec_pass_rate": spec_pass_rate,
        "numeric_score": numeric_score,
        "behavior_score": behavior_score,
        "overall": overall,
    }
    # Coverage vector: which required goldens the freeze emitted for this method,
    # so a coverage regression is visible in the summary itself. Added AFTER the reduction
    # fields so it never enters them.
    summary["coverage"] = golden_manifest.coverage_vector(_TEST_DIR / "golden", facts.get("method"))

    # Never write under tasks/: real grading pins T2J_SUMMARY_DIR to a location outside
    # the (often bind-mounted) source tree -- Harbor test.sh points it at /logs/verifier.
    # The fallback is a temp dir so an ad-hoc
    # ``pytest tasks/*/tests`` still emits a summary without polluting tasks/.
    out_dir = Path(os.environ.get("T2J_SUMMARY_DIR") or Path(tempfile.gettempdir()) / "t2j-grade")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if not os.environ.get("T2J_SUMMARY_DIR"):
        print(f"\n[t2j] eval_summary.json written to {out_path} (set T2J_SUMMARY_DIR to relocate)")
