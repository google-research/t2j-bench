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

"""Behavioral stage (generated): training-loop and framework-invariant equivalence
(TEST_PLAN §7, §8).

The numeric stage pins single-forward / step-0 math; this stage grades what only a
LOOP reveals: the training step actually descends (its optimizer is AdamW, its base is
frozen, its objective is preference-sensitive), and the forward obeys the framework
invariants a memorized ``(batch, params)`` pair cannot fake.

Continuous-similarity scoring (TEST_PLAN §4.2-§4.4, §10.4): every test reports a [0,1]
similarity via ``record_property("score", s)``; the surviving ``assert``s are STRUCTURAL
preconditions (the probe is supported / well-shaped), a failing one -> the test fails ->
conftest records ``0.0``. Two families of score:

  * **Continuous parity** (step-0 loss, padding/batch logit parity, softmax
    normalization, held-out + train-run loss): the recorded score IS the similarity, so a
    faithful port lands at the ≈0.999 noise ceiling (§4.7), not a boolean pass.
  * **{0,1} structural discriminators** (§4.7 exception -- an anti-hack boundary, not a
    parity tolerance): the golden-free / self-consistency invariants (train-loss-trend,
    AdamW-vs-SGD, base-frozen, DPO-preference, determinism, strict-causal). These record
    ``1.0`` when the invariant holds and ``0.0`` (test failure) when it does not.

All full training runs are non-deterministic across frameworks (shuffle + reduction
order, PyTorch-Philox vs JAX-threefry dropout), so nothing here is graded on absolute
magnitude beyond the RNG-independent step-0 loss (§8). Every current instance is LoRA,
which collapses the trajectory gates to **step-0 value-exact + finite + no-runaway**
(§7.2). Applicability + stage tagging come entirely from conftest (``_APPLICABLE``
predicates, ``_STAGE_OF`` by filename); a check that cannot apply is deselected there,
never skipped, so the reward denominator stays exactly the applicable set."""

from __future__ import annotations

import numpy as np
import pytest

from _t2j import similarity

def _finite(values) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(arr.size) and bool(np.all(np.isfinite(arr)))

def _avg(scores) -> float:
    """Low-tail-sensitive mean of a flat list of per-element [0,1] similarities (§4.7,
    P0.2b: power mean, p<1), matching similarity.score_mean / conftest._mean. A uniform
    faithful set is unchanged (power means are p-invariant on constant inputs, so honesty
    is not punished) but a single broken element within a battery/probe bites super-
    linearly (empty -> 0.0)."""
    scores = list(scores)
    return similarity.power_mean(scores) if scores else 0.0

def _thirds(losses) -> tuple[float, float]:
    """(first_third_mean, last_third_mean) of a loss curve (TEST_PLAN §7.1). With the frozen
    K=3-step replay each third is one step, so this is (loss[0], loss[-1]); written as thirds
    so it reads the same if K ever grows."""
    n = len(losses)
    k = max(1, n // 3)
    return float(np.mean(losses[:k])), float(np.mean(losses[-k:]))

def _no_runaway_band(first: float) -> float:
    """Calibration-free loop-divergence guard, replacing the retired per-task
    meta['train_trend_tol']. A genuine blow-up leaves the curve O(1)+ above its start (or
    NaN); a full relative doubling (or +1.0 for a near-zero curve) is far past any faithful
    fp32 run yet well below a real runaway. Used where the trend is NOT asserted down -- the
    PPO RL objective (non-monotone / budget-chaotic, §5, §7.4) and the step-0-parity
    trajectory probes, whose scored quantity is the step-0 loss, not the trend."""
    return max(1.0, abs(first))

# --------------------------------------------------------------------------- #
# §7.1 — does train.py run successfully?  ({0,1} discriminator, §4.7)
# --------------------------------------------------------------------------- #
def test_train_loss_trend(cand, golden, record_property):
    """Golden-free {0,1} discriminator (TEST_PLAN §7.1, §4.7): the candidate's reconstructed
    K-step training loop terminates and produces a finite loss curve that TRENDS DOWN. Runs
    under the sandbox's hard wall-clock kill (meta['train_timeout_s']), so a port that never
    finishes -- no jit, a per-step recompile, a runaway loop -- FAILS here (records 0.0)
    instead of hanging the suite.

    For SFT/DPO the supervised loss must descend over the window -- ``last_third_mean <
    first_third_mean`` (§7.1), scored {0,1} with NO tolerance constant: in fp32 every
    reference descends by >=7e-3 over the K=3 steps, orders above the fp32 noise floor. PPO
    is the documented exception (§5, §7.4): its K-step objective is the non-monotone,
    budget-chaotic RL loss, whose real improvement is a reward-INCREASE measured by the e2e
    T2 run (§7.4), not a K-step loss decrease -- so PPO asserts only finite + no-runaway
    (calibration-free ``_no_runaway_band``). Records 1.0 when the contract holds."""
    art = cand.train_trajectory()
    assert art.get("supported"), {"error": "candidate exposed no runnable training loop", **art}
    losses = [float(x) for x in art.get("losses", [])]
    assert _finite(losses), {"error": "non-finite / empty training loss curve", "losses": losses}
    first, last = _thirds(losses)
    if golden.training_method == "ppo":
        assert last <= first + _no_runaway_band(first), {
            "error": "PPO training loss ran away (loop diverges)",
            "first_third": first, "last_third": last, "losses": losses,
        }
    else:
        assert last < first, {
            "error": "training loss does not trend down (last_third_mean >= first_third_mean)",
            "first_third": first, "last_third": last, "losses": losses,
        }
    record_property("score", 1.0)

# --------------------------------------------------------------------------- #
# §7.2 — does training BEHAVE the same? (trend, not magnitude)
# --------------------------------------------------------------------------- #
def test_train_trajectory(cand, golden, record_property):
    """Reconstructed optax-AdamW loop vs the deterministic torch-AdamW golden
    (train_trajectory.json), collapsed to the LoRA contract (§7.2): the curve is finite and
    no-runaway (structural preconditions), and step-0 loss is graded CONTINUOUSLY against the
    golden by ``S_scalar`` (RNG-independent -- B=0 at init). Deselected where the golden froze
    no train_trajectory."""
    assert golden.train_trajectory, "golden froze no train_trajectory (should have been deselected)"
    ref = [float(s["loss"]) for s in golden.train_trajectory["trajectory"]]
    art = cand.train_trajectory()
    assert art.get("supported"), {"error": "candidate exposed no runnable training loop", **art}
    losses = [float(x) for x in art.get("losses", [])]
    assert _finite(losses), {"error": "non-finite / empty training loss curve", "losses": losses}
    assert len(losses) == len(ref), {"n_candidate": len(losses), "n_reference": len(ref)}
    first, last = _thirds(losses)
    assert last <= first + _no_runaway_band(first), {
        "error": "training loss ran away", "first_third": first, "last_third": last,
    }
    record_property("score", similarity.score_scalar(ref[0], losses[0]))

def test_loss_trajectory(cand, golden, record_property):
    """Bare-SGD replay (no optimizer state) vs loss_trajectory.json, same LoRA step-0
    collapse (§7.2): finite + no-runaway preconditions, step-0 loss scored by ``S_scalar``.
    Largely subsumed by ``test_train_trajectory``; kept as the optimizer-free numeric
    backstop. ``xfail`` on bloom, whose un-normalized-SGD fp32 replay is numerically fragile
    (an xfail stays out of the behavioral mean)."""
    ref = [float(s["loss"]) for s in golden.loss_trajectory["trajectory"]]
    art = cand.loss_trajectory()
    is_bloom = "bloom" in str(golden.meta.get("repo_config", {}).get("model_name_or_path", "")).lower()
    try:
        assert art.get("supported"), {"error": "candidate exposed no loss trajectory", **art}
        losses = [float(s["loss"]) for s in (art.get("trajectory") or [])]
        assert _finite(losses), {"error": "non-finite / empty loss trajectory", "losses": losses}
        assert len(losses) == len(ref), {"n_candidate": len(losses), "n_reference": len(ref)}
        # Bare-SGD (no optimizer state) at LR=1e-6 is numerically fragile (see docstring):
        # the fp32 reference can legitimately drift (measured: gemma3 torch +0.18 over the
        # window) while the bf16 candidate adds ~1 ULP of rounding noise on the near-flat
        # curve. Grade no-runaway RELATIVE to the reference's own drift plus a bf16 slack, so
        # a faithful port that tracks the reference cannot fail while a genuine O(1) blow-up
        # still trips.
        ref_drift = max(ref[-1] - ref[0], 0.0)
        band = ref_drift + _no_runaway_band(losses[0])
        assert losses[-1] <= losses[0] + band, {
            "error": "loss trajectory ran away", "first": losses[0], "last": losses[-1],
            "ref_drift": ref_drift, "band": band,
        }
    except AssertionError:
        if is_bloom:
            pytest.xfail("bloom un-normalized-SGD fp32 fragility (§7.2); subsumed by test_train_trajectory")
        raise
    record_property("score", similarity.score_scalar(ref[0], losses[0]))

def test_optimizer_identity(cand, golden, record_property):
    """Golden-free AdamW-vs-SGD {0,1} discriminator (TEST_PLAN §7.2, §13.1, §4.7). AdamW's
    first step is sign-normalized -- ``|dP| ~ lr0`` regardless of ``|g|`` -- so over the
    gradient-active adapter elements ``corr(|dP|,|g|) ~ 0`` and ``median(|dP|)/lr0 ~ 1``;
    a bare-SGD ``|dP| = c*lr0*|g|`` would give ``corr ~ 1`` and a median far below lr0.
    Scale/clip-invariant (the reference chains clip_by_global_norm). The near-flat LoRA
    curve cannot see this, so it is graded directly on the one-step update. Records 1.0 when
    the update carries the AdamW signature, 0.0 otherwise."""
    probe = cand.optimizer_probe()
    assert probe.get("supported"), {"error": "optimizer probe unsupported", **probe}
    corr = probe.get("corr_abs_dp_abs_g")
    med = probe.get("median_over_lr0")
    assert corr is not None and med is not None, {"error": "probe missing corr/median", **probe}
    assert float(corr) < 0.9, {
        "error": "|dP| correlates with |g| like SGD, not AdamW",
        "corr_abs_dp_abs_g": float(corr), "expected": "< 0.9",
        "lr0": probe.get("lr0"), "num_active": probe.get("num_active"),
    }
    assert 0.25 <= float(med) <= 4.0, {
        "error": "median|dP|/lr0 outside the AdamW plateau band",
        "median_over_lr0": float(med), "expected": "0.25..4.0",
    }
    record_property("score", 1.0)

def test_lora_base_frozen(cand, golden, record_property):
    """After one reconstructed step, every non-adapter (base) leaf is byte-unchanged
    (``base_max_delta <= 1e-12``) and >=1 adapter leaf moved (TEST_PLAN §7.2, §13.1). A {0,1}
    structural discriminator (§4.7): the base-freeze is byte-exact, not a parity tolerance.
    Catches a ``lora``-in-name-only repo / wrong freeze mask that the near-flat LoRA loss
    curve hides -- a full-finetune-as-LoRA lands inside the trajectory band but moves the base
    here. Records 1.0 when the base is frozen and an adapter moved."""
    probe = cand.optimizer_probe()
    assert probe.get("supported"), {"error": "optimizer probe unsupported", **probe}
    base_max_delta = float(probe.get("base_max_delta", 1.0))
    assert base_max_delta <= 1e-12, {
        "error": "a base (non-adapter) parameter moved -- base is not frozen",
        "base_max_delta": base_max_delta, "expected": "<= 1e-12",
        "base_moved_leaves": probe.get("base_moved_leaves"),
        "base_leaf_count": probe.get("base_leaf_count"),
    }
    assert probe.get("adapter_moved"), {
        "error": "no adapter leaf moved -- the step trained nothing",
        "adapter_max_delta": probe.get("adapter_max_delta"),
    }
    record_property("score", 1.0)

# --------------------------------------------------------------------------- #
# §7.2 — preference-objective sensitivity (DPO/ORPO/SimPO only, {0,1} discriminator)
# --------------------------------------------------------------------------- #
def test_dpo_objective(cand, golden, record_property):
    """DPO-only preference-sensitivity {0,1} discriminator (TEST_PLAN §7.2, §4.7): swapping the
    chosen (first half) and rejected (second half) rows must change the step-0 loss by
    ``> 1e-2``. Preference objectives (DPO/ORPO/SimPO) are asymmetric in (chosen, rejected); a
    plain causal-LM loss summed over all tokens is swap-invariant, so this catches a DPO repo
    that silently trains a causal objective -- which the trajectory curve can miss when a
    small-beta ORPO is dominated by its SFT term. Records 1.0 when the objective is
    preference-sensitive, 0.0 otherwise."""
    probe = cand.dpo_objective()
    assert probe.get("supported"), {"error": "DPO objective probe unsupported", **probe}
    delta = float(probe.get("delta", 0.0))
    assert delta > 1e-2, {
        "error": "swapping chosen/rejected barely changed the loss -- objective is not preference-sensitive",
        "delta": delta, "loss": probe.get("loss"), "loss_swapped": probe.get("loss_swapped"),
    }
    record_property("score", 1.0)

# --------------------------------------------------------------------------- #
# §7.5 — model invariants (golden-free, memorization-proof; deselect on MM)
# --------------------------------------------------------------------------- #
def test_determinism(cand, golden, record_property):
    """Two identical forwards over the frozen batch are bitwise identical (TEST_PLAN §7.5) --
    a {0,1} self-consistency discriminator (§4.7): a nondeterministic forward (uninitialized
    dropout, data-dependent control flow) cannot be memorized away. Records 1.0 when the two
    forwards agree bitwise."""
    probe = cand.determinism()
    assert probe.get("supported"), {"error": "determinism probe unsupported", **probe}
    assert probe.get("identical"), {
        "error": "two identical forwards disagreed", "max_abs_diff": probe.get("max_abs_diff"),
    }
    record_property("score", 1.0)

def test_causal_masking(cand, golden, record_property):
    """Perturbing the token at position t changes logits at positions ``>= t`` but leaves
    positions ``< t`` bitwise unchanged (TEST_PLAN §7.5) -- a {0,1} strict-causal-attention
    discriminator (§4.7). Text tasks only; the bidirectional prefix-LM reference (paligemma)
    is multimodal and deselects. Records 1.0 when attention is strictly causal."""
    probe = cand.self_consistency()
    assert probe.get("supported"), {"error": "self-consistency probe unsupported", **probe}
    assert probe.get("causal_expected", True), {"error": "instance is not strictly causal (should have deselected)"}
    before = float(probe["causal_before_max_abs_diff"])
    after = float(probe["causal_after_max_abs_diff"])
    assert before == 0.0, {"error": "positions before t changed -- attention is not causal", "before_max_abs_diff": before}
    assert after > 0.0, {"error": "positions at/after t did not change -- perturbation had no effect", "after_max_abs_diff": after}
    record_property("score", 1.0)

def test_padding_invariance(cand, golden, record_property):
    """Masked padding does not leak into the real positions (TEST_PLAN §7.5): (a) two
    different pad-token fills under the same mask give bitwise-identical real-position logits
    (a byte-exact structural precondition), and (b) a right-padded forward matches the
    unpadded forward over the real positions -- scored CONTINUOUSLY by ``S_dist`` (the
    distribution parity, no threshold)."""
    probe = cand.self_consistency()
    assert probe.get("supported"), {"error": "self-consistency probe unsupported", **probe}
    pad_samelen = float(probe["pad_samelen_max_abs_diff"])
    assert pad_samelen == 0.0, {
        "error": "masked pad CONTENT changed the real-position logits", "pad_samelen_max_abs_diff": pad_samelen,
    }
    record_property(
        "score",
        similarity.score_logits(
            np.asarray(probe["pad_unpadded_ref"]), np.asarray(probe["pad_difflen_unpadded"])
        ),
    )

def test_softmax_normalization(cand, golden, record_property):
    """The forward logits form a valid distribution: ``Σ softmax(logits) ≈ 1`` over the
    vocabulary at every position (TEST_PLAN §7.5). Scored CONTINUOUSLY as
    ``1 - clamp(max_abs_dev, 0, 1)`` -- a faithful softmax sits at the fp32 noise ceiling
    (dev ~1e-6 -> score ≈1), a broken normalization drops it."""
    probe = cand.self_consistency()
    assert probe.get("supported"), {"error": "self-consistency probe unsupported", **probe}
    dev = float(probe["softmax_max_abs_dev"])
    record_property("score", 1.0 - min(1.0, max(0.0, dev)))

def test_batch_independence(cand, golden, record_property):
    """Row i of a multi-row batch produces the same logits as that row run alone (TEST_PLAN
    §7.5): no cross-row leakage through attention/normalization. Scored CONTINUOUSLY as the
    mean of the two rows' ``S_dist`` (single vs batched)."""
    probe = cand.self_consistency()
    assert probe.get("supported"), {"error": "self-consistency probe unsupported", **probe}
    s0 = similarity.score_logits(np.asarray(probe["single1"]), np.asarray(probe["batch_row0"]))
    s1 = similarity.score_logits(np.asarray(probe["single2"]), np.asarray(probe["batch_row1"]))
    record_property("score", _avg([s0, s1]))

# --------------------------------------------------------------------------- #
# §7.7 — held-out validation-loss parity (L6, the top of the equivalence lattice)
# --------------------------------------------------------------------------- #
def test_validation_loss(cand, golden, record_property):
    """Held-out validation-loss parity (TEST_PLAN §2.5, §7.7 — L6), scored CONTINUOUSLY.

    Runs IN BAND: ``cand.validation_loss()`` executes the candidate's OWN training entrypoint
    (``train.run_{sft,dpo,ppo}``) end to end inside the torch-blocked sandbox, driven purely
    from its ``config.py`` defaults (which mirror ``torch_orig`` value for value) on the frozen
    ``(val_size, seed)`` split baked into the verifier image, and reports the post-training
    held-out loss ``V_jax``. That real run is SHARED with T3 (``test_train_run_end_to_end``) --
    one run, cached, two gates -- and is bounded by ``train_timeout_s`` (calibrated to the
    PyTorch reference runtime): a port that cannot reproduce the reference within that budget
    returns ``supported=False`` here and FAILS (records 0.0).

    Scored by ``similarity.score_scalar_banded(V_torch, V_jax, tol)`` (§4.7): a dead-zone of the
    frozen faithful tolerance ``tol`` (``clamp(factor*torch_self_std, floor, cap)`` from torch's
    OWN run-to-run variance (O1) -- the oracle's oscillation around itself, NOT the ref_jax-vs-torch
    gap) holds an honest port at 1.0, then the score ramps sharply to 0 for a real
    drift -- so a PPO neg-mean-reward that sign-flips around 0 (raw S_scalar -> 0.0) is no longer
    punished, while a wide-basin mediocrity no longer banks partial credit. ``tol`` absent (golden
    predates the re-freeze) falls back to raw ``S_scalar``. Deselects (conftest) unless
    ``golden/validation_loss.json`` exists."""
    ref = golden.validation_loss
    assert ref, {"error": "validation_loss.json absent (should have deselected)"}
    result = cand.validation_loss()
    assert result.get("supported"), {
        "error": "in-band validation run unsupported (crashed or exceeded train_timeout_s)",
        **result,
    }
    assert result.get("V_jax") is not None, {
        "error": "in-band run produced no V_jax (held-out eval loss)", "keys": sorted(result),
    }
    v_jax = float(result["V_jax"])
    v_torch = float(ref["V_torch"])
    assert np.isfinite(v_jax), {"error": "V_jax is not finite", "V_jax": result["V_jax"]}
    # Tolerance-banded parity (§4.7): a dead-zone of the frozen faithful tolerance ``tol``
    # (``clamp(factor*torch_self_std, floor, cap)`` from torch's OWN run-to-run variance (O1),
    # NOT the ref_jax-vs-torch gap) scores an
    # honest port 1.0, then the score ramps sharply to 0 for a real drift -- so a PPO
    # neg-mean-reward that sign-flips around 0 (raw S_scalar -> 0.0) is no longer punished,
    # while mediocrity (S_scalar's 0.67@2x wide basin) no longer banks partial credit.
    # ``tol`` absent (golden predates the re-freeze) -> falls back to raw S_scalar.
    record_property("score", similarity.score_scalar_banded(v_torch, v_jax, ref.get("tol")))

# --------------------------------------------------------------------------- #
# §7.4 — end-to-end train-run parity (T3, the tiered whole-run gate)
# --------------------------------------------------------------------------- #
@pytest.mark.train_run

def test_train_run_end_to_end(cand, golden, record_property):
    """End-to-end training-run parity (TEST_PLAN §7.4 -- T3), the tiered sibling of the L6
    held-out gate (§7.7), scored CONTINUOUSLY. Runs IN BAND via ``cand.train_run()`` -- the
    SAME cached real training run as the L6 gate: the candidate's own ``train.run_{sft,dpo,ppo}``
    trained end to end from ``config.py`` on the frozen ``(val_size, seed)`` split baked into
    the verifier image, reporting its whole-run summary (``train_loss``, ``eval_loss``,
    ``pre_eval_loss``). Bounded by ``train_timeout_s`` (pinned to the PyTorch reference
    runtime): a run that exceeds it returns ``supported=False`` and FAILS (records 0.0).

    STRUCTURAL preconditions: the run is supported, emits the required loss keys, and every
    loss is finite. Scored by ``similarity.score_scalar_banded(train loss vs torch, train_tol)``
    (§4.7): a dead-zone of the frozen faithful ``train_tol``
    (``clamp(factor*torch_self_std, floor, cap)`` from torch's OWN run-to-run variance (O1),
    NOT the ref_jax-vs-torch gap) holds an honest port at 1.0
    and the score ramps sharply to 0 beyond it. A
    non-learning or out-of-range run lands its final train loss far from torch and scores low,
    so the banded parity subsumes the old T1/T2 guards.

    ``@pytest.mark.train_run`` + conftest deselect unless ``golden/train_run.json`` exists."""
    ref = golden.train_run
    assert ref, {"error": "train_run.json absent (should have deselected)"}
    result = cand.train_run()
    assert result.get("supported"), {
        "error": "in-band train run unsupported (crashed or exceeded train_timeout_s)",
        **result,
    }
    for key in ("train_loss", "eval_loss", "pre_eval_loss"):
        assert result.get(key) is not None, {
            "error": f"in-band run produced no {key}", "keys": sorted(result),
        }
    train = float(result["train_loss"])
    evl = float(result["eval_loss"])
    pre = float(result["pre_eval_loss"])
    assert np.isfinite(train) and np.isfinite(evl) and np.isfinite(pre), {
        "error": "non-finite train/eval loss", "train_loss": result["train_loss"],
        "eval_loss": result["eval_loss"], "pre_eval_loss": result["pre_eval_loss"],
    }
    train_torch = float(ref["torch"]["train_loss"])
    # Tolerance-banded T3 parity (see test_validation_loss): dead-zone ``train_tol`` frozen
    # from torch's OWN run-to-run variance (O1), NOT the ref_jax gap; sharp ramp beyond it.
    # ``train_tol`` absent -> raw S_scalar.
    record_property("score", similarity.score_scalar_banded(train_torch, train, ref.get("train_tol")))
