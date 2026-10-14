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

"""Numeric stage (generated): math parity vs goldens + gradient authenticity.

Continuous-similarity scoring: every test computes a
similarity in **[0,1]** against the frozen golden and reports it via
``record_property("score", s)`` -- there is NO threshold and NO parity assertion. The
only surviving ``assert``s are STRUCTURAL preconditions (the candidate exposed the
artifact / a well-formed shape); a failing precondition, a crash, or a torch-leak makes
the test FAIL, which conftest records as ``0.0``. The reward is the spec-gated mean of
these similarities, so a weak module pulls its own score down proportionally instead of
tripping a boolean gate. With fp32 on both stacks + ``matmul=highest`` a faithful port
scores ≈0.999 (the noise ceiling), not 1.0.

The retired ``compare.check_logits``/``check_array``/``check_lr_schedule`` threshold gates
and the per-instance ``golden.thresholds`` (cosine_floor / max_kl / argmax_margin / abs_tol)
are gone from this stage; the metric toolkit now lives in ``_t2j.similarity``. Both sides
are still upcast to float32 (float64 for the LR schedule / finite difference) inside the
similarity primitives.
"""

from __future__ import annotations

import numpy as np

from _t2j import compare
from _t2j import similarity

def _valid_positions(logits, golden):
    """Restrict logits to non-padded positions before the distribution score.

    Padded-position logits are meaningless (the pad token is masked out of the loss) yet
    would dominate a per-position distribution mean -- especially for pairwise DPO /
    seq2seq PPO batches padded to a large length. The candidate and reference agree on the
    real tokens; comparing padding only measures bf16 noise on garbage.
    """
    logits = np.asarray(logits)
    mask = golden.inputs.get("attention_mask") if hasattr(golden, "inputs") else None
    if mask is None or logits.ndim < 3:
        return logits
    sel = np.asarray(mask).reshape(-1) > 0
    flat = logits.reshape(-1, logits.shape[-1])
    return flat[sel] if sel.shape[0] == flat.shape[0] else logits

def _valid_positions_by_mask(logits, mask):
    """``_valid_positions`` against an EXPLICIT mask (each battery variant carries its own
    attention_mask, not the canonical golden's)."""
    logits = np.asarray(logits)
    if mask is None or logits.ndim < 3:
        return logits
    sel = np.asarray(mask).reshape(-1) > 0
    flat = logits.reshape(-1, logits.shape[-1])
    return flat[sel] if sel.shape[0] == flat.shape[0] else logits

def _avg(scores) -> float:
    """Low-tail-sensitive mean of a flat list of per-element [0,1] similarities
    (power mean, p<1), matching similarity.score_mean / conftest._mean. A uniform
    faithful set is unchanged (power means are p-invariant on constant inputs, so honesty
    is not punished) but a single broken element within a battery/probe bites super-
    linearly (empty -> 0.0)."""
    scores = list(scores)
    return similarity.power_mean(scores) if scores else 0.0

def test_forward_logits(cand, golden, record_property):
    """bf16-native forward-logit distribution parity. ``S_dist`` (1 -
    JSD/ln2 on the softmax) over the non-padded positions -- gauge-invariant, sees a
    temperature change, no KL cap / argmax margin."""
    ref = _valid_positions(golden.forward["logits"], golden)
    got = _valid_positions(cand.forward()["logits"], golden)
    record_property("score", similarity.score_logits(ref, got))

def test_forward_logits_battery(cand, golden, record_property):
    """Input-space generality of the bf16-native forward logits.

    ``test_forward_logits`` pins ONE fixed probe (the canonical batch). This re-runs the
    forward-logit parity over a battery of shape/length/batch/mask variants the reference
    froze (short_len1, long_maxctx, batch_multi, padded_ragged), catching modeling.py bugs
    invisible to a single probe -- RoPE at a long position, a batch that collapses onto row
    0, a ragged attention mask, the length-1 edge case. Each variant scores by ``S_dist``
    over its OWN non-padded positions (the per-row JSD mean is inherently length-robust, so
    no cross-length floor is needed); for SFT the shifted-CE loss rides along (``S_scalar``)
    where a valid span exists. The test score is the mean over all variant terms. Deselected
    unless the instance froze a battery (text-only; see conftest ``_APPLICABLE``); where it
    applies, a candidate that cannot run it FAILS (score 0.0)."""
    assert golden.battery, "golden froze no input battery (should have been deselected)"
    got = cand.battery()
    assert got.get("supported"), {
        "error": "candidate produced no input battery (text-only forward_on required)",
        "reason": got.get("reason"),
    }
    got_variants = got.get("variants") or {}
    scores: list[float] = []
    for name in sorted(golden.battery):
        var = golden.battery[name]
        ref_logits = var.get("forward", {}).get("logits")
        if ref_logits is None:
            continue
        cv = got_variants.get(name)
        mask = var["inputs"].get("attention_mask")
        if not isinstance(cv, dict) or cv.get("logits") is None:
            scores.append(0.0)  # candidate produced no logits for this variant
        else:
            ref = _valid_positions_by_mask(ref_logits, mask)
            gc = _valid_positions_by_mask(cv["logits"], mask)
            scores.append(similarity.score_logits(ref, gc))
        ref_loss = var["forward"].get("loss")
        if ref_loss is not None:
            got_loss = cv.get("loss") if isinstance(cv, dict) else None
            scores.append(
                similarity.score_scalar(float(ref_loss), float(got_loss))
                if got_loss is not None else 0.0
            )
    record_property("score", _avg(scores))

def test_fp32_forward_battery(cand, golden, record_property):
    """FP32 input-space generality of the forward logits across the battery.

    The fp32 analogue of ``test_forward_logits_battery``: identical fp32 weights on both
    sides (tighter noise floor than bf16), so a shape-specific reduction-order / scaling bug
    the bf16-native tolerance masks shows here first. Each variant scores by ``S_dist`` over
    its own non-padded positions; the fp32 shifted-CE loss (SFT only) rides along via
    ``S_scalar``. Deselection / failure semantics as for the bf16 battery."""
    assert golden.battery, "golden froze no input battery (should have been deselected)"
    got = cand.battery()
    assert got.get("supported"), {
        "error": "candidate produced no input battery (dtype-following modules required)",
        "reason": got.get("reason"),
    }
    got_variants = got.get("variants") or {}
    scores: list[float] = []
    for name in sorted(golden.battery):
        var = golden.battery[name]
        ref_logits = var.get("fp32_forward", {}).get("logits")
        if ref_logits is None:
            continue
        cv = got_variants.get(name)
        mask = var["inputs"].get("attention_mask")
        if not isinstance(cv, dict) or cv.get("fp32_logits") is None:
            scores.append(0.0)  # candidate produced no fp32 logits for this variant
        else:
            ref = _valid_positions_by_mask(ref_logits, mask)
            gc = _valid_positions_by_mask(cv["fp32_logits"], mask)
            scores.append(similarity.score_logits(ref, gc))
        ref_loss = var["fp32_forward"].get("loss")
        if ref_loss is not None:
            got_loss = cv.get("fp32_loss") if isinstance(cv, dict) else None
            scores.append(
                similarity.score_scalar(float(ref_loss), float(got_loss))
                if got_loss is not None else 0.0
            )
    record_property("score", _avg(scores))

def test_fp32_forward(cand, golden, record_property):
    """FP32 exactness gate: identical fp32 weights on both sides, so the
    logits agree to fp32 kernel noise only -- catches reduction-order/dtype/scaling bugs the
    bf16_native tolerance masks. Scored by ``S_dist`` over the non-padded positions. Requires
    dtype-following modules (upcasting params -> genuine fp32 forward); a candidate that
    exposes no fp32 forward FAILS (score 0.0). Deselected only for an instance whose golden
    never froze an fp32 forward."""
    got = cand.fp32_forward()
    assert got.get("logits") is not None, {
        "error": "candidate exposes no genuine fp32 forward (dtype-following modules required)",
        "reason": got.get("reason"),
    }
    ref = _valid_positions(golden.fp32_forward["logits"], golden)
    gc = _valid_positions(got["logits"], golden)
    record_property("score", similarity.score_logits(ref, gc))

def test_hidden_states(cand, golden, record_property):
    """Per-layer intermediate hidden-state parity.

    Deselected unless the instance froze hidden states (meta.compare_hidden_states +
    hidden_ arrays in golden/forward.npz). Each layer's activations are scored against the
    frozen reference by ``S_tensor`` (direction AND magnitude) on the non-padded positions,
    and the test score is the per-layer mean -- localizing a forward divergence to a layer
    shows up as that layer's term dropping. A missing/mis-shaped candidate hidden stack fails
    the structural preconditions (score 0.0)."""
    gold_layers = golden.hidden_states()
    assert gold_layers, "golden froze no hidden states (should have been deselected)"
    got_layers = cand.forward().get("hidden_states")
    assert got_layers is not None, "candidate forward exposed no hidden_states"
    assert len(got_layers) == len(gold_layers), {
        "golden_layers": len(gold_layers),
        "candidate_layers": len(got_layers),
    }
    pairs = [
        (_valid_positions(ref, golden), _valid_positions(got, golden))
        for ref, got in zip(gold_layers, got_layers)
    ]
    record_property("score", similarity.score_mean(pairs, similarity.S_tensor))

def test_method_loss(cand, golden, record_property):
    """Method (SFT/DPO/PPO) training-loss parity on the frozen batch (``S_scalar``)."""
    record_property(
        "score", similarity.score_scalar(golden.loss_artifact["loss"], cand.loss()["loss"])
    )

def test_gradient_loss_and_norm(cand, golden, record_property):
    """Step-0 gradient-LOSS parity, scored by ``S_scalar``.

    Under LoRA -- every golden in this benchmark (all 45 grade finetuning_type=lora) -- the
    step-0 loss is RNG-independent (PEFT inits B=0, so the adapter is a no-op at init and
    the loss is the base model's), which is why it can be scored value-exact. The step-0
    grad_norm is ``||grad_B|| ~ (alpha/r)*A`` with ``A`` the RANDOMLY-initialised adapter
    (torch-PEFT RNG vs JAX RNG differ), so it cannot be matched value-exact: it is ONLY a
    structural precondition (finite + positive) and contributes NO parity to the score.
    Gradient CORRECTNESS is covered RNG-invariantly elsewhere -- ``test_gradient_authenticity``
    (analytic vs finite-difference), ``test_gradient_per_param`` (live/dead adapter pattern),
    and forward / loss parity -- so scoring the raw grad_norm here would only punish an honest
    port for its adapter RNG. The full-finetuning path (which alone would score loss + norm
    jointly) has no goldens, so it is asserted unreachable rather than left as silent dead
    code."""
    assert golden.is_lora, {"error": "no full-finetuning goldens exist in this benchmark"}
    grad = cand.gradients()
    gn = float(grad["grad_norm"])
    assert np.isfinite(gn) and gn > 0.0, {"grad_norm": gn}
    record_property("score", similarity.score_scalar(golden.gradients["loss"], grad["loss"]))

# Positions whose (shifted) label is IGNORE_INDEX contribute a meaningless per-token
# log-prob (the PPO formula gathers the log-prob of token 0 there). -100 is the shared
# IGNORE_INDEX across every reference (data.constants.IGNORE_INDEX). Used only to focus
# the token_logprobs comparison on real tokens; it is applied identically to both sides,
# so the exact value never affects fairness -- at worst it includes a few comparable
# extra positions.
_PPO_IGNORE_INDEX = -100

def _ppo_valid_mask(golden, width: int):
    """Boolean ``(batch, width)`` mask of real (attended, non-ignore) label positions,
    aligned to the frozen per-token array width. ``None`` if it cannot be derived (then
    the caller falls back to comparing every position)."""
    inputs = getattr(golden, "inputs", None) or {}
    labels = inputs.get("labels")
    if labels is None:
        return None
    lab = np.asarray(labels)
    if lab.ndim != 2 or lab.shape[1] < 2:
        return None
    shifted = lab[:, 1:]  # PPO gathers on the shifted labels
    mask = shifted != _PPO_IGNORE_INDEX
    attn = inputs.get("attention_mask")
    if attn is not None:
        am = np.asarray(attn)
        if am.ndim == 2 and am.shape == lab.shape:
            mask = mask & (am[:, 1:] > 0)
    if mask.shape[1] < width:  # normalize_ppo_time_axis only ever trims the tail
        return None
    mask = mask[:, :width]
    return mask if mask.any() else None

def test_ppo_loss_arrays(cand, golden, record_property):
    """PPO per-token / GAE array parity, scored continuously.

    The PPO *scalar* loss is logit-INDEPENDENT by construction -- ``_formulas.ppo_losses``
    sets ``old_logps = gathered.copy()`` so ``ratio ≡ 1`` and the clip is a no-op -- so
    ``test_method_loss`` gates the value head + GAE over deterministic rewards but never the
    candidate's token log-probs. This consumes the already-frozen per-token arrays so the
    candidate's real logits are scored too. ``token_logprobs`` is the sole logit-dependent
    field: scored by ``S_tensor`` over the valid positions (direction AND magnitude -- global,
    not per-element, so its ~-0.01 … -176 dynamic range does not flake) plus the ``S_scalar``
    of the masked mean. advantages / returns ride along via ``S_tensor``; policy_loss /
    value_loss via ``S_scalar`` (defense-in-depth). The test score is the mean of the field
    terms; a missing defense-in-depth field contributes 0.0. PPO-only (deselected otherwise,
    ``_APPLICABLE``). A missing/mis-shaped ``token_logprobs`` fails the precondition (0.0)."""
    ref = golden.loss_artifact
    got = cand.loss()

    # token_logprobs -- the load-bearing logit-dependent field (structural preconditions).
    assert got.get("token_logprobs") is not None, {"error": "candidate PPO loss exposed no token_logprobs"}
    ref_lp = np.asarray(ref["token_logprobs"], dtype=np.float64)
    got_lp = np.asarray(got["token_logprobs"], dtype=np.float64)
    assert got_lp.shape == ref_lp.shape, {
        "field": "token_logprobs", "ref_shape": list(ref_lp.shape), "cand_shape": list(got_lp.shape),
    }
    width = int(ref_lp.shape[1] if ref_lp.ndim == 2 else ref_lp.shape[-1])
    mask = _ppo_valid_mask(golden, width)
    if mask is not None and mask.shape == ref_lp.shape:
        ref_sel, got_sel = ref_lp[mask], got_lp[mask]
    else:
        ref_sel, got_sel = ref_lp.reshape(-1), got_lp.reshape(-1)
    scores = [
        similarity.S_tensor(ref_sel, got_sel),
        similarity.score_scalar(float(ref_sel.mean()), float(got_sel.mean())),
    ]

    # advantages / returns -- deterministic GAE arrays (S_tensor, direction AND magnitude).
    for field in ("advantages", "returns"):
        g = got.get(field)
        if g is None:
            scores.append(0.0)
            continue
        r = np.asarray(ref[field], dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        scores.append(similarity.S_tensor(r, g) if g.shape == r.shape else 0.0)

    # policy_loss / value_loss -- scalar, value-head-dependent (S_scalar).
    for field in ("policy_loss", "value_loss"):
        g = got.get(field)
        scores.append(similarity.score_scalar(float(ref[field]), float(g)) if g is not None else 0.0)

    record_property("score", _avg(scores))

# PPO clip range (TRL PPOConfig default; matches _formulas.ppo_losses cliprange=0.2).
_PPO_CLIPRANGE = 0.2

def test_ppo_old_policy_ratio(cand, golden, record_property):
    """Logit-dependent PPO scalar via a frozen old-policy snapshot.

    The PPO *scalar* policy loss in ``_formulas.ppo_losses`` sets ``old_logps =
    gathered.copy()`` so ``ratio ≡ 1`` and the clipped surrogate is a no-op -- the scalar
    objective never reads the candidate's logits (only the ``token_logprobs`` ARRAY does,
    ``test_ppo_loss_arrays``). This gate closes that: it treats the *golden's*
    already-frozen ``token_logprobs`` as the OLD (reference) policy and recomputes the
    clipped PPO surrogate with ``ratio = exp(cand_logp - old_logp)`` over the golden's own
    frozen (old-policy) ``advantages``, then scores it against the frozen ``policy_loss`` by
    ``S_scalar``. A candidate whose logits drift from the reference moves the surrogate; a
    matching candidate reproduces it.

    Golden-neutral by construction (no re-freeze). A faithfulness guard (STRUCTURAL
    precondition) asserts that ``ratio = exp(0) = 1`` on the reference's own log-probs
    reproduces the frozen ``policy_loss`` before the candidate is scored, so a
    mis-reconstructed mask / formula fails loudly (score 0.0) instead of scoring a broken
    gate. PPO-only (deselected otherwise, like ``test_ppo_loss_arrays``)."""
    ref = golden.loss_artifact
    assert (
        ref.get("token_logprobs") is not None
        and ref.get("advantages") is not None
        and ref.get("policy_loss") is not None
    ), {"error": "golden PPO loss froze no token_logprobs/advantages/policy_loss (should be deselected)"}
    old_logp = np.asarray(ref["token_logprobs"], dtype=np.float64)  # reference (old) policy log-probs
    adv = np.asarray(ref["advantages"], dtype=np.float64)           # old-policy GAE advantages (frozen)
    frozen_policy_loss = float(ref["policy_loss"])

    got = cand.loss()
    assert got.get("token_logprobs") is not None, {"error": "candidate PPO loss exposed no token_logprobs"}
    cand_logp = np.asarray(got["token_logprobs"], dtype=np.float64)
    assert cand_logp.shape == old_logp.shape, {
        "field": "token_logprobs", "ref_shape": list(old_logp.shape), "cand_shape": list(cand_logp.shape),
    }
    assert adv.shape == old_logp.shape, {
        "field": "advantages", "adv_shape": list(adv.shape), "logp_shape": list(old_logp.shape),
    }

    width = int(old_logp.shape[1] if old_logp.ndim == 2 else old_logp.shape[-1])
    mask = _ppo_valid_mask(golden, width)
    if mask is None or np.asarray(mask).shape != old_logp.shape:
        mask = np.ones_like(old_logp)
    mask = np.asarray(mask, dtype=np.float64)
    denom = max(float(mask.sum()), 1.0)

    def _surrogate(logp):
        # Clipped PPO policy surrogate with the golden as the OLD policy; mirrors
        # _formulas.ppo_losses (ratio -> clip -> _masked_mean) with old_logps frozen.
        ratio = np.exp(logp - old_logp)
        clipped = np.clip(ratio, 1.0 - _PPO_CLIPRANGE, 1.0 + _PPO_CLIPRANGE)
        pg = np.maximum(-adv * ratio, -adv * clipped)
        return float((pg * mask).sum() / denom)

    # Faithfulness guard (STRUCTURAL precondition): ratio ≡ 1 on the reference's own
    # log-probs must reproduce the frozen scalar policy_loss (validates the mask /
    # masked-mean reconstruction) before the candidate is scored.
    ref_surrogate = _surrogate(old_logp)
    assert abs(ref_surrogate - frozen_policy_loss) <= 1e-6 + 1e-4 * abs(frozen_policy_loss), {
        "field": "old_policy_reconstruction",
        "recomputed_ratio1": ref_surrogate, "frozen_policy_loss": frozen_policy_loss,
    }

    cand_policy_loss = _surrogate(cand_logp)
    record_property("score", similarity.score_scalar(frozen_policy_loss, cand_policy_loss))

def test_gradient_per_param(cand, golden, record_property):
    """Per-parameter gradient authenticity. A STRUCTURAL {0,1} discriminator
    (exception), not a parity tolerance: under LoRA -- every current golden -- only the
    adapters train and their per-param grad MAGNITUDES depend on the random adapter init, so
    they cannot be graded value-exact; what CAN is the reference's own live/dead adapter
    pattern + finiteness (a backward that never reaches the adapters, or leaks gradient where
    the reference has none, is a real bug). Passing the structural checks records 1.0; any
    failure records 0.0. Deselected on goldens without per-param grads.

    The value-exact per-param compare (aligning the LM subtree across the torch<->flax naming
    gap) is the legacy full-finetuning path, for which no goldens remain."""
    ref_grads = golden.gradients.get("grads")
    assert ref_grads, "golden has no per-parameter gradients (should have been deselected)"
    cand_grads = cand.gradients().get("grads")
    assert cand_grads, "candidate gradient_artifact() exposed no per-parameter 'grads'"
    if golden.is_lora:
        # LoRA: only the low-rank adapters are trained, and their per-param grad norms depend
        # on the random adapter init (RNG differs torch-PEFT vs JAX), so the MAGNITUDES cannot
        # be graded value-exact. Which of them are zero can be, and it is structural, not
        # stochastic:
        #   * lora_A is zero at step 0 in every LoRA instance -- PEFT initialises B to zero and
        #     d(loss)/dA is proportional to B;
        #   * lora_B is non-zero exactly when a gradient reaches the adapters at all, which
        #     under PPO is never: the value head is created zero-initialised, so `values` is
        #     identically zero and nothing propagates back into the trunk.
        # Requiring the candidate to reproduce the reference's own live/dead pattern catches a
        # backward pass that never reaches the adapters AND one that leaks gradient where the
        # reference has none. The RNG-independent forward / loss / authenticity gates stay exact.
        def _live(grads):
            def any_nonzero(marker):
                return any(
                    float(v.get("norm", 0.0)) > 0.0 for k, v in grads.items() if marker in k.lower()
                )

            return {"lora_A": any_nonzero("lora_a"), "lora_B": any_nonzero("lora_b")}

        adapters = {k: v for k, v in cand_grads.items() if "lora_a" in k.lower() or "lora_b" in k.lower()}
        assert adapters, {"error": "no LoRA adapter gradients found", "keys": list(cand_grads)[:8]}
        assert all(bool(v.get("finite", True)) for v in adapters.values()), {"error": "non-finite adapter grad"}
        assert _live(cand_grads) == _live(ref_grads), {
            "error": "adapter gradients are live in one framework and dead in the other",
            "reference": _live(ref_grads),
            "candidate": _live(cand_grads),
        }
        record_property("score", 1.0)
        return

def test_lr_schedule(cand, golden, record_property):
    """Learning-rate schedule parity. Scores the candidate's OWN optax
    schedule (reconstructed from the port's trainer) against the frozen config-derived golden
    by ``score_schedule`` (``S_L2`` in float64 -- magnitude-aware, so an all-zeros schedule
    scores ~0 rather than sailing through a loose abs band). ``assert supported`` makes a repo
    whose trainer exposes no usable schedule a hard FAIL (score 0.0)."""
    got = cand.lr_schedule()
    assert got.get("supported"), {
        "error": "candidate's trainer exposes no usable LR schedule to sample",
        "reason": got.get("reason"),
    }
    record_property(
        "score",
        similarity.score_schedule(golden.lr_schedule["learning_rates"], got["learning_rates"]),
    )

def test_gradient_authenticity(cand, golden, record_property):
    """Analytic directional derivative matches finite differences along several
    gradient-subspace directions: full support + even/odd leaf subsets. A
    faked/torch/constant forward, or a candidate whose gradients are only partially correct,
    cannot satisfy all of them. Consumes no golden -- it is a self-consistency
    check on the candidate alone.

    STRUCTURAL preconditions (fail -> score 0.0): the probe is supported, exposes >=2
    directions, and the FULL-support direction (probe 0) is non-vacuous
    (``||g_S|| > 1e-12``) -- a support that moves nothing is a probe that tested nothing.
    The even/odd SUBSET directions are allowed to be vacuous (a zero gradient on part of the
    support is a real property of some instances -- PPO's scalar loss never reads lm_head).

    Score: the mean over all directions of ``score_grad_authenticity(analytic, fd)`` where
    ``fd = (loss_perturbed - loss_base) / epsilon`` (``S_scalar`` of the analytic directional
    derivative against its finite-difference estimate). A severed/faked backward drives
    ``fd -> 0`` while ``analytic`` stays finite (or vice versa), collapsing the score; a
    genuine vacuous direction has ``analytic == fd == 0 -> 1.0``."""
    info = cand.gradient_authenticity()
    assert info.get("supported"), info
    probes = info.get("probes") or [info]  # back-compat with the single-probe shape
    coverage = {
        "support_params": info.get("support_params"),
        "total_params": info.get("total_params"),
        "support_coverage": info.get("support_coverage"),
        "support_leaves": info.get("support_leaves"),
        "total_leaves": info.get("total_leaves"),
    }
    assert len(probes) >= 2, {"reason": "expected multiple directions", "info": info}
    assert float(probes[0]["analytic_directional"]) > 1e-12, {
        "direction": 0,
        "reason": "vacuous probe: ||g|| over the WHOLE support is on the floor, so every "
        "direction takes a zero step and the check degenerates to 0 == 0",
        "probe": probes[0],
        **coverage,
    }
    scores: list[float] = []
    for probe in probes:
        eps = float(probe["epsilon"])
        analytic = float(probe["analytic_directional"])
        if eps == 0:
            scores.append(0.0)
            continue
        fd = (float(probe["loss_perturbed"]) - float(probe["loss_base"])) / eps
        scores.append(similarity.score_grad_authenticity(analytic, fd))
    record_property("score", _avg(scores))


def test_param_norms(cand, golden, record_property):
    """Per-parameter L2-norm parity vs the frozen reference weights,
    scored CONTINUOUSLY. The L2 norm is layout-invariant, so a conversion bug -- a missed
    transpose, a wrongly-ordered fused QKV, an unread shard, a duplicated tie, a misread bf16
    bit pattern, a tensor left at its random init -- reads here as that tensor's ``S_scalar``
    term dropping, not as an inscrutable downstream logit mismatch. Replaces the retired boolean
    ``compare.compare_param_norms`` tolerance gate; the structural 1:1 correspondence
    stays in Spec (``test_param_correspondence`` / ``test_param_spec``). Averaged over the gated,
    mapped tensors; a non-finite candidate norm forces that tensor to 0.0, and a conversion that
    maps no gated tensor scores 0.0 overall."""
    record_property("score", compare.score_param_norms(golden.param_norms, cand.param_norms()))


def test_pixel_values(cand, golden, record_property):
    """Candidate's OWN data pipeline reproduces the frozen FLOAT batch content -- ``pixel_values``
    from its image processor -- scored CONTINUOUSLY by ``S_tensor`` (direction AND magnitude).
    The integer/token keys (input_ids / attention_mask / labels / ...) are the
    EXACT structural gate in ``test_spec.test_data_pipeline``; the float keys move here because a
    JAX image-processor reimplementation can differ in interpolation, so a small resampling
    difference degrades the score continuously instead of tripping a hand-picked tolerance. The
    score is the mean over the float input keys (a missing / mis-shaped one scores 0.0).
    Deselected on text instances (no float inputs); a candidate exposing no runnable pipeline
    FAILS (score 0.0)."""
    result = cand.batch_arrays()
    assert result.get("supported"), {
        "error": "candidate exposes no runnable data pipeline "
        "(the reference reproduces golden/inputs.npz from its own processor + collator)",
        "reason": result.get("reason"),
    }
    cand_arrays = result["arrays"]
    gold = golden.inputs
    scores = []
    for key in sorted(gold):
        ref = np.asarray(gold[key])
        if ref.dtype.kind not in ("f", "c"):
            continue  # integer/token keys are the exact Spec gate, not a similarity
        if key not in cand_arrays:
            scores.append(0.0)  # candidate omitted a float key it must reproduce
            continue
        scores.append(similarity.S_tensor(ref, np.asarray(cand_arrays[key])))
    assert scores, "no float input keys (should have been deselected on a text instance)"
    record_property("score", _avg(scores))

# >>> REFUNITS BLOCK (generic, Route B) >>>  # noqa: E501
# =========================================================================== #
# Per-submodule parity vs torch_orig (golden/reference_units.*),
# Route B (stage: numeric). ONE fp32 forward with flax capture_intermediates
# exposes every named submodule's output BY PATH; each is SCORED (S_tensor) against
# the frozen torch reference, localizing an internal-activation mismatch to the op.
# See tests/refunits.py.
# =========================================================================== #
import pytest  # noqa: E402
import refunits as R  # noqa: E402

# Deselected (not skipped) when the golden froze no reference_units -- the conftest
# collection hook drops every ``@_needs_units`` check for such an instance, keeping it
# out of the reward denominator; where the artifact exists these are scored.
_needs_units = pytest.mark.refunits

# --- transformers_impl/modeling.py: per-submodule outputs (capture_intermediates) --- #
@_needs_units
@pytest.mark.parametrize("unit", R.modeling_units())
def test_reference_units_modeling(unit, record_property):
    """Each frozen submodule output (embedding, per-layer norms/attn/mlp, decoder block,
    final norm) scored (``S_tensor``) against the JAX candidate's own forward -- a low term
    localizes to the exact op (RoPE/attention bracketed by self_attn; activation by mlp)."""
    record_property("score", R.sim_array(unit, R.intermediate(unit)))

@_needs_units
def test_reference_units_logits(record_property):
    out, _ = R.jax_forward_capture()
    record_property("score", R.sim_array("logits", out.logits))

@_needs_units
def test_reference_units_hidden_states(record_property):
    out, _ = R.jax_forward_capture()
    n = int(R.scalars().get("num_hidden_states", len(out.hidden_states)))
    assert len(out.hidden_states) == n, {"jax_layers": len(out.hidden_states), "ref_layers": n}
    scores = [R.sim_array(f"hidden_{i}", out.hidden_states[i]) for i in range(n)]
    record_property("score", _avg(scores))

# =========================================================================== #
# PPO update-mechanics reference units.
#
# The only two behavioral gates that touch the real PPO trainer (L6 test_validation_loss,
# T3 test_train_run_end_to_end) score candidate-SELF-REPORTED train JSONs -- so a PPO port
# that never implements the actual update math can still bank ~1.0. This gate closes that:
# it executes the CANDIDATE's OWN ``ppo/ppo_utils.py`` on a fixed synthetic battery and
# scores it against the ORACLE (torch_orig) outputs frozen in golden/ppo_units.{npz,json}
# by the freezer. Non-spoofable (the grader supplies the inputs and calls the
# candidate's functions) and torch-sourced (never ref_jax). PPO-only + presence-gated on the
# ppo_units golden (conftest ``_APPLICABLE``).
# =========================================================================== #
def test_ppo_update_units(record_property):
    """Per-function parity of the candidate's own ``ppo/ppo_utils.py`` update math vs the
    frozen torch_orig outputs. Arrays (per-token log-probs, GAE advantages+returns,
    masked-whiten) score by ``S_tensor``; scalars (clipped policy & value loss + their
    clipfracs, masked mean/var, token-F1) by ``S_scalar``. The candidate module MUST import
    (a PPO port exposing no ppo_utils fails the whole gate -> 0.0); each function is then
    scored independently and a broken/missing one contributes 0.0, so a partial stub scores
    proportionally. The battery deliberately straddles BOTH clip ranges (clipfrac in (0,1)),
    so a port that drops the PPO/value clip is separated from the faithful one. Deselected off
    PPO / when the golden is absent (conftest)."""
    import importlib
    import json
    import os
    import sys
    from pathlib import Path

    gdir = Path(__file__).resolve().parent / "golden"
    meta = json.loads((gdir / "ppo_units.json").read_text())
    arr = {k: np.asarray(v) for k, v in np.load(gdir / "ppo_units.npz").items()}

    # Candidate JAX repo onto sys.path (same contract path candidate.py/refunits use), then
    # import the candidate's OWN update utilities -- self-contained so this block is byte-
    # identical across every PPO test_numeric variant (some do not import ``refunits``).
    repo = os.environ.get("T2J_JAX_REPO") or str(
        (Path(__file__).resolve().parent.parent / "solution" / "ref_jax_repo").resolve()
    )
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    U = importlib.import_module("ppo.ppo_utils")  # HARD precondition: a PPO port exposes this
    import jax.numpy as _jnp

    def J(x):
        return _jnp.asarray(x)

    def N(x):
        return np.asarray(x, dtype=np.float64)

    def _term(fn, n=1):
        """Score one function, contributing ``n`` zeros if it is missing/raises."""
        try:
            out = fn()
            return out if isinstance(out, list) else [out]
        except Exception:
            return [0.0] * n

    scores: list[float] = []

    # logprobs_from_logits -- per-token log-probs (array).
    scores += _term(lambda: similarity.S_tensor(
        arr["out_logprobs"],
        N(U.logprobs_from_logits(J(arr["in_logits"]), J(arr["in_labels"].astype("int32")))),
    ))
    # compute_gae -- advantages + returns (arrays).
    def _gae():
        adv, ret = U.compute_gae(J(arr["in_gae_rewards"]), J(arr["in_gae_values"]),
                                 J(arr["in_gae_mask"]), gamma=meta["gamma"], lam=meta["lam"])
        return [similarity.S_tensor(arr["out_gae_adv"], N(adv)),
                similarity.S_tensor(arr["out_gae_ret"], N(ret))]
    scores += _term(_gae, n=2)
    # compute_ppo_loss -- clipped policy loss + clipfrac (scalars).
    def _ppo():
        pg, cf = U.compute_ppo_loss(J(arr["in_ppo_new"]), J(arr["in_ppo_old"]),
                                    J(arr["in_ppo_adv"]), J(arr["in_ppo_mask"]),
                                    cliprange=meta["cliprange"])
        return [similarity.score_scalar(meta["out_ppo_pg"], float(pg)),
                similarity.score_scalar(meta["out_ppo_clipfrac"], float(cf))]
    scores += _term(_ppo, n=2)
    # compute_value_loss -- clipped value loss + clipfrac (scalars).
    def _val():
        vf, vcf = U.compute_value_loss(J(arr["in_vl_vpred"]), J(arr["in_vl_oldval"]),
                                       J(arr["in_vl_returns"]), J(arr["in_vl_mask"]),
                                       cliprange_value=meta["cliprange_value"])
        return [similarity.score_scalar(meta["out_vl_vf"], float(vf)),
                similarity.score_scalar(meta["out_vl_clipfrac"], float(vcf))]
    scores += _term(_val, n=2)
    # masked_whiten (array) + masked_mean / masked_var (scalars).
    scores += _term(lambda: similarity.S_tensor(
        arr["out_masked_whiten"],
        N(U.masked_whiten(J(arr["in_mw_values"]), J(arr["in_mw_mask"])))))
    scores += _term(lambda: similarity.score_scalar(
        meta["out_masked_mean"], float(U.masked_mean(J(arr["in_mw_values"]), J(arr["in_mw_mask"])))))
    scores += _term(lambda: similarity.score_scalar(
        meta["out_masked_var"], float(U.masked_var(J(arr["in_mw_values"]), J(arr["in_mw_mask"])))))
    # token_f1_reward -- pure-python multiset F1 (scalar).
    scores += _term(lambda: similarity.score_scalar(
        meta["out_token_f1"],
        float(U.token_f1_reward(list(meta["f1_pred"]), list(meta["f1_gold"])))))

    record_property("score", _avg(scores))
