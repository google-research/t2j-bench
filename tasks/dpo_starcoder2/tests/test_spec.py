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

"""Spec stage (generated, generic): independence + structural parity.

Three groups, all hardware-invariant (no H100 gate):

* Independence + params + data -- drive the candidate in the torch-blocked
  sandbox (``cand`` fixture) against the frozen goldens: no torch anywhere, a native JAX
  pytree, a pure traced forward, per-parameter weight parity, and the candidate's OWN data
  pipeline reproducing the frozen batch.
* Interface / structure -- STATIC AST analysis of the candidate SOURCE against
  ``golden/interface.json`` (parity, kind, signature, methods, constants, layout). No
  import, no torch, no candidate run, so a dropped/renamed/retyped entry point fails here,
  loudly, instead of surfacing as an inscrutable error in a downstream numeric check.
* Config-default parity -- ``config.get_train_args({})`` must resolve the same
  load-bearing config DEFAULTS torch_orig declares (``golden/config_spec.json``, frozen for
  all 45 by the freezer). Tokenizer-fact and held-out validation-split
  parity remain presence-gated on goldens a future freezer emits (deselected today).

No check skips at grade time: these gate the numeric/behavioral meaning, and every one
that APPLIES to this instance must pass for the stage to earn its share of the reward. A
check that cannot apply (its golden was never frozen for this instance/method) is
deselected in conftest -- kept out of both numerator and denominator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from _t2j import compare
from _t2j import golden_manifest

# Reference ignore_index for label masking (LLaMA-Factory / HF default); used only to
# report WHERE a label mismatch is when test_data_pipeline fails.
_IGNORE_INDEX = -100

# Sentinel so a config field that is genuinely ``None`` is distinguished from absent.
_MISSING = object()

# --------------------------------------------------------------------------- #
# Candidate SOURCE resolution for the static AST checks. Unlike the sandbox
# tests (which resolve the repo through the ``cand`` fixture), these read the candidate
# source directly, so they need their own resolver -- and the "guard the
# default": under real grading ``T2J_JAX_REPO`` points at the agent's repo, but if it is
# unset we must NOT silently grade the committed reference against its own frozen contract
# (that always passes, masking a broken harness). Fail unless ``T2J_ALLOW_SELF_GRADE``
# opts in -- which is how ``pytest tasks/*/tests`` self-validates the
# benchmark's own reference.
# --------------------------------------------------------------------------- #
def _candidate_repo() -> str:
    repo = os.environ.get("T2J_JAX_REPO")
    if repo:
        return str(Path(repo).resolve())
    if os.environ.get("T2J_ALLOW_SELF_GRADE"):
        return str((Path(__file__).resolve().parent / ".." / "solution" / "ref_jax_repo").resolve())
    raise AssertionError(
        "T2J_JAX_REPO is unset and T2J_ALLOW_SELF_GRADE is not set: refusing to grade the "
        "reference against its own frozen interface ('guard the default'). Set "
        "T2J_JAX_REPO to the candidate repo, or T2J_ALLOW_SELF_GRADE=1 to self-validate."
    )

def _interface_spec() -> dict:
    """Load the frozen interface contract (``golden/interface.json``).

    Required for every instance (``_COMMON``); a missing or corrupt one is a coverage
    failure the check reports directly (``test_golden_manifest`` names it too), rather than
    silently disabling every interface check.
    """
    path = Path(__file__).resolve().parent / "golden" / "interface.json"
    assert path.exists(), {
        "error": f"missing frozen contract: {path}",
        "hint": "regenerate it from the reference implementation",
    }
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise AssertionError({"error": f"corrupt interface.json: {exc}", "path": str(path)})


def _config_equal(a, b) -> bool:
    """Config-value equality: exact for ints/strings/bools, tolerant for floats, elementwise
    for lists (a learning rate re-parsed from the same string is float-equal, not bit-equal)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 + 1e-6 * abs(float(a))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_config_equal(x, y) for x, y in zip(a, b))
    return a == b

# =========================================================================== #
# Independence
# =========================================================================== #
def test_no_torch(cand):
    """Whole candidate repo is torch-free, statically AND at runtime.

    Two halves, because either alone is dodgeable:

    * STATIC -- an AST scan of the WHOLE repo (``convert_checkpoint`` included,
      ``allow_dirs=()``) for ``import torch`` and for the ``importlib`` / ``__import__`` /
      ``exec`` / ``eval`` obfuscations a scan of plain imports would miss.
    * RUNTIME -- a dynamic ``import torch`` evades the static scan entirely, so the
      sandbox worker also records any banned module (torch / torchvision / torchaudio /
      functorch / torch_xla) that leaked into ``sys.modules`` while the candidate ran. The
      one shared, cached sandbox run surfaces that as a "banned modules" error on the first
      accessor; trigger it here and fail on THAT signature so a runtime leak is attributed
      to this test. A non-torch sandbox failure is not this test's concern -- it is owned
      and reported by the structural / forward tests -- so it is let through the assert.
    """
    report = cand.purity_scan()
    assert report.scanned_files > 0, "no candidate sources scanned (is the candidate repo present?)"
    assert report.clean, [f.to_dict() for f in report.findings[:5]]
    try:
        cand.structural()  # any accessor drives the shared sandbox; a leak raises here
    except Exception as exc:  # noqa: BLE001 - only the banned-module signature is ours
        assert "banned modules" not in str(exc), str(exc)

def test_structural_probe(cand):
    """Params are a JAX pytree of jax.Array; module is flax; no pt_model."""
    info = cand.structural()
    assert info.get("is_pytree"), info
    assert info.get("all_jax_arrays"), info
    assert info.get("module_is_flax"), info
    assert not info.get("has_pt_model"), info

def test_forward_purity(cand):
    """Forward traces to non-trivial XLA ops with no host callbacks."""
    info = cand.forward_purity()
    assert info.get("supported"), info
    assert not info.get("callbacks"), info
    assert info.get("traced"), info

def test_determinism_static(cand):
    """Two forwards over the same frozen batch are bit-identical.

    A pure, jitted forward with fixed inputs and no live RNG is deterministic; a candidate
    whose forward reseeds per call (or leaks nondeterministic host state) is caught here,
    before any numeric parity check that would blame the mismatch on the weights.
    """
    info = cand.determinism()
    assert info.get("supported"), info
    assert info.get("identical"), info

# =========================================================================== #
# Parameters
# =========================================================================== #
def test_param_correspondence(cand, golden):
    """Every parameter you built lines up 1:1 with a checkpoint tensor.

    You convert the checkpoint yourself, so the grader does not dictate your pytree:
    it maps your paths and the torch state-dict names onto a common canonical key
    (``layers_0/self_attn/q_proj/kernel`` and ``model.layers.0.self_attn.q_proj.weight``
    both become ``model.layers.0.self_attn.q_proj.W``) and requires a shape that is
    equal or, for rank-2, transposed. Naming is yours; the correspondence is not.
    """
    ok, details = compare.compare_param_spec(golden.params_spec, cand.param_spec())
    assert ok, details

def test_param_spec(cand, golden):
    """The candidate exposes a well-formed param spec whose shapes correspond to the golden.

    Distinct from ``test_param_correspondence`` in its first failure mode: a port that
    returns an empty or malformed spec (no per-leaf shape) fails HERE regardless of any
    key-mapping outcome. Then the same shape-correspondence gate (1:1 up to a rank-2
    transpose; dtype not gated) is applied, so a spec that is well-formed but wrong is
    caught too.
    """
    spec = cand.param_spec()
    assert spec, {"error": "candidate exposed an empty parameter spec"}
    malformed = sorted(k for k, v in spec.items() if not isinstance(v, dict) or "shape" not in v)
    assert not malformed, {"malformed_param_spec_entries": malformed[:10], "total": len(malformed)}
    ok, details = compare.compare_param_spec(golden.params_spec, spec)
    assert ok, details

# =========================================================================== #
# Data pipeline
# =========================================================================== #
def test_batch_schema(cand, golden):
    """Candidate's own data pipeline reproduces golden/batch_schema.json.

    The reference always builds this batch from its processor + collator, so the
    check applies to every instance: a candidate that exposes no runnable data
    pipeline FAILS here (it does not skip) — a weights-only port is an incomplete
    port, and the continuous pass rate must reflect that.
    """
    result = cand.batch_schema()
    assert result.get("supported"), {
        "error": "candidate exposes no runnable data pipeline "
        "(the reference reproduces golden/batch_schema.json from its own processor + collator)",
        "reason": result.get("reason"),
    }
    cand_schema = result["schema"]
    gold = golden.batch_schema
    missing = sorted(set(gold) - set(cand_schema))
    assert not missing, {"missing_batch_keys": missing}
    shape_mismatch = {
        k: {"golden": gold[k]["shape"], "candidate": cand_schema[k]["shape"]}
        for k in gold
        if list(cand_schema[k]["shape"]) != list(gold[k]["shape"])
    }
    assert not shape_mismatch, shape_mismatch

def test_data_pipeline(cand, golden):
    """The candidate's OWN data pipeline must reproduce the frozen golden batch CONTENT,
    not just its schema.

    ``test_batch_schema`` checks only keys+shapes, and every numeric stage then consumes
    the FROZEN ``golden/inputs.npz`` -- so a wrong chat template, a mis-masked label span,
    a shape-preserving truncation / wrong pad-id, or a DPO collator that swaps
    chosen(row 0)/rejected(row 1) all pass at reward 1.0 while the port's real pipeline is
    broken. This compares the candidate's rebuilt batch to the golden key-by-key:

      * integer / token keys (input_ids, attention_mask, labels, token_type_ids, ...) must
        match EXACTLY -- deterministic tokenization + masking has no tolerance to spend;
      * the one float surface (pixel_values) is NOT gated here -- its value parity is a
        continuous ``S_tensor`` similarity in the Numeric dimension (``test_pixel_values``).

    A weights-only port with no runnable pipeline FAILS here (it does not skip). Applies to
    every instance (inputs.npz is frozen for all 45), so it is not gated in conftest.
    """
    result = cand.batch_arrays()
    assert result.get("supported"), {
        "error": "candidate exposes no runnable data pipeline "
        "(the reference reproduces golden/inputs.npz from its own processor + collator)",
        "reason": result.get("reason"),
    }
    cand_arrays = result["arrays"]
    gold = golden.inputs
    missing = sorted(set(gold) - set(cand_arrays))
    assert not missing, {"missing_batch_keys": missing}

    failures: dict = {}
    for key in sorted(gold):
        ref = np.asarray(gold[key])
        got = np.asarray(cand_arrays[key])
        if tuple(got.shape) != tuple(ref.shape):
            failures[key] = {"error": "shape", "golden": list(ref.shape), "candidate": list(got.shape)}
            continue
        # Integer/token keys are compared EXACTLY (a structural Spec check -- token ids are exact
        # or wrong, no similarity to spend). The one float surface (pixel_values) is NOT value-
        # gated here: its parity moves to the Numeric dimension (``test_pixel_values``, S_tensor),
        # since a float image tensor differing only by fp32 reduction order
        # should earn ≈0.999, not a binary pass/fail on a hand-picked tolerance. Its shape stays
        # structural and is gated above (and by ``test_batch_schema``).
        if ref.dtype.kind not in ("i", "u", "b"):
            continue
        if not np.array_equal(ref, got):
            entry = {"mismatched": int(np.count_nonzero(ref != got)), "total": int(ref.size)}
            if key == "labels":
                # A mis-masked supervision span is the classic content bug the schema
                # check misses: report how many positions differ in ignore-masking.
                entry["ignore_mask_diff"] = int(
                    np.count_nonzero((ref == _IGNORE_INDEX) != (got == _IGNORE_INDEX))
                )
            failures[key] = entry
    assert not failures, {"batch_content_mismatch": failures}

# =========================================================================== #
# Interface / structure (static AST; no import, no torch, no candidate run)
# =========================================================================== #
def test_interface_parity():
    """Candidate's public API is a superset of torch_orig's ported interface.

    ``golden/interface.json`` freezes, per module present in both trees, the public names
    torch_orig exposes that the reference port carries (``required``), plus the documented
    JAX-only additions such as ``model/loader.convert_checkpoint`` (``required_extras``).
    The candidate repo is AST-parsed and every required name must have a same-named
    counterpart in its module; extra names are allowed. Names only: the port legitimately
    adapts signatures (checked separately by ``test_interface_signature``).
    """
    spec = _interface_spec()
    missing = compare.compare_interface(spec, compare.candidate_public_api(_candidate_repo()))
    assert not missing, {
        "interface_missing": missing,
        "hint": "each torch_orig module's public functions/classes must have a same-named "
        "counterpart in your repo; documented JAX-only extras (convert_checkpoint) are "
        "required too, but you may define anything beyond them",
    }

def test_interface_kind():
    """Each required name is the SAME kind (class / function / constant) as the reference's.

    ``compare_interface`` is name-only, so a candidate that shadows a required class with a
    same-named function (or exposes a constant where a callable is required) passes parity
    but breaks every caller. ``golden/interface.json['kinds']`` freezes the reference kind
    for each contract name; this catches the swap.
    """
    spec = _interface_spec()
    assert "kinds" in spec, {
        "error": "interface.json has no 'kinds' section",
        "hint": "re-freeze from the reference implementation",
    }
    mismatch = compare.compare_interface_kinds(spec, compare.candidate_public_api_kinds(_candidate_repo()))
    assert not mismatch, {"interface_kind_mismatch": mismatch}

def test_interface_signature(cand):
    """Grader/CLI entry points ACCEPT the arguments the harness passes them.

    ``golden/interface.json['signatures']`` freezes, for the fixed entry points
    (``get_train_args``, ``load_tokenizer``, ``load_model``, ``load_ref_model`` on DPO, and
    ``run_<method>``), how the harness calls each: how many leading positionals it supplies
    and which parameters it passes by keyword. Each candidate callable must accept that call
    shape -- extra, reordered or renamed positional-only params are fine (the port adapts
    parameter lists), but a dropped keyword or a missing positional slot fails here rather
    than as a ``TypeError`` deep inside a numeric run.

    Union of the RUNTIME view (``cand.interface`` imports the module and reads the live
    ``inspect.signature`` -- so a factory-bound ``get_train_args = build_...()`` is seen for
    what the harness will actually call) and the AST view: a callable passes if EITHER
    accepts the call shape. The runtime probe is hardware-invariant (import + introspect, no
    model, no GPU); if it cannot run it returns ``supported=False`` and this degrades
    to the AST-only ``compare_signatures``.
    """
    spec = _interface_spec()
    assert "signatures" in spec, {
        "error": "interface.json has no 'signatures' section",
        "hint": "re-freeze from the reference implementation",
    }
    iface = cand.interface(spec)
    runtime_info = (iface.get("signatures") or {}) if iface.get("supported") else {}
    problems = compare.compare_signatures_runtime(spec, runtime_info, _candidate_repo())
    assert not problems, {
        "signature_problems": problems,
        "interface_probe_supported": iface.get("supported"),
        "interface_probe_reason": iface.get("reason"),
    }

def test_interface_methods(cand):
    """Each required class exposes at least the methods torch and the reference share.

    ``golden/interface.json['methods']`` freezes, per contract class, the torch ∩ reference
    method set (same superset policy as parity). A port that keeps the class name but drops
    ``DPOTrainer.concatenated_forward`` / ``Template.encode_multiturn`` / ``Dataset.map``
    fails here; extra methods are fine.

    Union of the RUNTIME view (``cand.interface`` reads ``dir()`` over the class MRO, so a
    method the port INHERITS -- e.g. ``keys``/``items``/``values`` on a ``ModelOutput(dict)``
    -- is credited) and the AST view (methods defined in the class body). A frozen method
    present in neither view is a real gap. The runtime probe is hardware-invariant; if
    it cannot run it returns ``supported=False`` and this degrades to the AST-only view.
    """
    spec = _interface_spec()
    assert "methods" in spec, {
        "error": "interface.json has no 'methods' section",
        "hint": "re-freeze from the reference implementation",
    }
    iface = cand.interface(spec)
    runtime_methods = (iface.get("methods") or {}) if iface.get("supported") else {}
    ast_methods = compare.candidate_class_methods(_candidate_repo())
    missing = compare.compare_interface_methods_runtime(spec, runtime_methods, ast_methods)
    assert not missing, {
        "interface_methods_missing": missing,
        "interface_probe_supported": iface.get("supported"),
        "interface_probe_reason": iface.get("reason"),
    }

def test_constant_values(cand):
    """Load-bearing constants have the reference's value / key set.

    ``golden/interface.json['constants']`` freezes, only where torch ITSELF literally states
    them, the constants a wrong value would silently corrupt training: ``IGNORE_INDEX``
    (must be -100), ``DEFAULT_TEMPLATE``, the ``SCHEDULER_MAP`` keys (which lr schedules are
    wired) and ``DATA_CONFIG``. Scalars must match exactly; dict constants must be a
    superset of the frozen keys.

    Union of the RUNTIME view (``cand.interface`` reads the live ``getattr`` on the same
    module, so a ``SCHEDULER_MAP = dict(cosine=..., linear=...)`` built non-literally is seen
    by its real keys) and the AST literal view: a constant matches if EITHER view matches.
    The runtime probe is hardware-invariant; if it cannot run it returns
    ``supported=False`` and this degrades to the AST-only ``compare_constants``.
    """
    spec = _interface_spec()
    assert "constants" in spec, {
        "error": "interface.json has no 'constants' section",
        "hint": "re-freeze from the reference implementation",
    }
    wanted = {rel: list(names) for rel, names in (spec.get("constants") or {}).items()}
    ast_constants = compare.candidate_module_constants(_candidate_repo(), wanted)
    iface = cand.interface(spec)
    runtime_constants = (iface.get("constants") or {}) if iface.get("supported") else {}
    problems = compare.compare_constants_runtime(spec, runtime_constants, ast_constants)
    assert not problems, {
        "constant_problems": problems,
        "interface_probe_supported": iface.get("supported"),
        "interface_probe_reason": iface.get("reason"),
    }

def test_repo_structure():
    """Candidate module tree mirrors torch_orig's layout, with no namespace-package dirs.

    ``golden/interface.json['structure']`` freezes the torch_orig ``.py`` layout plus the
    documented allowances. The candidate's ``.py`` set must be a SUPERSET of the required
    modules (extra files, incl. the reference's torch-free shims, are fine) and it must have
    NO directory that holds no ``.py`` at any depth -- a leftover package dir becomes a
    namespace package that can hijack ``import datasets`` to the torch-importing HF one.
    """
    spec = _interface_spec()
    assert "structure" in spec, {
        "error": "interface.json has no 'structure' section",
        "hint": "re-freeze from the reference implementation",
    }
    result = compare.compare_structure(spec, _candidate_repo())
    assert not result["missing_files"], {"missing_modules": result["missing_files"]}
    assert not result["empty_dirs"], {
        "empty_or_namespace_dirs": result["empty_dirs"],
        "hint": "delete leftover package directories that hold no .py at any depth",
    }

# =========================================================================== #
# Config-default parity (sandbox; get_train_args({}) defaults, no model/forward)
# =========================================================================== #
def test_config_spec(cand, golden):
    """The port resolves the same load-bearing config DEFAULTS as the reference.

    Config-DEFAULT parity: ``config.get_train_args({})`` -- the pure config.py dataclass
    defaults under an EMPTY override dict, NOT the explicit meta.json repo_config the numeric
    probes resolve -- must produce the same learning rate, scheduler, warmup, batch/accum,
    finetuning + LoRA knobs, preference/PPO/generation knobs, template and cutoff that
    torch_orig declares. ``golden/config_spec.json`` freezes those defaults from torch_orig
    (frozen for all 45 by the freezer). Floats compare within tolerance (a
    re-parsed learning rate need not be bit-identical); everything else exactly. A field
    absent on this method's config is neither frozen nor checked. This feeds the multiplicative
    spec gate: a changed default that would silently skew every downstream numeric/behavioral
    run fails HERE, loudly, instead of as an inscrutable mismatch deep in a training golden.
    """
    result = cand.config_spec()
    assert result.get("supported"), {
        "error": "candidate did not expose a resolved training config",
        "reason": result.get("reason"),
    }
    cand_cfg = result.get("config", {})
    gold = golden.config_spec or {}
    mismatches = {}
    for field, want in sorted(gold.items()):
        got = cand_cfg.get(field, _MISSING)
        if got is _MISSING or not _config_equal(want, got):
            mismatches[field] = {"golden": want, "candidate": None if got is _MISSING else got}
    assert not mismatches, {"config_mismatch": mismatches}

# =========================================================================== #
# Coverage
# =========================================================================== #
def test_golden_manifest(golden):
    """Every golden this METHOD must emit is present.

    Applicability deselection is golden-presence-driven, so a freeze bug that silently
    drops a required artifact reads as "not applicable" and never dents the pass rate.
    This hard-fails on any required-but-missing golden (by method, from
    ``_t2j/golden_manifest``), turning a dropped golden into a loud coverage FAIL. A
    token is a filename (``forward.npz``) or a within-file requirement
    (``fp32_forward:logits`` / ``forward:loss``). Consumes only the frozen golden dir.
    """
    missing = golden_manifest.missing_required(golden.dir, golden.training_method)
    assert not missing, {
        "method": golden.training_method,
        "missing_required_goldens": missing,
        "hint": "a correct freeze emits every required golden for the method; a missing "
        "one is a coverage regression, not an inapplicable check",
    }
    # Presence is not enough: a complete golden set can still contradict itself.
    problems = golden_manifest.consistency_problems(golden.dir)
    assert not problems, {
        "method": golden.training_method,
        "inconsistent_goldens": problems,
        "hint": "the golden set is internally contradictory -- re-freeze it; this is an "
        "artifact defect, not a candidate defect",
    }
