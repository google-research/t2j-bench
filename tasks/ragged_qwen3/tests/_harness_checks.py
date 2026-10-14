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

"""Validate inference harness inputs before grading a candidate.

CHECKS covers golden completeness, applicability/threshold consistency,
configuration provenance, and scenario coverage. These checks inspect the task
rather than the candidate, so they run in _grade_precheck and earn no reward.
A failure invalidates the grading run before pytest writes reward.json.
The check_* names keep these functions out of normal pytest collection.
This module reads JSON and Python source without executing a candidate.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from _t2j import compare, golden_manifest

_TEST_DIR = Path(__file__).resolve().parent


#: Threshold-band keys that are NOT ``compare.NumericThresholds`` fields and are
#: nonetheless legitimate. Read by
#: ``check_threshold_extras_name_something_a_check_reads``; the ``abs_tol.<surface>``
#: family is handled separately, by reading the surfaces the graded modules name.
#:
#: The last two are DELIBERATELY here before the code that adds them.
#: ``fp32_thresholds.logprob_atol`` and ``native_thresholds.decisive_margin`` are
#: calibrated later; a guard that rejected them would make this check an obstacle to
#: the work it exists to protect, and "add your key to the allow-list" is not a
#: lesson a later change should have to learn from a red test.
#: A key here that NOTHING reads is harmless -- it is a declared name, not a typo.
#:
#: BAND-AWARE, and the value says which bands. The allow-list used to be a flat
#: ``{name: owner}`` and never consulted the owner, so ``native_thresholds
#: .logprob_atol`` -- an fp32 quantity filed on the native band -- passed validation
#: and was then read by nobody. That is not a loosening on its own (the consumer
#: falls back to its own band's default, which for these keys is tighter), so it is
#: a completeness gap rather than a hole; it is closed here because the dict was
#: already shaped for it.
#:
#: ``_PROFILE_BAND`` is deliberately absent from every entry. See
#: :data:`_PROFILE_BAND` for why an extras key there is unread BY CONSTRUCTION.
_KNOWN_THRESHOLD_EXTRAS: dict[str, tuple[frozenset[str], str]] = {
    # test_numeric._centered_floor, on both graded bands.
    "centered_cosine_floor": (frozenset({"fp32_thresholds", "native_thresholds"}),
                              "test_numeric"),
    # The fp32 cross-backend per-token logprob gap. fp32
    # ONLY -- it is a bound on fp32 arithmetic noise and means nothing at bf16,
    # where the model's own error is four orders larger.
    "logprob_atol": (frozenset({"fp32_thresholds"}), "test_behavioral"),
    # The CUMULATIVE sequence-logprob bound, separately
    # calibrated because the derived `logprob_atol * N` form was mathematically
    # unreachable. fp32 only, for the same reason `logprob_atol` is.
    "logprob_cumulative_atol": (frozenset({"fp32_thresholds"}),
                                "test_behavioral"),
    # The top1-top2 decisiveness margin, NATIVE only -- it
    # exists because bf16 quantises the frozen gaps to ULP multiples, which is not
    # a condition the fp32 tier has.
    "decisive_margin": (frozenset({"native_thresholds"}), "test_behavioral"),
    # The L6 held-out band, read by
    # `test_behavioral.test_held_out_logprobs` against `golden/held_out.npz`:
    # `held_out_logprob_atol` bounds the per-token gap and `held_out_seq_atol` the
    # per-sequence sum. fp32 ONLY, on exactly the argument `logprob_atol` above
    # carries -- at bf16 the same quantity's cross-process spread is the size of
    # the effect, so a native band would grade nothing.
    #
    # LISTED BEFORE THE VALUES EXIST, deliberately and with precedent: the two
    # rows above were added the same way, because this check is an
    # allow-list and an allow-list that lags the work it protects turns into an
    # obstacle to it. An entry here asserts nothing on its own -- the loop only
    # consults it for keys `meta` actually carries -- so until the freeze that
    # measures them lands, these two names cost nothing and block nothing.
    #
    # THAT LAST SENTENCE IS STILL TRUE OF THIS TABLE AND IS NO LONGER TRUE OF THE
    # CHECK. What makes a key MANDATORY is a graded module reading it with a hard
    # subscript, which `_required_extras` derives from the source; this table only
    # says which band each name is legal ON. So a name parked here ahead of its
    # freeze is still inert -- add the reader and it becomes required at the same
    # commit, which is the ordering that cannot go stale in either direction.
    "held_out_logprob_atol": (frozenset({"fp32_thresholds"}),
                              "test_behavioral (L6)"),
    "held_out_seq_atol": (frozenset({"fp32_thresholds"}), "test_behavioral (L6)"),
}

#: The PROFILE band in ``meta``: ``thresholds_for_profile(precision_profile)``, the
#: class-wide band before any per-instance calibration.
#:
#: NO EXTRAS KEY IS LEGAL HERE, and that is a property of the reader rather than a
#: policy: ``Golden.thresholds`` and ``Golden.native_thresholds`` both filter this
#: band through ``fixtures._threshold_fields``, so anything that is not a
#: ``NumericThresholds`` field is dropped before any check can see it. Moving
#: ``centered_cosine_floor: 0.998`` and ``abs_tol.hidden_states: 1.0`` out of
#: ``native_thresholds`` and into this band leaves the consumers reading 0.98 and 0.1
#: -- bit for bit the same 10x loosening a one-character typo produces, which this
#: check catches -- while a band-blind rule would PASS, because it scans all three
#: bands with one rule and declares those names legal in all of them.
_PROFILE_BAND = "thresholds"

#: Every threshold band ``meta`` carries UNCONDITIONALLY, on every instance,
#: whether or not anything is calibrated on it. All three are scanned -- the two
#: graded ones for a misspelled or misfiled key, and the profile band because a
#: name parked there is unread and the reader cannot tell that from a value that
#: applied. The freezer is what makes "unconditional"
#: true here; see :data:`_CONSUMER_NAMESPACES` for the DIFFERENT family this is
#: deliberately not stretched to cover.
_BANDS: frozenset[str] = frozenset(
    {_PROFILE_BAND, "fp32_thresholds", "native_thresholds"})

#: What a band's NAME looks like. Any top-level ``meta`` key matching this and not
#: in :data:`_BANDS` OR :data:`_CONSUMER_NAMESPACES` is a misspelled band: the
#: freezer mirrors every ``instance.yaml`` key into ``meta`` including unknown
#: ones, so a one-character slip creates a stray "band" that no consumer reads
#: and every clause below skips, while the grader falls back to the class profile
#: for the whole band.
_BAND_SHAPED = re.compile(r"^[a-z0-9_]*thresholds?$")

#: Prefix of the per-surface absolute-band overrides. Mirrors
#: ``test_numeric._ABS_TOL_PREFIX``; the surface names themselves are NOT restated
#: here, they are read out of the graded modules by :func:`_named_surfaces`.
_ABS_TOL_PREFIX = "abs_tol."

#: Prefix of the per-surface CENTRED-COSINE overrides. Mirrors
#: ``test_numeric._CENTERED_COSINE_KEY + "."``. Same rule as ``abs_tol.<surface>``
#: above and validated by the same code path: the surface must be one a graded
#: module actually passes, on a band a call site actually grades it at.
#:
#: It exists because one centred cosine cannot serve every array on a tier any
#: more than one absolute band can. A residual stream that peaks at 1560 and
#: collapses to 59 at a deep layer shows why: the cross-framework
#: disagreement is one bf16 ULP of the peak at every layer, but the cancellation
#: turns that into a centred cosine of 0.9931 on the collapsed layers while the
#: LOGITS stay far inside 0.998. Without a per-surface key the only way to admit
#: the residual stream is to loosen the logit clause with it.
_CENTERED_COSINE_SURFACE_PREFIX = "centered_cosine_floor."

#: Every per-surface override prefix, so the validator below treats them alike
#: rather than hardcoding one. A prefix added here without a reader in a graded
#: module still fails: the surface itself is resolved by :func:`_named_surfaces`.
_SURFACE_PREFIXES: tuple[str, ...] = (
    _ABS_TOL_PREFIX, _CENTERED_COSINE_SURFACE_PREFIX)

#: Which graded module owns each ``conftest._APPLICABLE`` key.
#: Read by ``check_applicability_table_names_real_checks``: a gate keyed on a name no
#: module defines deselects nothing and is invisible -- it reads as full coverage.
#: Every key must appear here, so adding a gate without saying who owns it fails.
_APPLICABLE_OWNER: dict[str, str] = {
    # test_numeric.py
    "test_ragged_forward_logits": "test_numeric",
    "test_chunked_prefill_logits": "test_numeric",
    "test_hidden_states": "test_numeric",
    "test_fp32_decode_step_logits": "test_numeric",
    "test_reference_units": "test_numeric",
    "test_prefill_logits_battery": "test_numeric",
    "test_batch_composition_battery": "test_numeric",
    "test_paging_battery": "test_numeric",
    "test_chunk_boundary_battery": "test_numeric",
    # The multimodal input path. Numeric-tier because both compare a VALUE
    # against a frozen artifact -- this module's filing rule -- even though
    # `test_smart_resize_targets`' values are integers and no model runs.
    "test_image_preprocess": "test_numeric",
    "test_smart_resize_targets": "test_numeric",
    # The multimodal MODEL path. Numeric-tier by the same filing rule: each
    # compares a value against a frozen artifact or against another value the
    # same probe produced, even though two of the four compare integers with no
    # tolerance at all.
    "test_vision_tower_units": "test_numeric",
    "test_mrope_positions": "test_numeric",
    "test_image_embedding_merge": "test_numeric",
    "test_multimodal_prefill_logits": "test_numeric",
    # test_spec.py. The only Spec-tier gate, and the reason clause 2 of
    # check_applicability_table_names_real_checks is no longer vacuous.
    "test_static_shape_recompilation": "test_spec",
    # test_behavioral.py
    "test_batching_invariance": "test_behavioral",
    "test_no_cross_sequence_leakage": "test_behavioral",
    "test_chunking_invariance": "test_behavioral",
    "test_prefix_cache_invariance": "test_behavioral",
    # NO ENTRY FOR `test_serve_end_to_end` OR `test_held_out_logprobs`, and their
    # absence is the point rather than an omission: neither is in
    # `conftest._APPLICABLE`, because both always run, and this table is compared
    # to that one KEY FOR KEY below -- an owner without a gate is a stale gate
    # name and fails `check_applicability_table_names_real_checks`.
}


# --------------------------------------------------------------------------- #
# Static helpers. `_parse` is SHARED WITH ``test_spec.py``, which imports it from
# here rather than keeping a second copy: it is five lines, and two copies of "how
# this suite reads a source file" is exactly the kind of duplication that stops
# agreeing.
# --------------------------------------------------------------------------- #
def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None


def _module_functions(path: Path) -> set[str]:
    """Top-level ``def`` names in a test module (AST -- the module is never imported)."""
    tree = _parse(path)
    if tree is None:
        return set()
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


#: ``test_numeric._band``'s tier vocabulary, mapped to the ``meta`` band each tier
#: reads. Restated here rather than imported because this module must stay
#: importable without ``test_numeric``, and asserted against reality by
#: :func:`_named_surfaces`, which fails if a call site names a tier not in here.
_TIER_BAND: dict[str, str] = {"fp32": "fp32_thresholds",
                              "native": "native_thresholds"}


def _named_surfaces(paths) -> dict[str, set[str]]:
    """``{surface: {the meta bands it is actually graded at}}``, read off the source.

    Read off the SOURCE rather than declared in a table, for the same reason
    ``check_applicability_table_names_real_checks`` walks ``test_numeric.py``
    instead of trusting a list: a registry maintained beside the thing it
    describes is a registry that stops describing it. A module that cannot be
    parsed contributes nothing, so a broken file reads as "names no surface"
    (a FAIL for any override that depends on it) rather than raising here.

    THE TIER IS READ TOO, and that is the second half of the same question. A
    surface name that exists is not the same fact as a surface graded at the band
    the override is filed on: ``fp32_thresholds['abs_tol.hidden_states']``
    passes validation while ``hidden_states`` is only ever passed with tier
    ``"native"``, so the value is inert and a name-only check would say it is fine --
    the exact condition this check guards against, one clause over.

    The tier is matched by VOCABULARY (a positional or ``tier=`` string that is a
    key of :data:`_TIER_BAND`) rather than by argument index, so it survives a
    signature change; a surface found with no recognisable tier maps to the empty
    set and the caller reports that distinctly, rather than quietly widening.
    """
    out: dict[str, set[str]] = {}
    for path in paths:
        tree = _parse(Path(path))
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            surface = next(
                (kw.value.value for kw in node.keywords
                 if kw.arg == "surface" and isinstance(kw.value, ast.Constant)
                 and isinstance(kw.value.value, str)), None)
            if surface is None:
                continue
            literals = [a.value for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            literals += [kw.value.value for kw in node.keywords
                         if kw.arg == "tier" and isinstance(kw.value, ast.Constant)
                         and isinstance(kw.value.value, str)]
            out.setdefault(surface, set()).update(
                _TIER_BAND[t] for t in literals if t in _TIER_BAND)
    return out


#: ``Golden`` accessor -> the ``meta`` band it serves. The two extras accessors
#: (``fixtures.Golden.fp32_extras`` / ``.native_extras``) are where a calibrated
#: scalar that belongs to ONE CHECK rather than to the comparator lives, and they
#: are the only surface :func:`_required_extras` reads.
_EXTRAS_BAND: dict[str, str] = {"fp32_extras": "fp32_thresholds",
                                "native_extras": "native_thresholds"}


def _required_extras(paths) -> dict[str, set[str]]:
    """``{band: {extras keys a graded module reads with a HARD subscript}}``.

    THE DIFFERENCE BETWEEN THIS AND :data:`_KNOWN_THRESHOLD_EXTRAS` IS THE WHOLE
    POINT. That table is an ALLOW-list: it says which band each name is legal on,
    and -- as its own comment states -- "an entry here asserts nothing on its own",
    because the loop below only consults it for keys ``meta`` actually CARRIES. A
    key that is simply absent is legal by vacuity. This is the REQUIRE-list, and it
    is derived rather than declared.

    WHY IT HAD TO EXIST. ``test_behavioral`` reads its five calibrated scalars with
    a KeyError-loud ``golden.fp32_extras[...]``, deliberately: "a missing calibrated
    threshold must kill the check, not silently grade at a default nobody measured"
    (that module's header). ``fixtures.fp32_thresholds`` merges the band over
    ``compare.FP32_EXACT_THRESHOLDS`` so a PARTIAL ``NumericThresholds`` band cannot
    do that; ``fp32_extras`` has no floor to merge over and none can be invented,
    since the whole argument for these keys is that nobody may pick their value.
    So the loud KeyError is right -- but it fires inside a GRADED check, and a
    graded check that errors is charged to the candidate.

    With ``held_out_seq_atol`` deleted from BOTH ``instance.yaml`` and
    ``golden/meta.json``: ``check_threshold_extras_name_something_a_check_reads``
    PASSED (it iterates the keys meta carries, and this one is not among them),
    ``check_meta_mirrors_instance_yaml`` PASSED (it compares yaml -> meta only, and
    the yaml no longer has it either), all six self-checks PASSED -- and
    ``test_held_out_logprobs`` raised ``KeyError: 'held_out_seq_atol'``. One graded
    check ERRORS, 32 of 33, charged to a port that did nothing wrong. Neither of
    those three checks in :data:`_KNOWN_THRESHOLD_EXTRAS`'s bands is at fault: an
    allow-list cannot see an absence. This can.

    READ OFF THE SOURCE, for the reason ``_named_surfaces`` is: a hand-maintained
    list of "which calibrated keys are mandatory" is a second place for the answer
    to live and therefore a second place for it to go stale. Specifically, this
    derivation has three properties a table would not:

    * a ``.get(key, default)`` read is an ``ast.Call``, not an ``ast.Subscript``, so
      a key with a DOCUMENTED FALLBACK is automatically not required.
      ``centered_cosine_floor`` is exactly that -- ``test_numeric._centered_floor``
      falls back to the band's own ``cosine_floor``, which is itself a documented
      value from ``meta`` -- so it is legal on BOTH graded bands and required on
      NEITHER, and nothing here has to know that specially;
    * the BAND comes from the accessor the subscript is taken on, so a scalar that
      moves from the fp32 tier to the native one changes which band demands it at
      the same moment it changes which band supplies it;
    * a check that is deleted stops requiring its key, with no second edit.

    Two subscript spellings are resolved: a literal (``extras["logprob_atol"]``) and
    a module-level ``NAME = "literal"`` constant (``extras[_LOGPROB_ATOL_KEY]``,
    which is how every current reader spells it -- the constants carry the
    band-and-rationale docstrings). Anything else -- a computed key, an f-string, a
    name bound inside a function -- is NOT resolved and therefore NOT required, and
    that direction is the safe one: this check can only ever demand a key some
    module provably names.
    """
    out: dict[str, set[str]] = {band: set() for band in _EXTRAS_BAND.values()}
    for path in paths:
        tree = _parse(Path(path))
        if tree is None:
            continue
        # Module-level `NAME = "literal"` bindings, so `extras[_LOGPROB_ATOL_KEY]`
        # resolves. Module level ONLY: a name bound in a function body is not a
        # stable spelling of a threshold key and guessing at one would be the
        # fabricated requirement this check must not make.
        consts: dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, value = node.targets[0], node.value
            if (isinstance(target, ast.Name) and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                consts[target.id] = value.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            base = node.value
            if not (isinstance(base, ast.Attribute) and base.attr in _EXTRAS_BAND):
                continue
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out[_EXTRAS_BAND[base.attr]].add(key.value)
            elif isinstance(key, ast.Name) and key.id in consts:
                out[_EXTRAS_BAND[base.attr]].add(consts[key.id])
    return out


# =========================================================================== #
# The checks. Each names a distinct way this suite can silently stop grading
# things: a golden that went missing, a gate key that names nothing, a threshold
# key nothing reads, a golden frozen against a contract this tree no longer
# carries, a scenario key nobody classified, and a frozen decode run too short for
# the leg that grades it. None of them looks at the candidate.
# =========================================================================== #
def check_golden_manifest(golden):
    """Every golden this MODE requires is present -- PRESERVED artifacts INCLUDED.

    Applicability deselection is golden-presence-driven, so a freeze bug that silently
    drops a required artifact reads as "not applicable" and never dents the pass rate.
    This turns it back into a loud coverage FAIL.

    ``include_preserved=True``, which the freezer's own gate deliberately does not use.
    The canonical freeze runs before the interface freezer, the parameter-norms
    generator and the other producers, so counting their artifacts would abort every
    freeze; by grade time they have all run, and a missing ``interface.json`` or
    ``param_norms.json`` really is a broken golden set.
    Without this, a freeze that dropped either would grade GREEN on a reduced suite --
    every interface and parameter check would simply error, or, worse, a future
    presence-gate would deselect them.

    THE MODE IS CHECKED FIRST, because ``missing_required`` returns ``[]`` for an
    unrecognised mode -- documented, and correct for its other caller, but it means a
    typo'd ``meta['mode']`` would make this assertion pass on an empty golden tree.

    WHAT IS UNIQUE TO THIS CHECK ON A ``core`` INSTANCE, so nobody deletes it as
    redundant and nobody re-derives the answer. ``fixtures.load_golden`` opens every
    FILE-level catalog artifact eagerly, so a missing file raises in the fixture and
    errors the whole tier before this check runs. What is left, and is
    therefore this check's alone here, is every WITHIN-FILE token plus the
    unknown-mode trap above: the file is present and loads fine, and only the
    manifest's token grammar sees that an array inside it is gone.

    The manifest declares every within-file token the graded modules read by name --
    26 arrays in all -- not just a couple. Otherwise, with (for example)
    ``generation.npz::tokens`` and ``held_out.npz::ref_logprobs`` missing, this check
    would PASS, ``missing_required`` would return ``[]``, and both readers would raise
    ``KeyError`` inside a graded check -- an ``error`` in the reward reduction, charged
    to the candidate. See ``golden_manifest._COMMON`` for the derivation and for the
    four families deliberately left out of it.

    The check GROWS on ``ragged`` and ``chunked``, where the required battery axes are
    directory tokens that ``load_golden`` treats as optional (``_load_battery``
    returns ``None`` rather than raising), so a mode that lost its defining axis is
    invisible everywhere else.

    AND IT GROWS WITH THE INSTANCE'S GEOMETRY. ``decode_step.npz`` carries one
    ``logits_step_{i}`` per ``meta['num_decode_steps']``, a set of names no static
    table can hold; ``golden_manifest.meta_dependent_tokens`` derives them from the
    meta this check already has. Kept out of ``missing_required`` on purpose -- that
    function is also the freezer's integrity gate and speaks for a DIRECTORY, while
    this speaks for an INSTANCE.
    """
    mode = golden.mode
    assert mode in golden_manifest.REQUIRED_GOLDENS, {
        "error": f"golden/meta.json declares mode {mode!r}, which the manifest does "
                 "not know; missing_required answers [] for it, so this coverage "
                 "check would pass on any golden tree at all",
        "known_modes": sorted(golden_manifest.REQUIRED_GOLDENS),
    }
    missing = golden_manifest.missing_required(
        golden.dir, mode, include_preserved=True)
    missing += sorted(
        token for token in golden_manifest.meta_dependent_tokens(golden.meta)
        if not golden_manifest._has_token(golden.dir, token))
    assert not missing, {
        "mode": mode,
        "missing_required_goldens": missing,
        "hint": "a complete golden set is the canonical freeze from the golden "
                "freezer PLUS every producer: the interface freezer, the "
                "parameter-norms generator, the held-out freezer "
                "(held_out.npz, L6), and the paged-attention-unit freezer. A "
                "missing one is a coverage regression, not an inapplicable check.",
    }


def check_applicability_table_names_real_checks():
    """``conftest._APPLICABLE``'s bookkeeping -- and, once the graded modules exist,
    that every key names a real function.

    The gate deselects by comparing a key to ``item.name``. A key with a typo matches
    nothing, deselects nothing, and is INVISIBLE -- it reads as full coverage while the
    mode gating it was written for has quietly stopped applying. Nothing else in the
    suite can see that, because the symptom of a broken gate is a test that runs when
    it should not, and on the reference-as-candidate such a test usually passes.

    WHAT THIS VERIFIES TODAY. The verification is narrower than the name suggests,
    and it widened when ``test_numeric.py`` landed:

    * ATTRIBUTION IS TOTAL. Every ``_APPLICABLE`` key is named in
      :data:`_APPLICABLE_OWNER` and every entry there is still a key. It catches a
      ONE-SIDED typo -- a key renamed in the gate table and not here, or the
      reverse. **It does not catch a two-sided one:** rename the key in both
      places, which is the realistic single-edit mistake, and this clause passes.
      That is what the per-module clause below is for.
    * THE PER-MODULE CLAUSE IS LIVE FOR TEN OF THE FIFTEEN KEYS. Both
      ``test_spec.py`` (``test_static_shape_recompilation``, the only Spec-tier
      gate) and ``test_numeric.py`` (the other nine) exist, so those names are
      checked against real functions and a typo fails even when it is made in both
      tables. Typoing one
      ``_APPLICABLE`` key -- with :data:`_APPLICABLE_OWNER` edited to match, i.e.
      the two-sided mistake -- fails this check and only this check.
    * IT IS NO LONGER DORMANT ANYWHERE. ``test_behavioral.py`` exists,
      so the five keys that were waiting on it are live too and ``absent`` is now
      empty on this instance. **EVERY MODULE THIS TABLE NAMES MUST EXIST**, and the
      clause that used to tolerate two absent ones is gone -- see below.

    The AST walk is self-checked against ``test_spec.py`` -- specifically against
    ``test_static_shape_recompilation``, the one name :data:`_APPLICABLE_OWNER`
    attributes to it -- so that neither clause can come back vacuous. It self-checks
    against a graded module rather than ``Path(__file__)`` (this file), because
    a walk of THIS file would prove nothing about the walk of a graded one.
    """
    import conftest

    keys = set(conftest._APPLICABLE)
    assert sorted(keys - set(_APPLICABLE_OWNER)) == [], {
        "error": "an _APPLICABLE gate names no owning module. Add it to "
                 "_APPLICABLE_OWNER so the key can be checked against real function "
                 "names -- a gate keyed on a typo deselects nothing and is invisible.",
        "unattributed": sorted(keys - set(_APPLICABLE_OWNER)),
    }
    assert sorted(set(_APPLICABLE_OWNER) - keys) == [], {
        "error": "_APPLICABLE_OWNER names gates that no longer exist",
        "stale": sorted(set(_APPLICABLE_OWNER) - keys),
    }

    # ...and every gate says WHY it can deselect. `eval_summary.json` reports the
    # kind beside the flat list, and a gate missing from that table would report
    # "unclassified" -- which is the same invisibility this check exists for, one
    # table over: a reader cannot tell a mode gate that will open on another
    # instance from one that is dark on every instance in the class.
    kinds = set(conftest._DESELECTION_KIND)
    assert sorted(keys ^ kinds) == [], {
        "error": "conftest._APPLICABLE and conftest._DESELECTION_KIND disagree. "
                 "Every gate must declare why it can deselect, or eval_summary "
                 "reports it as 'unclassified'.",
        "gated_but_unclassified": sorted(keys - kinds),
        "classified_but_not_a_gate": sorted(kinds - keys),
    }

    # The AST walk must actually see a GRADED module's checks, or "the function is
    # not there" and "the parser found nothing" are the same answer.
    #
    # THE PROBE NAME IS DERIVED, NOT HARDCODED. If it named
    # `test_static_shape_recompilation` literally, a LEGITIMATE rename -- one that
    # updated all three tables, exactly as a careful author would -- would abort the
    # whole precheck with "the AST walk cannot read test_spec.py's test functions",
    # which is false and points at the parser.
    # Reading the probe out of `_APPLICABLE_OWNER` means a consistent rename carries
    # the self-check with it and nothing fires; an INCONSISTENT one still fails, in the
    # per-module clause below, which is the clause whose message is about exactly that.
    probe = _TEST_DIR / "test_spec.py"
    seen = _module_functions(probe)
    probe_names = sorted(name for name, stem in _APPLICABLE_OWNER.items()
                         if stem == "test_spec")
    # TWO CLAUSES, because they have two different repairs and one message cannot
    # serve both.
    #
    # (a) can the walk read the module at all? This is the vacuity guard, and it is
    # the only one entitled to say "the parser found nothing".
    assert len(seen) >= 10, {
        "error": "the AST walk read fewer than 10 top-level functions out of "
                 "test_spec.py, so it is not seeing the module and every per-module "
                 "clause below would pass vacuously. This is a harness/parser fault, "
                 "not a naming one.",
        "probe": str(probe), "found": sorted(seen),
    }
    # (b) does the module still define the name the table attributes to it? A miss
    # here is a table/module disagreement, which is what the loop below reports in
    # full -- so this clause only has to avoid claiming the parser broke.
    assert all(name in seen for name in probe_names), {
        "error": "test_spec.py does not define the check(s) _APPLICABLE_OWNER "
                 "attributes to it. If you RENAMED one, rename it in all three "
                 "tables together -- conftest._APPLICABLE, "
                 "conftest._DESELECTION_KIND and _harness_checks._APPLICABLE_OWNER -- "
                 "and this self-check follows automatically, because it reads the "
                 "probe name out of _APPLICABLE_OWNER rather than restating it. The "
                 "AST walk itself is fine: it read "
                 f"{len(seen)} functions out of this module.",
        "declared_by_the_table": probe_names,
        "missing_from_the_module": sorted(set(probe_names) - seen),
        "module_defines": sorted(n for n in seen if n.startswith("test_"))[:20],
    }

    absent, wrong = [], {}
    cache: dict[Path, set[str]] = {probe: seen}
    for name, stem in sorted(_APPLICABLE_OWNER.items()):
        module = _TEST_DIR / f"{stem}.py"
        if not module.exists():
            absent.append(stem)
            continue
        functions = cache.setdefault(module, _module_functions(module))
        if name not in functions:
            wrong[name] = {"declared_module": f"{stem}.py",
                           "module_defines": sorted(n for n in functions
                                                    if n.startswith("test_"))[:20]}
    assert not wrong, {
        "error": "an _APPLICABLE gate names a function its module does not define, so "
                 "it deselects nothing",
        "broken_gates": wrong,
    }
    # A GRADED MODULE THIS TABLE NAMES MUST BE ON DISK. An allow-list of modules
    # permitted to be absent becomes a standing hole once those modules are written,
    # so there is none.
    #
    # If an allow-list tolerated it, deleting `test_behavioral.py` from the image and
    # grading the reference would report `overall PASS`, `n_failed 0`, and a passing
    # reward, every self-check green. Seventeen graded checks -- the whole Behavioral
    # tier, including the benchmark's named equivalence claim -- would simply vanish,
    # and the only trace would be the `deselected` count dropping, which nothing
    # compares against anything. A collection ERROR is safe (pytest aborts the
    # session, reward 0.0); it is ABSENCE that was unguarded.
    #
    # NARROWED TO EMPTY, and the set is DERIVED rather than restated: an instance's
    # graded modules are exactly the modules `_APPLICABLE_OWNER` attributes gates to.
    # A future mode whose graded module has not been written adds no gate keys for it,
    # so it names no module here and this clause stays satisfied -- the tolerance an
    # allow-list would reach for, without the standing hole.
    #
    # THIS IS THE SPECIFIC BUG. The CLASS of bug -- a module that is present but stops
    # being collected -- is floored separately, in `conftest._require_complete_collection`.
    assert not absent, {
        "error": "a graded module this table depends on is missing from the tests "
                 "tree. Every check it holds is silently absent from the grade, and "
                 "the reward is a pass rate over whatever is LEFT -- so a suite that "
                 "lost a whole tier still reports 1.0. This is a broken image or a "
                 "broken checkout, not a candidate defect.",
        "absent": sorted(set(absent)),
        "expected_modules": sorted(set(_APPLICABLE_OWNER.values())),
        "tests_dir": str(_TEST_DIR),
    }


def check_threshold_extras_name_something_a_check_reads(golden):
    """Every per-instance threshold key is one a check actually consumes.

    THE SECOND TYPO'D TABLE, and it fails in the direction that matters. A
    ``NumericThresholds`` field is validated by the constructor, but
    ``instance.yaml``'s bands may also carry EXTRAS -- keys the shared comparator
    has no slot for, served through ``Golden.native_extras`` / ``fp32_extras`` and
    read by name in ``test_numeric``. Nothing validated those names:
    ``fixtures._threshold_extras`` accepts any key that is not a dataclass field,
    and the freezer mirrors the section verbatim, so it does not see it either.

    A ONE-CHARACTER typo -- ``centered_cosine_floorr``, and
    ``abs_tol.hidden_state`` for the ``s`` -- silently drops the native
    centered-cosine floor from 0.998 to 0.98 and the hidden-state absolute band
    from 1.0 to 0.1, because ``_centered_floor`` and ``_surface_band`` both fall
    back to the tier default when the key is absent. Nothing raises and every check
    still passes -- a 10x loosening back to precisely the value ``instance.yaml``'s
    argument was written to tighten away from, with no trace in the summary or in
    the pass rate.

    THE ASYMMETRY IS WHY IT READS AS SAFE, so it is written down: on the FP32 band
    the fallback (``cosine_floor`` 0.999999) is TIGHTER than the override
    (0.999995), so a typo there fails CLOSED -- the tier goes red and someone looks.
    Only the native direction fails open, and native is the tier where the fallback
    is the class budget the calibration argued against.

    Three legal shapes, and nothing else:

    * a ``compare.NumericThresholds`` field -- the constructor's own key set, read
      off the dataclass rather than transcribed;
    * a name in :data:`_KNOWN_THRESHOLD_EXTRAS`, ON A BAND THAT TABLE ALLOWS. It
      includes the two keys that are calibrated later, so this check
      cannot become an obstacle to the work it protects, but ``logprob_atol`` is an
      fp32 quantity and ``decisive_margin`` a native one, and each is legal only
      where its consumer will look;
    * ``abs_tol.<surface>`` where ``<surface>`` is a string some graded module
      passes as ``surface=`` **at this band's tier** -- both facts read out of the
      SOURCE by :func:`_named_surfaces`, not from a table here, so a per-surface
      band whose consumer was renamed, deleted or moved to the other tier stops
      being legal at the same moment. Without the tier half,
      ``fp32_thresholds['abs_tol.hidden_states']`` would pass while ``hidden_states``
      is only ever graded ``"native"``, leaving the value inert.

    ...and none of the three is legal on the PROFILE band, which takes
    ``NumericThresholds`` fields only. That clause closes the one defeat of this
    check that a reviewer found: an extras key parked in ``meta['thresholds']``
    passed validation while the consumer fell back to the class default, which is
    the same 10x loosening the typo produces. See :data:`_PROFILE_BAND`.
    It is reachable only from a hand-edited ``meta.json`` today --
    the freezer overwrites that band with
    ``asdict(thresholds_for_profile(...))``, so ``instance.yaml`` cannot express it
    -- and it is closed anyway, because "the freezer happens to overwrite it" is a
    property of another file that this check would not notice losing.

    AND THE BAND NAMES THEMSELVES, first of all, in two clauses. Everything above
    validates keys INSIDE the three known bands, which a misspelled band name
    escapes completely: it has no keys inside it, the loop reads an empty dict, and
    the grader falls back to the class profile for the whole band.

    THE ``instance.yaml`` HALF OF THAT IS NOT CLOSED HERE. The freezer writes all
    three bands UNCONDITIONALLY, so a misspelled yaml band produces a meta carrying
    the typo AND the canonical name (backfilled with class defaults), and
    the membership gate below passes on all four spellings. That path is refused at
    the producer instead, by the freezer's instance-key validation.
    What is left for these two clauses is a HAND-EDITED ``meta.json``, which the
    producer cannot see:

    * the OUTER gate is MEMBERSHIP: all three bands present by name. It does not
      have to recognise the corruption, only notice what went missing.
      **An instance that calibrates nothing on a band must write it empty**
      (``native_thresholds: {}``); absent and misspelled must not look the same,
      and the freezer emits all three for exactly that reason.
    * the INNER gate is SHAPE: no fourth band-shaped key. Membership cannot see a
      meta that carries both ``native_thresholds`` and a stray
      ``native_threshold``. Without this clause the check would pass while the
      grader's fp32 ``abs_tol`` goes
      1.0e-4 -> 1.0e-3 and its centered floor 0.9999995 -> 0.999999.

    Both are here because the same failure recurs at three different depths -- a
    typo'd key that is read by nobody and loosens what it was written to tighten --
    inside a band, across bands, and now the band names.

    AND THE OTHER DIRECTION, which every clause above is blind to: A KEY A READER
    DEMANDS AND META DOES NOT CARRY. Everything so far validates the keys meta HAS.
    A key it lacks is legal by vacuity here -- and five of them are read with a
    KeyError-loud ``golden.fp32_extras[...]`` / ``native_extras[...]`` by design,
    because a calibrated bound has no honest default to fall back to. That
    ``KeyError`` lands inside a GRADED check, which the reward reduction counts as
    an ``error`` and charges to the candidate. :func:`_required_extras` derives the
    mandatory set FROM THOSE READERS -- hard subscript = required, ``.get(key,
    default)`` = not -- and the clause built on it moves the failure to where the
    defect is. See that function for the derivation.

    Torch-free, candidate-free and golden-read-only: it costs a JSON parse and two
    AST walks, which is why it is a Spec check and not a Numeric one.
    """
    fields = set(compare.NumericThresholds.__dataclass_fields__)
    # The surfaces the graded modules actually name. `test_behavioral.py` is
    # included by path so the clause arms itself if that module introduces one.
    modules = [_TEST_DIR / "test_numeric.py", _TEST_DIR / "test_behavioral.py"]
    # BOTH AST DERIVATIONS BELOW ANSWER "NOTHING" FOR A MODULE THAT WILL NOT PARSE,
    # which is the safe direction for `_named_surfaces` (an override with no reader
    # FAILS) and the unsafe one for `_required_extras` (a requirement with no reader
    # simply disappears). Refuse the unparsable module outright instead: a graded
    # module this file cannot parse is one pytest cannot import either, so there is
    # no world in which reading it as "requires nothing" is the right answer.
    unparsable = [str(p) for p in modules if p.exists() and _parse(p) is None]
    assert not unparsable, {
        "error": "a graded module beside this check could not be parsed, so the "
                 "AST derivations below cannot see which thresholds it reads. "
                 "That reads as 'this module requires nothing', which would let "
                 "an uncalibrated golden through; refused instead.",
        "unparsable": unparsable,
    }
    surfaces = _named_surfaces(modules)

    # The walk must be able to SEE a surface, or "no module names this one" and
    # "the walk found nothing" are the same answer and every override below would
    # fail for the wrong reason. Only assert it when there is an override to check.
    bands = {name: dict(golden.meta.get(name) or {}) for name in sorted(_BANDS)}
    has_surface_override = any(
        key.startswith(_SURFACE_PREFIXES)
        for band in bands.values() for key in band)
    assert surfaces or not has_surface_override, {
        "error": "meta declares a per-surface absolute band but the AST walk found "
                 "no `surface=` literal in any graded module, so this check cannot "
                 "tell a real surface from a typo",
        "modules_scanned": [str(p) for p in modules if p.exists()],
    }

    # THE BAND NAMES THEMSELVES, before their contents, in two clauses -- an OUTER
    # membership gate and an INNER shape gate. Everything below hardens key names
    # INSIDE the three known bands, and a band whose NAME is wrong has no keys
    # inside it at all: the loop reads an empty dict and passes while the grader
    # falls back to the class profile for that whole band. Unlike the profile-band
    # clause this is reachable from `instance.yaml` -- the freezer mirrors EVERY
    # yaml key into meta, unknown ones included, by design.
    #
    # THE OUTER GATE: all three bands are PRESENT, by name. It covers a meta whose
    # band key is absent however that happened -- which today means a HAND-EDITED
    # meta.json, and nothing else.
    #
    # THE OUTER GATE ALONE DOES NOT CLOSE THE YAML PATH. Through the freezer, a
    # misspelling like `fp32_thresholds:` -> `fp32_thresholdz:` in `instance.yaml`
    # yields a meta carrying BOTH keys -- the typo (mirrored inert by the freezer)
    # and the canonical one, because the freezer writes all three bands
    # UNCONDITIONALLY and backfills the class defaults. This gate sees three bands
    # and PASSES. All four spellings pass. `missing_bands` cannot fire on any meta
    # the freezer can produce.
    #
    # SO THE YAML PATH IS CLOSED AT THE PRODUCER, not here, and it has to be: a
    # checker reading only `meta.json` structurally cannot tell "calibrated nothing"
    # from "misspelled it" once the defaults have been backfilled. The freezer
    # refuses the freeze outright for any yaml key the class does not recognise --
    # an allow-list at the ORIGIN is the shape that ends this family, where a pattern
    # here is the shape that keeps needing another round.
    #
    # THIS CLAUSE IS KEPT ANYWAY, and the rule is deliberate: the producer gate cannot
    # see a hand-edited `golden/meta.json`, and this one can. Two different inputs,
    # two gates. This clause does not by itself close the family.
    missing_bands = sorted(_BANDS - set(golden.meta))
    assert not missing_bands, {
        "error": "meta is missing a threshold band by name. Every band the grader "
                 "reads is served through `meta.get(...)`, so an absent one is an "
                 "empty dict and the whole band silently falls back to the class "
                 "profile -- which is where the calibration argued AGAINST. Any "
                 "corruption of the name lands here: a typo, a trailing space, a "
                 "capital, a truncation.",
        "missing_bands": missing_bands, "meta_keys": sorted(golden.meta),
        "hint": "fix the spelling in tests/instance.yaml and re-freeze meta.json. "
                "An instance that calibrates nothing on a band writes it EMPTY "
                "(`native_thresholds: {}`) rather than omitting it -- absent and "
                "misspelled must not look the same.",
    }

    # THE INNER GATE: no FOURTH band-shaped key. The membership gate above cannot
    # see this one -- a yaml carrying both `native_thresholds` and a stray
    # `native_threshold` satisfies it while half the calibration sits in a key
    # nothing reads. With one character wrong (`fp32_thresholds` ->
    # `fp32_threshold`), without this clause the check passes while the grader's fp32
    # `abs_tol` goes 1.0e-4 -> 1.0e-3 and its centered floor 0.9999995 -> 0.999999
    # -- the same 10x silent loosening the key-name clauses exist for, and a
    # fallback that no longer rejects the bf16-rotary port this instance's centered
    # floor was set to catch.
    stray_bands = sorted(
        name for name in golden.meta
        if _BAND_SHAPED.match(name) and name not in _BANDS
        and name not in _CONSUMER_NAMESPACES)
    assert not stray_bands, {
        "error": "meta carries a band-shaped key that no consumer reads. A "
                 "misspelled BAND name is invisible to every clause below -- they "
                 "validate keys inside the three known bands, and a fourth band's "
                 "keys are never looked at -- while the grader silently falls back "
                 "to the class profile for that whole band.",
        "stray_bands": stray_bands, "known_bands": sorted(_BANDS),
        "recognised_consumer_namespaces": sorted(_CONSUMER_NAMESPACES),
        "hint": "fix the spelling in tests/instance.yaml and re-freeze meta.json",
    }

    # THE REQUIRE-LIST, and it is the other half of this check's name. Everything
    # above and below asks "is this key that meta CARRIES legal?"; a key meta does
    # not carry is legal by vacuity, and a KeyError-loud reader then errors a GRADED
    # check for an instance-authoring omission. See :func:`_required_extras` for the
    # derivation.
    required = _required_extras(modules)

    # THE TWO TABLES MUST AGREE FIRST. A derived requirement on a band the
    # allow-list forbids is a contradiction inside this file -- one clause would
    # demand the key and the next would reject it as misfiled, so the instance
    # could not be authored at all. Reported as its own finding, because the repair
    # is an edit HERE (or in the reader) and not to any golden.
    contradictory = sorted(
        f"{band}.{key}"
        for band, keys in required.items() for key in keys
        if band not in _KNOWN_THRESHOLD_EXTRAS.get(key, (frozenset(), ""))[0])
    assert not contradictory, {
        "error": "a graded module reads a threshold extra on a band that "
                 "_KNOWN_THRESHOLD_EXTRAS does not declare legal there, so this "
                 "check would demand the key and then reject it. Fix the table or "
                 "the reader; no golden can satisfy both clauses as they stand.",
        "contradictory": contradictory,
        "known_extras": {k: sorted(v[0]) for k, v in
                         sorted(_KNOWN_THRESHOLD_EXTRAS.items())},
    }

    absent = sorted(
        f"{band}.{key}"
        for band, keys in required.items() for key in keys
        if key not in bands[band])
    assert not absent, {
        "error": "a graded module reads this calibrated threshold with a "
                 "KeyError-loud `[...]` and meta does not carry it. There is no "
                 "default to fall back to -- the entire argument for these keys is "
                 "that nobody may pick their value -- so the reader is right to "
                 "raise and this is where it has to be caught: a KeyError inside a "
                 "graded check is an ERROR charged to the CANDIDATE for an "
                 "instance-authoring omission.",
        "absent": absent,
        "carried": {band: sorted(bands[band]) for band in sorted(_EXTRAS_BAND.values())},
        "readers": {key: _KNOWN_THRESHOLD_EXTRAS[key][1]
                    for band, keys in sorted(required.items()) for key in sorted(keys)
                    if key not in bands[band] and key in _KNOWN_THRESHOLD_EXTRAS},
        "hint": "calibrate the value, record it in tests/instance.yaml under the "
                "band named above WITH its measured floor and ratio, and re-freeze "
                "meta.json. A key read with `.get(key, default)` instead -- like "
                "`centered_cosine_floor`, whose fallback is the band's own "
                "documented `cosine_floor` -- is deliberately NOT required here.",
    }

    unknown: dict[str, dict] = {}
    for band, values in sorted(bands.items()):
        for key in sorted(values):
            if key in fields:
                continue
            # THE PROFILE BAND TAKES NO EXTRAS AT ALL -- see _PROFILE_BAND. Checked
            # before the two allow-lists, because both of those answer "is this name
            # legal SOMEWHERE" and the answer here is "not in this band, whatever
            # the name".
            if band == _PROFILE_BAND:
                unknown[f"{band}.{key}"] = {
                    "reason": "the profile band carries only NumericThresholds "
                              "fields; fixtures._threshold_fields filters this band "
                              "for every consumer, so an extras key here is read by "
                              "nobody and the check that wanted it silently falls "
                              "back to the class default",
                    "numeric_thresholds_fields": sorted(fields),
                    "put_it_in": sorted(_BANDS - {_PROFILE_BAND}),
                }
                continue
            if key in _KNOWN_THRESHOLD_EXTRAS:
                legal_bands, owner = _KNOWN_THRESHOLD_EXTRAS[key]
                if band in legal_bands:
                    continue
                unknown[f"{band}.{key}"] = {
                    "reason": "a known extras key, on the wrong band: the check "
                              "that reads it reads the other one, so this value is "
                              "inert and that check falls back to its default",
                    "read_by": owner, "legal_bands": sorted(legal_bands),
                }
                continue
            prefix = next((p for p in _SURFACE_PREFIXES if key.startswith(p)), None)
            if prefix is not None:
                surface = key[len(prefix):]
                graded_at = surfaces.get(surface)
                if graded_at is None:
                    unknown[f"{band}.{key}"] = {
                        "reason": "no graded module passes this surface name",
                        "surface": surface, "known_surfaces": sorted(surfaces),
                    }
                elif not graded_at:
                    unknown[f"{band}.{key}"] = {
                        "reason": "the AST walk found this `surface=` but no tier "
                                  "literal beside it, so which band grades it is "
                                  "unknown -- the walk has stopped tracking the "
                                  "call signature (see _named_surfaces/_TIER_BAND)",
                        "surface": surface,
                    }
                elif band not in graded_at:
                    unknown[f"{band}.{key}"] = {
                        "reason": "a real surface, on a band no call site grades it "
                                  "at: the override is inert and the check that "
                                  "wanted it falls back to the tier default",
                        "surface": surface, "graded_at": sorted(graded_at),
                    }
                continue
            unknown[f"{band}.{key}"] = {
                "reason": "not a NumericThresholds field, not a known extras key, "
                          "and not <prefix><surface> for any per-surface prefix",
                "numeric_thresholds_fields": sorted(fields),
                "known_extras": sorted(_KNOWN_THRESHOLD_EXTRAS),
                "surface_prefixes": list(_SURFACE_PREFIXES),
            }
    assert not unknown, {
        "error": "a per-instance threshold key is read by nothing. On the native "
                 "band this FAILS OPEN -- the check that would have used it falls "
                 "back to the class default, silently loosening the gate the "
                 "calibration was written to tighten. This is an instance.yaml "
                 "defect, not a candidate defect.",
        "unread_keys": unknown,
        "hint": "fix the spelling in tests/instance.yaml and re-freeze meta.json, "
                "or add the name to _harness_checks._KNOWN_THRESHOLD_EXTRAS if a "
                "check really does read it",
    }


#: (dotted meta path, the check that reads it, the meta key gating that check).
#: Every entry is a key some graded check dereferences. Adding a read site without
#: adding a row here is the defect this table exists to make impossible.
#:
#: THE GATE FIELD IS NOT OPTIONAL POLISH. A check that is deselected does not
#: dereference anything, so requiring its threshold would demand a calibrated number
#: from an instance that has not run the calibration yet -- which is precisely how a
#: placeholder threshold gets invented and then never revisited. `None` means the
#: check is always collected and the key is always required.
_THRESHOLD_CONSUMERS: tuple[tuple[str, str, str | None], ...] = (
    ("perf_thresholds.delta", "test_perf.py::test_decode_speed", "perf_graded"),
)

#: Top-level ``meta`` namespaces :data:`_THRESHOLD_CONSUMERS` already owns --
#: derived from the dotted key before its first ``.`` rather than restated, so a
#: namespace that table adds or renames is recognised here with no second edit.
#:
#: THE BUG THIS TABLE CLOSES. ``perf_thresholds`` matches :data:`_BAND_SHAPED`
#: (``^[a-z0-9_]*thresholds?$``) but is not one of :data:`_BANDS`, so the
#: ``stray_bands`` clause in
#: :func:`check_threshold_extras_name_something_a_check_reads` would refuse every
#: grade the moment ``perf_graded: true`` makes ``meta['perf_thresholds']`` a real
#: dict. Without this exemption, no instance could ever set ``perf_graded: true``
#: and pass precheck, because the ``stray_bands`` clause would reject
#: ``perf_thresholds`` outright.
#:
#: WHY THIS IS A NAMESPACE EXEMPTION, NOT AN ADDITION TO ``_BANDS``. ``_BANDS``'s
#: own ``missing_bands`` clause (below) demands every named band be PRESENT, by
#: name, on EVERY instance's ``meta.json`` -- true today of the three legacy
#: bands only because the freezer writes them
#: unconditionally on every freeze. A consumer namespace is the opposite: it is
#: legal only where its own ``_THRESHOLD_CONSUMERS`` gate says so (here,
#: ``perf_graded``), and it is ABSENT on every instance that has not calibrated
#: it yet. Adding ``perf_thresholds`` to
#: ``_BANDS`` would have made ``missing_bands`` fire on every one of them, today,
#: for a key this check has no business demanding in phase 1 -- trading one
#: false refusal for another. Nor does exempting it here ask this check to
#: validate WHAT is inside ``perf_thresholds`` (e.g. that ``delta`` is a number):
#: ``check_every_threshold_consumer_has_a_key`` already owns that, correctly
#: gated on ``perf_graded``, and a namespace this table names is trusted to
#: police its own contents rather than being re-validated as if it were a
#: ``NumericThresholds`` band.
_CONSUMER_NAMESPACES: frozenset[str] = frozenset(
    dotted.split(".", 1)[0] for dotted, _, _ in _THRESHOLD_CONSUMERS)


def check_every_threshold_consumer_has_a_key(golden):
    """Every threshold a graded check READS is declared by this instance.

    THE MIRROR OF `check_threshold_extras_name_something_a_check_reads`, and it fails
    in the direction that one does not. That check catches a DECLARED key nothing
    reads -- a typo that silently drops a band to its default. This catches a READ key
    nothing declared, which has two equally bad outcomes: the read raises inside the
    grade, and a harness fault is charged to the port as a failed check; or the read
    is written defensively with a fallback, and every instance that forgot to declare
    the key is silently graded at the loosest value in the class.

    THE SECOND OUTCOME IS THE ONE THAT MOTIVATED THIS. A threshold set that declares
    only the keys an oracle grade happened to force leaves the rest at a class default,
    invisibly, and the symptom is a number that looks calibrated on one instance and is
    absent on five. There is no class default for `perf_thresholds.delta` -- this check
    is what makes that statement enforceable rather than aspirational.

    Fatal, in `_grade_precheck`, writing no `reward.json`: a missing threshold is a
    statement about the TASK, not about the port.
    """
    meta = getattr(golden, "meta", None) or {}
    missing = []
    for dotted, reader, gate in _THRESHOLD_CONSUMERS:
        if gate is not None and not meta.get(gate):
            continue          # the reader is deselected on this instance
        node = meta
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is None:
            missing.append(f"{dotted} (read by {reader})")
    assert not missing, (
        "this instance's golden/meta.json declares no value for "
        + "; ".join(missing)
        + ". There is NO class default for these keys, by design -- see this check's "
          "docstring. Declare it in instance.yaml and re-freeze.")


def check_perf_golden_if_graded(golden):
    """An instance that GRADES speed must CARRY the golden that gates it.

    THE MIRROR OF `check_every_threshold_consumer_has_a_key`, one artifact over: that
    check refuses a READER with no declared threshold; this refuses a GRADING SWITCH
    with no golden to grade against. Both exist because a phase-1/phase-2 flag and the
    data it depends on are two separate facts that can drift apart, and both times the
    drift is invisible in the pass rate rather than loud.

    WHY THIS IS NEEDED. `perf.json` is OPTIONAL in `golden_manifest` rather than
    REQUIRED: `core_qwen3`'s reference cannot resolve a slope at this K-range, and
    requiring the golden anyway would brick the whole instance (see that module's own
    comment). That is the RIGHT call for `core_qwen3`, but it means that on a
    **phase-2** instance (`perf_graded: true`), deleting `perf.json` no longer refuses
    anything: `conftest._golden_facts['has_perf_golden']` reads `False`,
    `test_decode_speed`'s gate (`perf_graded and has_perf_golden`) deselects the check,
    and the grade completes with the speed gate gone -- read as full coverage. That is
    this class's own named failure, stated in the `_DESELECTION_KIND` docstring one file
    over: "a gate that can never open reads as coverage."

    So the two facts are recombined here, at the same severity `check_golden_manifest`
    and `check_every_threshold_consumer_has_a_key` already use: an instance that
    DECLARES `perf_graded: true` must HAVE `golden/perf.json`, or the precheck refuses
    before pytest ever runs. A missing golden on a graded instance is a statement about
    the TASK -- nobody froze the reference, or it was deleted after the fact -- not
    about the port sitting in front of it.

    NOTHING FIRES TODAY. No committed `instance.yaml` sets `perf_graded: true`, so
    `meta.get("perf_graded")` is falsy on all six instances and this check is vacuously
    satisfied everywhere. It exists so the FIRST instance to flip the switch cannot
    ship without the golden that makes the switch mean anything, the same way
    `_KNOWN_THRESHOLD_EXTRAS`'s later-added rows were declared before their readers so
    as not to become an obstacle to the work they protect.

    Fatal, in `_grade_precheck`, writing no `reward.json` -- the same disposition
    `check_every_threshold_consumer_has_a_key` uses for the same reason.
    """
    meta = getattr(golden, "meta", None) or {}
    if not meta.get("perf_graded"):
        return          # phase 1 (or not yet calibrated): this check has nothing to say
    assert golden_manifest._has_token(golden.dir, "perf.json"), {
        "error": "golden/meta.json declares perf_graded: true, but golden/perf.json is "
                 "absent. perf.json is OPTIONAL in golden_manifest -- an instance whose "
                 "reference cannot resolve a slope (core_qwen3) degrades to a coverage "
                 "gap rather than an all-or-nothing refusal -- but an instance that has "
                 "already calibrated a threshold and flipped perf_graded to True has no "
                 "such excuse: without the golden, test_decode_speed silently deselects "
                 "(has_perf_golden is False) and the grade completes with the speed "
                 "gate gone, reading as full coverage. A gate that can never open must "
                 "not read as coverage.",
        "hint": "freeze golden/perf.json with the golden freezer before "
                "shipping perf_graded: true, or revert perf_graded until it is.",
    }


#: ``instance.yaml`` keys the freezer deliberately does NOT mirror into ``meta``:
#: two authoring-host paths, the golden directory (``meta.json`` is IN it), and
#: ``graded_checks`` -- a fact about the tests TREE rather than about the freeze,
#: so mirroring it would put it inside ``check_meta_mirrors_instance_yaml``'s
#: comparison and make adding a graded check demand a re-freeze of every golden.
#: IT IS NOW READ AT GRADE TIME, and that is not an argument for mirroring it:
#: ``conftest._instance_graded_check_count`` reads the YAML directly (the verifier
#: image bakes the whole ``tests/`` directory), exactly as the host grading path
#: does, so the in-container gate and the host gate share one number
#: and neither needs meta to carry it.
#: RESTATED from the freezer's own exemption table rather than imported -- this
#: module ships inside the verifier image, which cannot import the freezer -- and
#: held to EQUALITY with it by a freezer-side test that runs on the AUTHORING HOST,
#: the only place both halves are importable; it is not in this tree and cannot be.
#: Adding a key here that the freezer still mirrors is the SILENT direction --
#: ``check_meta_mirrors_instance_yaml`` below would just stop comparing it -- which
#: is why the guard is equality.
_INSTANCE_KEYS_NOT_MIRRORED: frozenset[str] = frozenset(
    {"torch_orig", "jax_repo", "golden_dir", "graded_checks"})

#: The one band the freezer MERGES rather than mirrors: ``meta['fp32_thresholds']``
#: is ``compare.FP32_EXACT_THRESHOLDS`` overridden by the yaml's block, so meta
#: legitimately carries keys the yaml does not. Compared as a subset for that reason.
_MERGED_INSTANCE_KEYS: frozenset[str] = frozenset({"fp32_thresholds"})


def check_meta_mirrors_instance_yaml(golden):
    """``golden/meta.json`` is a freeze of the ``instance.yaml`` sitting beside it.

    THE STALE-GOLDEN GATE, and it is the only one there is. **Nothing in the graded
    path reads ``instance.yaml``**: ``fixtures.py`` takes ``sampler_grid``, the
    geometry, the thresholds and ``repo_config`` from ``meta.json``, and
    ``_grade_precheck`` opened only ``meta.json`` too. So a host holding OLD goldens
    beside a NEW ``instance.yaml`` -- the exact state the artifact-upload ordering
    rule warns about -- graded silently green against the old contract.

    Without this gate, rolling the sampler goldens back to a previous 4-row grid
    against a 6-row ``instance.yaml`` and grading the reference passes precheck
    ("all self-checks pass") and grades fully green -- a stale host silently reopens
    the hole the newer grid closed. Harbor does not mask it: Harbor grades from the
    host ``tests/`` tree, which is where stale goldens live.

    There is no grid-length assertion that catches this combination; this check is
    what catches it.

    WHAT IT COMPARES. Every yaml key except the three authoring paths the freezer
    deliberately drops, value for value, JSON-normalised. ``fp32_thresholds`` is
    compared as a SUBSET because the freezer merges it over the class defaults.

    WHAT IT DOES NOT COMPARE, stated rather than left to be discovered: the
    direction is yaml -> meta only. A key DELETED from the yaml leaves a stale
    mirror in meta that this cannot see, because meta legitimately carries a dozen
    keys the yaml never had (``repo_config``, ``thresholds``, ``hashes``,
    ``provenance``, ``artifacts``, ``fp32_kv_cache_dtype``). Nor does it verify that
    the ARRAYS were re-frozen -- it verifies that the CONTRACT they were frozen
    against is the one in the tree. A hand-edited meta that merely copies the yaml
    would satisfy it; the freezer's own ``--check`` is what answers byte identity.

    NO ``instance.yaml`` AT ALL IS A FAILURE, not a pass. The verifier image bakes
    the whole ``tests/`` directory, so its absence means the image was built wrong,
    and "the file I compare against is missing" must not read as "they agree".

    ``perf_k_values`` is mirrored like every other geometry key. The perf stage
    reads it from ``meta.json``, not from ``instance.yaml``, because nothing in
    the graded path may read ``instance.yaml`` -- a host with stale goldens can
    otherwise grade the reference against a previous contract, which is why
    ``check_meta_mirrors_instance_yaml`` exists. There is no ``perf_page_ids``
    counterpart: nothing reads it (``perf_probe`` drives the candidate's own
    allocator).
    """
    import yaml                                  # pinned via flax

    path = _TEST_DIR / "instance.yaml"
    assert path.is_file(), {
        "error": "there is no instance.yaml beside this tests tree, so nothing can "
                 "say whether golden/meta.json is a freeze of the current contract. "
                 "The verifier image copies the whole tests/ directory; an absent "
                 "instance.yaml is a broken image, not an absent requirement.",
        "looked_for": str(path),
    }
    inst = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta = golden.meta
    drift: dict[str, dict] = {}
    for key in sorted(set(inst) - _INSTANCE_KEYS_NOT_MIRRORED):
        want = json.loads(json.dumps(inst[key]))
        if key in _MERGED_INSTANCE_KEYS:
            got = dict(meta.get(key) or {})
            for name in sorted(want or {}):
                if got.get(name) != want[name]:
                    drift[f"{key}.{name}"] = {"instance_yaml": want[name],
                                              "golden_meta": got.get(name)}
            continue
        if meta.get(key) != want:
            drift[key] = {"instance_yaml": want, "golden_meta": meta.get(key)}
    assert not drift, {
        "error": "golden/meta.json does not match the instance.yaml beside it, so "
                 "this golden set was frozen against a DIFFERENT contract than the "
                 "one in this tree. Nothing in the graded path reads instance.yaml, "
                 "so without this the grade would run green against the old "
                 "contract -- which is how a stale host silently loses the coverage "
                 "a re-freeze was performed to add.",
        "drift": drift,
        "hint": "re-freeze the goldens from the reference implementation, or run "
                "`git lfs pull` to fetch the current goldens. If you are the author, "
                "push the goldens BEFORE the instance.yaml that describes them.",
    }


def check_scenario_keys_are_classified(golden):
    """Every ``golden/scenario.json`` key is classified as an INPUT or an OUTPUT.

    ``fixtures.build_candidate`` builds the sandbox payload by intersecting the
    frozen scenario with ``_SCENARIO_INPUT_KEYS``, and the claim that rests on that
    intersection is the strong one: *no frozen OUTPUT crosses into the sandbox*.
    A key in neither list is therefore a decision nobody made -- an INPUT the
    candidate silently never receives (a probe reports ``unsupported`` for no
    reason a reader can find), or a frozen ANSWER that nobody looked at.

    WHY IT IS CHECKED HERE RATHER THAN RAISED AT THE FIXTURE. Raised as a
    ``RuntimeError`` inside ``build_candidate``, it reaches the graded suite through
    the SESSION-SCOPED ``cand`` fixture: raising there errors every check that
    requests ``cand`` -- 29 of the 31 on this instance, since only ``test_interface``
    and ``test_repo_structure`` are static -- for the non-importing floor, i.e. the
    score for a candidate that does not exist, awarded for a defect in the TASK's own
    freeze. Loud beats the silent drop, but at the wrong target; this module is the
    one that says "the grade did not happen and the repair is to the task".

    WHAT ``build_candidate`` STILL DOES, so the residual is not a surprise: it
    copies the allow-listed keys and drops anything else. That direction is the
    safe one for the claim above -- an unclassified key does NOT cross -- and the
    only thing lost is loudness, which is exactly what this check restores.

    THE TWO TUPLES ARE READ OUT OF ``fixtures``, never restated. A copy here would
    be a third table to keep in sync with a partition whose whole point is that it
    is total.
    """
    import fixtures

    inputs = set(fixtures._SCENARIO_INPUT_KEYS)
    outputs = set(fixtures._SCENARIO_OUTPUT_KEYS)
    # The partition must be a partition: a key in BOTH lists would be copied into
    # the payload by a list that also declares it an answer, and the two readers
    # would disagree about which fact is true.
    overlap = sorted(inputs & outputs)
    assert not overlap, {
        "error": "fixtures._SCENARIO_INPUT_KEYS and _SCENARIO_OUTPUT_KEYS both "
                 "name the same key, so the payload copies a value the same file "
                 "declares a frozen answer",
        "in_both": overlap,
    }
    unclassified = sorted(set(golden.scenario) - inputs - outputs)
    assert not unclassified, {
        "error": "golden/scenario.json carries key(s) that are in neither "
                 "fixtures._SCENARIO_INPUT_KEYS nor _SCENARIO_OUTPUT_KEYS. A new "
                 "INPUT must be added to the allow-list or it silently never "
                 "reaches the candidate; a new frozen OUTPUT must be added to the "
                 "deny-list so the claim 'no frozen output crosses into the "
                 "sandbox' stays a checked one. This is a TASK-authoring failure, "
                 "not a port defect.",
        "unclassified": unclassified,
        "declared_inputs": sorted(inputs),
        "declared_outputs": sorted(outputs),
    }


def check_scenario_carries_a_gradable_decode_run(golden):
    """The frozen decode run is long enough for the steady-state compilation leg.

    ``test_compilation_is_cached``'s third clause counts XLA compilations over
    decode steps 2..N. Step 1 is the decode-shape warm-up and is excluded by
    design, so the leg needs at least
    ``_t2j.candidate.MIN_STEADY_STATE_DECODE_IDS`` frozen ids -- the constant is
    IMPORTED from the probe rather than restated, because a gate and the thing it
    gates written as two expressions stop agreeing.

    WHY THIS IS FATAL RATHER THAN A SKIP, AND WHY IT IS NOT A DESELECTION. Both
    were tried and both are wrong here:

    * A SKIP shrinks the denominator. ``summary_to_reward`` computes
      ``passed / (passed + failed + error)``, so a skipped check leaves the
      numerator AND the denominator -- 30/(30+0+0) = **1.0**. A port that rebuilds
      its ``nnx.jit`` wrapper inside ``decode_step`` is bit-identical to the
      reference in every value comparison; the compile count is the ONLY thing that
      separates them, and a skip does not weaken that check, it deletes it while
      the summary still reads as a perfect score. ``conftest`` states the rule
      twice ("NO check may skip at grade time"; "DESELECT, never skip") and
      ``summary_to_reward``'s own docstring rests on it.
    * A DESELECTION would keep ``n_skipped == 0`` and stay out of both terms, but
      it would also LEGITIMISE the gap: deselection is for instances where a
      concept genuinely does not apply -- no chunked path, no second prompt length
      -- and it is reported in ``eval_summary.json`` as exactly that. A short decode
      run is not that. It is an AUTHORING DEFECT: ``num_decode_steps`` is a free
      choice in ``instance.yaml``, the shortest gradable value is a hard constraint
      rather than an instance property, and
      the freezer now refuses anything below it
      before a freeze starts -- so the supported authoring path cannot produce a
      golden this deselection would apply to. Deselecting would file a defect as a
      variation and leave the re-jit hole ungraded with nothing saying so.

      (Note what does NOT justify the ruling: "every instance in this class decodes"
      is true -- ``decode_step.npz`` and
      ``scenario.json`` are required goldens in all three modes -- but it argues
      only that the run is non-empty, and what this check needs is that it is longer
      than the warm-up. The two are different claims and only the second is the
      minimum.)

    So it lands here, at the severity the wrong card gets: fatal, before pytest, no
    ``reward.json``. Unreachable on any instance authored through the freezer, whose
    geometry gate enforces the same minimum -- this one is the belt to that braces,
    and it is what covers a hand-edited or partially re-frozen golden tree.

    THE KEY IS THE GOLDEN'S OWN NAME. ``scenario.json`` calls the ids
    ``decode_token_ids``; ``build_candidate`` renames them to
    ``decode_input_token_ids`` at the sandbox boundary, because on the far side they
    are a teacher-forcing INPUT. This reads the near side.
    """
    from _t2j.candidate import MIN_STEADY_STATE_DECODE_IDS

    ids = list(golden.scenario.get("decode_token_ids") or [])
    assert len(ids) >= MIN_STEADY_STATE_DECODE_IDS, {
        "error": "golden/scenario.json carries too few `decode_token_ids` for "
                 "test_compilation_is_cached's decode steady-state clause. Step 1 "
                 "is the decode-shape warm-up and is excluded by design, so the "
                 "counted region needs at least one step after it. Grading with "
                 "fewer would either charge the CANDIDATE for a freeze gap or skip "
                 "the check and shrink the denominator, which scores a re-jitting "
                 "port 1.0.",
        "decode_token_ids": len(ids),
        "minimum": MIN_STEADY_STATE_DECODE_IDS,
        "hint": "re-freeze the goldens from the reference implementation; "
                "instance.yaml's num_decode_steps is what sizes this run.",
    }


# =========================================================================== #
# The entry point _grade_precheck.py calls.
# =========================================================================== #
#: ``(name, needs_golden)`` for each check, in the order the precheck runs them.
#: Cheapest and most fundamental first: with no golden set at all the others
#: report second-order symptoms of the same fault. The two scenario checks come
#: last because both read a PARSED scenario, i.e. they presuppose everything above.
CHECKS = (
    ("golden_manifest", True),
    ("applicability_table_names_real_checks", False),
    ("threshold_extras_name_something_a_check_reads", True),
    ("every_threshold_consumer_has_a_key", True),
    ("perf_golden_if_graded", True),
    ("meta_mirrors_instance_yaml", True),
    ("scenario_keys_are_classified", True),
    ("scenario_carries_a_gradable_decode_run", True),
)


def run(golden) -> list[tuple[str, str]]:
    """Run all of them. ``[(check_name, failure_message)]`` -- empty means healthy.

    EVERY CHECK RUNS EVEN AFTER ONE FAILS. These are independent statements about
    the benchmark's own health and an author fixing them wants the whole list, not
    the first one; the graded suite has the same property for the same reason.

    Only ``AssertionError`` is caught, and it is caught by TYPE rather than
    broadly. The three bodies state their findings with bare ``assert``, so an
    AssertionError is a finding; anything else -- a missing golden file surfacing
    as ``FileNotFoundError``, a parse error, a bad ``meta.json`` -- is a fault in
    the harness itself and must propagate, because reporting it as "check failed"
    would file a broken grader under the same heading as a broken golden.
    """
    out: list[tuple[str, str]] = []
    for name, needs_golden in CHECKS:
        fn = globals()[f"check_{name}"]
        try:
            fn(golden) if needs_golden else fn()
        except AssertionError as exc:
            out.append((name, str(exc)))
    return out
