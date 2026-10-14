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

"""Vendored numeric comparison utilities (mirrors the engine's metrics.py).

Reimplements the comparison math so grade time needs no engine.
Both inputs are cast to float32; a check passes only if all relevant metrics are
within the materialized threshold (see ``golden/meta.json``). Vendored from the
authoring engine's ``metrics.py`` and frozen at generation time.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._arrays import to_numpy
from .similarity import S_scalar, power_mean


# --- Per-parameter gradient comparison -------------------- #
# Map torch state-dict names and JAX pytree paths to a common canonical key so
# per-parameter gradients can be paired across frameworks. Values are compared via
# grad_stats (norm/mean/std), which are transposition/permutation-invariant, so the
# torch weight [out,in] vs flax kernel [in,out] layout difference does not matter.
_PARAM_PREFIX_RE = re.compile(r"^(transformer|gpt_neox|model)\.")
# Torch-side WRAPPER paths with no flax counterpart, stripped before anything else:
# TRL's value-head wrapper contributes a `pretrained_model.` prefix to every parameter
# of a PPO instance, and PEFT's injection contributes `base_model.model.` plus a
# `.base_layer.` infix on every adapted Linear of a LoRA instance. Both are facts about
# how the *torch reference* was assembled, not about the model, and a candidate's flax
# tree cannot and should not reproduce them -- left in, NOT ONE parameter of those
# instances maps to its counterpart.
_WRAPPER_PREFIX_RE = re.compile(r"^(?:pretrained_model\.|base_model\.model\.)+")
_WRAPPER_INFIX = ".base_layer."


def canonical_param_key(name: str) -> str:
    """Normalize a torch state-dict name or JAX pytree path to a common key."""
    s = str(name).replace("/", ".")
    s = _WRAPPER_PREFIX_RE.sub("", s)            # pretrained_model. / base_model.model. -> ''
    s = s.replace(_WRAPPER_INFIX, ".")           # q_proj.base_layer.weight -> q_proj.weight
    s = s.replace("vision_tower.vision_model.", "vision_tower.")  # gemma3/paligemma loaders drop the HF vision_model wrapper; collapse both sides so the towers pair
    s = re.sub(r"_(\d+)(?=[.]|$)", r".\1", s)   # layers_0 / h_0 -> layers.0 (NOT dense_4h_to_h)
    s = _PARAM_PREFIX_RE.sub("model.", s)        # transformer./gpt_neox. -> model.
    parts = s.split(".")
    if parts and parts[-1] in ("weight", "kernel", "embedding", "scale"):
        parts[-1] = "W"                          # unify Linear/Embed/Norm leaf tokens
    return ".".join(parts)


# --- Per-parameter CONVERSION checks -------------------------- #
# Since the candidate converts the HF checkpoint itself, the grader no longer demands a
# verbatim pytree: it checks CORRESPONDENCE (does every parameter you built line up with
# a checkpoint tensor, at a compatible shape?) and per-parameter L2 norms (do they hold
# the right numbers?). Both are keyed by canonical_param_key, so the candidate's module
# naming is its own choice.
#
# The parameter gate below is deliberately WIDER than the gradient gate above. A
# parameter is READ from the checkpoint, not computed through a backward pass, so the
# numerical-noise argument that keeps `norm` and `embed` out of _EXCLUDED_MARKERS does
# not apply — those are exactly the tensors a conversion bug corrupts, so they stay in.
# What is excluded here, and why:
#   * vision/multimodal subtrees — their torch<->flax module naming legitimately
#     diverges (the paligemma / gemma3 / llava_next vision towers do not
#     canonicalize onto their torch names). Covered by forward parity, which runs on
#     real images.
#   * `v_head` — no checkpoint contains it; PPO synthesizes it zero-initialized.
#   * `lm_head` — the candidate may legitimately TIE it to the embedding and never
#     materialize a leaf. Its values are checked exactly by the logit parity gate.
#   * `lora` — adapters are randomly initialized, not converted.
_PARAM_EXCLUDED_MARKERS = (
    "vision", "visual", "image", "patch", "audio", "talker", "token2wav",
    "multi_modal", "mm_", "v_head", "lm_head", "lora",
)


def _is_param_gated(canon_key: str) -> bool:
    """True for a parameter covered by the strict STRUCTURAL conversion gate (Spec)."""
    return not any(mk in canon_key for mk in _PARAM_EXCLUDED_MARKERS)


# --- Per-parameter L2-norm gate (NUMERIC) --------------------------------- #
# The numeric norm score (``score_param_norms``) is DELIBERATELY NARROWER than the
# structural spec gate above. A modality (vision/audio) tower's weights ARE read from the
# checkpoint and converted, so their L2 norms are exactly what a conversion bug corrupts
# and they MUST be graded. They are held out of the structural 1:1 spec gate only because
# torch<->flax module naming across those subtrees can legitimately diverge and an unmapped
# leaf would be a false spec FAIL -- but pairing IS layout-invariant for the norm score, so
# nothing stops us from grading their numbers. What still cannot be norm-scored, and why:
#   * ``v_head`` -- PPO synthesizes it zero-initialized; no checkpoint tensor to match.
#   * ``lm_head`` -- may be legitimately tied to the embedding and never materialized; its
#     values are gated exactly by the logit-parity forward check.
#   * ``lora`` -- adapters are randomly initialized, not converted from the checkpoint.
_NORM_EXCLUDED_MARKERS = ("v_head", "lm_head", "lora")


def _is_norm_scored(canon_key: str) -> bool:
    """True for a parameter whose converted-weight L2 norm is numerically graded.

    Wider than ``_is_param_gated``: modality-tower weights (vision/audio) ARE converted
    from the checkpoint, so their norms are scored here even though they are held out of
    the structural spec gate.
    """
    return not any(mk in canon_key for mk in _NORM_EXCLUDED_MARKERS)


def _canon_first(entries: dict) -> dict[str, tuple[str, Any]]:
    """{canonical key: (original key, value)}, keeping the first of any collision."""
    out: dict[str, tuple[str, Any]] = {}
    for key, value in entries.items():
        out.setdefault(canonical_param_key(key), (key, value))
    return out


def _shape_of(entry: Any) -> tuple[int, ...]:
    if isinstance(entry, dict):
        entry = entry.get("shape", ())
    return tuple(int(d) for d in entry)


def _shape_compatible(cand_entry: Any, ref_entry: Any) -> bool:
    """Candidate/reference shapes match, allowing the rank-2 flax<->torch transpose."""
    rs = _shape_of(ref_entry)
    cs = _shape_of(cand_entry)
    return rs == cs or (len(rs) == 2 and rs == cs[::-1])


def _common_suffix_len(a: list[str], b: list[str]) -> int:
    """Number of trailing dotted segments ``a`` and ``b`` share."""
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


# The residual naming offset a faithful conversion leaves after ``canonical_param_key``
# is a leading-PREFIX difference: a dropped ``model.`` root
# (``h.0.…`` vs ``model.h.0.…``), a reordered wrapper (``language_model.model.`` vs
# ``model.language_model.``). The tail -- module path + layer index + leaf -- is identical.
# So an unmatched candidate key is paired to the reference key with which it shares the
# LONGEST dotted suffix, but ONLY when that pairing is unambiguous: the winning suffix must
# be uniquely-longest from BOTH sides (a mutual best match) and shapes must be compatible.
# This never fires for a genuine DROP -- an omni text-only tree's ``model.layers.0.…`` ties
# ``thinker.model.layers.0.…`` and ``talker.model.layers.0.…`` at equal length, so the
# ambiguity guard refuses it and the missing tower stays correctly unmatched. It is applied
# only to keys left over after exact-canonical matching, so a reference (oracle) tree, whose
# keys all match exactly, is completely unaffected -- no re-freeze, oracle stays 1.0.
_MIN_FALLBACK_SUFFIX = 2  # dotted segments; a bare leaf (`.W`) is never specific enough


def _match_param_keys(
    cand_by: dict[str, tuple[str, Any]],
    ref_by: dict[str, tuple[str, Any]],
    shape_ok: Any = None,
) -> dict[str, str]:
    """Map candidate canonical key -> reference canonical key (exact, then suffix fallback).

    Exact-canonical matches are taken first (identical to the historical behaviour). Only
    the leftover candidate/reference keys go through the layout-invariant longest-common-
    suffix fallback, accepting a pair iff it is a mutual unique best and (when ``shape_ok``
    is given) shape-compatible. ``shape_ok(cand_entry, ref_entry) -> bool`` gates the spec
    use; the norm use passes ``None`` because norm entries carry no shape and the norm
    VALUE comparison is itself the guard against a wrong pairing.
    """
    mapping: dict[str, str] = {}
    used_ref: set[str] = set()
    for ck in cand_by:
        if ck in ref_by:
            mapping[ck] = ck
            used_ref.add(ck)
    rem_cand = {ck: cand_by[ck] for ck in cand_by if ck not in mapping}
    rem_ref = {rk: ref_by[rk] for rk in ref_by if rk not in used_ref}
    if not rem_cand or not rem_ref:
        return mapping

    cparts = {ck: ck.split(".") for ck in rem_cand}
    rparts = {rk: rk.split(".") for rk in rem_ref}

    # Iterate to a fixed point: each pass accepts every mutual unique-best pair at once
    # (so a clean offset resolves in one pass) and removes them from both pools, letting
    # any pairing that a just-removed key was blocking settle on the next pass.
    changed = True
    while changed and rem_cand and rem_ref:
        changed = False
        best_cand: dict[str, tuple[int, list[str]]] = {}
        for ck in list(rem_cand):
            blen, winners = 0, []
            for rk in rem_ref:
                if shape_ok is not None and not shape_ok(rem_cand[ck][1], rem_ref[rk][1]):
                    continue
                length = _common_suffix_len(cparts[ck], rparts[rk])
                if length < _MIN_FALLBACK_SUFFIX:
                    continue
                if length > blen:
                    blen, winners = length, [rk]
                elif length == blen:
                    winners.append(rk)
            if blen >= _MIN_FALLBACK_SUFFIX:
                best_cand[ck] = (blen, winners)
        best_ref: dict[str, tuple[int, list[str]]] = {}
        for rk in list(rem_ref):
            blen, winners = 0, []
            for ck in rem_cand:
                if shape_ok is not None and not shape_ok(rem_cand[ck][1], rem_ref[rk][1]):
                    continue
                length = _common_suffix_len(cparts[ck], rparts[rk])
                if length < _MIN_FALLBACK_SUFFIX:
                    continue
                if length > blen:
                    blen, winners = length, [ck]
                elif length == blen:
                    winners.append(ck)
            if blen >= _MIN_FALLBACK_SUFFIX:
                best_ref[rk] = (blen, winners)
        for ck, (blen, winners) in best_cand.items():
            if ck not in rem_cand or len(winners) != 1:
                continue
            rk = winners[0]
            if rk not in rem_ref:
                continue
            rblen, rwin = best_ref.get(rk, (0, []))
            if rblen == blen and len(rwin) == 1 and rwin[0] == ck:
                mapping[ck] = rk
                del rem_cand[ck]
                del rem_ref[rk]
                changed = True
    return mapping


def compare_param_spec(ref_spec: dict, cand_spec: dict) -> tuple[bool, dict[str, Any]]:
    """1:1 correspondence between the candidate pytree and the torch checkpoint.

    ``ref_spec`` is ``golden/params_spec.json`` — the frozen torch reference's
    ``{state_dict name: {shape, dtype}}``, i.e. the tensors the HF checkpoint actually
    holds. ``cand_spec`` is the candidate's own ``{pytree path: {shape, dtype}}``.

    Passes iff every GATED candidate parameter maps, by canonical key, onto exactly one
    reference parameter of a compatible shape. Compatible means equal, or — for rank-2 —
    the transpose, because a flax ``nn.Dense`` kernel is ``[in, out]`` where a torch
    ``nn.Linear`` weight is ``[out, in]``. dtype is not gated: the compute dtype
    is the candidate's choice and the values are gated by compare_param_norms.

    Only the candidate side is required to map. A candidate that ties or fuses fewer
    leaves is not penalised here; every leaf it *does* build is norm-checked, which is
    what makes an unloaded or garbage tensor impossible to hide.
    """
    ref_by = _canon_first(ref_spec)
    cand_by = _canon_first(cand_spec)

    # Exact-canonical first, then the shape-guarded longest-common-suffix fallback for the
    # residual leading-prefix offsets a faithful conversion leaves (see _match_param_keys).
    mapping = _match_param_keys(cand_by, ref_by, shape_ok=_shape_compatible)

    gated = [ck for ck in cand_by if _is_param_gated(ck)]
    unmapped = sorted(ck for ck in gated if ck not in mapping)
    shape_mismatch: dict[str, Any] = {}
    for ck in gated:
        rk = mapping.get(ck)
        if rk is None:
            continue
        # Fallback pairs are already shape-gated; an EXACT (rk == ck) pair is not, so a
        # correctly-named-but-wrong-shape conversion still surfaces here as a real fail.
        if not _shape_compatible(cand_by[ck][1], ref_by[rk][1]):
            shape_mismatch[ck] = {
                "checkpoint": {"key": ref_by[rk][0], "shape": list(_shape_of(ref_by[rk][1]))},
                "candidate": {"key": cand_by[ck][0], "shape": list(_shape_of(cand_by[ck][1]))},
            }

    passed = (not unmapped) and (not shape_mismatch) and bool(gated)
    details = {
        "candidate_params": len(cand_by),
        "reference_params": len(ref_by),
        "gated": len(gated),
        "matched": len(gated) - len(unmapped),
        "excluded": len(cand_by) - len(gated),
        "num_unmapped": len(unmapped),
        "num_shape_mismatch": len(shape_mismatch),
        # An unmapped key is nearly always a naming slip; show the candidate's own path
        # next to the canonical form it produced so the fix is obvious.
        "unmapped": [{"canonical": ck, "candidate_key": cand_by[ck][0]} for ck in unmapped[:10]],
        "shape_mismatch": dict(list(shape_mismatch.items())[:10]),
    }
    return passed, details


# --- Per-parameter L2-norm SIMILARITY (Numeric) ----------------- #
# Continuous-scoring replacement for the retired boolean ``compare_param_norms`` tolerance
# gate. Same layout-invariant pairing by canonical key and the same conversion-gate exclusions
# (``_is_param_gated``), but each matched tensor is scored per-statistic by the shared GAIN=2
# ``similarity.S_scalar`` and the per-statistic scores are aggregated by the low-tail-sensitive
# ``power_mean`` (p<1). Both sides compute the same fp32 statistics the same way, so a faithful
# conversion sits at the ≈0.999 noise ceiling.


def score_param_norms(ref_norms: dict, cand_norms: dict) -> float:
    """Mean per-parameter converted-weight similarity in [0,1].

    Pairs the candidate's converted-weight statistics with the reference's by canonical key (the
    scored statistics are layout-invariant, so this is indifferent to naming and the
    [in,out]/[out,in] convention), restricts to the norm-scored conversion set (``_is_norm_scored``
    -- WIDER than the structural spec gate: modality vision/audio towers ARE scored here), and
    scores TWO robustly-positive statistics per matched tensor -- the L2 ``norm`` (overall scale)
    and ``abs_mean`` (mean magnitude, which catches a distribution-shape corruption that leaves the
    norm intact) -- each via the shared GAIN=2 ``S_scalar``. ``mean`` (≈0 for zero-mean linear
    weights) and ``std`` (≈0 for all-ones norm/scale weights) are DELIBERATELY not scored: near
    zero the symmetric-relative-error metric is fp32-noise-dominated and would false-fail a
    faithful port. A non-finite candidate tensor scores 0.0; a conversion that maps no scored
    tensor at all scores 0.0 overall. Structural 1:1 correspondence is gated separately by
    ``compare_param_spec`` (Spec).
    """
    ref_by = _canon_first(ref_norms)
    cand_by = _canon_first(cand_norms)
    # Same exact-then-suffix pairing as the spec gate so a faithful-but-offset naming is
    # scored, not zeroed. No shape gate here (norm entries carry no shape); a wrong pairing
    # can only LOWER the candidate's own score, and a genuine drop stays ambiguous->unpaired.
    mapping = _match_param_keys(cand_by, ref_by, shape_ok=None)
    scores: list[float] = []
    for ck, (_cand_key, cand_entry) in cand_by.items():
        rk = mapping.get(ck)
        if not _is_norm_scored(ck) or rk is None:
            continue
        if not bool(cand_entry.get("finite", True)):
            scores.append(0.0)  # a non-finite converted tensor fails every statistic
            continue
        ref_entry = ref_by[rk][1]
        for stat in ("norm", "abs_mean"):
            scores.append(S_scalar(ref_entry.get(stat, 0.0), cand_entry.get(stat, 0.0)))
    # Low-tail-sensitive aggregation (power_mean, p<1): one badly-converted tensor bites
    # super-linearly instead of being diluted by many good ones; a uniform faithful set
    # (all ≈0.999) is p-invariant, so honesty is not punished. Empty -> 0.0.
    return power_mean(scores) if scores else 0.0


# --------------------------------------------------------------------------- #
# Interface parity (spec stage). The candidate's per-module PUBLIC API must be a
# SUPERSET of the frozen contract: torch_orig's ported public interface plus the
# documented JAX-only extras (e.g. ``model/loader.convert_checkpoint``). This is static
# AST analysis of the candidate tree -- no import, no torch, no candidate run -- so a
# dropped entry point fails here loudly instead of surfacing as an inscrutable import
# error downstream. The authoring side that freezes ``golden/interface.json`` uses
# the SAME name rule below so the two agree.
# --------------------------------------------------------------------------- #
def public_api(source: str) -> set[str]:
    """Top-level public classes, functions and UPPER_CASE constants of a module source.

    Underscore-prefixed names are private and excluded. A syntactically broken file
    yields the empty set, so its contract names then read as missing (a FAIL) rather than
    raising at collection time.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                out.add(node.target.id)
    return {n for n in out if not n.startswith("_")}


def candidate_public_api(repo_dir: str) -> dict[str, set[str]]:
    """``{repo-relative .py path -> public API}`` for every module in the candidate tree."""
    out: dict[str, set[str]] = {}
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                src = open(full, encoding="utf-8").read()
            except OSError:
                continue
            out[os.path.relpath(full, repo_dir)] = public_api(src)
    return out


def compare_interface(spec: dict, cand_api: dict[str, set[str]]) -> dict[str, list[str]]:
    """Contract names the candidate does not expose, per module (empty dict == parity).

    ``spec`` is ``golden/interface.json``: ``required`` (torch_orig's ported public API
    per shared module) plus ``required_extras`` (documented JAX-only names). Policy is
    SUPERSET -- every required name must be present (a missing one is a gap); any extra
    names the candidate defines are allowed. Names only, not signatures.
    """
    want: dict[str, set[str]] = {}
    for group in ("required", "required_extras"):
        for rel, names in (spec.get(group) or {}).items():
            want.setdefault(rel, set()).update(names)
    missing: dict[str, list[str]] = {}
    for rel, names in sorted(want.items()):
        gap = sorted(names - cand_api.get(rel, set()))
        if gap:
            missing[rel] = gap
    return missing


# --------------------------------------------------------------------------- #
# Interface KIND / METHODS / CONSTANTS / STRUCTURE / SIGNATURE (spec stage).
# All are static AST / file-walk analysis of the candidate tree -- no import, no
# torch, no candidate run -- so a required entry point that is the wrong kind, a
# class that dropped a method, a retuned load-bearing constant, a missing module,
# or a callable that will not accept the arguments the harness passes it FAILS
# here, loudly, rather than surfacing as an inscrutable error in a downstream
# numeric check. The authoring side freezes these
# same sections into ``golden/interface.json`` with the SAME rules used below, so
# the frozen contract and these checks agree by construction (exactly as the name
# rule in ``public_api`` is mirrored on both sides).
# --------------------------------------------------------------------------- #
#: Load-bearing named constants whose VALUE (or, for a dict, whose key set) the port
#: must reproduce. Dict-valued constants
#: (``SCHEDULER_MAP`` -> function objects, ``DATA_CONFIG`` -> nested structure) are not
#: literals, so only their keys are comparable; scalar constants are compared by value.
WANTED_CONSTANTS: frozenset[str] = frozenset({
    "IGNORE_INDEX",       # label-mask sentinel; must be exactly -100 (HF / LlamaFactory)
    "DEFAULT_TEMPLATE",   # chat-template fallback name
    "SCHEDULER_MAP",      # lr-scheduler registry: KEYS gate which schedules are wired
    "DATA_CONFIG",        # dataset registry structure: KEYS gate the dataset contract
})


def _iter_py_sources(repo_dir: str):
    """Yield ``(repo-relative .py path, source text)`` for the candidate tree.

    Skips ``__pycache__`` and any file that cannot be read as UTF-8 (a binary blob or
    permission error is simply not part of the interface). Shared by the kind / methods
    walks so they see exactly the same module set as ``candidate_public_api``.
    """
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                src = open(full, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            yield os.path.relpath(full, repo_dir), src


# --- kinds ----------------------------------------------------------------- #
def public_api_kinds(source: str) -> dict[str, str]:
    """``{public top-level name -> "class" | "function" | "constant"}`` for a module.

    Same visibility rule as ``public_api`` (drop ``_``-prefixed, UPPER-only constants); a
    syntactically broken file yields ``{}`` so its names read as absent (a parity FAIL)
    rather than raising.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = "function"
        elif isinstance(node, ast.ClassDef):
            out[node.name] = "class"
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out.setdefault(t.id, "constant")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                out.setdefault(node.target.id, "constant")
    return {n: k for n, k in out.items() if not n.startswith("_")}


def candidate_public_api_kinds(repo_dir: str) -> dict[str, dict[str, str]]:
    """``{rel.py -> {public name -> kind}}`` for every module in the candidate tree."""
    return {rel: public_api_kinds(src) for rel, src in _iter_py_sources(repo_dir)}


def compare_interface_kinds(spec: dict, cand_kinds: dict[str, dict[str, str]]) -> dict[str, dict[str, dict]]:
    """Required names the candidate defines as the WRONG kind (empty dict == all correct).

    ``spec['kinds']`` freezes ``{rel: {name: kind}}`` for the reference's contract names. A
    candidate that shadows a required class with a same-named function (or vice versa)
    satisfies name-only ``compare_interface`` but is caught here. A required name the
    candidate does not define at all is a ``compare_interface`` miss, reported there, so it
    is skipped here to avoid double-counting.
    """
    kinds = spec.get("kinds") or {}
    mismatch: dict[str, dict[str, dict]] = {}
    for rel, want in sorted(kinds.items()):
        have = cand_kinds.get(rel, {})
        bad = {
            name: {"expected": kind, "candidate": have.get(name)}
            for name, kind in sorted(want.items())
            if name in have and have[name] != kind
        }
        if bad:
            mismatch[rel] = bad
    return mismatch


# --- class methods --------------------------------------------------------- #
def public_class_methods(source: str) -> dict[str, set[str]]:
    """``{top-level class name -> its non-underscore method names}`` for a module.

    Only methods DEFINED in the class body are seen (AST, no import), never inherited ones
    -- the same rule on both sides, so a method the reference inherits (and the candidate
    also inherits) is neither required nor credited. A broken file yields ``{}``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                m.name for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")
            }
    return out


def candidate_class_methods(repo_dir: str) -> dict[str, dict[str, set[str]]]:
    """``{rel.py -> {class name -> method names}}`` across the candidate tree."""
    return {rel: public_class_methods(src) for rel, src in _iter_py_sources(repo_dir)}


def compare_interface_methods(spec: dict, cand_methods: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    """Frozen methods a candidate class is missing, per ``rel::Class`` (empty == superset).

    ``spec['methods']`` freezes ``{rel: {Class: [methods]}}`` -- for each class present in
    BOTH torch_orig and the reference, the methods they share (torch ∩ ref, the same
    superset policy as ``compare_interface``). The candidate's same-named class at the same
    module must expose at least those (extra methods are fine). A class the candidate does
    not define reports every frozen method as missing -- a class-level parity gap is a
    methods gap too.
    """
    methods = spec.get("methods") or {}
    missing: dict[str, dict[str, list[str]]] = {}
    for rel, classes in sorted(methods.items()):
        have_classes = cand_methods.get(rel, {})
        rel_missing: dict[str, list[str]] = {}
        for cls, names in sorted(classes.items()):
            gap = sorted(set(names) - have_classes.get(cls, set()))
            if gap:
                rel_missing[cls] = gap
        if rel_missing:
            missing[rel] = rel_missing
    return missing


# --- constants ------------------------------------------------------------- #
def _constant_repr(value_node: ast.AST) -> dict | None:
    """Freeze-able view of a constant's VALUE from its AST, or None if not literal.

    A dict literal is reduced to its sorted string keys (``{"kind":"dict","keys":[...]}``):
    ``SCHEDULER_MAP`` maps names to FUNCTION objects and ``DATA_CONFIG`` to nested
    structures, neither literal, so only the keys are comparable. Everything else is
    ``literal_eval``'d (``{"kind":"value","value": -100}``). A non-literal scalar (a call,
    a Name) or a ``{**splat}`` dict yields None -> the constant is simply not frozen.
    """
    if isinstance(value_node, ast.Dict):
        keys: list[str] = []
        for k in value_node.keys:
            if k is None:  # ``{**base}`` splat -> keys unknown statically
                return None
            try:
                keys.append(str(ast.literal_eval(k)))
            except (ValueError, SyntaxError, TypeError):
                return None
        return {"kind": "dict", "keys": sorted(keys)}
    try:
        return {"kind": "value", "value": ast.literal_eval(value_node)}
    except (ValueError, SyntaxError, TypeError):
        return None


def module_constants(source: str, names) -> dict[str, dict]:
    """``{NAME -> repr}`` for each wanted top-level constant a module defines literally.

    First top-level binding wins. A wanted name that is absent, or bound to a non-literal
    (a computed value), is omitted -- the freeze is data-driven, pinning only what the
    reference actually states as a literal.
    """
    wanted = set(names)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, dict] = {}
    for node in tree.body:
        target = value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value_node = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value_node = node.target.id, node.value
        if target in wanted and target not in out:
            rep = _constant_repr(value_node)
            if rep is not None:
                out[target] = rep
    return out


def candidate_module_constants(repo_dir: str, wanted_by_rel: dict) -> dict[str, dict[str, dict]]:
    """``{rel -> {NAME -> repr}}`` for the candidate, restricted to the frozen (rel, name) set."""
    out: dict[str, dict[str, dict]] = {}
    for rel, names in wanted_by_rel.items():
        try:
            src = open(os.path.join(repo_dir, rel), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            out[rel] = {}
            continue
        out[rel] = module_constants(src, names)
    return out


def compare_constants(spec: dict, cand_constants: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Load-bearing constants the candidate states differently (empty dict == all match).

    ``spec['constants'] = {rel: {NAME: repr}}`` from the reference. Policy:
      * value constants (``IGNORE_INDEX``, ``DEFAULT_TEMPLATE``) must match EXACTLY;
      * dict constants (``SCHEDULER_MAP`` / ``DATA_CONFIG`` keys) must be a SUPERSET -- every
        frozen key present, extras allowed.
    A frozen constant the candidate does not define (or defines non-literally) is a miss.
    """
    constants = spec.get("constants") or {}
    problems: dict[str, dict[str, dict]] = {}
    for rel, wanted in sorted(constants.items()):
        have = cand_constants.get(rel, {})
        rel_problems: dict[str, dict] = {}
        for name, want in sorted(wanted.items()):
            got = have.get(name)
            if got is None:
                rel_problems[name] = {"error": "missing or non-literal", "expected": want}
            elif want.get("kind") == "dict":
                missing_keys = sorted(set(want.get("keys", [])) - set(got.get("keys", [])))
                if got.get("kind") != "dict" or missing_keys:
                    rel_problems[name] = {
                        "error": "dict keys not a superset",
                        "missing_keys": missing_keys,
                        "expected_keys": want.get("keys"),
                        "candidate_keys": got.get("keys"),
                    }
            elif got.get("kind") != "value" or got.get("value") != want.get("value"):
                rel_problems[name] = {
                    "error": "value mismatch",
                    "expected": want.get("value"),
                    "candidate": got.get("value"),
                }
        if rel_problems:
            problems[rel] = rel_problems
    return problems


# --- module / directory structure ------------------------------------------ #
def candidate_py_files(repo_dir: str) -> set[str]:
    """Repo-relative ``.py`` paths in the candidate tree (excludes ``__pycache__``)."""
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                out.add(os.path.relpath(os.path.join(dirpath, fn), repo_dir))
    return out


def _imported_top_level_names(repo_dir: str) -> set[str]:
    """Top-level module names any candidate ``.py`` imports (absolute imports only).

    A ``.py``-free directory can only silently HIJACK an import if its name is one the
    candidate actually imports -- that is the sole purpose of the empty-dir check (a hollow
    ``datasets/`` shadowing the real HF ``datasets`` and dragging in torch). A directory
    nothing imports -- ``output/``, ``checkpoint-500/``, ``my_run/``, any run artefact or
    tooling cache, whatever it is named -- cannot hijack anything, so it is not a source
    fault. Checking IMPORT INTENT instead of a fixed catalog of output-dir names is both
    complete for the threat and immune to whatever an agent happens to call its run dirs
    (avoids the idiom-catalog overfitting the taxonomy warns against).
    """
    names: set[str] = set()
    for rel in candidate_py_files(repo_dir):
        try:
            tree = ast.parse(open(os.path.join(repo_dir, rel), encoding="utf-8").read())
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".", 1)[0])  # absolute `from X[.y] import`
    return names


def candidate_empty_dirs(repo_dir: str) -> list[str]:
    """``.py``-free directories whose name could hijack a candidate import (namespace trap).

    Mirrors the harness's structure check: a leftover package dir (kept alive by a
    stray ``__pycache__``) becomes a namespace package that loses to a same-named real
    package on ``sys.path`` -- how a hollow ``datasets/`` silently handed ``import datasets``
    to the torch-importing HF one.

    A ``.py``-free directory is a fault ONLY if importing its name would shadow a module
    the candidate uses -- i.e. its basename is a top-level name the candidate imports
    (:func:`_imported_top_level_names`). Run artefacts and tooling caches (``output/``,
    ``checkpoint-500/``, ``wandb/``, any name) import to nothing, so they are never flagged
    regardless of what they are called; a candidate that instead DROPS a required package
    fails ``missing_files``. This is a semantic check, not an output-name blocklist.
    """
    imported = _imported_top_level_names(repo_dir)
    out: list[str] = []
    for dirpath, dirnames, _ in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if os.path.abspath(dirpath) == os.path.abspath(repo_dir):
            continue
        has_py = any(
            fn.endswith(".py")
            for _, _, fns in os.walk(dirpath)
            for fn in fns
        )
        if has_py:
            continue
        if os.path.basename(dirpath) in imported:
            out.append(os.path.relpath(dirpath, repo_dir))
    return out


def compare_structure(spec: dict, repo_dir: str) -> dict:
    """Candidate module tree vs the frozen torch_orig layout (all-empty == satisfies).

    ``spec['structure'] = {files:[torch .py], allow_missing:[...], allow_extra:[...]}``.
    Required = ``files - allow_missing`` (every torch module the port must carry; today
    ``allow_missing`` is empty -- every torch module has a counterpart). The candidate's
    ``.py`` set must be a SUPERSET (extra files, incl. the reference's documented
    ``allow_extra`` torch-free shims, are fine) with NO empty/namespace-package directory.
    """
    structure = spec.get("structure") or {}
    files = set(structure.get("files") or [])
    allow_missing = set(structure.get("allow_missing") or [])
    required = files - allow_missing
    cand_files = candidate_py_files(repo_dir)
    return {
        "missing_files": sorted(required - cand_files),
        "empty_dirs": sorted(candidate_empty_dirs(repo_dir)),
    }


# --- callable signatures --------------------------------------------------- #
def _callable_param_info(func_node) -> dict:
    """Parameter shape of a def node: positional slots, keyword-acceptable names, varargs."""
    a = func_node.args
    pos = [p.arg for p in getattr(a, "posonlyargs", [])] + [p.arg for p in a.args]
    by_keyword = {p.arg for p in a.args} | {p.arg for p in a.kwonlyargs}
    return {
        "positional_slots": len(pos),
        "accepts_by_keyword": sorted(by_keyword),
        "has_var_positional": a.vararg is not None,
        "has_var_keyword": a.kwarg is not None,
    }


def candidate_callable_params(repo_dir: str, rel: str, name: str) -> dict | None:
    """Parameter info for top-level function ``name`` in candidate module ``rel``.

    None when the module or the function is absent (a parity/structure miss, reported
    there). AST-only, so default values / annotations need not import.
    """
    try:
        tree = ast.parse(open(os.path.join(repo_dir, rel), encoding="utf-8").read())
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _callable_param_info(node)
    return None


def check_callable_signature(info: dict, *, positional: int = 0, keywords=()) -> dict:
    """Does a candidate callable ACCEPT what the harness passes it?

    ``positional`` = leading positional args the harness supplies (the callable needs at
    least that many positional slots, or ``*args``). ``keywords`` = keyword names the
    harness supplies (each must be a positional-or-keyword / keyword-only parameter, or the
    callable must take ``**kwargs``). Extra, reordered or renamed positional-only params are
    fine -- the port adapts parameter lists. ``info`` is ``candidate_callable_params(...)``;
    None (callable absent) returns ``ok=False`` so the reason is explicit.
    """
    if info is None:
        return {"ok": False, "reason": "callable not found", "missing_keywords": list(keywords)}
    positional_ok = info["has_var_positional"] or info["positional_slots"] >= positional
    accepts_kw = set(info["accepts_by_keyword"])
    missing_kw = [] if info["has_var_keyword"] else [k for k in keywords if k not in accepts_kw]
    return {
        "ok": bool(positional_ok and not missing_kw),
        "positional_ok": bool(positional_ok),
        "positional_slots": info["positional_slots"],
        "needs_positional": positional,
        "missing_keywords": missing_kw,
    }


def compare_signatures(spec: dict, repo_dir: str) -> dict[str, dict]:
    """Grader-entry callables that will not accept the harness's arguments (empty == ok).

    ``spec['signatures'] = {"rel::name": {"positional": N, "keywords": [...]}}`` freezes, for
    the fixed grader entry points, how the harness calls each. Every listed callable must
    ACCEPT that call shape (see ``check_callable_signature``). Existence itself is a parity
    concern; a callable that is absent still reports here (``ok=False``) so a dropped entry
    point is unambiguous.
    """
    signatures = spec.get("signatures") or {}
    problems: dict[str, dict] = {}
    for ref, want in sorted(signatures.items()):
        rel, name = ref.split("::", 1)
        info = candidate_callable_params(repo_dir, rel, name)
        result = check_callable_signature(
            info, positional=int(want.get("positional", 0)), keywords=tuple(want.get("keywords", ()))
        )
        if not result["ok"]:
            problems[ref] = result
    return problems


# --------------------------------------------------------------------------- #
# Runtime-augmented interface comparators.
#
# Each takes the UNION of two views of the candidate: the RUNTIME view -- imported
# and introspected in the torch-blocked interface sandbox by
# ``candidate._interface_worker_entry`` (``inspect.signature`` for entry points,
# ``dir()`` over the MRO for class methods, ``getattr`` for constants) -- and the
# AST view above. A signature / method / constant is satisfied iff EITHER view
# satisfies it. Union is strictly more lenient than the AST-only gate, so the frozen
# goldens need NO re-freeze and the oracle stays 1.0 by construction, while a
# faithful port whose entry point the AST cannot see -- a factory-bound
# ``get_train_args = build_...()``, a ``ModelOutput(dict)`` whose ``keys``/``items``
# are inherited, a non-literal ``SCHEDULER_MAP`` -- is no longer failed for it. A
# symbol absent in BOTH views is a real gap and still fails. When the runtime probe
# could not run (``supported=False``), its view is empty ``{}`` and each function
# degrades exactly to the AST-only gate it augments (today's behaviour).
# --------------------------------------------------------------------------- #
def compare_signatures_runtime(spec: dict, runtime_info: dict | None, repo_dir: str) -> dict[str, dict]:
    """Frozen entry points that accept the harness call shape in NEITHER view (empty == ok).

    For each ``spec['signatures']`` entry, first check the RUNTIME signature
    (``runtime_info[ref]`` is the live ``inspect.signature`` param shape, or None if the
    module/callable was absent); if it does not accept the call shape, fall back to the AST
    signature (``candidate_callable_params``). Only a callable that accepts in NEITHER view
    is a problem -- reported with both views for debuggability. ``runtime_info`` is ``{}``
    when the probe could not run, degrading this to ``compare_signatures``.
    """
    signatures = spec.get("signatures") or {}
    runtime_info = runtime_info or {}
    problems: dict[str, dict] = {}
    for ref, want in sorted(signatures.items()):
        rel, name = ref.split("::", 1)
        positional = int(want.get("positional", 0))
        keywords = tuple(want.get("keywords", ()))
        runtime_res = check_callable_signature(runtime_info.get(ref), positional=positional, keywords=keywords)
        if runtime_res["ok"]:
            continue
        ast_res = check_callable_signature(
            candidate_callable_params(repo_dir, rel, name), positional=positional, keywords=keywords
        )
        if not ast_res["ok"]:
            problems[ref] = {"runtime": runtime_res, "ast": ast_res}
    return problems


def compare_interface_methods_runtime(
    spec: dict, runtime_methods: dict | None, ast_methods: dict | None
) -> dict[str, dict[str, list[str]]]:
    """Frozen methods missing from a candidate class in BOTH views, per ``rel::Class`` (empty == ok).

    ``runtime_methods`` and ``ast_methods`` are both ``{rel: {Class: iterable-of-names}}`` --
    the runtime ``dir()``-over-MRO set (which credits INHERITED methods, e.g.
    ``ModelOutput(dict).keys``) and the AST DEFINED-in-body set. For each frozen class the
    candidate must expose each frozen method in at least one view; a method present in
    neither is a real gap. ``runtime_methods`` is ``{}`` when the probe could not run,
    degrading this to ``compare_interface_methods``.
    """
    methods = spec.get("methods") or {}
    runtime_methods = runtime_methods or {}
    ast_methods = ast_methods or {}
    missing: dict[str, dict[str, list[str]]] = {}
    for rel, classes in sorted(methods.items()):
        rt = runtime_methods.get(rel, {})
        ast = ast_methods.get(rel, {})
        rel_missing: dict[str, list[str]] = {}
        for cls, names in sorted(classes.items()):
            have = set(rt.get(cls, ())) | set(ast.get(cls, ()))
            gap = sorted(set(names) - have)
            if gap:
                rel_missing[cls] = gap
        if rel_missing:
            missing[rel] = rel_missing
    return missing


def _constant_matches(want: dict, got: dict | None) -> bool:
    """One frozen constant repr vs one candidate repr: scalar value EXACT, dict keys SUPERSET.

    Same policy as ``compare_constants`` (a value must equal; a dict's frozen keys must all be
    present, extras allowed). ``got`` None/empty -> no match.
    """
    if not got:
        return False
    if want.get("kind") == "dict":
        if got.get("kind") != "dict":
            return False
        return not (set(want.get("keys", [])) - set(got.get("keys", [])))
    return got.get("kind") == "value" and got.get("value") == want.get("value")


def compare_constants_runtime(
    spec: dict, runtime_constants: dict | None, ast_constants: dict | None
) -> dict[str, dict[str, dict]]:
    """Load-bearing constants the candidate states differently in BOTH views (empty == ok).

    For each ``spec['constants']`` entry, the candidate matches if EITHER its RUNTIME value
    (``getattr`` on the live module, so a non-literal ``SCHEDULER_MAP`` built at import is
    seen by its keys) OR its AST literal matches the reference. Only a constant that matches
    in neither view is a problem. ``runtime_constants`` is ``{}`` when the probe could not
    run, degrading this to ``compare_constants``.
    """
    constants = spec.get("constants") or {}
    runtime_constants = runtime_constants or {}
    ast_constants = ast_constants or {}
    problems: dict[str, dict[str, dict]] = {}
    for rel, wanted in sorted(constants.items()):
        rt = runtime_constants.get(rel, {})
        ast = ast_constants.get(rel, {})
        rel_problems: dict[str, dict] = {}
        for name, want in sorted(wanted.items()):
            if _constant_matches(want, rt.get(name)) or _constant_matches(want, ast.get(name)):
                continue
            rel_problems[name] = {
                "error": "value/keys match neither the runtime nor the AST view",
                "expected": want,
                "runtime": rt.get(name),
                "ast": ast.get(name),
            }
        if rel_problems:
            problems[rel] = rel_problems
    return problems
