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

"""Spec stage: the L0 contract a parity number is only meaningful under.

Every assertion here is over SOURCE, API SURFACE, REPO LAYOUT, PARAMETER METADATA or
EXACT INTEGER CONTENT. Not one is a float from a model forward, and
that is the filing rule: a Spec check whose verdict could change by moving to another
card is in the wrong file.

WHY THE TIER STILL DEMANDS AN H100. "Hardware-invariant" is a
CLASSIFICATION of what these checks measure, not an execution exception. The conftest's
pinned-device gate fires for any session that collects a graded stem, this module
included, and refuses to run anywhere but ``NVIDIA H100 80GB HBM3``. Two reasons it
must: the probes below share ONE sandbox worker with the numeric and behavioral tiers
-- running it twice on two devices would double a 2.4 GB conversion and let the two
halves of a grade disagree about which machine produced them -- and a green Spec tier
recorded on the wrong device is a green ``eval_summary`` stage for a session whose
other two stages could not have run at all.

WHAT EACH GROUP GRADES

* Independence (``test_independence``) -- the L0 precondition. Static AST purity over
  the whole candidate repo, PLUS the runtime evidence that the torch blocker was live
  in the process that produced every other number in this suite.
* Structure and interface (``test_repo_structure``, ``test_interface``) -- static AST
  against ``golden/interface.json``, frozen from the reference implementation.
  No import, no torch, no candidate run: a dropped or retyped entry point fails here,
  loudly, instead of surfacing as an inscrutable traceback inside a numeric probe.
  A renamed public function and a class replaced by a same-named factory
  are both named by these checks while the sandbox reports only "candidate sandbox
  failed" or nothing at all.
* Parameters (``test_param_correspondence``, ``test_param_norms``) -- the candidate
  converts the checkpoint itself, so the grader does not dictate a pytree;
  it asks whether the tree the candidate built lines up 1:1 with the POST-FUSION torch
  state dict and holds the right numbers.
* Engine contracts (``test_kv_cache_spec``, ``test_attention_metadata_schema``,
  ``test_block_table_contract``, ``test_static_shape_bucketing``,
  ``test_jit_traceability``) -- the cache layout, the metadata schema, the allocator's
  exact integers, the compiled bucket, and a lowered jaxpr with no host callback.
* Config and loader (``test_config_spec``, ``test_loader_spec``) -- the resolved
  configuration and the ``/data/{repo_id}`` contract under no-network.

THE THREE COVERAGE CHECKS ARE NO LONGER HERE. ``check_golden_manifest``,
``check_applicability_table_names_real_checks`` and
``check_threshold_extras_name_something_a_check_reads`` -- a golden that was never
frozen, a mode gate keyed on a name no module defines, and a per-instance threshold
keyed on a name no check reads -- moved to ``_harness_checks.py`` and now run in
``_grade_precheck.py``, fatal and before pytest. They were the
only three checks in this module that take no ``cand`` fixture: their subject is
whether the SUITE can still grade, not whether the port is correct, and while they
were graded a submission of nothing could score above zero. See that module's
header. Fourteen checks remain here and all fourteen read the candidate.

NO CHECK SKIPS. Applicability deselection lives in ``conftest._APPLICABLE``, and
everything collected must resolve to pass or fail -- a skip would leave the check in
both the numerator and the denominator of ``passed/(passed+failed+error)``.

EXACTLY ONE check here is gated: ``test_static_shape_recompilation``, on
``has_second_prompt_length``. Its discriminating leg prefills at a second, shorter
prompt length, and a geometry that admits none cannot express the property -- so it is
deselected (out of both numerator and denominator) and named in ``eval_summary.json``'s
``deselected`` list. The other thirteen apply to all three modes and to every geometry.
The gate is ``_t2j.candidate.second_prompt_length``, THE SAME function the probe calls,
imported rather than restated.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np

from _t2j import compare, purity, sandbox
from _harness_checks import _parse

_TEST_DIR = Path(__file__).resolve().parent

#: Exception types that mean "your loader is broken", not "this checkpoint is not
#: baked". ``test_loader_spec`` requires ``resolve_model_dir`` to REFUSE an unknown
#: repo id, and refusing is not the same fact as raising: a loader with a typo raises
#: too. Kept as a denylist rather than an allow-list of one type, because
#: ``FileNotFoundError`` is the reference's choice and a port may reasonably raise
#: ``ValueError`` or ``OSError`` for a missing checkpoint.
_PROGRAMMING_ERRORS = frozenset({
    "AttributeError", "TypeError", "NameError", "ImportError",
    "ModuleNotFoundError", "SyntaxError", "IndentationError", "UnboundLocalError",
    # Arithmetic and indexing faults are bugs too, not lookups that came up empty.
    "ZeroDivisionError", "IndexError", "RecursionError", "AssertionError",
})

#: The three callback primitives named for ``test_jit_traceability``.
#: ``purity.CALLBACK_PRIMITIVES`` must be able to SEE each of them, or the traceability
#: check silently stops covering one -- see that test.
_CALLBACK_PRIMITIVES = ("pure_callback", "io_callback", "debug_callback")

#: Floor on the lowered forward's equation count, as a multiple of the model's layer
#: count. A genuine transformer forward is thousands of primitives at 16 layers
#: (measured on the reference: 2 950, i.e. ~184 per layer); a host-callback shim that
#: merely calls out to torch lowers to a near-trivial jaxpr. Expressed per LAYER
#: rather than as an absolute so it means the same thing on the 20B instances, and set
#: an order of magnitude below the measurement so it can only ever catch the shim,
#: never a differently-factored port.
_MIN_PRIMITIVES_PER_LAYER = 20

#: Ceiling on the XLA compilations the MODEL BODY may cost at a new prompt length
#: (``test_static_shape_recompilation``), i.e. on ``compilation.body`` -- the count
#: taken after the engine's eager bookkeeping at that length has already been warmed
#: through the driver's own jitted path.
#:
#: THE BOUND IS ON THE SIGNAL, NOT ON THE GLUE, and that distinction is the whole
#: reason for the extra leg. The counts are:
#:
#:   raw ``new_shape``   (glue + body):  16 jitted   /  82 unjitted   -- 5.1x apart
#:   ``body``  (glue warmed away):        9 jitted   /  98 unjitted   -- 10.9x apart
#:
#: Bounding the raw total meant bounding a number of which ~15 were metadata
#: primitives (iota, less, add, select_n, gather) that a jitted and an unjitted
#: engine both pay. The residual is not zero because the driver's warming path does
#: not exercise every one of the engine's own eager primitives. Eight of the 9
#: measured leftovers are that glue -- convert_element_type x2, gather, _squeeze,
#: scatter, dynamic_slice, squeeze, _reduce_max -- but the NINTH is
#: ``jit(_forward)``, the model body itself, not glue at all. The bound's headroom is
#: not sitting entirely on primitives a correct port pays anyway.
#: Recalibrate per FAMILY if that residual grows; ``compilation.body`` on a correct
#: port is the number to watch.
#:
#: NOT DEPTH-INVARIANT BY DESIGN -- IT IS DEPTH-BLIND, AND THAT IS A WEAKER CLAIM.
#: A constant (not ``num_hidden_layers``-scaled) bound is tempting on the theory
#: that "every layer of a transformer shares op signatures with every other, so the
#: eager count does not grow with depth, and neither does the jitted one." Giving
#: every ``LlamaDecoderLayer.__call__`` its OWN jit boundary instead of the one the
#: whole model forward already shares makes ``compilation.body`` come back **9** --
#: BIT-IDENTICAL to a correctly single-jitted port's own **9**. The number really
#: doesn't grow with depth, but not because a correct port's cost is depth-invariant:
#: it is that
#: ``jax.log_compiles`` never logs a compilation for a nested ``jax.jit``/``nnx.jit``
#: invoked from inside an ambient trace (Engine's own outer ``nnx.jit``, the driver's
#: ``_jit_forward``, or ``jax.make_jaxpr``) -- only the outermost dispatch-time
#: compile is logged, so a port sliced into 16 separate jit boundaries is
#: STRUCTURALLY INVISIBLE to this counter, at ANY depth and at ANY bound. Sixteen
#: separate XLA programs is worse than one by every measure that matters --
#: compile time, memory for sixteen live executables, no cross-layer fusion -- and
#: this check cannot tell it apart from the reference. Tightening the constant does
#: not close that hole; it is left named here because the alternative, silence, is
#: how such a rationale goes unchecked.
#: The bound below is tightened to the correct count plus 4 (9 + 4 = 13) purely to
#: remove headroom that this showed was never earning its keep, not
#: because it catches the per-layer port -- it does not, and nothing in this file
#: currently does.
_MAX_BODY_COMPILATIONS = 13

# --------------------------------------------------------------------------- #
# Small static helpers. Each mirrors a rule the freezer applies across the
# freeze/grade boundary, exactly as ``compare.public_api`` is mirrored in the
# structure checker: the freezer states the reference's value, this side
# reads the candidate's, and the frozen JSON is what the two are compared through.
#
# ``_parse`` is imported from ``_harness_checks`` rather than defined twice. The
# dependency runs in that direction because the precheck must NOT import this module:
# doing so would pull numpy, ``purity`` and the sandbox into an entrypoint whose whole
# value is that it is cheap and runs before anything else.
# --------------------------------------------------------------------------- #
def _dataclass_fields(path: Path) -> dict[str, list[str]]:
    """``{class -> public annotated field names, in declaration order}``.

    Exactly what ``@dataclass`` turns into ``__init__`` parameters, which is how
    ``_t2j/candidate.py`` builds an ``AttentionMetadata``.
    """
    tree = _parse(path)
    if tree is None:
        return {}
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = [s.target.id for s in node.body
                  if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                  and not s.target.id.startswith("_")]
        if fields:
            out[node.name] = fields
    return out


def _declared_path_constant(path: Path, name: str) -> str | None:
    """``NAME = Path("literal")`` (or a bare string) -> the literal, else ``None``.

    ``DATA_ROOT`` cannot be read at runtime for this purpose: ``_t2j/candidate.py``
    REBINDS it on an authoring host that has no ``/data`` (``T2J_INFERENCE_DATA_ROOT``),
    so the live value proves nothing about the contract. The SOURCE is not rebindable.
    """
    tree = _parse(path)
    if tree is None:
        return None
    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target != name or value is None:
            continue
        func = getattr(value, "func", None)
        callee = getattr(func, "id", None) or getattr(func, "attr", None)
        if isinstance(value, ast.Call) and callee == "Path" and len(value.args) == 1:
            arg = value.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _importable_dir(rel: str) -> bool:
    """Could a directory at ``rel`` ever be picked up as a namespace package?

    ``compare.candidate_empty_dirs`` reports EVERY directory holding no ``.py`` at any
    depth, because a leftover package dir becomes a namespace package that loses to a
    same-named real package on ``sys.path`` -- how a stale ``datasets/`` once handed
    ``import datasets`` to the torch-importing HF one. That hazard needs the directory
    name to be an importable module name. ``.pytest_cache`` and ``.ipynb_checkpoints``
    are not: a leading dot is not a valid identifier, so no import statement can ever
    name them, and failing a candidate for tool litter would be charging it for a
    hazard that does not exist. ``datasets/`` is still caught, which is the case the
    rule was written for.
    """
    return all(part.isidentifier() for part in Path(rel).parts)


def _in_order(want: list[str], have: list[str]) -> bool:
    """Does ``have`` contain every element of ``want``, in the same relative order?"""
    it = iter(have)
    return all(name in it for name in want)


def _expected_block_table(page_ids, pages_per_seq: int) -> np.ndarray:
    """One request's page list, right-padded with 0 to the compiled width.

    The rule ``cache/block_table.py::block_table_row`` states, restated here so the
    candidate's allocator can be graded against it. Validated against the FROZEN
    golden in ``test_block_table_contract`` before it is applied to the candidate, so
    a wrong rule here fails on the reference rather than passing a wrong port.
    """
    row = np.zeros(int(pages_per_seq), dtype=np.int64)
    ids = np.asarray([int(p) for p in page_ids], dtype=np.int64)
    row[: ids.size] = ids
    return row


def _expected_slot_mapping(page_ids, page_size: int, num_tokens: int,
                           start: int = 0) -> np.ndarray:
    """``slot = page_ids[pos // page_size] * page_size + pos % page_size``.

    Over positions ``start .. num_tokens - 1``. ``start`` defaults to 0, which is
    the whole of a request's mapping and the only span a `core` scenario has. It
    is non-zero for the two spans a MULTI-STAGE batch splits a `decode` request
    into: the prefix its warm-up forward writes (``0 .. start``) and the one query
    token the graded forward writes (``start .. n``). One formula, stated once --
    a second copy specialised to the warm-up is how the two halves of one rule
    start disagreeing.
    """
    ids = [int(p) for p in page_ids]
    return np.asarray(
        [ids[pos // int(page_size)] * int(page_size) + pos % int(page_size)
         for pos in range(int(start), int(num_tokens))], dtype=np.int64)


def _declared_sequences(meta: dict, pinned, prompt_len: int) -> list[dict]:
    """The frozen batch as ``{page_ids, n, stage}``, one entry per request.

    ``meta["sequences"]`` is what a multi-sequence freeze writes. A `core`
    scenario carries none, and the fallback reconstructs exactly what it has
    always meant: one full prefill of ``prompt_len`` tokens on the pinned pages.
    The same reconstruction ``_t2j/candidate.py::_ragged_requests`` performs.

    A SHARED PREFIX (``shares_prefix_with``/``shared_prefix_len``)
    resolves the same way the golden freezer
    resolves it: the sharing request's own ``page_ids`` names only what it
    OWNS, and the physical page it borrows -- the source request's own first
    page -- is prepended before this function's caller reconstructs a block
    table row from it. Without this a sharing request's declared page list is
    short by exactly the borrowed page, and the derived block table disagrees
    with the frozen one at that request's row (a declared `[5]` vs. a
    frozen `[2, 5]`).

    ALSO CARRIES ``start``/``warm_from`` -- the ABSOLUTE position each
    request's GRADED query begins at, and the position its OWN warm-up
    contribution begins at -- because a `prefill_chunked` or sharing request's
    graded range is not ``[0, n)`` the way a plain `prefill`'s is. Both default
    to 0 (a plain `prefill`'s unaffected shape): a literal `prefill_chunked`
    stage sets ``start`` from the scenario's own final ``chunk_split_points``
    entry (the same value the golden freezer reads for
    a request with no ``shares_prefix_with``); a sharing request sets BOTH
    ``start`` and ``warm_from`` to its own ``shared_prefix_len`` -- the
    borrowed page was already written by the source request's own processing,
    so this request contributes NOTHING to the warm-up (``warm_from == start``
    makes ``_expected_frozen_slots``'s warm range empty by construction, not
    by a special case).
    """
    raw = meta.get("sequences") or []
    splits = [int(s) for s in (meta.get("chunk_split_points") or [])]
    declared = []
    for seq in raw:
        pages = [int(p) for p in seq["page_ids"]]
        stage = str(seq.get("stage") or "prefill")
        shares_with = seq.get("shares_prefix_with")
        start = warm_from = 0
        if shares_with is not None:
            source = raw[int(shares_with)]
            pages = [int(source["page_ids"][0]), *pages]
            start = warm_from = int(seq.get("shared_prefix_len") or 0)
        elif stage == "prefill_chunked":
            start = int(splits[-1]) if splits else 0
        declared.append({"page_ids": pages,
                          "n": len(seq["prompt_token_ids"]),
                          "stage": stage, "start": start, "warm_from": warm_from})
    return declared or [{"page_ids": [int(p) for p in pinned],
                         "n": int(prompt_len), "stage": "prefill",
                         "start": 0, "warm_from": 0}]


def _expected_frozen_slots(declared: list[dict],
                           page_size: int) -> tuple[np.ndarray, np.ndarray]:
    """``(slot_mapping, warmup_slot_mapping)`` for the whole declared batch.

    The rule the golden freezer composes both arrays
    with, restated here so leg 1 can require them back ELEMENT FOR ELEMENT: a
    plain `prefill` request contributes all ``n`` of its tokens to the graded
    forward and nothing to the warm-up; a `decode` request contributes its LAST
    token to the graded forward and its ``n - 1``-token prefix to the warm-up;
    a `prefill_chunked` request (or a request sharing another's prefix) contributes
    ``n - start`` tokens to the graded forward (``start`` from
    ``_declared_sequences``, its ABSOLUTE chunk/share boundary) and
    ``start - warm_from`` tokens -- possibly zero, for a sharing request -- to
    the warm-up; and the requests concatenate in schedule order, graded group
    first.
    """
    graded: list[np.ndarray] = []
    warm: list[np.ndarray] = []
    for request in declared:
        n, pages = int(request["n"]), request["page_ids"]
        if request["stage"] == "decode":
            start, warm_from = n - 1, 0
        else:
            start = int(request.get("start") or 0)
            warm_from = int(request.get("warm_from") or 0)
        graded.append(_expected_slot_mapping(pages, page_size, n, start))
        warm.append(_expected_slot_mapping(pages, page_size, start, warm_from))
    empty = np.zeros(0, dtype=np.int64)
    return (np.concatenate(graded) if graded else empty,
            np.concatenate(warm) if warm else empty)


def _default_page_ids(num_tokens: int, page_size: int) -> list[int]:
    """What a free list popped in ascending order hands a fresh request.

    ``BlockPool`` is documented as "the simplest thing that is still a real allocator:
    a free list popped in ascending order, and therefore DETERMINISTIC", and
    the structure checker holds ``allocate``/``free``/``num_free`` to
    being the SAME CODE on both reference sides (AST equality, in CI). So the page
    assignment a correct candidate produces for the frozen prompt is not a matter of
    taste -- it is pinned, and this is it.
    """
    need = max(1, (int(num_tokens) + int(page_size) - 1) // int(page_size))
    return list(range(need))


# =========================================================================== #
# Independence (L0)
# =========================================================================== #
def test_independence(cand):
    """The candidate is torch-free, statically AND in the process that produced
    every other number in this suite.

    Parity numbers are meaningless if the candidate calls torch, so this is the
    precondition the whole grade rests on. Two halves, because either alone is
    dodgeable:

    STATIC -- an AST scan of the WHOLE repo with ``allow_dirs=()`` (so an offline
    ``convert/`` directory is scanned too), covering plain imports and the
    ``importlib`` / ``__import__`` / ``exec`` / ``eval`` obfuscations a grep for
    ``import torch`` would miss.

    RUNTIME -- a dynamically-built module name defeats the static scan entirely
    (``__import__(chr(116) + "orch")`` shares no string literal with "torch"), so the
    worker also reports what the sandbox did. The ERROR TYPE is the evidence and it is
    the only thing that can be: the verifier image has no torch installed at all, so a
    missing blocker and a working blocker BOTH "fail to import torch", and only
    ``TorchBlockedError`` distinguishes "refused" from "absent".

    TRANSITIVE -- neither of the above notices a candidate that does its real work
    inside ``subprocess.run([sys.executable, ...])``: that child inherits this
    process's ENVIRONMENT, never its ``sys.meta_path``, so the RUNTIME check above
    (which only ever inspects the worker's own process) would pass while the child
    ran torch unblocked. ``_worker_entry`` writes a ``sitecustomize.py`` into a
    private temp directory of its OWN -- deliberately not the candidate's scratch
    cwd, which the candidate can rewrite -- and exports that directory on
    ``PYTHONPATH`` before anything else is imported, so any subprocess a candidate
    spawns installs the same blocker before its own first import. That narrows the
    cheat rather than closing it, and ``candidate.py`` lists the routes that remain;
    ``subprocess_torch_import_error_type`` is what a REAL child of
    the worker reported when it tried, and only ``TorchBlockedError`` there means
    the propagation actually happened rather than being merely intended (see the
    (now corrected) ``SANDBOX_ENV_FLAG`` comment in ``candidate.py``).

    ``cand.banned_leaks`` is asserted last and deliberately, with its limitation
    written down rather than left to be rediscovered: it CANNOT return a non-empty
    list, because ``Candidate._ensure_run`` converts any leak into a ``CandidateError``
    before the accessor is reachable. The live assertion is therefore the one above it
    -- reaching ``cand.sandbox_report`` at all requires a run that leaked nothing. The
    ``== []`` line stays because it is the accessor the L0 contract names, and if that
    upstream conversion is ever relaxed this becomes load-bearing without an edit here.
    """
    report = cand.purity_scan()
    assert report.scanned_files > 0, {
        "error": "no candidate sources were scanned",
        "repo_dir": cand.repo_dir,
        "hint": "is the candidate repo present, and does it contain any .py?",
    }
    assert report.clean, {
        "torch_in_candidate_source": [f.to_dict() for f in report.findings[:5]],
        "total_findings": len(report.findings),
        "scanned_files": report.scanned_files,
    }

    # Drives the shared sandbox. A runtime leak raises CandidateError from here.
    got = cand.sandbox_report
    assert got.get("blocker_installed") is True, {
        "error": "the torch blocker was not on sys.meta_path when the candidate "
                 "finished, so nothing was actually preventing a torch import",
        "sandbox_report": got,
    }
    # A SHAPE CHECK ON THE REPORT, NOT A CONSTRAINT ON THE CANDIDATE, documented
    # here the way `banned_leaks == []` at the foot of this check is -- because
    # an undocumented clause that cannot fail is indistinguishable from one that
    # stopped working. TRACED END TO END: no caller passes `banned=`, so
    # `Candidate._banned` is `sandbox.DEFAULT_BANNED_MODULES`,
    # `install_torch_blocker` freezes exactly that into `blocker.banned`, and
    # `_worker_entry` reports it as `banned_roots`. The SET comparison is
    # therefore `S <= S` for every candidate and cannot fail.
    #
    # WHAT IT CAN STILL FAIL ON is the FIELD, and that is why it is not deleted:
    # an absent or empty `banned_roots` makes the subset false (S is non-empty),
    # which is the only way the sandbox report can stop describing the blocker it
    # says was installed -- the clause above asserts `blocker_installed`, a bare
    # boolean, and would not notice. It also becomes load-bearing with no edit
    # here the day a caller uses the `banned=` parameter, which exists.
    assert set(sandbox.DEFAULT_BANNED_MODULES) <= set(got.get("banned_roots") or ()), {
        "error": "the sandbox report does not name the contract's banned roots. "
                 "No caller narrows the set, so this is the report having stopped "
                 "describing the blocker rather than a narrower blocker: a HARNESS "
                 "defect, not a port defect.",
        "expected_at_least": sorted(sandbox.DEFAULT_BANNED_MODULES),
        "blocked": sorted(got.get("banned_roots") or ()),
    }
    assert got.get("torch_import_error_type") == "TorchBlockedError", {
        "error": "`import torch` inside the sandbox did not raise the BLOCKER's "
                 "exception. Any other outcome means torch was reachable, or was "
                 "merely absent -- neither is evidence that it was refused.",
        "raised": got.get("torch_import_error_type"),
        "message": got.get("torch_import_error"),
    }
    # TRANSITIVE, not just local: a candidate that does its work inside
    # `subprocess.run([sys.executable, ...])` -- or any other nested spawn --
    # inherits this environment, not this process's `sys.meta_path`. Before
    # `_worker_entry` wrote a `sitecustomize.py` into a private temp directory and
    # exported THAT on `PYTHONPATH`, such a child ran completely unblocked; the
    # env flag's old comment claimed propagation that no code ever read.
    assert got.get("subprocess_torch_import_error_type") == "TorchBlockedError", {
        "error": "a subprocess spawned from inside the sandbox was not "
                 "torch-blocked -- the blocker did not survive the spawn, so a "
                 "candidate doing its real work in a child process would run "
                 "with no independence guarantee at all",
        "raised": got.get("subprocess_torch_import_error_type"),
        # A `probe-failed:` value is a HARNESS fault, not a port defect: the
        # worker could not run the child at all (a timeout, a fork/pipe
        # failure), so this check reports "unanswered" rather than "answered
        # wrong". It is deliberately still a FAILURE -- the probe fails closed
        # (`_worker_entry`) so an unanswerable question cannot read as a pass --
        # but the repair is to the host or the harness, not to the candidate.
        "if_it_says_probe_failed": "the transitive probe could not run in the "
                                   "grader's own worker; this is a harness or "
                                   "host fault, not a candidate defect. See "
                                   "`_worker_entry`'s wrap of subprocess.run.",
    }
    assert cand.banned_leaks == [], {
        "banned_modules_loaded_in_sandbox": cand.banned_leaks}

    # THE FROZEN ANSWERS ARE NOT ONE RELATIVE PATH AWAY, asserted rather than
    # assumed -- the filesystem analogue of the blocker clause above, and here for
    # the same reason: the evidence is the graded thing, not the intent.
    #
    # Without the chdir `_worker_entry` performs: from the cwd the worker
    # inherited, `Path('.').rglob('golden/fp32_generation.npz')` returned the frozen
    # token ids and `step_logprobs`, so a probe could have returned the answer it was
    # being graded against. Falsifiable: revert the chdir and this clause
    # fails with the golden files listed.
    #
    # WHAT IT IS NOT. This says no golden is reachable RELATIVELY. It does not say
    # the goldens are unreachable -- an absolute path still reads them, and no
    # in-process check can change that. See `fixtures.build_candidate`.
    hits = got.get("relative_golden_hits")
    assert hits == [], {
        "error": "a frozen golden artifact is reachable by a RELATIVE path from the "
                 "candidate worker's working directory, so a probe could return the "
                 "answer instead of computing it. The worker must start in a scratch "
                 "directory (see _t2j.candidate._worker_entry).",
        "worker_cwd": got.get("cwd"),
        "reachable_goldens": hits,
    }

    # THE FROZEN ANSWERS ARE ALSO NOT ONE IMPORT AWAY -- a separate route from
    # the relative-path one above. A `spawn` child inherits the parent's
    # `sys.path`, so the grader's own tests directory would be importable
    # from inside the worker: `import fixtures; fixtures.load_golden(...)`
    # returned every frozen answer, no model involved at all. `_worker_entry`
    # takes that directory off the child's `sys.path` before anything else is
    # imported.
    #
    # WHAT THIS DOES NOT SAY: that no grader code is reachable at all.
    # `_t2j.candidate` -- the worker's own module -- was already imported
    # before the prune could run (multiprocessing had to import it to find
    # `_worker_entry`), so it stays cached and `sys.modules['_t2j.candidate']
    # .__file__` still names an absolute path a couple of `.parent`s from
    # `golden/`. That is the ABSOLUTE-PATH route from `relative_golden_hits`
    # above, restated rather than reopened, and it is not claimed closed here
    # either -- see `_t2j.candidate._worker_entry` and `fixtures.build_candidate`.
    assert got.get("tests_dir_on_sys_path") == [], {
        "error": "the grader's own tests directory is reachable on the candidate "
                 "worker's sys.path, so `import fixtures; "
                 "fixtures.load_golden(...)` returns every frozen answer",
        "tests_dir_on_sys_path": got.get("tests_dir_on_sys_path"),
    }
    assert got.get("grader_modules_importable") == [], {
        "error": "a grader module is still reachable by a FRESH import from "
                 "inside the candidate sandbox after the tests-directory prune",
        "grader_modules_importable": got.get("grader_modules_importable"),
    }


# =========================================================================== #
# Structure and interface (static AST against golden/interface.json)
# =========================================================================== #
def test_repo_structure(golden, cand):
    """The candidate mirrors ``torch_orig``'s module layout, with no namespace dirs.

    ``interface.json['structure']`` freezes torch_orig's ``.py`` tree
    (authoring scaffolding excluded -- not part of what a
    candidate ships). The candidate's ``.py`` set must be a SUPERSET of the required
    modules: extra files are fine, a missing one is not, because the whole premise of
    the class is that the port mirrors the reference so the two can be compared module
    by module.

    NO EMPTY DIRECTORY at any depth, either. A leftover package dir with no ``.py``
    under it becomes a namespace package that LOSES to a same-named real package on
    ``sys.path`` -- which is how a stray ``datasets/`` once handed ``import datasets``
    to the torch-importing HF one in the training class.
    """
    spec = golden.interface
    structure = spec.get("structure") or {}
    files = list(structure.get("files") or [])
    # A contract with nothing in it would make the superset test vacuous, and an
    # interface.json frozen against an absent torch_orig is exactly that shape.
    assert len(files) >= 2, {
        "error": "interface.json froze a trivial module layout",
        "structure": structure,
        "hint": "re-freeze the interface golden from the reference implementation",
    }
    result = compare.compare_structure(spec, cand.repo_dir)
    assert not result["missing_files"], {
        "missing_modules": result["missing_files"],
        "required": len(files) - len(structure.get("allow_missing") or []),
        "hint": "every torch_orig module must have a same-path counterpart in your "
                "repo; you may add files beyond them",
    }
    hijackable = [rel for rel in result["empty_dirs"] if _importable_dir(rel)]
    assert not hijackable, {
        "empty_or_namespace_dirs": hijackable,
        "ignored_non_importable": [rel for rel in result["empty_dirs"]
                                   if not _importable_dir(rel)],
        "hint": "delete leftover package directories that hold no .py at any depth",
    }


def test_interface(golden, cand):
    """The candidate's API surface satisfies the frozen contract.

    Five sections of ``golden/interface.json``, all static AST over the candidate
    SOURCE -- no import, no torch, no candidate run:

    * ``required`` / ``required_extras`` -- NAMES. Per shared module, torch_orig's
      public API intersected with the reference's, plus the documented JAX-only
      ``model/loader.convert_checkpoint``. Superset policy: extras are fine.
    * ``kinds`` -- a required name must be the same KIND. A candidate that shadows
      ``Engine`` with a function satisfies name parity and breaks every caller.
    * ``methods`` -- per contract class, the methods torch and the reference share.
      This is what pins ``Engine.prefill`` / ``decode_step`` / ``generate`` and the
      ``BlockPool`` allocator surface.
    * ``dataclass_fields`` -- the field NAMES of ``AttentionMetadata``, ``RunConfig``,
      ``ModelConfig`` and the three engine results. ``_t2j/candidate.py`` constructs
      ``AttentionMetadata`` entirely by keyword and splats ``RunConfig``'s fields in,
      so a renamed or dropped field is a broken call, not a style choice. Superset
      plus RELATIVE ORDER: ``dataclasses`` generates ``__init__`` in declaration
      order, so a reordered field list silently changes what every positional
      construction in the candidate's own engine means while every keyword call keeps
      working.
    * ``signatures`` -- the fixed grader entry points must ACCEPT the call shape the
      harness uses (how many leading positionals, which keywords). Renaming or adding
      parameters is fine; dropping one the harness passes is a ``TypeError`` deep
      inside a numeric probe, surfaced here instead.

    Frozen by the freezer, which emits a section only
    where the REFERENCE itself satisfies it -- so the contract can never demand
    something the worked example fails.
    """
    spec = golden.interface
    repo = cand.repo_dir
    for section in ("required", "required_extras", "kinds", "methods",
                    "dataclass_fields", "signatures"):
        assert section in spec, {
            "error": f"interface.json has no '{section}' section",
            "hint": "re-freeze the interface golden from the reference implementation",
        }
    # The contract must be non-trivial, or every assertion below passes on nothing.
    assert sum(len(v) for v in spec["required"].values()) >= 10, {
        "error": "interface.json froze an implausibly small `required` set",
        "required": spec["required"],
    }
    assert spec["required_extras"], {
        "error": "interface.json froze no required_extras; the conversion entry "
                 "point (model/loader.convert_checkpoint) is the deliverable and "
                 "must be contracted",
    }

    missing = compare.compare_interface(spec, compare.candidate_public_api(repo))
    assert not missing, {
        "interface_missing": missing,
        "hint": "each torch_orig module's public functions/classes needs a same-named "
                "counterpart in your repo, plus convert_checkpoint; anything beyond "
                "them is yours",
    }
    kind_mismatch = compare.compare_interface_kinds(
        spec, compare.candidate_public_api_kinds(repo))
    assert not kind_mismatch, {"interface_kind_mismatch": kind_mismatch}

    method_gap = compare.compare_interface_methods(
        spec, compare.candidate_class_methods(repo))
    assert not method_gap, {"interface_methods_missing": method_gap}

    field_problems: dict = {}
    for rel, classes in sorted((spec.get("dataclass_fields") or {}).items()):
        have = _dataclass_fields(Path(repo) / rel)
        for cls, want in sorted(classes.items()):
            got = have.get(cls)
            if got is None:
                field_problems[f"{rel}::{cls}"] = {
                    "error": "class defines no annotated fields", "expected": want}
            elif set(want) - set(got):
                field_problems[f"{rel}::{cls}"] = {
                    "error": "missing fields",
                    "missing": sorted(set(want) - set(got)), "candidate": got}
            elif not _in_order(want, got):
                field_problems[f"{rel}::{cls}"] = {
                    "error": "fields present but reordered; dataclass __init__ is "
                             "generated in declaration order",
                    "expected_order": want, "candidate": got}
    assert not field_problems, {"dataclass_field_problems": field_problems}

    # Signatures. BOTH forms go through the shipped comparator: ten of the nineteen
    # frozen targets are ``rel::Class.method``, and ``compare.candidate_callable_params``
    # resolves those as of this class's copy (it is the one documented divergence from
    # the byte-copied training original -- the top-level-only lookup it inherited
    # returned "callable not found" for all ten against the CORRECT reference).
    # Both halves are asserted PRESENT so a contract that lost one is not mistaken for
    # a contract that is satisfied.
    signatures = spec.get("signatures") or {}
    functions = [ref for ref in signatures if "." not in ref.split("::", 1)[1]]
    methods = [ref for ref in signatures if "." in ref.split("::", 1)[1]]
    assert functions and methods, {
        "error": "interface.json froze no callable signatures on one of the two "
                 "sides; the reference satisfies both, so this is a stale contract",
        "module_level": sorted(functions), "methods": sorted(methods),
    }
    signature_problems = compare.compare_signatures(spec, repo)
    assert not signature_problems, {
        "signature_problems": signature_problems,
        "hint": "you may rename, reorder or add parameters; you may not drop one the "
                "harness passes",
    }


# =========================================================================== #
# Parameters (L0)
# =========================================================================== #
def _by_canonical(spec: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    """``({canonical key: original key}, {canonical key: [colliding originals]})``.

    Collisions are REPORTED rather than resolved. ``compare._canon_first`` keeps the
    first of a collision, which is right for its purpose (a tied weight legitimately
    appears twice) and wrong for a 1:1 claim: two candidate leaves that canonicalize
    onto one name mean one of them is being graded twice and the other not at all.
    """
    by: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for key in spec:
        canon = compare.canonical_param_key(key)
        if canon in by:
            collisions.setdefault(canon, [by[canon]]).append(key)
        else:
            by[canon] = key
    return by, collisions


def test_param_correspondence(golden, cand):
    """CANONICAL-KEY 1:1 onto the POST-FUSION torch state dict, shapes up to transpose.

    A port that keeps q/k/v separate does not mirror vLLM and fails here -- which is
    the intended verdict, not an over-strict one: ``STACKED_PARAMS`` fuses q, k, v
    along dim 0 into ``qkv_proj.weight`` and gate, up into ``gate_up_proj.weight``, and
    that post-fusion key set is the contract both reference trees state. A
    port whose converter emits the five shards separately and re-fuses them only
    inside ``apply_params`` -- so it loads and runs correctly -- fails this check
    (and ``test_param_norms``, whose gated keys then map onto nothing).

    NAMING, HOWEVER, IS THE CANDIDATE'S OWN BUSINESS, and the comparison is made
    through ``compare.canonical_param_key`` on BOTH sides for that reason. A
    flax-idiomatic port that returns
    ``model/layers_0/self_attn/qkv_proj/kernel`` canonicalizes onto torch's
    ``model.layers.0.self_attn.qkv_proj.weight`` and is correct; requiring the raw
    torch spelling would fail it -- that exact port fails raw-key equality
    and passes the whole tier under canonical keys -- and would enforce a naming
    contract that no ``instruction.md`` states to a candidate. ``test_param_norms`` already
    canonicalizes, so raw-key equality here would also have left the tier's two
    parameter checks disagreeing about whose business naming is.

    Equality is BOTH directions -- "1:1 onto", not "into". A tied port that never
    materializes ``lm_head`` is short one entry and fails, which is the intended
    verdict: ``convert_checkpoint`` is the graded entry point and torch's
    ``state_dict()`` carries both halves of the tie.

    THE SHAPE RULE IS TRANSPOSE-TOLERANT, ON PURPOSE, AND NOTHING IN THIS TIER
    COMPENSATES. ``sorted(shape)`` cannot tell a ``[H, W]`` weight from a ``[W, H]``
    one -- deliberately, because a flax ``nn.Dense`` kernel is ``[in, out]`` where a
    torch ``nn.Linear`` weight is ``[out, in]`` and a candidate that transposes
    everything consistently is CORRECT. The consequence, stated flatly because it was
    understated once: **no Spec assertion sees a per-weight transposition at any
    shape.** ``test_param_norms`` cannot -- the L2 norm is transposition-invariant by
    construction -- and a square weight (``o_proj`` is [2048, 2048] here) is invisible
    even to a non-sorted shape comparison. A candidate whose own ``apply_params`` is
    strict about orientation will refuse the tensor and die, but that is the
    REFERENCE's strictness and a transpose-tolerant port has none. This break belongs
    to the L1/L2 numeric tiers and is written into their requirements.
    """
    want, got = golden.params_spec, cand.params_spec
    assert want, {"error": "golden/params_spec.json is empty; re-freeze it"}
    assert got, {"error": "the candidate exposed an empty parameter spec"}

    want_by, want_collisions = _by_canonical(want)
    got_by, got_collisions = _by_canonical(got)
    assert not want_collisions, {
        "error": "two frozen state-dict names canonicalize onto one key, so the "
                 "golden cannot state a 1:1 correspondence. This is an artifact "
                 "defect, not a candidate defect.",
        "collisions": want_collisions,
    }
    assert not got_collisions, {
        "error": "two of your parameters canonicalize onto the same key, so one of "
                 "them is graded twice and the other not at all",
        "collisions": got_collisions,
    }
    assert set(got_by) == set(want_by), {
        "missing": sorted(set(want_by) - set(got_by)),
        "unexpected": sorted({got_by[c] for c in set(got_by) - set(want_by)}),
        # THE HINT A FAILING CANDIDATE ACTUALLY READS, so it has to describe the
        # comparator rather than flatter it. A claim like "your own naming and
        # nesting are free" would be the opposite of what `canonical_param_key`
        # does: it normalises separators, `layers_N` -> `layers.N`, a
        # `transformer.` / `gpt_neox.` prefix, two wrapper prefixes and the LEAF
        # token -- and nothing else. Every intermediate path component must be
        # torch's spelling. That over-promise is the same shape as the two
        # undocumented contracts published alongside it (the submodule attribute path
        # and the form of DATA_ROOT): a rule the grader enforces and the text denies.
        "hint": "keys are compared through canonical_param_key, which normalises "
                "ONLY: `/` to `.`, `layers_N` to `layers.N`, a leading "
                "`transformer.`/`gpt_neox.`/wrapper prefix, and the leaf token "
                "(weight|kernel|embedding|scale all become W). EVERY OTHER PATH "
                "COMPONENT MUST MATCH THE CHECKPOINT'S SPELLING -- "
                "`model.layers.0.self_attn.qkv_proj` and nothing else. What must "
                "also match is the POST-FUSION key SET: q/k/v fused into "
                "self_attn.qkv_proj, gate/up into mlp.gate_up_proj",
    }
    for canon, want_key in sorted(want_by.items()):
        got_key = got_by[canon]
        assert sorted(got[got_key]["shape"]) == sorted(want[want_key]["shape"]), {
            "canonical": canon,
            "golden": {"key": want_key, "shape": list(want[want_key]["shape"])},
            "candidate": {"key": got_key, "shape": list(got[got_key]["shape"])},
        }
    # The shipped comparator too, for its gated-subset diagnostics: the assertions
    # above are over the WHOLE key set, this one reports the excluded/matched split.
    #
    # `multimodal` off the GOLDEN, never a literal and never a family name: it is
    # `instance.yaml`'s own key mirrored into meta.json, the same source
    # `conftest._golden_facts` reads. On a VL instance it stops the tower's 326
    # parameters being reported as `excluded` beside a failure that is about them.
    ok, details = compare.compare_param_spec(
        want, got, multimodal=bool(golden.meta.get("multimodal")))
    assert ok, details


def test_param_norms(golden, cand):
    """...and holds the right NUMBERS: per-parameter L2 norm at conversion tolerance.

    Runs in the Spec tier on purpose. The tensors are COPIED out of the checkpoint, so
    the only legitimate gap between two norms is float32 accumulation order -- roughly
    1e-6 relative, against a 1e-3 gate. A conversion bug therefore reads here as
    "parameter X is wrong" instead of arriving downstream as an inscrutable logit
    mismatch. On a text-only family, one converted tensor scaled by 1.01 fails
    this check and only this check.

    THAT LAST CLAUSE IS NOT CLASS-WIDE. On a multimodal instance the same
    mutation -- one converted tensor, scaled by 1.01, applied to a vision-tower
    tensor -- is caught by FOUR checks: this check, ``test_vision_tower_units``,
    ``test_fp32_prefill_logits`` and ``test_decode_trajectory``. A scaled *tower*
    tensor also moves the tower's own recorded submodule output and the fp32-tier
    logits, and a text-only instance has neither surface. So "only this check" is a
    fact about where the mutation was applied, not about the mutation itself.

    WHAT IT DOES NOT CATCH. **A wrongly-ordered fused QKV does not move the norm.** The
    L2 norm of a concatenation is invariant to the order of its parts, so swapping k and
    v inside ``qkv_proj`` is invisible here: that mutation passes this check and is
    caught only by ``test_loader_spec``, via ``STACKED_PARAMS``. A transposition is
    invisible for the same reason. The reduction being layout-invariant is what makes
    naming, nesting and the ``[in,out]``/``[out,in]`` convention the candidate's own
    business, and it is the same property that costs it these two.

    An unread shard, a duplicated tie, a misread bf16 bit pattern and a tensor left at
    its random init all DO move a norm by percent to orders -- by construction, since
    they change the values.

    ``golden/param_norms.json`` is produced offline on CPU, from the reference's own
    ``convert_checkpoint``, reduced through this instance's own
    ``_t2j._formulas.param_stats``.
    """
    assert golden.param_norms, {
        "error": "golden/param_norms.json is empty or absent",
        "hint": "produce the param-norms golden from the reference (CPU, inside "
                "t2j-model:{family})",
    }
    # `multimodal` off the GOLDEN -- and THIS is the call site the flag exists for.
    # This is the only check in the class that reads a parameter's VALUE, so before
    # it was threaded a perturbed vision-tower tensor changed no name and no shape
    # and was caught by nothing at all. See `compare._param_excluded`.
    ok, details = compare.compare_param_norms(
        golden.param_norms, cand.param_norms,
        multimodal=bool(golden.meta.get("multimodal")))
    assert ok, details
    # `compare_param_norms` already requires a non-empty gated set, but not that the
    # gate covered the whole tree: a candidate whose keys all canonicalized into the
    # EXCLUDED markers would pass on one gated parameter.
    assert details["matched"] == details["gated"] and details["gated"] >= 10, {
        "error": "too few parameters were actually compared",
        "details": details,
    }


# =========================================================================== #
# Engine contracts
# =========================================================================== #
def test_kv_cache_spec(golden, cand):
    """The KV cache tensor's shape, dtype and LAYOUT.

    ``[num_layers, num_pages, page_size, 2, num_kv_heads, head_dim]`` on both reference
    sides -- upstream's ``kv_packing`` dimension and its ``head_dim % 128 == 0``
    assertion are TPU lane alignment, dropped on both sides because an H100 has no such
    constraint and keeping them would make Llama-3.2-1B's head_dim of 64 inexpressible.

    The layout is asserted COMPOSITIONALLY, not just as a shape equality: the frozen
    shape must be exactly that product in that order, so the golden cannot silently
    stop describing the layout it claims to.

    NOT VERIFIED: that a candidate allocating the right number of elements in a
    different ARRANGEMENT fails here. It should, by inspection, but no mutation backs
    it -- and the reason is worth recording rather than repeating the attempt. Every
    minimal permutation of ``make_kv_cache``'s shape breaks the reference's own
    ``write_kv`` (its ``reshape(num_pages * page_size, 2, *shape[3:])``) before this
    check is reached, so the mutation would measure the reference's strictness rather
    than this assertion -- the same trap the transpose concession fell into. A
    genuinely self-consistent alternative layout needs coordinated edits to
    ``make_kv_cache``, ``write_kv`` and ``paged_attention``'s gather, which is a port,
    not a mutation. A ``make_kv_cache`` that ignores its dtype
    fails this check and ONLY this check.

    KNOWN BLIND SPOT on this instance: ``page_size`` and ``num_kv_heads`` are both 8,
    so swapping those two axes is invisible to any shape comparison. The numeric tier's
    cache-write parity is what sees it. Other instances of the class have distinct
    values and are not blind.
    """
    want, got = golden.cache_spec, cand.cache_spec
    assert want, {"error": "golden/cache_spec.json is empty; re-freeze it"}

    # THE CACHE IS A CONTAINER, and the spec says which kind of family froze it.
    has_shape = "shape" in want
    has_groups = "groups" in want
    assert has_shape != has_groups, (
        "cache_spec.json must carry EXACTLY ONE of `shape` (a single-geometry "
        "family, six integers) or `groups` (a heterogeneous one). Both is "
        "ambiguous and neither is undeclared; a silent default here would let a "
        "heterogeneous family grade against a rectangular contract.")
    assert ("shape" in got) == has_shape and ("groups" in got) == has_groups, {
        "error": "the candidate's cache_spec declares a different KIND of cache "
                 "than the golden: one is single-geometry (`shape`) and the "
                 "other heterogeneous (`groups`). A shape comparison across the "
                 "two would be comparing a cache to part of one.",
        "golden_keys": sorted(k for k in want if k in ("shape", "groups")),
        "candidate_keys": sorted(k for k in got if k in ("shape", "groups")),
    }

    if has_shape:
        composed = [want["num_layers"], want["num_pages"], want["page_size"], 2,
                    want["num_kv_heads"], want["head_dim"]]
        assert list(want["shape"]) == composed, {
            "error": "the frozen cache_spec contradicts itself -- its shape is not "
                     "[num_layers, num_pages, page_size, 2, num_kv_heads, head_dim]. "
                     "This is an artifact defect, not a candidate defect.",
            "frozen_shape": list(want["shape"]), "composed": composed,
        }
        assert list(got["shape"]) == list(want["shape"]), {
            "error": "the candidate's KV cache has the wrong shape/layout",
            "golden": list(want["shape"]), "candidate": list(got["shape"]),
        }
    else:
        # A heterogeneous family: every group keeps the class layout, and the
        # SLOT COUNTS SUM to the frozen `num_layers` only when no layer shares a
        # slot -- which is exactly what such a family does not promise. So the
        # per-group shapes are compared elementwise and nothing is recomposed.
        golden_groups = [list(g) for g in want["groups"]]
        assert all(len(g) == 6 and g[1] == want["num_pages"]
                   and g[2] == want["page_size"] and g[3] == 2
                   for g in golden_groups), {
            "error": "a frozen cache group is not "
                     "[n_slots, num_pages, page_size, 2, num_kv_heads, head_dim]. "
                     "This is an artifact defect, not a candidate defect.",
            "frozen_groups": golden_groups,
        }
        assert [list(g) for g in got["groups"]] == golden_groups, {
            "error": "the candidate's KV cache groups have the wrong "
                     "shapes/layout",
            "golden": golden_groups,
            "candidate": [list(g) for g in got["groups"]],
        }

    assert got["dtype"] == want["dtype"], {
        "error": "the candidate's KV cache has the wrong dtype; make_kv_cache must "
                 "honour the dtype it is given",
        "golden": want["dtype"], "candidate": got["dtype"],
    }


def test_attention_metadata_schema(golden, cand):
    """``AttentionMetadata``'s fields, dtypes and shapes -- and one field's VALUE.

    FIVE OF THE SIX FIELDS ARE GRADED ON SHAPE AND DTYPE ONLY, and that is
    correct for them: their values are the allocation, the positions and the
    cu_seqlens, every one of which is graded elementwise somewhere else
    (``test_block_table_contract``, the numeric tier's frozen forward).
    ``request_distribution`` is the exception -- it is graded NOWHERE else,
    because neither reference reads it -- so its value is asserted here. See
    the clause after the shape/dtype loop.

    The candidate's OWN ``Engine._metadata`` built the object under test (the metadata
    probe is one of the four ``USE_FROZEN_ALLOCATION=False`` entries -- handing it the
    frozen arrays would make this an assertion about the golden), and every array field
    is compared against the counterpart the reference froze into ``inputs.npz``: same
    shape, same dtype. A candidate whose ``seq_lens`` is ``float32`` instead
    of ``int32`` fails this check and only this check -- the forward still runs and
    every other tier is content, which is exactly the "wrong gather three tiers later"
    this exists to pre-empt. The shape half (a ``query_start_loc`` of ``[num_seqs]``
    rather than ``[num_seqs + 1]``) is asserted by the same loop but is NOT VERIFIED
    by a mutation: it breaks ``paged_attention``'s cu_seqlens arithmetic, so the
    worker dies before the assertion is reached.

    THE FIELD LIST IS A CROSS-ARTIFACT TIE, and its limits are worth stating. The list
    the probe reports is hardcoded in ``_t2j/candidate.py``, so comparing it to the
    frozen contract cannot catch a candidate that renamed a field -- ``getattr`` would
    raise inside the worker first, taking the whole run down with a traceback. What it
    DOES catch is the drift that has no other alarm: a reference whose metadata gains
    or loses a field, re-frozen into ``interface.json``, while the driver still probes
    the old list. That is a grader that has silently stopped looking at part of the
    contract, and nothing else in the suite would notice.
    """
    schema = cand.metadata_schema
    frozen_fields = ((golden.interface.get("dataclass_fields") or {})
                     .get("attention/metadata.py", {}).get("AttentionMetadata"))
    assert frozen_fields, {
        "error": "interface.json froze no AttentionMetadata field list",
        "hint": "re-freeze the interface golden from the reference implementation",
    }
    assert list(schema["fields"]) == list(frozen_fields), {
        "error": "the driver probes a different metadata field list than the frozen "
                 "contract states -- the grader has drifted from the reference",
        "contract": list(frozen_fields), "probed": list(schema["fields"]),
    }

    # field -> the frozen array in inputs.npz that pins its shape and dtype.
    array_fields = {
        "input_positions": "positions",
        "block_tables": "block_tables",
        "seq_lens": "seq_lens",
        "query_start_loc": "query_start_loc",
        "request_distribution": "request_distribution",
    }
    # ...EXCEPT `input_positions`, WHOSE LENGTH IS PER QUERY TOKEN AND NOT
    # BUCKETED. The other four are `(max_num_seqs,)`, `(max_num_seqs + 1,)`,
    # `(max_num_seqs * pages_per_seq,)` and `(3,)`: properties of the COMPILED
    # BUCKET, identical for any batch the instance can schedule, so the frozen
    # array pins them. `input_positions` is `(total_num_scheduled_tokens,)`, a
    # property of the BATCH -- and this probe describes a single-request prefill
    # of the frozen prompt while the frozen scenario may describe a mixed batch
    # of several. On `core` the two coincide (one request, `prompt_len` tokens)
    # and this substitution changes nothing; on a multi-sequence instance the
    # frozen array is longer and comparing against it would fail a correct
    # engine on a shape it was never asked to produce. So the LENGTH comes from
    # `prompt_len` -- the number of tokens the probe actually forwarded -- and
    # the DTYPE still comes from the frozen array, which is the half a mutation
    # has ever moved (a float32 seq_lens; the shape half is documented above as
    # unverified).
    #
    # ONLY THE LAST AXIS IS THE ONE THIS SUBSTITUTES. `input_positions` is
    # `(R, total_num_scheduled_tokens)` for a model that builds R rope sections
    # and `(total_num_scheduled_tokens,)` for one that builds a single band;
    # `attention/metadata.py` and `engine.py::_metadata` both say in as many
    # words that the rank is PASSED THROUGH and that `layers/rotary.py` is where
    # the two ranks are made uniform. The leading axes are therefore a property
    # of the MODEL, pinned by the frozen array exactly like the other four
    # fields' whole shapes; the trailing axis is the property of the BATCH that
    # the paragraph above is about. Spelling the expected shape as `[prompt_len]`
    # would assert rank 1 outright: on a multimodal engine that
    # correctly reports `(3, 11)`, this check would fail a correct engine with
    # "golden [11], candidate [3, 11]". On every text-only instance the frozen
    # array is 1-D, so the leading part is empty and the expectation is the
    # `[prompt_len]` it has always been -- byte for byte, nothing is loosened.
    prompt_len = int(golden.meta["prompt_len"])
    problems: dict = {}
    for field, key in sorted(array_fields.items()):
        entry = schema["schema"].get(field)
        reference = golden.inputs.get(key)
        if reference is None:
            problems[field] = {"error": f"golden/inputs.npz has no '{key}'"}
            continue
        if entry is None or entry.get("kind") != "array":
            problems[field] = {"error": "not an array field", "candidate": entry}
            continue
        if field == "input_positions":
            want = [*(int(d) for d in np.shape(reference)[:-1]), prompt_len]
            if list(entry["shape"]) != want:
                problems[field] = {
                    "error": "shape (one position per query token on the last "
                             "axis -- the probe forwarded prompt_len of them -- "
                             "and the frozen array's own rope-section rank ahead "
                             "of it)",
                    "golden": want, "candidate": list(entry["shape"])}
            elif entry.get("dtype") != str(np.asarray(reference).dtype):
                problems[field] = {"error": "dtype",
                                   "golden": str(np.asarray(reference).dtype),
                                   "candidate": entry.get("dtype")}
            continue
        if list(entry["shape"]) != [int(d) for d in np.shape(reference)]:
            problems[field] = {"error": "shape",
                               "golden": [int(d) for d in np.shape(reference)],
                               "candidate": list(entry["shape"])}
        elif entry.get("dtype") != str(np.asarray(reference).dtype):
            problems[field] = {"error": "dtype",
                               "golden": str(np.asarray(reference).dtype),
                               "candidate": entry.get("dtype")}
    assert not problems, {"attention_metadata_schema_problems": problems}

    # ----------------------------------------------------------------------- #
    # `request_distribution`'s VALUE. Every clause above is a shape or a dtype;
    # this is the one field of the six whose CONTENT is graded, and it needs its
    # own clause because nothing else in the suite -- or in either reference --
    # ever reads it. `paged_attention` does not touch it on either side, so it is
    # numerically inert and a wrong value can never surface as a wrong logit.
    # Without this clause, a reference whose `Engine._metadata`
    # emits a constant `[0, 1, 0]` for every call, decode steps included, is
    # caught by nothing.
    #
    # `[n_decode, n_prefill, n_mixed]`. The candidate's engine is single-request
    # on every mode (`interface.json` pins `prefill`/`decode_step`/`generate`/
    # `reset`, none of them batched), so its two answers are pinned and are the
    # SAME on `core`, `ragged` and `chunked`: a prefill is one prefill request,
    # a decode step is one decode request. On `core` the prefill half is
    # trivially true -- one prefill is the only composition the mode has -- and
    # the DECODE half is what discriminates there, which is why the probe pays
    # for a decode step to produce it.
    stages = (schema.get("request_distribution") or {})
    want_stage = {"prefill": [0, 1, 0], "decode": [1, 0, 0]}
    wrong = {stage: {"golden": want, "candidate": stages.get(stage)}
             for stage, want in sorted(want_stage.items())
             if list(stages.get(stage) or []) != want}
    assert not wrong, {
        "error": "Engine._metadata built the wrong request_distribution. It is "
                 "[n_decode, n_prefill, n_mixed] and it NAMES the stage of the "
                 "batch: a prefill of one request is [0, 1, 0] and a decode step "
                 "on one request is [1, 0, 0]. Nothing else in this suite reads "
                 "the field, so a constant here is invisible everywhere else.",
        "stages": wrong,
    }
    # ...and the FROZEN one must name the composition the scenario declares --
    # the same rule the golden freezer applies. This is the
    # `ragged` half: on a multi-sequence batch the frozen value is the only place
    # the mix is written down, `test_ragged_forward_logits` asserts it is mixed,
    # and an artifact whose distribution disagreed with its own `sequences` would
    # make that assertion a statement about nothing.
    declared = golden.meta.get("sequences") or []
    n_decode = sum(1 for seq in declared
                   if str(seq.get("stage") or "prefill") == "decode")
    # THE THIRD (CHUNKED) BUCKET, restated from the golden freezer's own
    # resolution rule rather than re-derived from stage
    # names alone: a request whose declared stage is literally `prefill_chunked`
    # (its own earlier chunk already wrote history) counts, and so does a request
    # that only carries `shares_prefix_with` (its history was WRITTEN by another
    # request's chunk, via the prefix cache, but its own declared stage stays
    # `prefill` -- the freezer resolves it to `prefill_chunked` internally
    # for the batch it builds, but that resolution is not restated in
    # `meta["sequences"]`, which mirrors the AUTHORED instance.yaml stage). Without
    # the second clause a chunked instance whose only chunking is an
    # automatic-prefix-cache share (no request literally named `prefill_chunked`)
    # would be graded as if it had none.
    n_chunked = sum(1 for seq in declared
                    if str(seq.get("stage") or "prefill") == "prefill_chunked"
                    or seq.get("shares_prefix_with") is not None)
    n_prefill = len(declared) - n_decode - n_chunked
    want_frozen = [n_decode, n_prefill, n_chunked] if declared else [0, 1, 0]
    got_frozen = [int(x) for x in
                  np.asarray(golden.inputs["request_distribution"]).reshape(-1)]
    assert got_frozen == want_frozen, {
        "error": "golden/inputs.npz's request_distribution does not count the "
                 "stages golden/meta.json declares; this is an artifact defect, "
                 "not a candidate defect",
        "declared_stages": [str(s.get("stage") or "prefill") for s in declared],
        "golden": want_frozen, "frozen": got_frozen,
    }
    warm = golden.inputs.get("warmup_request_distribution")
    if warm is not None:
        # A warm-up forward writes every request's ALREADY-KNOWN cached
        # prefix, so it is all-prefill over exactly the requests that HAVE
        # one to write -- which is NOT the same set as "declared stage ==
        # decode". A `decode` request with more than one prompt token has
        # one (the original formula's whole case); so does a
        # `prefill_chunked` request whose OWN preceding chunk needs writing
        # (chunked_llama3's r0: 5 warm tokens); a request that only
        # SHARES another's prefix (`shares_prefix_with`) has NONE of its own
        # to write -- the source request's chunk already wrote those exact
        # slots -- even though it is also, in the third-bucket sense above,
        # "chunked". A decode-only formula undercounts the warm-up
        # composition -- e.g. `[0, 1, 0]` where the frozen value is `[0, 2, 0]`,
        # because two requests each contribute a nonzero warm-up prefix while
        # one contributes none.
        chunk_splits = [int(s) for s in (golden.meta.get("chunk_split_points") or [])]
        chunk_start = int(chunk_splits[-1]) if chunk_splits else 0

        def _has_warm_prefix(seq: dict) -> bool:
            stage = str(seq.get("stage") or "prefill")
            if seq.get("shares_prefix_with") is not None:
                return False
            if stage == "decode":
                return len(seq["prompt_token_ids"]) > 1
            if stage == "prefill_chunked":
                return chunk_start > 0
            return False

        want_warm = [0, sum(1 for seq in declared if _has_warm_prefix(seq)), 0]
        got_warm = [int(x) for x in np.asarray(warm).reshape(-1)]
        assert got_warm == want_warm, {
            "error": "golden/inputs.npz's warmup_request_distribution does not "
                     "count the requests whose cached prefix the warm-up forward "
                     "writes; this is an artifact defect",
            "golden": want_warm, "frozen": got_warm,
        }

    # `padded_num_reqs` is the compiled bucket: STATIC, never an array -- putting it in
    # a pytree's data fields retraces on every value and defeats the bucketing.
    bucket = schema["schema"].get("padded_num_reqs") or {}
    assert bucket.get("kind") == "static", {
        "error": "padded_num_reqs must be a static int, not a traced array",
        "candidate": bucket,
    }
    assert bucket.get("value") == golden.meta["max_num_seqs"], {
        "error": "padded_num_reqs is not the configured bucket",
        "golden": golden.meta["max_num_seqs"], "candidate": bucket.get("value"),
    }
    assert schema["num_seqs"] == golden.meta["max_num_seqs"], {
        "num_seqs": schema["num_seqs"], "golden": golden.meta["max_num_seqs"]}
    assert schema["pages_per_seq"] == golden.meta["pages_per_seq"], {
        "pages_per_seq": schema["pages_per_seq"],
        "golden": golden.meta["pages_per_seq"]}


def test_block_table_contract(golden, cand):
    """Block table + slot mapping: EXACT integer content, from the candidate's own
    allocator.

    THE COMPARISON IS NOT AGAINST THE FROZEN ARRAYS, and that is not a weakening --
    it is the only correct comparison available. ``golden/inputs.npz``'s
    ``block_tables`` / ``slot_mapping`` record the allocation ``instance.yaml`` PINS
    (``page_ids: [2, 0, 3]``), chosen deliberately NOT to be what the allocator hands
    out, so that ``slot != pos`` everywhere and a port that ignored the block table and
    indexed the cache flat could not reproduce ``kv_cache_state.npz``. This probe runs
    the candidate's own ``Engine.prefill`` on a fresh pool, which correctly returns the
    DEFAULT ascending assignment. Asserting the two are equal -- as a literal reading
    of "compare to the frozen scenario" would -- fails the worked reference itself.

    So one rule is checked twice. ``_expected_block_table`` / ``_expected_slot_mapping``
    restate ``cache/block_table.py``'s two index formulas; leg 1 applies them to the
    golden's PINNED page ids and requires the frozen arrays back, which is what stops
    the restatement from drifting; leg 2 applies them to the DEFAULT assignment a free
    list popped in ascending order produces and requires the candidate's arrays back.
    A wrong rule fails leg 1 on the reference; a wrong allocator or a wrong slot
    formula fails leg 2.

    LEG 1 COVERS THE WHOLE FROZEN MAPPING, both groups, element for element:
    ``slot_mapping`` over every scheduled query token of every declared sequence and
    ``warmup_slot_mapping`` over every cached prefix, composed by
    ``_expected_frozen_slots`` from the same rule
    the golden freezer composes them with. On `core` that is
    one full prefill and an empty warm-up, i.e. exactly the single-sequence
    assertion this leg has always made; on `ragged` it is 27 slots across three
    requests where a head-only slice reached 12. See the comment at the assertion
    for the two artifact defects the slice admitted.

    WHY PINNING THE ALLOCATOR IS LEGITIMATE.
    The structure checker holds ``BlockPool.allocate`` / ``free`` /
    ``num_free`` to being the same code on both reference sides by AST equality, in
    CI, and the reason is leg 2 above: the allocator is graded DIRECTLY, integer for
    integer, and a contract cannot demand exact integers unless the rule producing
    them is the same rule on both sides. An allocator that pops the free
    list from the top instead of the bottom fails this check and only this check.

    Two wrong reasons, named so neither is left to be re-derived:

    * NOT the cache goldens. ``USE_FROZEN_ALLOCATION["kv_cache_state"]`` is ``True``,
      so the cache-state probe is HANDED the frozen allocation and is indifferent to
      what the candidate's ``BlockPool`` would have chosen -- as is every
      allocation-dependent artifact in the numeric tier.
    * NOT paging-invariance. Its two page lists come from the DRIVER
      (``_default_pages`` / ``_permuted_pages`` in ``_t2j/candidate.py``), not from
      the pool, and the only pool methods it reaches are ``slot_mapping`` and
      ``block_table_row`` -- the two INDEX methods, which the AST pin deliberately
      excludes because they return the framework's array type.

    ``_invariance_cache_reuse`` does call the pinned ``allocate`` and ``free``
    directly, so ``test_cache_reuse_safety`` inherits the dependency. Two mutations
    of the pinned allocator show where that dependency lands: one that makes
    ``allocate`` stop consuming the free list trips ``free``'s own double-free guard
    and takes the whole worker down (one identical ``CandidateError``, nothing
    attributable), and one that makes ``free`` forget which pages are live fails
    ``test_cache_reuse_safety`` and nothing else. So the dependency is real and it is
    that check's, not this one's:
    neither allocator mutation moves ``test_block_table_contract``, whose leg 2
    asks only for the DEFAULT ascending assignment and gets it from both.
    """
    page_size = int(golden.meta["page_size"])
    pages_per_seq = int(golden.meta["pages_per_seq"])
    max_num_seqs = int(golden.meta["max_num_seqs"])
    prompt_len = int(golden.meta["prompt_len"])
    pinned = golden.scenario_page_ids
    assert pinned, {"error": "scenario.json pins no page_ids"}
    # THE FROZEN TABLE IS THE WHOLE BUCKET, one row per request slot -- which is
    # what `AttentionMetadata` declares (`(max_num_seqs * pages_per_seq,)`) and
    # what `test_static_shape_bucketing` asserts the size of. On `core`
    # (max_num_seqs == 1) that is one row and this list is `[pinned]`, i.e.
    # exactly the single-request expectation this leg has always had. A
    # multi-sequence scenario states each request's pages in `meta["sequences"]`;
    # sequence 0's are `page_ids`, which `validate_sequences` requires.
    declared = _declared_sequences(golden.meta, pinned, prompt_len)
    per_request = [request["page_ids"] for request in declared]
    assert per_request[0] == [int(p) for p in pinned], {
        "error": "sequence 0's page list is not the scenario's `page_ids`; the "
                 "decode slot arithmetic reads the second and the attention "
                 "gather the first",
        "page_ids": list(pinned), "sequences[0]": per_request[0],
    }

    def _bucketed_table(rows: list[list[int]]) -> np.ndarray:
        parts = [_expected_block_table(row, pages_per_seq) for row in rows]
        parts += [np.zeros(pages_per_seq, dtype=np.int64)] * (max_num_seqs - len(rows))
        return np.concatenate(parts)

    # Leg 1 -- the formulas reproduce the frozen artifacts from the pinned pages.
    np.testing.assert_array_equal(
        np.asarray(golden.scenario_block_table).reshape(-1).astype(np.int64),
        _bucketed_table(per_request),
        err_msg="the frozen block table is not each request's `page_ids` "
                "right-padded with 0 to pages_per_seq, one row per request slot; "
                "the golden and cache/block_table.py disagree")
    # EVERY SLOT THE SCENARIO FREEZES, ELEMENT FOR ELEMENT -- both groups.
    #
    # A head-only comparison (`frozen_slots[:head.size]` against
    # sequence 0's mapping) with a compensating `written == want_total` COUNT
    # clause standing in for the rest is not enough. That is byte-equivalent on
    # `core`, where sequence 0 IS the whole batch and there is no warm-up, but on
    # a ragged batch the head covers only the first sequence and every other slot
    # is checked by its COUNT alone. Two size-preserving artifact defects pass such
    # a check -- sequence 1's decode slot moved onto a page sequence 0 owns,
    # and sequences 1..N rewritten as a pure `slot == pos` identity -- both of
    # which make the paged tier gradeable by a port that never reads a block
    # table. Both are caught here: leg 1
    # touches no `cand` fixture, so the whole leg runs host-side against a
    # `fixtures.load_golden` of a MUTATED COPY of the golden tree, with a stub
    # candidate that raises the moment leg 2 reaches it. That host-side instrument
    # is needed because the standard mutation controls patch the reference
    # implementation, and leg 1's defect class is a wrong ARTIFACT.
    #
    # THE COUNT CLAUSE IS GONE, not moved. It asserted that `slot_mapping` plus
    # `warmup_slot_mapping` held exactly as many entries as the declared
    # sequences have tokens; `_expected_frozen_slots` builds both arrays from
    # those same counts and `assert_array_equal` fails on a length mismatch, so
    # no input could fail the count clause and pass the two below. A clause that
    # cannot fail independently reads as coverage and is worse than no clause --
    # the same argument `instance.yaml` records for the cumulative-logprob bound
    # it replaced.
    frozen_slots = np.asarray(golden.scenario_slot_mapping).reshape(-1).astype(np.int64)
    warm = golden.inputs.get("warmup_slot_mapping")
    frozen_warm = (np.zeros(0, dtype=np.int64) if warm is None
                   else np.asarray(warm).reshape(-1).astype(np.int64))
    want_slots, want_warm = _expected_frozen_slots(declared, page_size)
    np.testing.assert_array_equal(
        frozen_slots, want_slots,
        err_msg="the frozen slot mapping is not page_ids[pos // page_size] * "
                "page_size + pos % page_size over every scheduled query token of "
                "every declared sequence, concatenated in schedule order; the "
                "golden and cache/block_table.py disagree")
    np.testing.assert_array_equal(
        frozen_warm, want_warm,
        err_msg="the frozen WARM-UP slot mapping is not the same formula over "
                "each `decode` request's cached prefix; the golden and "
                "cache/block_table.py disagree")
    # ...and the pinned allocation must still be the non-identity one the whole paged
    # tier depends on. A re-freeze that quietly restored `page_ids: [0, 1, 2]` makes
    # slot == pos, at which point flat indexing reproduces every cache golden.
    # Asserted over SEQUENCE 0's full span -- `page_ids` is its page list by
    # `validate_sequences`, and it is the sequence every single-request probe
    # drives. Not over the concatenated batch: `np.any(x != arange(x.size))` gets
    # EASIER to satisfy as `x` grows, so widening it would weaken the guard.
    head = _expected_slot_mapping(declared[0]["page_ids"], page_size,
                                  declared[0]["n"])
    assert np.any(head != np.arange(head.size)), {
        "error": "the frozen slot mapping is the identity, so a port that ignored "
                 "block_tables entirely would reproduce the cache goldens",
        "slot_mapping": head.tolist(),
        "hint": "instance.yaml's page_ids must not be the ascending default",
    }

    # Leg 2 -- the candidate's own allocator, on its own fresh pool. ONE request:
    # the probe runs `Engine.prefill` on the frozen prompt, so the other
    # max_num_seqs - 1 rows of its bucket are the unscheduled zeros.
    default_pages = _default_page_ids(prompt_len, page_size)
    np.testing.assert_array_equal(
        np.asarray(cand.block_table).reshape(-1).astype(np.int64),
        _bucketed_table([default_pages]),
        err_msg=f"the block table your engine built for the frozen prompt is not the "
                f"free list popped ascending ({default_pages}) right-padded with 0 to "
                f"pages_per_seq={pages_per_seq}, in a table of max_num_seqs="
                f"{max_num_seqs} rows")
    np.testing.assert_array_equal(
        np.asarray(cand.slot_mapping).reshape(-1).astype(np.int64),
        _expected_slot_mapping(default_pages, page_size, prompt_len),
        err_msg="the slot mapping your engine built for the frozen prompt does not "
                "follow slot = page * page_size + offset over those pages")


def test_static_shape_bucketing(golden, cand):
    """Metadata is padded to the COMPILED BUCKET, not to the request's occupancy.

    ``padded_num_reqs`` is ``max_num_seqs`` and the block table is
    ``max_num_seqs * pages_per_seq`` wide REGARDLESS of how many pages this request
    actually needs -- that constant shape is what lets one compiled program serve every
    batch, and it is the whole reason ``padded_num_reqs`` is a static meta field rather
    than a traced value. The frozen prompt needs 2 pages of the 4 the bucket
    advertises, so a candidate that sized its block table to the occupancy produces 2
    here. An engine that builds its block table at ``len(page_ids)`` instead
    of ``pages_per_seq`` fails this check, ``test_block_table_contract`` and
    ``test_attention_metadata_schema`` -- three of this module's fourteen, all three
    legitimately,
    since the same wrong width is a wrong allocation and a wrong metadata shape too.

    ONLY ``block_table_size`` DISCRIMINATES ON THIS INSTANCE. With ``max_num_seqs ==
    1`` the bucket and the occupancy coincide for ``seq_lens`` (1) and
    ``query_start_loc`` (2), and ``padded_num_reqs`` is 1 either way -- so those three
    are satisfied by an unbucketed engine too. The block table is the one place the
    two differ here: the frozen prompt needs 2 pages of the 4 the bucket advertises.
    On a ``ragged`` or ``chunked`` instance, where ``max_num_seqs > 1``, all four
    discriminate.

    The COMPILATION half of the bucketing contract -- which the shapes only imply,
    since an engine that emits perfect shapes and re-traces anyway produces identical
    numbers -- lives in ``test_static_shape_recompilation``. It is separate because
    it needs a second prompt length that the frozen scenario may not admit, and a
    check that cannot apply must deselect rather than fail.
    """
    shapes = cand.static_shapes
    meta = golden.meta
    max_num_seqs = int(meta["max_num_seqs"])
    pages_per_seq = int(meta["pages_per_seq"])
    want = {
        "padded_num_reqs": max_num_seqs,
        "max_num_seqs": max_num_seqs,
        "pages_per_seq": pages_per_seq,
        "num_query_tokens": int(meta["prompt_len"]),
        # The BUCKET widths, cross-checked against what the reference actually froze
        # so they are read off an artifact and not only off the config.
        "block_table_size": max_num_seqs * pages_per_seq,
        "seq_lens_size": max_num_seqs,
        "query_start_loc_size": max_num_seqs + 1,
    }
    frozen = {
        "block_table_size": int(np.size(golden.inputs["block_tables"])),
        "seq_lens_size": int(np.size(golden.inputs["seq_lens"])),
        "query_start_loc_size": int(np.size(golden.inputs["query_start_loc"])),
    }
    disagree = {k: {"from_config": want[k], "frozen_in_inputs_npz": v}
                for k, v in frozen.items() if want[k] != v}
    assert not disagree, {
        "error": "golden/meta.json's geometry and golden/inputs.npz disagree about "
                 "the compiled bucket; this is an artifact defect",
        "disagreement": disagree,
    }
    mismatch = {k: {"expected": v, "candidate": shapes.get(k)}
                for k, v in sorted(want.items()) if shapes.get(k) != v}
    assert not mismatch, {
        "static_shape_mismatch": mismatch,
        "hint": "pad metadata to max_num_seqs / max_num_seqs * pages_per_seq, not to "
                "the number of sequences or pages this request happens to use",
    }


def test_compilation_is_cached(cand):
    """The forward is compiled, and calling it again at the same shape costs nothing.

    ALWAYS RUNS. Half of the compilation contract needs no second prompt length, and
    for one week it rode the geometry gate that the other half needs -- which meant an
    instance with a short probe prompt lost the ONLY detector of a port that rebuilds
    its ``jit`` wrapper per call, while its summary read as full coverage. That is the
    shape of a silent coverage loss, so the two halves are two checks.

    * THE COUNTER IS LIVE. One ``nnx.jit``-ed full-model forward that this grader
      wrote itself costs exactly **1** compilation. Asserted first, because a counter
      that silently observed nothing would satisfy everything below it for every
      candidate -- the failure mode a compilation check is most exposed to. A
      PARENT-side counter reads 0 while the child reports 16, which is why the
      counter lives inside the worker.
    * A REPEAT AT THE SAME SHAPE COSTS ZERO. Threshold-free. A port that
      builds its ``nnx.jit`` wrapper inside the call fails here and nowhere else in
      the tier.
    * DECODE STEADY STATE COSTS ZERO. Every clause above is a PREFILL; without this
      clause, "one compiled program per shape" is graded for exactly half of
      instruction.md's contract ("the graded loop calls prefill once and then decodes
      step by step; EACH of those must reuse a single compiled program"). This clause
      is what closes that half -- kept IN THIS NODE, not a new test function, so the
      graded denominator does not move and an instance that cannot express a second
      prompt length still gets this coverage.

    THE DECODE CLAUSE FAILS -- IT DOES NOT SKIP -- ON AN ARTIFACT GAP, and the
    previous round had this exactly backwards. An instance whose golden carries
    fewer than ``_t2j.candidate.MIN_STEADY_STATE_DECODE_IDS``
    ``decode_input_token_ids`` cannot express this clause; that round made it
    ``pytest.skip``, reasoning that a skip is the only outcome that does not charge
    the candidate for a freeze gap. It is also the only outcome that DELETES THE
    CHECK: ``summary_to_reward`` computes ``n_passed / (n_passed + n_failed +
    n_error)``, so a skip leaves the numerator AND the denominator -- 30/(30+0+0) =
    **1.0** -- and a port that rebuilds its ``nnx.jit`` wrapper inside
    ``decode_step`` is bit-identical to the reference in every value comparison.
    The compile count is the only thing that separates them, so the skip scored that
    port a perfect grade. ``conftest`` states the rule in two places ("NO check may
    skip at grade time"; "DESELECT, never skip") and ``summary_to_reward``'s own
    docstring rests on it.

    THE ROUTING IS THE PRECHECK'S JOB, NOT THIS CHECK'S.
    ``_harness_checks.check_scenario_carries_a_gradable_decode_run`` refuses such a
    golden in ``_grade_precheck``: fatal, before pytest, no ``reward.json``, and a
    message naming the freezer. So the clause below is unreachable in a graded run;
    it exists for a bypassed precheck, and it fails LOUDLY as a harness fault rather
    than silently vanishing. ``n_skipped`` stays 0 and the denominator stays 31.
    """
    compiles = cand.compilation
    assert compiles.get("supported"), {
        "error": "the compilation probe did not run",
        "reason": compiles.get("reason"), "probe": compiles,
    }
    assert int(compiles["control"]) >= 1, {
        "error": "the compilation counter observed nothing even for a forward this "
                 "grader jitted itself, so it cannot be used to judge the candidate",
        "probe": {k: v for k, v in compiles.items() if not k.endswith("_names")},
    }
    assert int(compiles["same_shape"]) == 0, {
        "error": "re-running the same prefill re-compiled. A jit wrapper built inside "
                 "the call re-traces every time, which is the same cost as no jit at "
                 "all and defeats the bucketing test_static_shape_bucketing grades.",
        "compilations_on_repeat": compiles["same_shape"],
        "probe": {k: v for k, v in compiles.items() if not k.endswith("_names")},
    }
    assert compiles.get("decode_steady_state") is not None, {
        "error": "the compilation probe could not run its decode leg, so the only "
                 "detector of a port that re-jits per decode token has nothing to "
                 "measure. This is a HARNESS fault -- an artifact gap in "
                 "golden/scenario.json -- and not a port defect, but it is a "
                 "FAILURE rather than a skip: a skip leaves both terms of "
                 "passed/(passed+failed+error) and would score a re-jitting port "
                 "1.0. _grade_precheck should have refused this golden before "
                 "pytest started; see check_scenario_carries_a_gradable_decode_run.",
        "reason": compiles.get("decode_steady_state_reason"),
        "probe": {k: v for k, v in compiles.items() if not k.endswith("_names")},
    }
    # A KEY-PRESENCE GUARD, and no more than that: it does not independently verify
    # that "the counted region actually counted something". Given the clause above,
    # the probe's own
    # `steady_state_decode_tokens` guarantees at least one counted step whenever
    # `decode_steady_state` is not None, so this can only fire if the probe stops
    # REPORTING the key (a rename, a partial return dict). That is worth one line:
    # without it, `.get(...) or 0` would read a missing key as zero steps and the
    # assertion below would pass for any candidate at all.
    assert int(compiles.get("decode_steady_state_steps") or 0) >= 1, {
        "error": "the decode steady-state leg reported a compilation count with no "
                 "`decode_steady_state_steps` beside it, so `decode_steady_state "
                 "== 0` below would pass without anything having been measured. "
                 "The probe stopped reporting the key -- a harness regression.",
        "steps_counted": compiles.get("decode_steady_state_steps"),
        "probe": {k: v for k, v in compiles.items() if not k.endswith("_names")},
    }
    assert int(compiles["decode_steady_state"]) == 0, {
        "error": "decoding re-compiled after the decode shape was already warm. Every "
                 "token then pays a full trace and compile, which is the same cost as "
                 "no jit at all -- and it is invisible to every value comparison, "
                 "because the outputs of a per-token-retraced decode are "
                 "bit-identical to a correctly cached one; nothing but the compile "
                 "count and the wall clock can tell them apart.",
        "compilations_over_steady_state_decode_steps": compiles["decode_steady_state"],
        "first_few": compiles.get("decode_steady_state_names"),
        "hint": "build the jit wrapper once, at construction; do not build it inside "
                "decode_step",
    }


def test_static_shape_recompilation(golden, cand):
    """A NEW shape costs one more compiled program, not the whole graph again.

    The half of the compilation contract that genuinely needs a second prompt length,
    and therefore the only Spec check in ``conftest._APPLICABLE``. GEOMETRY-GATED, on
    ``has_second_prompt_length`` -- which is
    ``_t2j.candidate.second_prompt_length(prompt_len, page_size) is not None``, THE
    SAME CALL the probe makes. Gate and probe were two hand-written expressions once
    and disagreed in both directions; one function is the fix.

    Deselected, not failed, when the geometry admits no second length: the reward is
    ``passed/(passed+failed+error)``, so charging a candidate for a property this
    instance's frozen prompt cannot express is the same accounting error as a free
    pass with the sign flipped. The deselection is named in ``eval_summary.json``'s
    ``deselected`` list, so it cannot pass for coverage.

    THE BOUND IS ON THE MODEL BODY, not on the raw new-shape total. The probe warms
    the engine's eager bookkeeping at the target length first, through the driver's
    own jitted path, so what remains is the body. The raw new-shape total is 16
    jitted against 82 unjitted (5.1x); the body is 9 against 98 (10.9x). A port with no
    ``jit`` at all fails here and nowhere else in the tier.

    SECOND CLAUSE: THE BUCKET AT THAT SECOND LENGTH, which is where "padded to the
    compiled bucket" is actually falsifiable. ``test_static_shape_bucketing`` grades
    the same four sizes at the FROZEN prompt length only, and on a geometry where
    that prompt's page occupancy equals ``pages_per_seq`` the bucket and the
    occupancy are the same integer -- so an engine that sizes its block table at
    ``len(page_ids)`` emits the right width and that check, plus
    ``test_block_table_contract`` and ``test_attention_metadata_schema``, all agree
    with it. For example, a prompt of 135 at ``page_size`` 32 occupies 5 pages, so
    with ``pages_per_seq`` 5 all three pass an occupancy-sized engine and only this
    clause (the bucket at the SECOND length) separates the bucket from the
    occupancy; a ``pages_per_seq`` of 6 separates the two at the frozen length as
    well. The two halves are complementary and neither replaces the other.

    The prefill at the second length ALREADY RAN, one leg above, and its metadata
    was discarded at a ``del``. Reading four integers off it costs no forward, no
    compilation and no golden. Occupancy at that length against ``pages_per_seq``,
    on the four shipped instances: 1/4 (core_llama3), 1/4 (core_qwen2), 4/6
    (core_gpt_oss), 1/4 (ragged_llama3) -- so the clause discriminates on all four.

    IT DOES NOT DISCRIMINATE ON EVERY GEOMETRY, and the report says which case it is
    rather than implying coverage it has not got. ``second_prompt_length`` subtracts
    ``page_size // 2``, which lowers the occupancy UNLESS ``prompt_len`` is an exact
    multiple of ``page_size`` (16 at ``page_size`` 8 gives 12: two pages either
    way). The guard is computed here from ``meta``'s own geometry -- never assumed,
    and never taken from the candidate, which is the party being graded. Where the
    two coincide the assertion still holds and is a consistency check on the
    bucket rather than a detector of occupancy sizing; it is NOT skipped, because a
    skip would take this whole check out of the reward's denominator.
    """
    compiles = cand.compilation
    assert compiles.get("supported"), {
        "error": "the compilation probe did not run",
        "reason": compiles.get("reason"), "probe": compiles,
    }
    # The gate said this instance has a second length; the probe is the authority on
    # whether it found one. They call the same function, so a disagreement is a
    # harness defect and says so -- it is not a verdict on the candidate.
    assert compiles.get("body") is not None, {
        "error": "conftest collected this check but the probe found no second prompt "
                 "length to use. The gate and the probe call the same predicate "
                 "(_t2j.candidate.second_prompt_length), so this is a harness defect.",
        "prompt_len": compiles.get("prompt_len"),
        "probe": {k: v for k, v in compiles.items() if not k.endswith("_names")},
    }
    assert int(compiles["body"]) <= _MAX_BODY_COMPILATIONS, {
        "error": "prefilling at a second prompt length compiled the model body "
                 "primitive by primitive rather than as one program, so every new "
                 "shape pays for the whole graph again",
        "body_compilations": compiles["body"],
        "budget": _MAX_BODY_COMPILATIONS,
        "raw_new_shape_total_including_glue": compiles["new_shape"],
        "one_jitted_forward_costs": compiles["control"],
        "lengths": [compiles.get("prompt_len"), compiles.get("other_prompt_len")],
        "first_few": compiles.get("body_names"),
    }

    shapes = compiles.get("other_static_shapes")
    assert shapes is not None, {
        "error": "the compilation probe reported no metadata shapes for the second "
                 "prompt length. It prefills at that length either way, so this is "
                 "a harness defect and not a verdict on the port.",
        "probe": {k: v for k, v in compiles.items() if not k.endswith("_names")},
    }
    meta = golden.meta
    max_num_seqs = int(meta["max_num_seqs"])
    pages_per_seq = int(meta["pages_per_seq"])
    page_size = int(meta["page_size"])
    short_len = int(shapes["prompt_len"])
    want = {
        "block_table_size": max_num_seqs * pages_per_seq,
        "seq_lens_size": max_num_seqs,
        "query_start_loc_size": max_num_seqs + 1,
        "padded_num_reqs": max_num_seqs,
    }
    # THE GUARD, from meta's geometry and not from the candidate's metadata: how
    # many pages this shorter prompt actually needs. Only reported -- an instance
    # whose occupancy already equals the bucket is not a failing instance.
    occupancy = (short_len + page_size - 1) // page_size
    mismatch = {k: {"bucket_from_meta": v, "candidate_at_len_%d" % short_len:
                    shapes.get(k)}
                for k, v in sorted(want.items()) if shapes.get(k) != v}
    assert not mismatch, {
        "error": "at a second prompt length the engine's metadata is not padded to "
                 "the compiled bucket. Static shapes are what let one compiled "
                 "program serve every request; sizing them to what this request "
                 "happens to occupy re-traces per length.",
        "static_shape_mismatch": mismatch,
        "second_prompt_len": short_len,
        "pages_this_prompt_occupies": occupancy,
        "pages_per_seq": pages_per_seq,
        "clause_discriminates_on_this_geometry": occupancy < pages_per_seq,
        "hint": "pad metadata to max_num_seqs / max_num_seqs * pages_per_seq, not "
                "to the number of sequences or pages this request happens to use",
    }


def test_jit_traceability(golden, cand):
    """The forward lowers to a jaxpr with NO host-callback primitive.

    Asserted over the LOWERED JAXPR, never over source text: a grep for ``callback``
    passes for a candidate that imports it under an alias, and the escape hatch this
    exists to close -- calling torch from inside a "pure" JAX forward through
    ``pure_callback`` / ``io_callback`` / ``debug_callback`` -- is invisible in the
    import graph once aliased. ``jax.make_jaxpr`` sees the primitive whatever it was
    called at the call site.

    The PRIMITIVE COUNT is the second half and it is not decoration, because a forward
    can be perfectly pure and traceable and still do nothing. The two
    halves are independent: a ``pure_callback`` reached through an alias in the live
    forward fails the CALLBACK clause alone (``callbacks == ['pure_callback']``), and
    a forward that returns zeros without touching the model -- no callback anywhere --
    fails the FLOOR clause alone (``num_primitives`` 1 against a floor of 320). Each
    fails this check and no other. A genuine 16-layer transformer lowers to 2 950
    equations, so the per-layer floor sits an order of magnitude below the real thing
    and means the same on the 20B instances.

    THE DETECTOR'S OWN COVERAGE IS ASSERTED TOO. ``purity.CALLBACK_PRIMITIVES`` is a
    substring list; narrowing it to ``("pure_callback",)`` would leave this check
    green while it silently stopped seeing ``io_callback`` and ``debug_callback``. A
    check that cannot fail is worse than no check, so the list is required to cover all
    three names.
    """
    for name in _CALLBACK_PRIMITIVES:
        assert any(needle in name for needle in purity.CALLBACK_PRIMITIVES), {
            "error": f"the purity scanner cannot detect the {name!r} primitive, so "
                     "this check would pass a candidate that used it",
            "detector": list(purity.CALLBACK_PRIMITIVES),
        }
    info = cand.traceability
    assert info.get("supported"), {
        "error": "the candidate's forward could not be traced at all",
        "reason": info.get("reason"), "info": info,
    }
    assert info.get("callbacks") == [], {
        "host_callbacks_in_forward": info.get("callbacks"),
        "hint": "a pure JAX forward has no pure_callback / io_callback / "
                "debug_callback; they are the escape hatch back to host (torch) code",
    }
    num_layers = int(golden.cache_spec["num_layers"])
    floor = _MIN_PRIMITIVES_PER_LAYER * num_layers
    assert int(info.get("num_primitives", 0)) >= floor, {
        "error": "the lowered forward is too small to be a real transformer -- this "
                 "is the shape of a host-callback shim",
        "num_primitives": info.get("num_primitives"),
        "floor": floor, "num_layers": num_layers,
    }
    # DELIBERATELY FREE TODAY, and said so rather than left for a reader to
    # re-derive. The probe computes `traced` as `primitives > 0 and not
    # callbacks` (`_t2j/candidate.py`'s traceability probe), and the two clauses
    # directly above assert both conjuncts with STRICTLY stronger bounds --
    # `callbacks == []` and `num_primitives >= 20 * num_layers`, a floor two
    # orders above zero. So this cannot fire while they hold.
    #
    # It stays because it is the field the L1 contract names, and because it is
    # the one clause that would survive the probe's definition of `traced`
    # diverging from those two conjuncts -- the same reason `banned_leaks == []`
    # stays in `test_no_torch_in_candidate`. A free clause that is documented as
    # free is a statement; an undocumented one reads as coverage.
    assert info.get("traced"), info


# =========================================================================== #
# Config and loader
# =========================================================================== #
def test_config_spec(golden, cand):
    """The port resolves the same configuration the reference did.

    NO SEPARATE GOLDEN. There is no ``config_spec.json`` in the golden catalog, and
    none is needed:
    the resolved config is already mirrored into ``meta.json`` by the freezer (every
    ``instance.yaml`` key, plus the ``repo_config`` the grader hands the candidate), so
    a third file would restate it and add a way for the two to disagree.

    Two halves, and neither is an echo. Both read the candidate's OWN objects:

    * ``RunConfig`` -- read back from the constructed object AFTER its
      ``__post_init__``, not from the payload the grader passed in. The distinction is
      measurable: a port that clamps, defaults or silently overrides a knob returns a
      value it was never given, and that is a real mutation this catches. Its FIELD
      LIST is the candidate's own too, so a ``RunConfig`` that dropped a contracted
      knob names the knob here -- but only for 2 of the 8: the driver dereferences
      ``page_size``, ``num_pages``, ``pages_per_seq``, ``max_num_seqs``, ``dtype``
      and ``num_decode_steps`` eagerly in ``_CandidateRuntime.__init__``, so a port
      missing one of those still kills the worker before this check is reached.
      ``max_model_len`` and ``seed`` are the two this actually rescues.
    * ``ModelConfig`` -- ``num_layers`` / ``num_kv_heads`` / ``head_dim`` are NOT
      config knobs. They are resolved from the checkpoint's own ``config.json`` by the
      candidate's ``ModelConfig.from_hf``. This clause is DIAGNOSTIC on this
      reference, not discriminating, and the distinction is measured rather than
      guessed: a port that reads ``num_attention_heads`` where it should read
      ``num_key_value_heads`` -- the classic GQA slip -- does fail here, but it fails
      everything, because the shapes come from the checkpoint and ``apply_params``
      refuses the mismatched ``qkv_proj`` first, taking the whole worker down. All
      three quantities are load-bearing for the model build, so a
      wrong one always kills the run on this reference. What this clause buys is the
      one legible message -- ``num_kv_heads: golden 8, candidate 32`` -- next to ten
      tracebacks, and coverage for a port whose ``apply_params`` is permissive.

    THE TWO GEOMETRY INVARIANTS ARE DIAGNOSTIC, NOT DISCRIMINATING, and are labelled
    so nobody later reads them as independent coverage. They are evaluated over the
    candidate's values, not over ``golden.meta`` -- evaluating them with the
    candidate absent from both sides would make them constant-true and therefore a
    guaranteed point for every candidate in a reward of
    ``passed/(passed+failed+error)``. Because they read the candidate, they are real
    and reachable: a port that drops ``RunConfig``'s guards and defaults
    ``max_model_len`` from the pool size fails on the first of them, and on no other
    check. But they cannot fail
    unless the field comparison below WOULD ALSO have failed, because the configured
    geometry is self-consistent. Their value is localization -- "this is what is
    unrunnable about it" instead of a list of numbers that do not match -- which is
    why they run first. Not a second, independent gate.

    Also the matmul-precision pin: ``JAX_DEFAULT_MATMUL_PRECISION=highest`` is a Global
    Constraint, and the value asserted is the one JAX reported from INSIDE the worker,
    against what ``meta.provenance`` says the grade must run at. An fp32 tier measured
    at TF32 is the failure this pin exists to prevent, and this is where the two ends
    of it are tied together.

    ``model_name_or_path`` IS DELIBERATELY NOT CHECKED HERE. The candidate never
    resolves a repo id -- the grader hands it in and ``loader_facts`` echoes it back --
    so comparing the two compares a value to itself. What the candidate does with it
    is graded in ``test_loader_spec``, where the resolved ``model_dir`` and the
    rejection of an unbaked id both have two sides.
    """
    meta = golden.meta
    repo_config = dict(meta["repo_config"])

    declared = ((golden.interface.get("dataclass_fields") or {})
                .get("config.py", {}).get("RunConfig") or [])
    assert declared, {
        "error": "interface.json froze no RunConfig field list",
        "hint": "re-freeze the interface golden from the reference implementation",
    }
    unsupplied = sorted(set(declared) - set(repo_config))
    assert not unsupplied, {
        "error": "golden/meta.json's repo_config does not carry every RunConfig field "
                 "the contract declares, so the grader cannot configure a correct "
                 "candidate. This is an artifact defect.",
        "unsupplied": unsupplied,
    }

    resolved = cand.resolved_config
    values = dict(resolved["values"])
    absent = sorted(set(declared) - set(resolved["fields"]))
    assert not absent, {
        "error": "your RunConfig does not carry every field the contract declares, so "
                 "the grader cannot configure it",
        "missing_fields": absent,
        "your_fields": sorted(resolved["fields"]),
    }
    # The two geometry invariants RunConfig.__post_init__ states, over the values the
    # candidate's own object holds -- BEFORE the field-by-field comparison, so a port
    # that resolved an unrunnable geometry is told what is unrunnable about it rather
    # than being handed a list of numbers that do not match. (Order matters here: any
    # concrete mutation trips both, and the geometry message is the useful one.)
    assert (values["pages_per_seq"] * values["page_size"]
            >= values["max_model_len"]), {
        "error": "your resolved geometry cannot hold a full-length sequence: "
                 "pages_per_seq * page_size < max_model_len", "resolved": values}
    assert (values["num_pages"]
            >= values["max_num_seqs"] * values["pages_per_seq"]), {
        "error": "your resolved pool cannot fill the block table it advertises: "
                 "num_pages < max_num_seqs * pages_per_seq", "resolved": values}

    mismatch = {name: {"configured": repo_config[name], "your_runconfig_holds": values.get(name)}
                for name in sorted(declared)
                if name in repo_config and values.get(name) != repo_config[name]}
    assert not mismatch, {
        "resolved_run_config_mismatch": mismatch,
        "hint": "these are read back off your RunConfig object after __post_init__; a "
                "value that is not the one you were handed means something rewrote it",
    }

    # The model SHAPE, read out of the checkpoint by the candidate's own from_hf.
    model_shape = {k: cand.cache_spec[k]
                   for k in ("num_layers", "num_kv_heads", "head_dim")}
    shape_mismatch = {k: {"golden": golden.cache_spec[k], "candidate": v}
                      for k, v in sorted(model_shape.items())
                      if golden.cache_spec[k] != v}
    assert not shape_mismatch, {
        "resolved_model_config_mismatch": shape_mismatch,
        "hint": "ModelConfig.from_hf reads the checkpoint's config.json; "
                "num_key_value_heads is not num_attention_heads under GQA",
    }

    want_precision = (meta.get("provenance") or {}).get("matmul_precision")
    assert want_precision, {
        "error": "meta.provenance carries no matmul_precision", "meta": meta.get(
            "provenance")}
    assert cand.sandbox_report.get("matmul_precision") == want_precision, {
        "error": "the sandbox ran at a different matmul precision than the grade is "
                 "pinned to; an fp32 tier measured at TF32 is not an fp32 tier",
        "pinned": want_precision,
        "effective": cand.sandbox_report.get("matmul_precision"),
        "env": cand.sandbox_report.get("matmul_precision_env"),
    }


def test_loader_spec(golden, cand):
    """The loader's observable contract: ``/data/{repo_id}``, offline, fusion order.

    NO SEPARATE GOLDEN, for the same reason as ``test_config_spec``: everything here
    is already reachable through ``interface.json`` plus the live loader probe, and a
    ``loader_spec.json`` would only add a second place for the same facts to drift.

    THE ``/data`` CONTRACT IS CHECKED STATICALLY, and it has to be. ``resolve_model_dir``
    hardcodes ``/data/{repo_id}`` -- both images bake the checkpoint there and grading
    has no network -- but ``_t2j/candidate.py`` REBINDS ``model/loader.DATA_ROOT`` on an
    authoring host that has no ``/data`` (``T2J_INFERENCE_DATA_ROOT``, unset in every
    graded run). So the runtime value proves nothing about the contract, while the
    SOURCE constant cannot be rebound. Both are checked: the source must declare the
    frozen root, and the resolution must be ``root / repo_id`` under whichever root was
    effective.

    ``STACKED_PARAMS`` is the fusion LAYOUT DECLARED, not a naming detail: the third
    field is the shard's index within the fused tensor, so ``q, k, v`` in that order
    along dim 0 is the difference between a working ``qkv_proj`` and one whose heads
    are permuted. Frozen only where torch_orig and the reference state the same table.

    DECLARED IS ALL IT IS, and the difference is a measured coverage hole rather than
    a quibble. This check compares the candidate's TABLE to the contract, so it sees a
    swap written INTO the table -- k and v exchanged in ``STACKED_PARAMS``
    fails here and nowhere else in the tier, since the shape is unchanged and the L2
    norm of a concatenation is invariant to the order of its parts, which makes both
    parameter checks blind by construction.

    A converter that concatenates q, v, k while the table still declares q(0), k(1),
    v(2) is NOT caught: it scores a clean sweep of this tier. It is not a no-op --
    the candidate's own KV cache state moves -- but
    nothing in Spec looks at a value. No Spec check sees fusion order AS IMPLEMENTED.
    Closing it needs a value-level comparison, which is Numeric's job; the
    end-to-end generated-sequence check is where it lands.

    AND THE LOADER MUST REFUSE AN UNBAKED CHECKPOINT. That is the "under no-network"
    half of this check and the only loader fact here that the candidate computes
    rather than receives: ``resolve_model_dir`` is handed a repo id nothing bakes, and
    must raise. A loader that falls through -- returning a path it never checked, or
    reaching for the hub -- hands back a directory whose weights do not exist, and the
    failure surfaces hundreds of lines later inside ``convert_checkpoint``.

    WHAT IS NOT CHECKED, and why: ``loader_facts['requested_model_name_or_path']`` is
    the repo id the GRADER passed in, echoed back by the probe. Asserting it equals
    ``meta['repo_config']['model_name_or_path']`` compares a value to itself, and an
    assertion that cannot fail is a guaranteed point in a continuous pass rate. It is
    used in failure messages only.

    THE OFFLINE CLAUSE MOVED FOR THAT SAME RULE. It read ``os.environ.get(name) ==
    "1"`` in the GRADER's process, which was falsifiable only because ``conftest``
    set the two variables with ``setdefault`` -- so an inherited ``HF_HUB_OFFLINE=0``
    both broke hermeticity AND failed this check, charging a correct port for
    the operator's shell. ``conftest`` now ASSIGNS them, which is the right fix for
    hermeticity and would have left this clause asserting that ``conftest``
    executed. It reads the SANDBOX CHILD's own environment instead
    (``sandbox_report['offline_env']``): the child is the process that would do the
    downloading, it is where the ``spawn`` inheritance is actually observable, and
    a candidate that clears the variable inside ``run_all`` -- which is what a port
    reaching for the hub would do -- shows up there and nowhere else.
    """
    loader = golden.interface.get("loader") or {}
    facts = cand.loader_facts
    assert loader.get("data_root") and loader.get("stacked_params"), {
        "error": "interface.json froze no loader contract",
        "loader": loader,
        "hint": "re-freeze the interface golden from the reference implementation",
    }

    declared = _declared_path_constant(
        Path(cand.repo_dir) / "model" / "loader.py", "DATA_ROOT")
    assert declared == loader["data_root"], {
        "error": "your loader does not declare the contracted checkpoint root",
        "contract": loader["data_root"], "candidate_source_declares": declared,
        "hint": "both images bake the checkpoint at /data/{repo_id} and grading runs "
                "with no network, so nothing can be downloaded",
    }

    effective_root = Path(facts["data_root"])
    repo_id = str(golden.meta["repo_config"]["model_name_or_path"])
    assert Path(facts["model_dir"]) == effective_root / repo_id, {
        "error": "resolve_model_dir did not resolve the repo id under the data root",
        "data_root": str(effective_root),
        "repo_id": repo_id,
        "requested": facts.get("requested_model_name_or_path"),
        "resolved": facts["model_dir"],
    }
    rejection = facts.get("rejects_unknown_model") or {}
    assert rejection.get("raised"), {
        "error": "resolve_model_dir accepted a repo id that is baked into no image. "
                 "It must refuse: grading has no network, so a path nothing checked "
                 "is a directory whose weights do not exist, and the failure lands "
                 "far from here inside convert_checkpoint.",
        "probe": rejection,
        "hint": "raise when {data_root}/{repo_id}/config.json is absent",
    }
    # ...and it must have refused the LOOKUP, not crashed. A bare "it raised
    # something" is satisfied by a loader that is simply broken, which is what the
    # probe's type report exists to separate. The exclusion is a denylist of the
    # programming-error family rather than an allow-list of one type, because
    # ``FileNotFoundError`` is the reference's choice and a port may reasonably raise
    # ``ValueError`` or ``OSError`` for a missing checkpoint.
    assert rejection["raised"] not in _PROGRAMMING_ERRORS, {
        "error": "resolve_model_dir did not refuse the unknown repo id, it crashed on "
                 "it -- the exception is the shape of a bug in the loader rather than "
                 "of a lookup that came up empty",
        "probe": rejection,
    }

    assert facts["stacked_params"] == loader["stacked_params"], {
        "error": "your STACKED_PARAMS is not the contracted fusion table; the third "
                 "field is the shard's index within the fused tensor, so its order IS "
                 "the layout of qkv_proj / gate_up_proj",
        "contract": loader["stacked_params"], "candidate": facts["stacked_params"],
    }

    assert facts["has_tokenizer"], {
        "error": "load_tokenizer returned nothing for a checkpoint that ships "
                 "tokenizer.json",
        "model_dir": facts["model_dir"],
    }

    # The no-network half, READ IN THE SANDBOX CHILD -- the process that would do
    # the downloading -- and not in this one.
    #
    # Reading `os.environ.get(name)` HERE, IN THE GRADER'S OWN PROCESS, would be
    # unfalsifiable given the pin. `conftest` ASSIGNS both variables (assigning,
    # not `setdefault`, so an inherited `HF_HUB_OFFLINE=0` cannot win), so nothing
    # between that assignment and this line can move the value, and the candidate
    # runs in a `spawn` child that cannot write back to this process's environment.
    # Hermeticity is mandatory, so the pin is right; but a graded clause that
    # cannot fail is worse than no clause: it reads as coverage of the no-network
    # requirement while asserting only that conftest executed.
    #
    # So it is asserted where it still has content. `sandbox_report['offline_env']`
    # is read inside the worker AFTER `run_all` has produced every probe, so it
    # covers two things this file cannot otherwise see: that the `spawn` carried
    # the environment across at all, and that the CANDIDATE's own code did not
    # clear it -- which is exactly what a port would do to reach the hub, and the
    # failure this clause exists for. Both are real, reachable states; neither
    # is reachable from the grader's side.
    offline = dict(cand.sandbox_report.get("offline_env") or {})
    assert offline and all(value == "1" for value in offline.values()), {
        "error": "the candidate's sandbox did not run offline. Grading has no "
                 "network: a port that reaches the hub is graded on weights "
                 "nobody pinned, and a `1` here is the only thing standing "
                 "between the two.",
        "worker_environment": offline,
        "grader_environment": {name: os.environ.get(name)
                               for name in ("HF_HUB_OFFLINE",
                                            "TRANSFORMERS_OFFLINE")},
        "hint": "both are ASSIGNED by tests/conftest.py before anything imports "
                "and inherited across the spawn; an empty dict means the worker "
                "reported no offline_env at all, and a non-'1' value means "
                "something inside the sandbox changed it",
    }
