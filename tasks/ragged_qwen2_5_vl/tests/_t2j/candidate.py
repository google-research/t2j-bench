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

"""Run inference probes against a candidate JAX repository.

Candidate code executes in one freshly spawned, torch-blocked worker; the
pytest parent receives NumPy results. Reusing the worker avoids repeating
checkpoint conversion for each probe. Results are cached by the parent.

USE_FROZEN_ALLOCATION selects allocation policy per probe: cache comparisons
use frozen page/slot assignments, while allocator checks derive their own.
REQUIRED_PROBES must all return non-None values. Unsupported frozen inputs
produce an explicit supported=False result. Nonessential probe failures are
stored and raised when consumed; initialization and COMMON_MODE_PROBES failures
abort the worker. Failures outside probe execution can also abort the run.

JAX x64 is disabled for this class to preserve the integer/float precision of
its frozen inference contract. No training finite-difference probe runs here.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import queue
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import purity, sandbox
from .perf import nvml

# Marker env var read by the `sitecustomize.py` `_worker_entry` writes into a
# private temp directory of its own (and exports on PYTHONPATH): this is what makes a
# `subprocess.run([sys.executable, ...])` -- or any other nested spawn that
# inherits this process's environment -- install the SAME blocker before its
# own first import. The env var alone never did that: setting it here is not a
# self-fulfilling propagation, it is one half of a mechanism whose other half
# is the sitecustomize file (see `_worker_entry`, and
# `test_the_torch_blocker_survives_into_a_subprocess`).
SANDBOX_ENV_FLAG = "T2J_TORCH_SANDBOX"


class CandidateError(RuntimeError):
    """Raised in the parent when the sandboxed candidate failed to produce artifacts."""


class CandidateTimeout(CandidateError):
    """The sandbox budget ran out. A DIFFERENT KIND OF EVENT from a wrong port.

    Every other `CandidateError` is a statement about the candidate: it returned
    the wrong shape, it imported a banned module, it contradicted itself. This
    one is a statement about the WALL CLOCK, and on a loaded host it fires on a
    correct port.

    Under host contention, the unmodified reference can exceed the budget and
    score the same reward as the `torch_only` control -- a correct port becomes
    indistinguishable from one that cheats by importing torch. Raising the budget
    restores a full score once the load falls.

    A subclass rather than a new exception, so every existing
    `pytest.raises(CandidateError)` still catches it; `conftest` keys on the
    subclass to end the session with exit 4 instead of scoring it.
    """


# --------------------------------------------------------------------------- #
# The probe surface. Both tables are read by tests, so a probe cannot be added,
# renamed or re-policied without the change being visible.
# --------------------------------------------------------------------------- #
#: Keys the worker MUST return. Absent or ``None`` -> ``CandidateError``.
REQUIRED_PROBES: tuple[str, ...] = (
    "params_spec",
    "param_norms",
    "cache_spec",
    "loader_facts",
    "resolved_config",
    "compilation",
    "metadata_schema",
    "block_table",
    "slot_mapping",
    "static_shapes",
    "traceability",
    "prefill",
    "chunked_prefill",
    "fp32_prefill",
    "kv_cache_state",
    "decode_steps",
    "paged_attention_unit",
    "fp32_paged_attention_unit",
    "sampler_outputs",
    "generation",
    "fp32_generation",
    "invariance",
    "battery",
    "reference_units",
    # MULTIMODAL ONLY, AND STILL REQUIRED. Both answer `supported: False` on a
    # text-only family -- the port there ships no `image_processing` module and
    # nothing hands the worker an image -- which is a VALUE, so the required-probe
    # contract holds unchanged. The two checks that read them are deselected on
    # those instances by `conftest._APPLICABLE`'s `multimodal` gate, so the marker
    # is never graded; it exists so that a MULTIMODAL instance whose port is
    # missing the module fails with that sentence instead of an ImportError
    # traceback from a probe nobody expected to run.
    "image_preprocess",
    "smart_resize",
    # ...AND THE THREE THAT REACH THE MODEL. Same contract as the pair above --
    # each answers `supported: False` on a text-only family, which is a VALUE, so
    # the required-probe rule holds unchanged and the checks that read them are
    # deselected there anyway. `multimodal_prefill` feeds TWO checks
    # (`test_mrope_positions` and `test_multimodal_prefill_logits`); see its
    # docstring for why the positions and the logits come out of one engine call
    # rather than two.
    "multimodal_prefill",
    "image_embedding_merge",
    "vision_tower_units",
    "serve",
    "held_out",
    "perf",
    "sandbox_report",
    "allocation_source",
)

#: Probes whose failure genuinely takes everything with it, so ``run_all``'s
#: per-probe capture (below) re-raises instead of isolating them. The checkpoint
#: conversion and the resolved configuration are INPUTS to every other probe --
#: `params_spec` / `cache_spec` build the module tree every numeric probe reads,
#: and `resolved_config` is the `RunConfig` every probe below constructs its
#: engine from. Isolating a failure in one of those would not buy partial credit;
#: it would produce 29 identical tracebacks instead of one, for no gain in signal.
#:
#: THE LOADER-FACTS PROBE USED TO BE THE FOURTH ENTRY AND IT DOES NOT BELONG
#: HERE, because the rationale above is the one claim in this file that was
#: measurably false of it. The test above ("INPUTS to every other probe") is a
#: statement about READERS, and that probe has none: it is the ONLY reader of
#: `loader_mod.DATA_ROOT` on a graded path -- the other `DATA_ROOT` site is
#: guarded by `if data_root:` and dead in a graded run, because `test.sh` unsets
#: `T2J_INFERENCE_DATA_ROOT` -- and no other probe consumes its RESULT. Nothing
#: is built from it; it only reports. So its failure cost the whole run for a
#: fact one check reads, and `instruction.md` promises a partial port partial
#: credit.
#:
#: THE FACTS AND THE ARITHMETIC, split on purpose:
#:   `loader_facts` has exactly ONE graded reader,
#:   `test_spec.test_loader_spec`, confirmed by an AST scan of all four graded
#:   modules (the harness keeps that scan as a test); the probe has no consumer
#:   among the other probes; and a raise inside it aborts the whole worker unless
#:   it is captured, which it now is.
#:   By arithmetic: with the worker dead, nearly every check reports the one
#:   `CandidateError` -> the floor the `torch_only` CHEATING control scores;
#:   isolated, the same port loses only the one check.
COMMON_MODE_PROBES: tuple[str, ...] = (
    "params_spec", "cache_spec", "resolved_config",
)

#: Per-probe allocation policy. ``True`` -- the probe is HANDED the
#: frozen block table / slot mapping / page ids from ``scenario.json`` +
#: ``inputs.npz``. ``False`` -- the probe lets the candidate's own allocator
#: derive them, because that allocator is what the probe grades.
#:
#: THIS TABLE IS THE DECLARATION, NOT THE MECHANISM. What each probe actually
#: honours is recorded AT ITS POINT OF USE -- every read of a frozen array goes
#: through :meth:`_CandidateRuntime._frozen` and every candidate-chosen page list
#: through :meth:`_CandidateRuntime._candidate_pages` -- and surfaces as
#: ``Candidate.allocation_source``. The two are compared by
#: ``test_the_allocation_policy_the_probes_EXECUTED_matches_the_declared_table``,
#: which is what makes this table a control rather than a comment: a code-only
#: change to which allocation a probe honours moves the observed map away from
#: this one and fails that test. (A ``battery`` whose variants disagree with each
#: other records ``"mixed"``, which is also a mismatch -- deliberately, because a
#: half-frozen battery axis is a freezer bug worth a loud failure.)
#:
#: The three ``False`` entries the frozen-allocation policy names are
#: ``block_table`` / ``slot_mapping``
#: (``test_block_table_contract``), ``invariance.paging_*``
#: (``test_paging_invariance``) and ``invariance.cache_reuse``
#: (``test_cache_reuse_safety``); ``invariance.decode_prefill`` joined them as
#: well, see its entry below. ``metadata_schema`` / ``static_shapes`` ride
#: along on the same engine run -- they grade the metadata the candidate BUILDS,
#: which is the same question, and handing them the frozen arrays would make
#: them assertions about the golden. ``generation`` and ``fp32_generation`` are
#: the fourth and fifth deliberate ``False``: what they compare (token ids and
#: their logprobs) is allocation-INVARIANT for any correct allocator, and running
#: them through the candidate's own ``engine.generate`` is the whole content of
#: ``test_generate_runs`` and of ``test_greedy_sequence_fp32``; re-implementing
#: the decode loop here with frozen slots would grade this file's loop instead of
#: the candidate's. ``perf`` is the sixth, for the identical reason: it is
#: :meth:`_CandidateRuntime.generation_probe`'s pattern run at several step
#: counts, and what it measures (wall-clock, not a numeric artifact) has no
#: allocation-dependent seam to hand a frozen table into.
USE_FROZEN_ALLOCATION: dict[str, bool] = {
    "prefill": True,
    "fp32_prefill": True,
    "kv_cache_state": True,
    "decode_steps": True,
    "paged_attention_unit": True,
    "fp32_paged_attention_unit": True,
    "sampler_outputs": True,
    "traceability": True,
    "battery": True,
    "invariance.determinism": True,
    "metadata_schema": False,
    "static_shapes": False,
    "block_table": False,
    "slot_mapping": False,
    "generation": False,
    "fp32_generation": False,
    "perf": False,
    # `chunked_prefill` drives the candidate's own `Engine.prefill`, N times on
    # one instance, exactly as `generation` drives `Engine.generate` -- the
    # `BlockPool` inside that Engine is what chooses every chunk's pages, so
    # handing this probe a frozen block table would mean reimplementing
    # per-chunk allocation here (see `chunked_prefill_probe`'s own docstring).
    # `False` on every instance today: `core`/`ragged`/`core_gpt_oss` freeze an
    # empty `chunk_split_points`, so the probe never reaches the `Engine` call
    # that would record an allocation source at all (see the "unsupported,
    # without spending a forward" branch), and this entry is exercised only
    # once a `chunked` instance ships a non-empty one.
    "chunked_prefill": False,
    # The two `serve.py` probes, and they are ``False`` for a stronger reason than
    # the generation pair above: nothing frozen is even REACHABLE from them.
    # ``run_serve`` takes a config dict and allocates through its own ``Engine``,
    # so there is no seam at which a frozen block table could be handed in without
    # reimplementing ``serve.py`` here -- which would grade this file instead of
    # the candidate's entry point. Both record ``"candidate"`` at their point of
    # use, and this table is compared to what they RECORDED by
    # ``test_the_allocation_policy_the_probes_EXECUTED_matches_the_declared_table``
    # -- a probe that records without declaring fails it as loudly as the reverse.
    "serve": False,
    "held_out": False,
    # `multimodal_prefill` drives the candidate's own `Engine.prefill`, so the
    # `BlockPool` inside that Engine chooses the pages -- the same shape as
    # `generation` above, and for the same reason: what it compares (a logit row)
    # is allocation-INVARIANT for any correct allocator: on this class's own
    # reference the handed-allocation prefill and `Engine.generate`
    # agree bit for bit on that row. The other two multimodal model probes are
    # absent from this table on purpose: `image_embedding_merge` and
    # `vision_tower_units` never build an Engine and never touch a KV cache, so
    # they have no allocation to honour and record none.
    "multimodal_prefill": False,
    # `compilation` drives the candidate's own ``Engine.prefill`` twice, on a fresh
    # pool. What it counts is XLA compilations, which no frozen array could supply,
    # and handing it the frozen allocation would only add a second thing that could
    # differ between its two legs.
    "compilation": False,
    "invariance.paging_default": False,
    "invariance.paging_permuted": False,
    "invariance.cache_reuse": False,
    # ``False`` deliberately. While this probe was HANDED the positions, the
    # seq_lens, the block table and the slot, no candidate-side defect in any of
    # them could reach its comparison -- three such defects passed it. It now runs
    # both legs through the candidate's own `Engine`, so the allocation and the
    # metadata are the candidate's and the check grades the decode PATH rather
    # than the model's attention op.
    "invariance.decode_prefill": False,
    # Both drive the candidate's own `Engine` -- `invariance.chunking` reuses
    # `chunked_prefill_probe` for its chunked leg and builds a fresh `Engine`
    # for its single-shot leg; `invariance.prefix_cache` builds two `Engine`s
    # of its own to produce a cache hit and a cold miss. Neither has a frozen
    # array to hand in without reimplementing per-chunk allocation or the
    # prefix-cache's own acquire/release bookkeeping here, the same argument
    # `chunked_prefill`'s own entry above makes.
    "invariance.chunking": False,
    "invariance.prefix_cache": False,
}

#: Every self-consistency probe name ``Candidate.invariance`` will answer to.
#: Anything else raises ``RuntimeError("unknown probe: ...")`` -- a typo in a
#: test module must not read as "this candidate does not support that", which is
#: how a behavioral check silently stops grading anything.
INVARIANCE_KINDS: frozenset[str] = frozenset({
    # Every mode.
    "paging_default",
    "paging_permuted",
    "decode_prefill",
    "cache_reuse",
    "determinism",
    # `ragged` / `chunked` only. Registered here (so the name is KNOWN) and
    # answered with an unsupported marker on `core`, whose engine runs one
    # sequence at a time. Their tests are deselected on `core`; when the mode
    # gate opens in I2 an unimplemented probe FAILS the check rather than
    # quietly passing.
    "batching",
    "cross_sequence_leakage",
    "chunking",
    "prefix_cache",
})

#: Invariance kinds that need a second concurrent sequence, hence a mode that
#: has one.
_MULTI_SEQUENCE_KINDS: frozenset[str] = frozenset({
    "batching", "cross_sequence_leakage", "chunking", "prefix_cache",
})


def second_prompt_length(prompt_len: Any, page_size: Any) -> int | None:
    """The SHORTER prompt length the compilation probe prefills at, or ``None``.

    ONE SOURCE, called by both sides. The probe uses it to decide whether its
    new-shape leg can run at all, and ``conftest._golden_facts`` uses it to decide
    whether to collect the check that grades that leg. Those were two separate
    expressions for one week and they did not agree in either direction: the gate
    ``prompt_len > max(1, page_size // 2)`` deselected a 4-token prompt at
    ``page_size`` 8 that the probe supports (it prefills at 1), and it collected a
    ``page_size=1`` geometry that the probe does not support, which then hard-failed
    with a "harness failure, not a candidate one" message -- the exact error the
    deselection existed to remove. A predicate that must be kept in agreement with
    the thing it gates is a predicate that will stop agreeing.

    Pure arithmetic over two integers: no jax, no candidate, no golden beyond the two
    numbers, so ``conftest`` can call it during collection.

    ``None`` means this geometry has no second length. Two ways to get there:
    ``page_size < 2`` (the step is zero, so the "shorter" prompt is the same length),
    and ``prompt_len < 2`` (there is nothing shorter than a single token).
    """
    n, step = int(prompt_len), int(page_size) // 2
    shorter = max(1, n - step)
    return shorter if shorter < n else None


def interior_boundary_position(prompt_len: Any, page_size: Any) -> int | None:
    """The last page boundary strictly INSIDE the prompt, or ``None``.

    THE FALLBACK FOR A PROMPT THAT ALREADY FILLS ITS PAGE BUDGET.
    ``_invariance_decode_prefill``'s far leg exists to make a decode step cross a
    page boundary, because the near leg (position ``n-1``) never leaves the pages
    the prefill already wrote and no page-boundary defect can reach it. It looks
    for the boundary AFTER the prompt, which needs continuation tokens and
    ``max_model_len`` headroom the frozen scenario may not have.

    On core_gpt_oss, prompt 135 at ``page_size`` 32, so the next boundary
    is 160 -- 26 continuation tokens away against 8 frozen, and one past
    ``max_model_len``. The far leg is therefore absent, and with it the ONLY place
    in the graded run where ``Engine._grow`` has to extend an existing allocation:
    prefill takes 5 pages and the deepest position the suite reaches is 143, still
    5 pages. A port whose ``_grow`` allocates on the first call and then never
    grows again is bit-identical to the reference on that instance.

    A boundary inside the prompt reaches the same property from the other side:
    prefill ``ids[:b]``, decode ``ids[b]``, and the decode step is the call that
    must allocate the next page. 128 on core_gpt_oss, 8 on the three
    ``page_size``-8 instances. ``None`` when the prompt spans less than two pages,
    which is the only geometry with no interior boundary to use.

    Pure arithmetic over two integers, for the reason
    :func:`second_prompt_length` is: the probe decides whether the leg can run and
    ``test_behavioral`` decides whether to require it, and two expressions for one
    predicate stop agreeing.
    """
    n, size = int(prompt_len), int(page_size)
    if size < 1 or n < 2:
        return None
    boundary = size * ((n - 1) // size)
    return boundary if 1 <= boundary <= n - 1 else None


#: How many frozen decode ids the steady-state compilation leg needs. Step 1 is the
#: decode-shape WARM-UP and is excluded from the count by design (the decode shape is
#: new relative to the prefill shape, so one compilation there is correct), so a
#: steady-state region needs at least one more id after it.
#:
#: ONE NUMBER, TWO READERS, and the second one is in another file:
#: ``_harness_checks.check_scenario_carries_a_gradable_decode_run`` imports THIS
#: constant to refuse a golden the leg could not grade. Same discipline as
#: :func:`second_prompt_length` -- a gate and the thing it gates written as two
#: expressions stop agreeing.
MIN_STEADY_STATE_DECODE_IDS = 2

#: R = 3, median of the three slopes. A global instrument
#: parameter, not per-instance geometry, so it lives here rather than in
#: `instance.yaml` -- the same status `generate_timeout_s` has, with the same
#: override seam (`meta.get("perf_rounds")`, see `fixtures.build_candidate`)
#: for an instance that needs a different value without every instance having
#: to declare one.
DEFAULT_PERF_ROUNDS = 3

#: How many throttled discards one `perf_probe` call will absorb before it stops
#: trying and returns fewer than `perf_rounds` clean rounds. A throttled round
#: is discarded and retried up to a round budget, and the budget number is not
#: fixed elsewhere. THIS IS A CHOSEN BUDGET, NOT A DERIVED ONE: no throttle event
#: has been available to calibrate against, so the
#: multiplier is picked generously (enough slack to absorb one discard per
#: target round) rather than derived. `perf_probe` always returns every
#: attempt, discarded or not, so a budget that turns out too small fails loudly
#: (quorum unmet, raw pairs still in the result) rather than silently.
PERF_ROUND_ATTEMPT_MULTIPLIER = 2


def steady_state_decode_tokens(scenario: Any) -> tuple[list[int], str | None]:
    """``(tokens, reason_it_cannot_run)`` for the decode steady-state leg.

    Pure: a dict in, a list and an optional string out. No jax, no engine, no
    candidate -- which is what makes the unsupported branch reachable from a unit
    test instead of only from a golden nobody can freeze. The harness exercises
    this function in BOTH directions.

    BELT AND BRACES, NOT A LIVE GRADING PATH. A non-``None`` reason is UNREACHABLE
    at grade time: ``check_scenario_carries_a_gradable_decode_run`` runs in
    ``_grade_precheck`` and aborts the whole grade (exit 2, no ``reward.json``)
    before pytest starts if the golden carries fewer than
    :data:`MIN_STEADY_STATE_DECODE_IDS` ids. It is kept because the precheck can be
    bypassed -- a bare ``pytest tasks/*/tests``, or
    the mutation harness if its precheck leg were ever dropped --
    and because the two failures it separates are different defects:

      * fewer than 1 id -> ``decode_tokens[0]`` is an ``IndexError``, which
        ``run_all``'s ``produce`` turns into a probe error, CHARGING THE CANDIDATE
        FOR A GOLDEN GAP;
      * exactly 1 id    -> the counted block runs zero iterations, ``steady.n`` is 0
        and ``decode_steady_state == 0`` passes VACUOUSLY.

    ``test_compilation_is_cached`` FAILS (it does not skip) on the reason, naming it
    a harness fault -- see that check for why a skip was the wrong instrument.
    """
    tokens = [int(t) for t in (scenario.get("decode_input_token_ids") or [])]
    if len(tokens) >= MIN_STEADY_STATE_DECODE_IDS:
        return tokens, None
    return tokens, (
        f"scenario.json carries fewer than {MIN_STEADY_STATE_DECODE_IDS} "
        f"`decode_input_token_ids` ({len(tokens)}); step 1 is the decode-shape "
        "warm-up and is excluded by design, so a steady-state count needs at least "
        "one more. This is an artifact gap, not a port defect -- and one "
        "`_grade_precheck` should already have refused to grade at all")


def resolve_module_path(root: Any, path: str) -> Any:
    """Walk a dotted attribute path, treating an all-digit component as an INDEX.

    ``model.layers.0.self_attn.qkv_proj`` is ``root.model.layers[0].self_attn.
    qkv_proj`` on BOTH sides: torch's ``nn.ModuleList`` and flax's
    ``nnx.data([...])`` are both indexed rather than attributed, and the two
    reference trees use the same attribute names throughout (``ref_jax_repo/
    modeling.py``: "same classes, same attribute names, same order of
    operations"). That is what lets ``reference_units`` grade a candidate
    submodule against the torch submodule it was captured from.

    ``None`` when any component is absent -- the caller reports WHICH path failed,
    which is a legible verdict ("your model has no
    ``model.layers.0.self_attn.qkv_proj``"), where a raised ``AttributeError``
    three frames into the worker would take the whole grade down with no context.

    Deliberately the same rule the freezer uses to resolve a submodule path,
    restated rather than imported: the grader is hermetic and the freezer does
    not ship with it.
    """
    node = root
    for part in str(path).split("."):
        if part.isdigit():
            try:
                node = node[int(part)]
            except (TypeError, KeyError, IndexError):
                return None
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def assert_frozen_allocation_agrees(block_tables: Any, page_ids: Any) -> None:
    """The frozen allocation is pinned in TWO files; refuse to run if they differ.

    ``scenario.json``'s ``page_ids`` drives the decode slot arithmetic
    (:meth:`_CandidateRuntime._decode_slot`) while ``inputs.npz``'s
    ``block_tables`` drives the attention gather. That is one fact written down
    twice, and until now nothing compared them. A freezer that updates one and
    not the other produces a decode that READS pages the prefill never WROTE --
    which arrives as a numeric failure attributed to the candidate, in a probe
    whose inputs both came from the golden. The freezer writes both keys, so this
    guard earns itself the moment the real freeze lands.

    Only the leading ``len(page_ids)`` entries are compared: the block table row
    is padded out to ``pages_per_seq`` with whatever the freezer used for unused
    slots, and that padding is not part of the allocation. A 2-D table is read at
    row 0 -- ``core`` is one sequence, and a mode with more of them pins the
    others in its own scenario.
    """
    if block_tables is None or not len(page_ids):
        return
    table = np.asarray(block_tables)
    if table.size == 0:
        return
    row = (table.reshape(-1) if table.ndim <= 1 else np.asarray(table[0]).reshape(-1))
    want = np.asarray(page_ids, dtype=np.int64)
    got = row[: want.size].astype(np.int64)
    if got.size != want.size or not np.array_equal(got, want):
        raise ValueError(
            "the frozen scenario contradicts itself: scenario.json pins "
            f"page_ids={want.tolist()} but inputs.npz's block_tables starts "
            f"{got.tolist()}. The decode slot arithmetic reads the first and the "
            "attention gather reads the second, so a candidate graded against "
            "this golden would be blamed for the freezer's desync."
        )


def _is_probe_error(value: Any) -> bool:
    """True for the marker ``run_all``'s per-probe capture leaves behind.

    A single, narrow shape (``dict`` with exactly this one key) so a candidate's
    own legitimate return value -- an ``{"unsupported": ...}`` marker, say -- is
    never mistaken for a captured exception. Shared by ``run_all`` (to decide
    whether a dependent result must inherit the failure) and by
    ``Candidate._probe`` / ``_validate_results`` (to decide whether to raise on
    read versus grade normally).
    """
    return isinstance(value, dict) and "__probe_error__" in value


def _to_np(value: Any) -> np.ndarray:
    """jax/np -> np.ndarray (bf16 -> fp32), torch-free."""
    if value is None:
        raise ValueError("cannot convert None to numpy")
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32)
    return arr


def _to_np_with_provenance(value: Any) -> tuple[np.ndarray, dict[str, str]]:
    """``_to_np``, plus what the value WAS before numpy flattened it.

    `np.asarray` is lossy in two ways that the grade depends on and cannot see
    afterwards. bf16 has no numpy dtype, so a leg that silently ran at float32 arrives
    indistinguishable from one that ran at bfloat16 -- dropping one
    `.astype` in RMSNorm runs the WHOLE native forward at fp32 and still passes
    every check. And
    `jax.Array` and `np.ndarray` coerce identically, so a port that returns numpy
    across its API boundary -- which instruction.md forbids in as many words -- is
    invisible.

    `test_paged_attention_op` already had a bespoke version of the dtype half (the
    `out_dtype`/`requested_dtype` pair on `paged_attention_unit_probe`). This is that
    clause generalised, so a new probe gets both halves by construction instead of by
    someone remembering.

    THE DTYPE READS THROUGH ``np.dtype(...).name``, not ``str(...)``, for the same
    reason ``paged_attention_unit_probe`` reads ``requested_dtype`` through
    ``jnp.dtype(dtype).name`` rather than ``str(dtype)``: this module is deliberately
    jax-free (the parent side of this file may not import it), so it cannot call
    ``jnp.dtype`` directly, but ``np.dtype`` normalises the same way for any value
    that already carries a real dtype instance -- which every array this is called on
    does. The two spellings render identically on a concrete array (``np.dtype(x.dtype
    ).name == str(x.dtype)``); this one is picked to match the runtime's own
    convention rather than add a second one beside it.

    ...AND IT FALLS BACK RATHER THAN RAISING, which is a repair. ``np.dtype`` does
    NOT accept every dtype object in the world -- a ``torch.dtype`` raises
    ``TypeError: Cannot interpret 'torch.float32' as a data type`` -- while
    ``_to_np`` directly above copes with a torch tensor fine through
    ``np.asarray``. So the two helpers disagreed about what they accept, and the
    stricter one was the one with the docstring saying it generalises the other.
    UNREACHABLE ON THE GRADED PATH (the sandbox blocks torch, so every probe is
    handed a ``jax.Array``), which is why it stayed latent; it is reachable the
    moment anyone points a probe body at the torch reference tree from an
    authoring script, and there it turned a diagnostic field into a crash. The
    fallback renders the object's own ``str()``, which for a torch tensor is
    ``'torch.float32'`` -- deliberately NOT normalised by stripping a prefix,
    because this module must not carry knowledge of a library the verifier is
    built to exclude, and a value that says where it came from is the more useful
    diagnostic anyway.
    """
    raw_dtype = getattr(value, "dtype", None)
    if raw_dtype is None:
        rendered = ""
    else:
        try:
            rendered = np.dtype(raw_dtype).name
        except TypeError:
            rendered = str(raw_dtype)
    return _to_np(value), {
        "dtype": rendered,
        "type": type(value).__module__ + "." + type(value).__name__,
    }


#: Depth past which :func:`_json_safe` stops recursing and renders the rest as a
#: string. A self-referential container (``d = {}; d["d"] = d``) is the case that
#: matters: without a bound the walk below is a ``RecursionError``, and the whole
#: point of this helper is that a candidate's return shape cannot cost more than
#: the checks that read it.
_JSON_SAFE_MAX_DEPTH = 12

#: Element count past which an array is summarised rather than converted.
#: ``tolist()`` on a large array costs several times its own memory INSIDE the
#: worker, which still holds the model and the GPU -- a new way for one probe to
#: kill the run, which is the exact defect this helper exists to remove. Nothing
#: graded comes near it: ``generations[0]['tokens']`` and ``step_logprobs`` are
#: ``num_decode_steps`` long (single digits to low tens on every instance).
_JSON_SAFE_MAX_ELEMS = 1_000_000


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Anything -> ``dict`` / ``list`` / ``str`` / ``int`` / ``float`` / ``bool`` / ``None``.

    THE RECURSIVE FORM OF ``resolved_config_probe``'s ``_plain``, and it exists
    for the same stated reason -- "so the map survives the pickle back across
    the sandbox boundary whatever a candidate annotated them as" -- applied to
    the one probe payload that is a candidate OBJECT GRAPH rather than a map of
    scalars.

    WHAT IT DEFENDS. ``run_all``'s per-probe capture isolates an
    exception raised INSIDE a probe. It does not isolate the pickle: all 34
    probe results cross the pipe in ONE ``conn.send`` and one
    ``parent_conn.recv()``, so a value that fails to (un)pickle aborts the whole
    grade -- a common-mode surface, and one no ``try`` in a probe body can
    reach. A class defined in the candidate's own ``serve.py`` unpickles in the
    pytest parent as ``ModuleNotFoundError: No module named 'serve'``, because
    the parent never has the candidate repo on ``sys.path``: the score is the
    floor a repo that does not exist earns, and a message naming the
    grader instead of ``serve.py``'s return shape.

    IT ALSO KEEPS jax OUT OF THE PARENT. A ``jax.Array`` in that payload pickles
    with a ``jax._src.array`` reconstructor, so the parent's ``recv()`` imports
    jax -- which this module's docstring forbids in as many words -- and does so
    before ``_hard_kill`` has run, while the worker still holds the device.
    Every array here becomes a plain list of Python scalars, so no jax (and no
    ``ml_dtypes``) symbol survives into the pickle at all.

    NOT A REPAIR OF THE CANDIDATE'S SHAPE. An object this cannot structurally
    represent renders as its ``str()`` -- the same fallback ``_plain`` uses --
    which fails the check that reads it, by name, with the object printed. It is
    deliberately not ``dataclasses.asdict``/``vars()``: guessing at a return
    shape ``interface.json`` does not pin would make the grader responsible for
    a contract it never stated, and would let a genuinely wrong payload through.

    EXACT TYPES ON THE LEAF CLAUSE, ``type(...) in`` AND NOT ``isinstance``, and
    that one word was this function reproducing the collapse it exists to
    prevent. ``isinstance`` is true of SUBCLASSES, so a scalar type defined in
    the candidate's own ``serve.py`` -- ``class Dtype(str, Enum)`` is an
    ordinary way to spell that field, and ``json.dumps`` accepts it, so the port
    writes a perfect ``serve_results.json`` -- was returned untouched, put
    ``serve`` back into the pickle and scored the non-existent-port
    floor. Without the unwrap,
    ``b"serve" in pickle.dumps(_json_safe({"dtype": Dtype.BF16}))`` -> ``True``,
    and the unpickle in a child without the repo -> ``ModuleNotFoundError``.
    The container clauses were never exposed: a ``dict``/``list``/``tuple``
    subclass is REBUILT as a builtin, so only the leaf clause leaked.

    AND THE SUBCLASSES ARE UNWRAPPED TO THEIR BUILTIN VALUE, not rendered with
    ``str()``, because rendering them is a second wrong answer wearing a smaller
    number. ``str(Dtype.BF16)`` is ``'Dtype.BF16'`` (``Enum`` supplies
    ``__str__``, the mixin does not), so the fallback would echo that into
    ``resolved_dtype`` and fail ``test_serve_end_to_end``'s dtype clause for a
    port whose ``serve_results.json`` correctly reads ``"bfloat16"``. The unwrap
    below reproduces exactly what ``json.dumps`` does with the same values
    (namely ``{"d": "bfloat16", "s": 3, "ms": "abc", "mi": 7, "mf": 2.5}``),
    which is the only defensible reading -- ``instruction.md`` asks ``serve.py``
    for a JSON file, so JSON's own treatment of a scalar subclass is the
    contract the candidate was held to. ``bool`` needs no clause: CPython
    refuses to subclass it.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if _depth >= _JSON_SAFE_MAX_DEPTH:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in value]
    # THE SCALAR SUBCLASSES -- see the docstring. `str.__str__` and not `str()`:
    # the latter dispatches to the subclass's own `__str__`, which is the whole
    # problem for an `Enum` mixin; the former is the builtin, so it hands back
    # the underlying characters as a plain `str`. Ahead of the array clause
    # deliberately, and it changes two answers for the better: `np.float64` and
    # `np.str_` ARE subclasses of `float`/`str` (`np.int64` and
    # `np.bool_` are not), so before this they were the two numpy scalars the
    # leaf clause returned unconverted.
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        # numpy / jax arrays, INCLUDING the 0-d numpy scalars `np.int64` and
        # `np.float32` (neither of which is an `int`/`float` in numpy 2, so they
        # fall through the first clause). `_to_np` rather than `np.asarray` for
        # its bf16 -> fp32 leg: `tolist()` on an `ml_dtypes.bfloat16` array
        # yields `ml_dtypes` scalars, which pickle by importing that package
        # into the parent.
        #
        # BOTH ATTRIBUTES, and `shape` alone is not enough. In numpy
        # 2.2.6, a candidate object carrying a `shape` plus
        # `__len__`/`__getitem__` that return itself makes `np.asarray` LOOP
        # FOREVER inside numpy, where no recursion bound in this function can
        # reach it -- a hang, which the parent only ends at
        # `generate_timeout_s` and reports as a `CandidateTimeout` for the whole
        # grade.
        #
        # A NARROWING, NOT A GUARANTEE, and the difference is worth the line: an
        # object carrying BOTH `shape` and `dtype` and the same self-returning
        # `__getitem__` still reaches `np.asarray` and still hangs. What this
        # buys is that the accidental duck-types -- anything with a `shape` and
        # nothing else -- render as `str()` below instead, costing the check
        # that reads them and nothing more. Bounding the deliberate case needs a
        # watchdog around the call, which this function does not have.
        try:
            arr = _to_np(value)
        except Exception:  # noqa: BLE001 - a value we cannot read is a string
            return str(value)
        if arr.size > _JSON_SAFE_MAX_ELEMS:
            return (f"<{arr.dtype} array of shape {tuple(arr.shape)}, "
                    f"{arr.size} elements: too large to carry across the "
                    f"sandbox boundary>")
        if arr.dtype == object:
            # `tolist()` hands back the very objects the coercion exists to
            # remove, so walk them.
            return _json_safe(arr.tolist(), _depth + 1)
        return arr.tolist()
    return str(value)


#: The ``RunConfig`` fields ``serve.py``'s config dict forwards. Named here rather
#: than read off the CANDIDATE's ``RunConfig`` -- the way ``_run_config_kwargs``
#: below deliberately does -- because the two questions are different. There, the
#: question is "what will this candidate's constructor accept", and a candidate
#: with one extra field is explicitly legal (``test_spec.test_interface``:
#: "Superset policy: extras are fine"). Here, the question is "what serving
#: geometry is this instance graded at", and the answer belongs to the GOLDEN. A
#: candidate that dropped a field must fail ``test_config_spec`` by name; it must
#: not silently change what ``serve.py`` is handed and thereby move both legs of
#: ``test_serve_end_to_end`` together.
_RUN_FIELD_KEYS: tuple[str, ...] = (
    "page_size", "num_pages", "max_num_seqs", "pages_per_seq", "max_model_len",
    "dtype", "seed", "num_decode_steps",
)

#: The held-out arrays that may cross into the sandbox: the four INPUT ones.
#: ``held_out.npz``'s fifth array, ``ref_logprobs``, is the frozen ANSWER
#: ``test_held_out_logprobs`` compares against and it stays in the grader's
#: process -- selecting by this tuple in :meth:`Candidate.__init__` is what makes
#: that true rather than merely intended. Same discipline as
#: ``fixtures._SCENARIO_INPUT_KEYS``, at the one other boundary a frozen output
#: could cross.
HELD_OUT_INPUT_KEYS: tuple[str, ...] = (
    "prompt_ids", "prompt_lens", "cont_ids", "cont_lens",
)


def _serve_run_fields(repo_config: dict[str, Any]) -> dict[str, Any]:
    """The :data:`_RUN_FIELD_KEYS` subset of the GRADER's ``repo_config``.

    A missing key raises HERE, naming it and naming whose defect it is. The bare
    ``repo_config[k]`` this replaces raised ``KeyError`` from inside the probe,
    which the parent reports as "the candidate's 'serve' probe raised" -- a
    golden-authoring gap charged to the port, which is the misattribution
    ``_run_config_kwargs`` (below) already exists to prevent on the other side of
    the same construction.
    """
    missing = [name for name in _RUN_FIELD_KEYS if name not in repo_config]
    if missing:
        raise RuntimeError(
            f"the grader's own repo_config carries no {missing} (it supplies "
            f"{sorted(repo_config)}), so serve.py cannot be driven at this "
            "instance's serving geometry. This is an instance.yaml / meta.json "
            "gap, not a port defect.")
    return {name: repo_config[name] for name in _RUN_FIELD_KEYS}


def _run_config_kwargs(run_config_cls: Any, supplied: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``supplied`` that ``run_config_cls(**...)`` accepts.

    TWO BUGS IN ONE LINE, both of which cost a correct port the entire grade. The old
    form was ``{f: supplied[f] for f in cls.__dataclass_fields__}``:

    * it iterated the CANDIDATE's field list and indexed the harness's fixed 9-key
      ``repo_config``, so ONE extra field -- which ``test_spec.test_interface``
      explicitly permits ("Superset policy: extras are fine") -- was a bare ``KeyError``
      inside ``__init__``, killing the worker before any check could attribute it;
    * ``__dataclass_fields__`` includes ``ClassVar`` pseudo-fields, unlike
      ``dataclasses.fields()``, so a module constant declared inside the dataclass was
      the same total loss.

    A field the CONTRACT declares and the grader did not supply is still an error, and
    it is raised here with a name rather than as a ``TypeError`` from the constructor.
    """
    import dataclasses

    names = [f.name for f in dataclasses.fields(run_config_cls) if f.init]
    kwargs = {name: supplied[name] for name in names if name in supplied}
    required = [f.name for f in dataclasses.fields(run_config_cls)
                if f.init and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING]
    unsupplied = [name for name in required if name not in kwargs]
    if unsupplied:
        raise RuntimeError(
            f"your RunConfig requires {unsupplied}, which the grader's repo_config "
            f"does not carry (it supplies {sorted(supplied)}). A required field "
            "outside the contract cannot be configured.")
    return kwargs


# =========================================================================== #
# In-sandbox runtime.
# =========================================================================== #
def _rows_are(positions, want: list) -> bool:
    """Every rope row of ``positions`` equals ``want``. Rank 1 or rank 2.

    ``AttentionMetadata.input_positions`` is ``(T,)`` for a text-only family and
    ``(R, T)`` for an M-RoPE one, and the engine passes it through at whatever
    rank it arrives. A caller that wants "the query is exactly this range" has to
    ask it of every row rather than of a flattened copy.
    """
    array = np.asarray(positions)
    rows = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
    return bool(rows.shape[-1] == len(want)
                and all([int(v) for v in row] == want for row in rows))


class _CandidateRuntime:
    """Built inside the torch-blocked subprocess; computes every probe."""

    def __init__(self, payload: dict[str, Any]) -> None:
        import importlib
        import sys

        self.payload = payload
        self.repo_dir = str(Path(payload["repo_dir"]).resolve())
        self.mode = str(payload.get("mode", "core"))
        self.seed = int(payload.get("seed", 0))
        self.compare_hidden_states = bool(payload.get("compare_hidden_states", False))
        self.logprobs_top_n = int(payload.get("logprobs_top_n", 5))
        self.sampler_grid = list(payload.get("sampler_grid") or [])
        self.repo_config = dict(payload["repo_config"])
        self.scenario = dict(payload["scenario"])
        self.inputs = {k: np.asarray(v) for k, v in (self.scenario.get("inputs") or {}).items()}

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        self.config_mod = importlib.import_module("config")
        self.loader_mod = importlib.import_module("model.loader")
        self.modeling_mod = importlib.import_module("modeling")
        self.engine_mod = importlib.import_module("engine")
        self.metadata_mod = importlib.import_module("attention.metadata")
        self.paged_mod = importlib.import_module("attention.paged")
        self.kv_cache_mod = importlib.import_module("cache.kv_cache")
        self.block_table_mod = importlib.import_module("cache.block_table")
        self.sampler_mod = importlib.import_module("sample.sampler")

        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self.device = _F().resolve_jax_device(payload.get("device"))

        # AUTHORING-HOST ESCAPE HATCH, off in every graded run. `resolve_model_dir`
        # hardcodes /data and that hardcoding is itself graded (test_loader_spec),
        # so it is not touched; instead the module-level DATA_ROOT it reads is
        # rebound -- the same seam ref_jax_repo's own parity runner and unit test
        # use. Only fires when the GRADER's process sets T2J_INFERENCE_DATA_ROOT,
        # which nothing a candidate can do sets, and the effective root is reported
        # back in `loader_facts` so a spec test can assert it was /data.
        data_root = payload.get("data_root")
        if data_root:
            self.loader_mod.DATA_ROOT = Path(data_root)

        self.model_dir = self.loader_mod.resolve_model_dir(
            self.repo_config["model_name_or_path"])
        self.model_config = self.config_mod.ModelConfig.from_hf(self.model_dir)
        self.run_config = self.config_mod.RunConfig(
            **_run_config_kwargs(self.config_mod.RunConfig, self.repo_config))

        self.page_size = int(self.run_config.page_size)
        self.num_pages = int(self.run_config.num_pages)
        self.pages_per_seq = int(self.run_config.pages_per_seq)
        self.max_num_seqs = int(self.run_config.max_num_seqs)
        self.num_decode_steps = int(payload.get("num_decode_steps")
                                    or self.run_config.num_decode_steps)
        #: The perf stage's K sweep (`instance.yaml::perf_k_values`, mirrored into
        #: `meta.json`) and its round count. `perf_rounds`
        #: falls back to :data:`DEFAULT_PERF_ROUNDS` rather than 0 so a bare
        #: `_CandidateRuntime` built without going through `fixtures.build_candidate`
        #: -- as the unit-test style harness builds it --
        #: still gets a sane round count.
        self.perf_k_values = [int(k) for k in (payload.get("perf_k_values") or [])]
        self.perf_rounds = int(payload.get("perf_rounds") or DEFAULT_PERF_ROUNDS)
        self.native_dtype = getattr(jnp, self.run_config.dtype)
        #: The dtype the fp32 leg must allocate its KV CACHE at, from
        #: ``meta['fp32_kv_cache_dtype']`` -- a pin the freezer stamps off the
        #: config its own fp32 window actually built.
        #:
        #: NOT ``repo_config['dtype']``, and the difference matters. Compared with
        #: the reference's own ``Engine.generate`` on the same fp32
        #: model, with the cache dtype the only variable, the per-token logprob gap
        #: against ``fp32_generation.npz`` is orders of magnitude larger on the bf16
        #: serving cache than on an fp32 cache. Every token still agrees in both arms
        #: and every check still passes, so an fp32 leg built on the serving config
        #: calibrates ``logprob_atol`` two orders too loose with nothing warning
        #: it. Read from the golden rather than assumed here so the probe and the
        #: freezer cannot drift, and reported back in the probe result so
        #: ``test_greedy_sequence_fp32`` asserts what ran rather than what was
        #: intended.
        self.fp32_kv_cache_dtype = str(payload.get("fp32_kv_cache_dtype") or "float32")
        #: The L6 workload: ``{prompt_ids, prompt_lens, cont_ids, cont_lens}``,
        #: right-padded, straight out of ``golden/held_out.npz`` (see
        #: :data:`HELD_OUT_INPUT_KEYS` for what is deliberately NOT in it).
        #: ``.get``, not ``[...]``: a caller that supplies no held-out set must
        #: reach :meth:`held_out_probe`'s unsupported branch -- one check failing
        #: with a reason -- rather than raising here, which is construction, hence
        #: common-mode, hence all 30-odd checks for one absent artifact.
        self.held_out_spec = payload.get("held_out") or None

        # The frozen scenario, read and CHECKED before the expensive part: a
        # golden whose two pins disagree must cost a spawn, not a 2.4 GB
        # conversion, to find out.
        #
        # TWO TOKEN ARRAYS, AND ON `core` THEY ARE THE SAME ONE.
        #
        # `prompt_ids` is SEQUENCE 0 -- the canonical single-sequence probe. Every
        #   probe that drives one request reads it: the generation loop, the
        #   decode steps, the paging invariants, cache reuse, serve, the
        #   compilation legs, `static_shapes.num_query_tokens`.
        # `forward_ids` is what the FROZEN FORWARD actually forwards. On a
        #   multi-sequence scenario that is the batch's query tokens concatenated
        #   in `query_start_loc` order -- 14 tokens over three requests where
        #   `prompt_ids` is 12 -- and it is what `prefill_forward.npz` was frozen
        #   from.
        #
        # Reading one from the other is what breaks: `input_ids` alone makes
        # `generate` continue a concatenation of three unrelated requests, and
        # `prompt_token_ids` alone makes the frozen-prefill probe forward twelve
        # tokens against a metadata that describes fourteen. On `core` the two
        # keys hold the same array and every path below is unchanged.
        self.prompt_ids = np.asarray(
            self.scenario.get("prompt_token_ids")
            if self.scenario.get("prompt_token_ids") is not None
            else self.inputs.get("input_ids", []),
        ).reshape(-1).astype(np.int32)
        self.forward_ids = np.asarray(
            self.inputs.get("input_ids", self.scenario.get("prompt_token_ids", [])),
        ).reshape(-1).astype(np.int32)
        if self.prompt_ids.size == 0 or self.forward_ids.size == 0:
            raise ValueError(
                "the frozen scenario carries no prompt: golden/inputs.npz must "
                "hold `input_ids`"
            )
        self.page_ids = [int(p) for p in (self.scenario.get("page_ids") or [])]
        assert_frozen_allocation_agrees(self.inputs.get("block_tables"), self.page_ids)

        #: ``{probe: "frozen" | "candidate" | "mixed"}``, written at each point of
        #: use rather than copied off :data:`USE_FROZEN_ALLOCATION`.
        self._allocation_source: dict[str, str] = {}

        # THE conversion. Done once; every probe is a view onto its result.
        self.params = self.loader_mod.convert_checkpoint(self.model_dir, self.model_config)
        self._model = None          # native-dtype model, built on demand
        self._fp32_model = None
        self._fwd = None            # the ONE compiled forward (see _jit_forward)
        #: `((n_slots, num_kv_heads, head_dim), ...)`, resolved once by
        #: :meth:`_cache_groups` and reused by every probe that allocates a cache.
        #: Cached because resolving it constructs an `Engine`, and six call sites
        #: doing that independently would allocate six containers to read three
        #: integers off each.
        self._cache_group_spec = None

    # --- model construction ------------------------------------------------ #
    def _build_model(self, dtype):
        from flax import nnx

        model = self.modeling_mod.Qwen2_5_VLForConditionalGeneration(
            self.model_config, rngs=nnx.Rngs(self.seed), dtype=dtype)
        self.loader_mod.apply_params(model, self.params)
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self._build_model(self.native_dtype)
        return self._model

    @property
    def fp32_model(self):
        if self._fp32_model is None:
            self._fp32_model = self._build_model(self._jnp.float32)
        return self._fp32_model

    def _release_fp32(self) -> None:
        self._fp32_model = None
        gc.collect()

    # --- frozen-allocation plumbing ---------------------------------------- #
    # Every allocation a probe honours passes through one of the three methods
    # below, so `allocation_source` reports what the code DID rather than what
    # USE_FROZEN_ALLOCATION says. That distinction is the whole point: the table
    # drives nothing, and an accessor built by iterating the table would assert
    # the table against itself.
    def _record_allocation(self, probes: Any, source: str) -> None:
        names = (probes,) if isinstance(probes, str) else tuple(probes)
        for name in names:
            previous = self._allocation_source.get(name)
            # "mixed" rather than a raise: a probe that legitimately spans several
            # inputs (the battery, one variant per axis) must be able to say so,
            # and the declared-vs-observed test turns it into a failure there --
            # where it can be read -- instead of a traceback three frames deep.
            self._allocation_source[name] = (
                source if previous in (None, source) else "mixed")

    def _frozen(self, probes: Any, key: str) -> np.ndarray:
        """A frozen array out of ``golden/inputs.npz``, recording the honoured policy."""
        self._record_allocation(probes, "frozen")
        return self.inputs[key]

    def _frozen_pages(self, probes: Any, page_ids: Any = None) -> list[int]:
        """The frozen page assignment, recording the honoured policy."""
        self._record_allocation(probes, "frozen")
        return [int(p) for p in (self.page_ids if page_ids is None else page_ids)]

    def _candidate_pages(self, probes: Any, page_ids: Any) -> list[int]:
        """Pages the CANDIDATE's own allocator chose, recording the honoured policy."""
        self._record_allocation(probes, "candidate")
        return [int(p) for p in page_ids]

    def _bucket(self, seq_lens, query_start_loc, block_tables):
        """Pad the three per-request fields out to the COMPILED bucket.

        ``AttentionMetadata``'s own docstring declares ``block_tables`` to be
        ``(max_num_seqs * pages_per_seq,)`` and ``seq_lens`` ``(max_num_seqs,)``,
        and derives ``pages_per_seq`` as ``block_tables.shape[0] // num_seqs``.
        A probe that built a one-request metadata against a max_num_seqs-wide
        table would make that derivation read the whole table as one request's
        pages -- a different gather, and a different softmax width, from the one
        the reference ran.

        A NO-OP ON `core`, where ``max_num_seqs`` is 1 and every array is already
        at its bucket width. It matters on a mode whose bucket is wider than one
        request, which is what `ragged` is.
        """
        seq_lens = np.asarray(seq_lens).reshape(-1)
        query_start_loc = np.asarray(query_start_loc).reshape(-1)
        block_tables = np.asarray(block_tables).reshape(-1)
        if seq_lens.size < self.max_num_seqs:
            seq_lens = np.concatenate(
                [seq_lens, np.zeros(self.max_num_seqs - seq_lens.size,
                                    dtype=seq_lens.dtype)])
        if query_start_loc.size < self.max_num_seqs + 1:
            tail = np.full(self.max_num_seqs + 1 - query_start_loc.size,
                           query_start_loc[-1], dtype=query_start_loc.dtype)
            query_start_loc = np.concatenate([query_start_loc, tail])
        want = self.max_num_seqs * self.pages_per_seq
        if block_tables.size < want:
            block_tables = np.concatenate(
                [block_tables,
                 np.zeros(want - block_tables.size, dtype=block_tables.dtype)])
        return seq_lens, query_start_loc, block_tables

    def _metadata(self, *, positions, block_tables, seq_lens, query_start_loc,
                  request_distribution):
        """Build the candidate's OWN ``AttentionMetadata`` around given arrays."""
        jnp = self._jnp
        seq_lens, query_start_loc, block_tables = self._bucket(
            seq_lens, query_start_loc, block_tables)
        return self.metadata_mod.AttentionMetadata(
            input_positions=jnp.asarray(positions, dtype=jnp.int32),
            block_tables=jnp.asarray(block_tables, dtype=jnp.int32),
            seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
            query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
            request_distribution=jnp.asarray(request_distribution, dtype=jnp.int32),
            padded_num_reqs=self.max_num_seqs,
        )

    def _frozen_positions(self) -> np.ndarray:
        """The frozen forward's ``input_positions``, off ``golden/inputs.npz``.

        ONE READ, shared by :meth:`_frozen_prefill_metadata` and
        :meth:`_advanced_positions`, because the decode column below is defined
        as an offset FROM this array and two expressions for one array stop
        agreeing.

        PASSED THROUGH AT WHATEVER RANK IT WAS FROZEN AT -- ``(T,)`` for a
        text-only family and ``(R, T)`` for one whose model builds R rope
        sections. Normalising the rank here would contradict
        ``AttentionMetadata``'s own contract, which the reference's
        ``engine.py::_metadata`` states in as many words and which
        ``test_attention_metadata_schema`` grades.

        The ``arange`` fallback is for a golden frozen before the key existed:
        it is the only value a single full prefill's positions can take at
        rank 1, and it is what every text-only instance's frozen array holds
        anyway.
        """
        positions = self.inputs.get("positions")
        if positions is None:
            positions = np.arange(int(self.forward_ids.shape[0]), dtype=np.int32)
        return np.asarray(positions)

    def _advanced_positions(self, step: int) -> np.ndarray:
        """The position column ``step`` decode steps past the frozen prefill.

        THE REFERENCE ENGINE'S OWN RULE, and it is stated here because nothing
        freezes it: ``golden/decode_step.npz`` carries logits and no positions,
        so a probe that drives the decode by hand has to derive the column the
        engine would have produced. ``engine.Engine._advance`` is that rule --
        one past the furthest position ANY rope section reached, at the same
        rank -- and ``decode_step`` then adds one per step, so step ``k`` sits
        at ``max + 1 + k``.

        IT IS NOT A MULTIMODAL SPECIAL CASE. At rank 1 one position per token
        ascending is the definition, so ``max + 1 + k`` is ``n + k`` -- the
        literal expression this method replaced, digit for digit, on every
        text-only instance. At rank R it is the only expression that is also
        right, because a VL prompt's last position is generally larger than its
        length (an image occupies ``max(h, w)`` positions, not one per token)
        and ``n + k`` would rewind the whole image back under the text that
        follows it.

        THE MAX IS TAKEN OVER REQUEST 0'S OWN QUERY ROWS, NOT THE WHOLE FROZEN
        ARRAY, and that is a correctness fix rather than a tidy-up. It is NOT true
        that "at rank 1 the frozen positions are ``arange(n)``": that is FALSE on
        two shipped instances. In ``golden/inputs.npz``, `chunked_llama3` freezes
        ``[5,6,7,8,9,10,11,8,9,10,8]`` (a chunk-relative window plus two other
        requests) and `ragged_llama3` freezes ``[0..11,5,8]``. A whole-array
        ``max()`` happens to give the right answer on both only because request
        0 is the deepest request in each of those batches. A freeze whose
        request 1 sat past request 0's last position would silently shift this
        probe's rope column ON A TEXT-ONLY INSTANCE, which is the worst shape a
        defect can take here. ``query_start_loc[1]`` is where request 0's query
        rows end -- the same array ``_frozen_prefill_metadata`` already hands the
        candidate -- so slicing to it makes the answer right by construction
        instead of by coincidence. Verified against all seven frozen goldens:
        the sliced max is unchanged on every one.

        AND THE RANK-1 IDENTITY IS ASSERTED, because it is the premise the
        caller's ``base_len`` arithmetic rests on. One position per token
        ascending means ``max + 1`` IS the prompt length; if a golden disagrees,
        the frozen positions are not what this probe assumes and continuing
        would produce a plausible wrong number. It cannot fire at rank R, where
        the two legitimately differ (9 against 11 on this class's VL instance).
        """
        positions = self._frozen_positions()
        # Request 0's own query rows. Absent on a golden frozen before the key
        # existed, where a single full prefill is the only thing it can be.
        end = self.inputs.get("query_start_loc")
        end = (int(np.asarray(end).reshape(-1)[1]) if end is not None
               else int(positions.shape[-1]))
        own = positions[..., :end]
        top = int(own.max())
        base_len = int(self.prompt_ids.shape[0])
        if own.ndim == 1 and top + 1 != base_len:
            raise ValueError(
                f"golden/inputs.npz's `positions` puts request 0's last query "
                f"token at {top}, so the next decode position is {top + 1} -- "
                f"but sequence 0's prompt is {base_len} tokens long, and at "
                f"rank 1 (one position per token, ascending) those two are the "
                f"same number by definition. The decode probe advances from "
                f"this column and would rotate every step by the difference.")
        return np.full((*own.shape[:-1], 1), top + 1 + int(step), dtype=np.int32)

    def _image_inputs(self, arrays: Any = None, *, prefix: str = "") -> dict[str, Any]:
        """The frozen image as the two forward keywords -- ``{}`` when there is none.

        THE ONE PLACE THIS FILE DECIDES WHAT AN ENGINE CALL IS HANDED, and it is
        deliberately a DICT rather than a pair plus a branch at every call site.
        Every probe below spells ``**self._image_inputs()``; on a family whose
        golden froze no ``pixel_values`` that expands to nothing and the call is
        byte for byte the one it has always been, and on one that did it carries
        the picture. No probe asks whether the instance is multimodal, which is
        the same delegation ``engine.py`` uses for positions and for host-side
        inputs -- see :meth:`_prepare_model_inputs`.

        ``arrays`` names WHICH frozen mapping to read, defaulting to the
        canonical ``inputs.npz``. :meth:`_handed_forward` passes a battery
        variant's own mapping instead, so a variant is forwarded with the image
        its own ids were frozen against and never with the canonical scenario's.

        ``prefix`` names WHICH FORWARD's image to read out of that mapping, and
        it is the same ``warmup_`` prefix every other member of the warm-up
        group already carries (``warmup_input_ids``, ``warmup_positions``, ...).
        A mixed-stage batch runs TWO forwards and a sequence gives its image to
        whichever of them covers its ``<|image_pad|>`` placeholders -- the
        warm-up's, when the sequence is a decode whose query token is past every
        placeholder. Before the prefix existed the schema recorded ONE top-level
        pair with nothing saying whose it was, and freezer and candidate agreed
        only when the image-bearing sequence had ``start == 0``; see
        :meth:`_handed_forward` for the details.

        float32 pixel_values REGARDLESS OF THE SERVING DTYPE and int32 for the
        grid -- see :meth:`_frozen_image_inputs`, which is this method's
        two-element view and carries the full argument for both dtypes.
        """
        source = self.inputs if arrays is None else arrays
        pixel_values = source.get(f"{prefix}pixel_values")
        grid = source.get(f"{prefix}image_grid_thw")
        if pixel_values is None or grid is None:
            return {}
        jnp = self._jnp
        return {
            "pixel_values": jnp.asarray(np.asarray(pixel_values),
                                        dtype=jnp.float32),
            "image_grid_thw": jnp.asarray(np.asarray(grid), dtype=jnp.int32),
        }

    def _graded_image_inputs(self) -> dict[str, Any]:
        """The image keywords THE GRADED FORWARD owns -- ``{}`` when it owns none.

        THE SAME RULE :meth:`_ragged_forward` APPLIES, factored out because three
        probes need it and two of them would otherwise spell ``self._image_inputs()``
        unconditionally. A sequence gives its picture to whichever forward covers
        its ``<|image_pad|>`` placeholders, and the freezer records WHICH by the
        ``warmup_`` prefix. So a ``warmup_pixel_values`` in the frozen mapping is a positive
        statement that the GRADED forward's ids carry no placeholders, and
        handing it the picture anyway raises "the prompt carries 0 <|image_pad|>
        ... but the tower produced 4 embeddings".

        On `chunked_qwen2_5_vl`, the first instance whose canonical
        sequence 0 is BOTH multimodal and a `prefill_chunked`: its run is [1, 5)
        and its graded query starts at 5, so the warm-up owns the image and the
        graded forward's ten ids carry none. Without this, `test_jit_traceability`
        and `test_image_embedding_merge` both fail on exactly that.

        INERT ON EVERY INSTANCE THAT PREDATES IT: no `core` golden has a warm-up
        at all, and on the two `ragged` multimodal ones the image-bearing
        sequence is a `prefill` with `start == 0`, so the graded forward owns the
        picture and this returns exactly what `_image_inputs()` returns.
        """
        return {} if self._image_inputs(prefix="warmup_") else self._image_inputs()

    def _image_owner_forward(self):
        """``(ids, image_kwargs)`` for whichever forward the picture belongs to.

        For the probes whose SUBJECT is the merge itself rather than the graded
        row: the ids and the image have to come from the same forward or the
        placeholder count cannot match the tower's row count. Falls back to the
        graded forward, which is every instance with no warm-up group.
        """
        warm = self._image_inputs(prefix="warmup_")
        if warm:
            warm_ids = self.inputs.get("warmup_input_ids")
            if warm_ids is not None:
                return np.asarray(warm_ids), warm
        return np.asarray(self.forward_ids), self._image_inputs()

    def _image_placeholder_runs(self, input_ids) -> list:
        """``[(start, stop)]`` -- each maximal run of the image placeholder id.

        THE ID IS RESOLVED BY NAME OFF THE CANDIDATE'S OWN CONFIG, never spelled
        here and never read from a golden. ``image_token_id`` is the attribute
        the reference's ``get_input_embeddings`` masks on and the attribute its
        ``get_mrope_input_positions`` recovers its spans from, so this asks the
        PORT the same question the port asks itself -- the same by-name
        delegation :meth:`_prepare_model_inputs` makes for ``INPUTS_HOOK``.
        Nothing here branches on a family, a model name or a ``multimodal``
        fact.

        ``[]`` when the config declares no such attribute, which is every
        text-only family, and the one caller treats that as "no image".

        HALF-OPEN. A chunk boundary AT ``start`` puts the whole run in the later
        chunk and one AT ``stop`` puts it in the earlier one; both reproduce an
        unchunked prefill. Only a boundary strictly between them
        tears the run, which is the case :meth:`chunked_prefill_probe` reports
        unsupported.
        """
        image_id = getattr(getattr(self.model, "config", None),
                           "image_token_id", None)
        if image_id is None:
            return []
        flat = [int(t) for t in np.asarray(input_ids).reshape(-1)]
        runs = []
        index = 0
        while index < len(flat):
            if flat[index] == int(image_id):
                start = index
                while index < len(flat) and flat[index] == int(image_id):
                    index += 1
                runs.append((start, index))
            else:
                index += 1
        return runs

    def _prepare_model_inputs(self, model, input_ids, **model_kwargs) -> dict:
        """``Engine._model_inputs``: the extra forward keywords, PREPARED ON THE HOST.

        FOR THE PROBES THAT DRIVE THE MODEL DIRECTLY, not through the
        candidate's ``Engine`` -- :meth:`_handed_forward`,
        :meth:`_prefill_on_pages`, :meth:`decode_steps_probe` and
        ``compilation_probe``'s control leg. A probe that goes through
        ``Engine.prefill`` needs none of this: it passes the keywords in and the
        CANDIDATE's engine resolves its own hook, which is the stronger
        arrangement and is why those call sites are one ``**`` and nothing else.

        THE HOOK IS RESOLVED BY NAME OFF THE CANDIDATE'S OWN ``Engine``, exactly
        as ``engine.Engine._model_inputs`` does, and for the identical reason:
        on jax 0.11 every array born inside a ``jit`` trace is a tracer, so a
        subsystem that reads its own intermediates concretely -- a vision tower
        whose window permutation and per-chunk attention bounds are Python
        control flow over ``jnp`` values -- cannot execute under the trace. The
        model gets one chance, on the host, to turn such keywords into ones the
        trace can carry, typically by running the tower and returning
        ``inputs_embeds``. That is a property of how the tower is WRITTEN and
        not a limit of jit; see the reference's own method for the measurement.

        A PORT THAT DEFINES NO HOOK GETS ITS KEYWORDS BACK UNCHANGED, which is
        the shared default (``engine.default_model_inputs``) and is what every
        text-only family sees. ``INPUTS_HOOK`` is not in ``interface.json``'s
        required set, so a port whose tower IS traceable may legitimately have
        neither -- it passes ``pixel_values`` straight into its own forward and
        this method must not stand in the way.
        """
        hook = getattr(self.engine_mod.Engine, "INPUTS_HOOK", None)
        prepare = getattr(model, hook, None) if hook else None
        if prepare is None:
            return dict(model_kwargs)
        return dict(prepare(input_ids, **model_kwargs))

    def _built_positions(self, model, input_ids, **model_kwargs):
        """``Engine._input_positions``: ask the MODEL, fall back to the shared default.

        The same protocol lookup :meth:`_prepare_model_inputs` makes, for the
        other hook, and used by the same four direct-drive probes. ``positions``
        is the model's to build on both frameworks -- a family whose position is
        a ``(t, h, w)`` triple says so by defining ``POSITIONS_HOOK``, and this
        file never learns that it did.

        THE FALLBACK IS RESOLVED DEFENSIVELY, IN THREE STEPS, because
        ``engine.default_input_positions`` is NOT guaranteed to be in
        ``interface.json``'s required set: a bare
        ``self.engine_mod.default_input_positions`` would regress an instance.
        `chunked_llama3`'s
        `engine.py` defines ``POSITIONS_HOOK`` and defines NO
        ``default_input_positions`` at all; its ``interface.json``
        ``required["engine.py"]`` is the four result classes and nothing else.
        (The hook is defined but no module-level default is: `chunked`'s prefill
        takes the request's whole run as ``full_input_ids``, asks the hook for it
        and windows this chunk out of the answer, so the defect below is reachable
        exactly as described.)
        So the hook name resolved, the Llama model had no such
        method, and the module attribute raised ``AttributeError``, taking
        ``test_paging_invariance``, ``test_cache_reuse_safety``,
        ``test_compilation_is_cached`` and ``test_static_shape_recompilation``
        red on a text-only instance.

        THE SYMBOL IS PER-MODE, NOT CLASS-WIDE. Six of the seven instances
        require it; `chunked` does not, and its own module docstring says the
        delegation machinery is "absent from this fork rather than carried dead".
        So the port's own builder is PREFERRED when it exists -- that is the more
        faithful drive, and a broken one is visible through it -- and the local
        :meth:`_default_input_positions` answers when it does not. Note the
        asymmetry that was the bug: the hook NAME lookup was already
        ``getattr(..., None)`` while the module attribute was not.
        """
        hook = getattr(self.engine_mod.Engine, "POSITIONS_HOOK", None)
        build = getattr(model, hook, None) if hook else None
        if build is None:
            build = getattr(self.engine_mod, "default_input_positions",
                            self._default_input_positions)
        return build(input_ids, **model_kwargs)

    @staticmethod
    def _default_input_positions(input_ids, **_):
        """``(num_tokens,)`` ascending positions -- for a mode whose engine has none.

        Byte for byte what ``engine.default_input_positions`` computes, and byte
        for byte what the three call sites that now delegate spelled inline
        (``np.arange(n, dtype=np.int32)``) before they did. ``**_`` swallows the
        keywords a rank-1 builder has no use for, exactly as the shared default
        does. NOT a second opinion about positions: it exists so that a mode
        which deliberately ships no shared builder does not take an
        ``AttributeError`` in a helper that was only ever meant to ask.
        """
        return np.arange(int(np.asarray(input_ids).reshape(-1).shape[0]),
                         dtype=np.int32)

    def _frozen_prefill_metadata(self, probe: Any):
        """Metadata for the frozen forward, straight out of ``golden/inputs.npz``.

        Every field is HANDED IN, and ``probe`` is the name that
        records having honoured it. ``positions`` and ``request_distribution``
        fall back to the only values they can take for a single full prefill, so
        a golden frozen before those keys existed still drives the probe rather
        than erroring three frames down.
        """
        n = int(self.forward_ids.shape[0])
        dist = self.inputs.get("request_distribution")
        if dist is None:
            dist = np.array([0, 1, 0], dtype=np.int32)
        return self._metadata(
            positions=self._frozen_positions(),
            block_tables=self._frozen(probe, "block_tables"),
            seq_lens=self.inputs.get("seq_lens", np.array([n], dtype=np.int32)),
            query_start_loc=self.inputs.get("query_start_loc",
                                            np.array([0, n], dtype=np.int32)),
            request_distribution=dist,
        )

    def _empty_cache(self, dtype):
        """The candidate's KV cache CONTAINER, allocated the way its engine does.

        Always through the candidate's own `make_kv_cache`, at the geometry
        :meth:`_cache_groups` resolved; `dtype` is the caller's, because the fp32
        window allocates the same container at a different precision.
        """
        return self.kv_cache_mod.make_kv_cache(
            self._cache_groups(), self.num_pages, self.page_size, dtype)

    def _cache_groups(self):
        """`((n_slots, num_kv_heads, head_dim), ...)` -- one row per KV geometry.

        THE RECTANGULAR LITERAL IS STILL THE ANSWER FOR EVERY FAMILY THAT HAS ONE
        GEOMETRY, and it is the same three integers off `ModelConfig` this method
        replaced, so nothing about the fourteen instances shipped today moves.
        What changed is that the literal is no longer the ONLY answer: a family
        whose layers disagree -- different head_dims, or shared slots -- allocates
        several groups, and one rectangular tensor cannot express it at all.

        SO THE CANDIDATE'S OWN `Engine` IS ASKED FIRST, and only its GEOMETRY is
        taken: `(n_slots, num_kv_heads, head_dim)` per group, fed back through the
        candidate's `make_kv_cache` so every probe still gets a freshly zeroed
        container at the dtype it asked for. Deriving the grouping from
        `golden/cache_spec.json` instead would make `cache_spec_probe` -- which
        calls this method -- compare the golden against itself, which is the one
        thing `test_kv_cache_spec` must not do.

        AND THE LITERAL IS THE FALLBACK, deliberately, in BOTH the cases where the
        engine did not answer with more than one group:

        * discovery found nothing (`interface.json` pins no `Engine` attribute, so
          a port that names its container something else is a LEGAL port -- see
          :meth:`_observed_cache_dtype`, where requiring the name would cost a
          bit-identical port its reward). Falling back means such a port is
          driven exactly as it is today rather than losing every numeric probe;
        * discovery found exactly ONE group, where the literal and the discovered
          row agree for any correct port and the literal is what the goldens were
          frozen against.

        A FALLBACK ON A HETEROGENEOUS FAMILY IS LOUD, NOT SILENT, which is what
        makes it safe to have one. The container the literal builds has a single
        group, so the first layer whose `layer_slot` names group 1 raises out of
        `cache/kv_cache._require_slot` ("KV cache group 1 is out of range"), and
        `cache_spec_probe` reports `shape` where the golden declares `groups`, so
        `test_kv_cache_spec` fails with the two contracts side by side. There is
        no arrangement in which a wrong container is graded green.

        WHAT THIS DOES NOT CLOSE, reported rather than fixed: on a family whose
        golden declares `shape`, a candidate whose engine allocates one group of
        the WRONG geometry still has the literal handed to it, so `cache_spec_probe`
        answers with the literal and `test_kv_cache_spec` cannot see it. That hole
        predates this method and is unchanged by it; closing it means making the
        discovered geometry authoritative for one group too, which converts a
        localised spec failure into a whole-tier one.
        """
        if self._cache_group_spec is None:
            rectangular = ((int(self.model_config.num_hidden_layers),
                            int(self.model_config.num_key_value_heads),
                            int(self.model_config.head_dim)),)
            discovered = self._engine_cache_groups()
            self._cache_group_spec = (
                discovered if discovered and len(discovered) > 1 else rectangular)
        return self._cache_group_spec

    def _engine_cache_groups(self):
        """The geometry of the container the candidate's `Engine` allocates, or None.

        `Engine(model, run_config)` and nothing else: the reference's own
        `__init__` is where the grouping rule lives (`make_kv_cache(((...),), ...)`
        there, two rows for a family whose layers disagree), so this asks the
        candidate the same question at the same seam rather than restating the
        rule. The container is read and dropped; only three integers per group
        survive.

        NEVER RAISES. Every failure -- an `Engine` that will not construct without
        a live pool, an attribute that is not there, a member that is not a tuple
        of 6-D arrays -- answers `None`, and the caller falls back. This method is
        reached from `cache_spec_probe`, which is a COMMON_MODE_PROBE: a raise here
        would take every check with it, in service of a lookup that is an
        optimisation for thirteen of fourteen families.

        THE SHAPE IS VALIDATED, not assumed. A 6-D array whose axes 1..3 are
        `(num_pages, page_size, 2)` is a KV cache group at this class's layout and
        an `Engine` attribute that happens to be a tuple of something else is not;
        without that clause a port holding, say, a tuple of per-layer position
        tensors would be read as a 32-group cache.
        """
        want_middle = (self.num_pages, self.page_size, 2)
        try:
            engine = self.engine_mod.Engine(self.model, self.run_config)
        except Exception:  # noqa: BLE001 - a port whose Engine needs more is not a defect here
            return None
        try:
            container = getattr(engine, "kv_cache", None)
            if not isinstance(container, (list, tuple)) or not container:
                return None
            rows = []
            for group in container:
                shape = getattr(group, "shape", None)
                if shape is None or len(shape) != 6:
                    return None
                shape = tuple(int(d) for d in shape)
                if shape[1:4] != want_middle:
                    return None
                rows.append((shape[0], shape[4], shape[5]))
            return tuple(rows)
        except Exception:  # noqa: BLE001 - see NEVER RAISES above
            return None
        finally:
            del engine
            gc.collect()

    @staticmethod
    def _cache_arrays(cache) -> tuple:
        """Every group of the container as numpy, IN CONTAINER ORDER.

        The one shape a cache-derived probe value takes, for one group and for
        several alike: a TUPLE. `test_kv_cache_write` compares its length against
        the frozen member count before it compares anything else, so a candidate
        that allocated the wrong number of geometries fails by name -- which the
        old helper (`_one_group`, which refused a heterogeneous container outright)
        could not do, because it made the first freeze of such a family impossible.

        Container order is the order `layer_slot` addresses, which is the order the
        golden's `cache_group{i}` members are written in; nothing here sorts.
        """
        return tuple(_to_np(group) for group in cache)

    def _jit_forward(self):
        """The ONE ``nnx.jit``-ed forward, mirroring what ``engine.Engine`` compiles.

        Jitted rather than eager on purpose: the golden was produced by an engine
        whose forward is jitted, and an eager 16-layer decode loop dispatches
        every op individually. ``output_hidden_states`` is static -- it changes
        the number of returned values.

        COMPILED ONCE, cached on the runtime, and the ground for that is COMPILE
        COST. A fresh ``nnx.jit`` wrapper per call carries a fresh trace cache, so
        every probe re-traced and re-compiled the whole 16-layer model. Caching it
        roughly halves the number of full-model compilations over a worker run, a
        deterministic count that is the whole argument. Wall clock on a shared card
        is occupancy-dependent and a single sample of it proves nothing on its own,
        so it is not the argument.
        The cost matters most at this class's 20B end, where the recompiles --
        not the forwards -- would dominate.

        WHAT THIS IS *NOT* FOR: separate compilations of one graph INSIDE ONE
        PROCESS do not diverge. XLA's autotune results are cached per process and
        keyed by the HLO fingerprint, so a recompile in the same worker reuses the
        winning algorithm. The bf16 divergence this class does have is
        CROSS-PROCESS under contention, which no amount of
        caching inside one worker would address, and which is why the invariance
        probes run both of their legs in the same worker to begin with.

        ``model_kwargs`` MIRRORS ``engine.Engine.__init__``'s own ``_forward``,
        down to the argument position: a pytree of arrays, already prepared on
        the host (:meth:`_prepare_model_inputs`), splatted into the model's
        forward inside the trace. ``{}`` for every text-only family, so the
        lowered function is the one it has always been -- an empty dict flattens
        to no leaves and its treedef is part of the jit cache key, so the two
        shapes get two programs rather than one program that has to branch.
        It is a TRACED argument and not a static one on purpose: what it carries
        is a merged embedding table, which is data.
        """
        if self._fwd is None:
            from flax import nnx

            def _forward(model, input_ids, kv_cache, meta, slot_mapping, hidden,
                         model_kwargs):
                return model(input_ids, kv_cache, meta, slot_mapping,
                             output_hidden_states=hidden, **model_kwargs)

            self._fwd = nnx.jit(_forward, static_argnums=(5,))
        return self._fwd

    def _handed_forward(self, arrays, model, dtype, *, want_hidden: bool, probe: Any):
        """A forward whose EVERY input was handed in, out of one arrays mapping.

        The one composition, used twice: by :meth:`_run_frozen_prefill` over
        ``golden/inputs.npz`` and by :meth:`battery_probe` over a variant's own
        ``inputs.npz``. Extracted rather than duplicated because the two must not
        be able to drift -- a battery variant graded through a slightly different
        composition than the canonical scenario is comparing two things.

        ``warmup_*`` keys, when present, are a FIRST forward on the same cache
        (see :meth:`_warmup_group`). Absent -- which is every `core` golden and
        every all-prefill battery variant -- this is the single call it has
        always been.

        THE IMAGE IS HANDED IN LIKE EVERY OTHER ARRAY, out of the SAME mapping
        the ids came from (:meth:`_image_inputs`), and it is prepared on the
        host before the trace (:meth:`_prepare_model_inputs`). Without it this
        composition forwarded an image-conditioned prompt with the
        ``<|image_pad|>`` rows left as their placeholder EMBEDDINGS -- the right
        shapes, the right positions, a clean run and a wrong answer that diverges
        sharply from
        ``prefill_forward.npz``, which is the golden this very forward produced.
        A frozen mapping that carries no image expands to nothing here, which is
        every text-only instance.

        ...AND WHICH OF THE TWO FORWARDS GETS IT IS READ OFF THE MAPPING, not
        assumed to be the graded one. ``warmup_pixel_values`` /
        ``warmup_image_grid_thw`` are the WARM-UP's image, frozen beside its own
        ``warmup_input_ids``; the top-level pair is the graded forward's.
        Exactly one forward owns any given image (``_build_batch``: a sequence's
        ``start`` is either 0 or it is not, and ``validate_sequences`` refuses a
        second image), so a mapping that records a warm-up image is a mapping
        whose GRADED forward has none -- and the top-level pair it also carries
        is the instance's picture for the twenty single-sequence probes, not
        this forward's.

        On ``ragged_qwen2_5_vl``'s two `batch_composition` variants,
        which are the first frozen mixed-stage variants on a multimodal
        instance, ``compositions()`` stages sequence 0 as a `decode`, so its four
        ``<|image_pad|>`` rows move into the warm-up; with the warm-up leg given
        no image the graded row diverges from the frozen answer on both variants
        (``d1p2`` and ``d2p1``). This is a different arrangement from the GRADED
        forward losing its own image on a `core` geometry (the paragraph above):
        the two produce different divergences, so the warm-up case had to be
        checked directly rather than inferred from the other.
        """
        jnp = self._jnp
        cache = self._empty_cache(dtype)
        fwd = self._jit_forward()
        # POINT OF USE: every array below came out of a frozen inputs.npz.
        self._record_allocation(probe, "frozen")
        warm_image = self._image_inputs(arrays, prefix="warmup_")
        image = {} if warm_image else self._image_inputs(arrays)
        if "warmup_input_ids" in arrays:
            warm_meta = self._metadata(
                positions=arrays["warmup_positions"],
                block_tables=arrays["block_tables"],
                seq_lens=arrays["warmup_seq_lens"],
                query_start_loc=arrays["warmup_query_start_loc"],
                request_distribution=arrays["warmup_request_distribution"],
            )
            warm_ids = jnp.asarray(arrays["warmup_input_ids"], dtype=jnp.int32)
            _hidden, cache = fwd(
                model, warm_ids, cache, warm_meta,
                jnp.asarray(arrays["warmup_slot_mapping"], dtype=jnp.int32), False,
                # THE WARM-UP'S OWN IMAGE, from beside its own `warmup_input_ids`
                # in the same npz -- the schema carries it explicitly.
                # It is NOT the top-level pair: merging the graded prompt's
                # picture into a different token array would scatter the tower's
                # rows at whatever offsets that array happens to hold a
                # placeholder at. Empty on every text-only instance and on every
                # variant whose graded forward owns the image.
                self._prepare_model_inputs(model, warm_ids, **warm_image))
        ids_np = np.asarray(arrays["input_ids"]).reshape(-1)
        n = int(ids_np.shape[0])
        meta = self._metadata(
            positions=arrays.get("positions", np.arange(n, dtype=np.int32)),
            block_tables=arrays["block_tables"],
            seq_lens=arrays.get("seq_lens", np.array([n], dtype=np.int32)),
            query_start_loc=arrays.get("query_start_loc",
                                       np.array([0, n], dtype=np.int32)),
            request_distribution=arrays.get("request_distribution",
                                            np.array([0, 1, 0], dtype=np.int32)),
        )
        slot = jnp.asarray(arrays["slot_mapping"], dtype=jnp.int32)
        ids = jnp.asarray(ids_np, dtype=jnp.int32)
        out = fwd(model, ids, cache, meta, slot, bool(want_hidden),
                  self._prepare_model_inputs(model, ids, **image))
        if want_hidden:
            hidden, hidden_states, cache = out
        else:
            hidden_states = None
            hidden, cache = out
        return model.compute_logits(hidden), hidden_states, cache, meta, slot

    def _run_frozen_prefill(self, model, dtype, *, want_hidden: bool, probe: Any):
        """The frozen forward: frozen ids, frozen slot mapping, frozen metadata."""
        return self._handed_forward(self.inputs, model, dtype,
                                    want_hidden=want_hidden, probe=probe)

    def _decode_slot(self, position: int, probe: Any) -> int:
        """The frozen flat cache slot for absolute ``position``.

        Pure arithmetic over the FROZEN page ids, so it needs no extra golden key
        and cannot drift from the allocation ``scenario.json`` pins.
        """
        page_ids = self._frozen_pages(probe)
        page_index = position // self.page_size
        if page_index >= len(page_ids):
            raise ValueError(
                f"the frozen scenario allocates {len(page_ids)} page(s) but "
                f"position {position} needs page index {page_index}: "
                "scenario.json's page_ids do not cover prompt_len + num_decode_steps"
            )
        return page_ids[page_index] * self.page_size + (position % self.page_size)

    # --- Spec-tier probes --------------------------------------------------- #
    def params_spec_probe(self) -> dict[str, Any]:
        """``{post-fusion name: {shape, dtype}}`` from ``convert_checkpoint``.

        Read off the CONVERTER's output and not off the nnx parameter pytree, for
        a measured reason: flax dedups a tied reference, so the pytree carries
        ``lm_head.weight`` and NOT ``model.embed_tokens.weight`` while torch's
        ``named_parameters()`` carries exactly the opposite pair. The frozen
        ``golden/params_spec.json`` comes from torch's ``state_dict()``, which
        keeps both -- and so does ``convert_checkpoint``, which is also the
        graded entry point (``interface.json`` ``required_extras``). Comparing
        converter output to state dict is therefore an honest set equality;
        comparing the pytree to it would report one spurious missing key and one
        spurious extra on every tied model.
        """
        return {
            name: {"shape": [int(d) for d in np.shape(arr)],
                   "dtype": str(np.asarray(arr).dtype)}
            for name, arr in self.params.items()
        }

    def param_norms_probe(self) -> dict[str, Any]:
        return {name: _F().param_stats(_to_np(arr)) for name, arr in self.params.items()}

    def cache_spec_probe(self) -> dict[str, Any]:
        """`shape` for a single-geometry cache, `groups` for a heterogeneous one.

        EXACTLY ONE OF THE TWO, never both and never neither -- the freezer emits
        the same pair of alternatives and ``test_kv_cache_spec`` requires the
        choice to be made. A single default here would let a family whose layers
        disagree grade against a rectangular contract that cannot describe it.
        """
        cache = self._empty_cache(self.native_dtype)
        spec = {
            "dtype": str(cache[0].dtype),
            "num_layers": int(self.model_config.num_hidden_layers),
            "num_pages": self.num_pages,
            "page_size": self.page_size,
            "num_kv_heads": int(self.model_config.num_key_value_heads),
            "head_dim": int(self.model_config.head_dim),
        }
        if len(cache) == 1:
            spec["shape"] = [int(d) for d in cache[0].shape]
        else:
            spec["groups"] = [[int(d) for d in group.shape] for group in cache]
        del cache
        gc.collect()
        return spec

    def allocator_probe(self) -> dict[str, Any]:
        """One run of the candidate's OWN engine -- the allocator-grading probe.

        ``use_frozen_allocation=False`` for everything this returns: the block
        table, the slot mapping and the metadata are whatever the candidate's
        ``BlockPool`` and ``Engine._metadata`` produce for the frozen prompt, and
        ``test_block_table_contract`` compares them to the allocation
        ``scenario.json`` pins. Handing them in here would compare the golden to
        itself.

        TWO ENGINE CALLS, not one: a ``prefill`` and then a ``decode_step`` on the
        same engine. Everything but ``request_distribution`` is read off the
        prefill; the decode exists solely so the stage-dependent half of that one
        field is observable at all. See the comment at ``stage_distribution``.

        ``**self._image_inputs()`` -- the frozen image, forwarded verbatim, and
        nothing else about this probe changes. ``Engine.prefill``'s contract is
        that extra keywords reach BOTH the position builder and the host-side
        input hook, so on a family whose prompt carries ``<|image_pad|>`` this
        is what lets the engine build its own M-RoPE columns at all; without it
        the position builder is handed placeholders it has no grid for and
        raises. Empty on a text-only family, where the call is unchanged.
        """
        jnp = self._jnp
        engine = self.engine_mod.Engine(self.model, self.run_config)
        result = engine.prefill(jnp.asarray(self.prompt_ids, dtype=jnp.int32),
                                **self._image_inputs())
        meta = result.attn_metadata
        # POINT OF USE: every field below is read off the allocation the
        # candidate's own BlockPool + Engine._metadata just produced.
        derived_pages = self._candidate_pages(
            ("metadata_schema", "block_table", "slot_mapping", "static_shapes"),
            result.page_ids)

        # Fields, never instances: the frozen dataclass's generated __eq__
        # compares field tuples and bool() of a multi-element array is ambiguous,
        # so `a == b` RAISES (ValueError here, RuntimeError on the torch side).
        # hash() raises too. Neither may appear in a schema comparison.
        fields = ["input_positions", "block_tables", "seq_lens", "query_start_loc",
                  "request_distribution", "padded_num_reqs"]
        schema: dict[str, Any] = {}
        for name in fields:
            value = getattr(meta, name)
            if isinstance(value, (int, np.integer)) or np.ndim(value) == 0:
                schema[name] = {"kind": "static", "value": int(value)}
            else:
                arr = _to_np(value)
                schema[name] = {
                    "kind": "array",
                    "shape": [int(d) for d in arr.shape],
                    "dtype": str(np.asarray(value).dtype),
                }
        # THE ONE METADATA FIELD WHOSE *VALUE* IS GRADED, and the only stage pair
        # that can grade it. `request_distribution` is `[n_decode, n_prefill,
        # n_mixed]` -- it NAMES the batch composition -- and NOTHING reads it:
        # neither reference's `paged_attention` touches it, so it is numerically
        # inert on both sides and a wrong value cannot surface as a wrong logit.
        # Without this leg, a reference whose `Engine._metadata`
        # emits a constant `[0, 1, 0]` for EVERY call -- reporting "one prefill
        # request" while decoding -- scores full reward, caught
        # by nothing.
        #
        # A PREFILL ALONE CANNOT CATCH THAT, which is why this costs a decode
        # step. `Engine.prefill`'s correct answer IS `[0, 1, 0]`, so the constant
        # is indistinguishable from the truth there; `Engine.decode_step`'s is
        # `[1, 0, 0]`, and that is the call the constant contradicts. Both are
        # single-request on every mode -- `interface.json` pins no batched entry
        # point -- so the two expected values are the same on `core`, `ragged`
        # and `chunked`, and the check that reads them needs no mode gate.
        #
        # AFTER the three arrays above are read off the PREFILL result: the decode
        # step advances the same engine, and `test_block_table_contract`'s leg 2
        # grades the prefill allocation.
        tokens = self.scenario.get("decode_input_token_ids") or []
        token = (int(tokens[0]) if tokens
                 else int(np.asarray(self.prompt_ids).reshape(-1)[-1]))
        decode = engine.decode_step(0, token)
        stage_distribution = {
            "prefill": [int(x) for x in
                        _to_np(meta.request_distribution).reshape(-1)],
            "decode": [int(x) for x in
                       _to_np(decode.attn_metadata.request_distribution).reshape(-1)],
        }
        out = {
            "metadata_schema": {
                "fields": fields,
                "schema": schema,
                "num_seqs": int(meta.num_seqs),
                "pages_per_seq": int(meta.pages_per_seq),
                "request_distribution": stage_distribution,
            },
            "block_table": _to_np(meta.block_tables).astype(np.int64),
            "slot_mapping": _to_np(result.slot_mapping).astype(np.int64),
            "page_ids": derived_pages,
            "static_shapes": {
                "padded_num_reqs": int(meta.padded_num_reqs),
                "max_num_seqs": self.max_num_seqs,
                "pages_per_seq": int(meta.pages_per_seq),
                "num_query_tokens": int(self.prompt_ids.shape[0]),
                "block_table_size": int(np.size(_to_np(meta.block_tables))),
                "query_start_loc_size": int(np.size(_to_np(meta.query_start_loc))),
                "seq_lens_size": int(np.size(_to_np(meta.seq_lens))),
            },
        }
        del engine, result, decode
        gc.collect()
        return out

    def _count_compilations(self):
        """Context manager yielding a counter of XLA compilations in this process.

        ``jax.log_compiles(True)`` makes jax emit one ``Finished XLA compilation of
        <name> in <t> sec`` record per compiled program on the ``jax`` logger. That
        is the only compilation observable jax exposes -- ``nnx.jit`` has no
        cache-size accessor in flax 0.10/0.12, and the reference ``Engine``'s own
        ``_traces`` counter is a property of the REFERENCE, so a candidate cannot be
        required to have one.

        IT WORKS ACROSS THE SANDBOX BOUNDARY because it never crosses it: the
        counter runs INSIDE the worker process, next to the compilations it counts,
        and only the resulting integers are pickled back. A parent-side counter
        would see nothing at all.

        The handler is removed and the level restored in ``finally``, so the grade's
        own logging is unaffected either way.
        """
        import contextlib
        import logging

        class _Counter(logging.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.n = 0
                self.names: list[str] = []

            def emit(self, record) -> None:  # noqa: ANN001 - logging hook
                message = record.getMessage()
                if "Finished XLA compilation" in message:
                    self.n += 1
                    if len(self.names) < 32:
                        self.names.append(message.split(" in ")[0][-60:])

        jax = self._jax

        @contextlib.contextmanager
        def _counting():
            handler = _Counter()
            logger = logging.getLogger("jax")
            level = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            with jax.log_compiles(True):
                try:
                    yield handler
                finally:
                    logger.removeHandler(handler)
                    logger.setLevel(level)

        return _counting()

    def compilation_probe(self) -> dict[str, Any]:
        """How many XLA compilations the candidate's engine costs, over five legs.

        THE HOLE THIS CLOSES. Every other Spec check is satisfied by an engine that
        never calls ``jit`` at all: the shapes are identical, the numbers are
        identical, and the jaxpr the traceability probe captures is the DRIVER's own
        ``nnx.jit`` of the candidate's model, not the candidate's. "Static shape
        bucketing" is the discipline that makes one compiled program serve every
        batch, so a port that pays for a fresh compilation per call has not done the
        thing the tier says it graded.

        Five legs. THE FIRST FOUR ALWAYS RUN; only the fifth (``body``) needs a
        second prompt length, and it is the only one whose absence deselects
        anything.

        * ``control`` -- one ``nnx.jit``-ed full-model forward, from a cleared
          cache. Jitted BY CONSTRUCTION (this file wrote it), so it calibrates the
          counter against a known answer: exactly **1**. A spec check
          asserts it is non-zero, which is what stops a broken counter from making
          the real legs pass vacuously.
        * ``same_shape`` -- a second prefill of the same prompt on a warm engine.
          Must be **0**. This is the threshold-free leg and the ONLY detector of a
          port that builds a fresh ``jit`` wrapper per call, which is why it is not
          allowed to ride on the geometry gate: an instance with a short prompt would
          otherwise lose that coverage entirely while reading as fully covered.
        * ``decode_steady_state`` -- until this leg existed, every leg above was a
          PREFILL, so "one compiled program per shape" was graded for exactly half
          of instruction.md's contract ("the graded loop calls prefill once and then
          decodes step by step; EACH of those must reuse a single compiled
          program"). One decode step warms the (new, post-prefill) decode shape and
          is excluded; the next several must cost **0**. Without this leg, a decode
          that rebuilds its ``nnx.jit`` wrapper inside ``decode_step`` is
          bit-identical to the reference and passes every other Spec check.
          ``None`` -- with ``decode_steady_state_reason`` set -- when the golden
          carries fewer than :data:`MIN_STEADY_STATE_DECODE_IDS`
          ``decode_input_token_ids``, the same unsupported shape
          :meth:`decode_steps_probe` uses for the same missing input. That branch
          is BELT AND BRACES behind ``_grade_precheck``, which refuses such a
          golden outright, and the check that reads it FAILS on it rather than
          skipping (a skip would shrink the reward's denominator).
          ``decode_steady_state_steps`` is ALWAYS reported and is how many steps
          the counted region actually ran, so the check that reads the count can
          assert the measurement happened instead of passing on zero iterations.
        * ``new_shape`` -- a prefill at a DIFFERENT prompt length. Reported, not
          graded: on a correct port 15 of its 16 compilations are the engine's own
          eager metadata primitives, so it is mostly glue.
        * ``body`` -- the same prefill, after the glue at that length has been warmed
          through the driver's own jitted path. This is the DISCRIMINATOR, and it is
          the signal with the glue removed. ``None`` when the geometry admits no
          second length (see :func:`second_prompt_length`), in which case
          ``conftest`` has already deselected the check that reads it.

        ``cold`` is reported and not graded either: it folds one-time setup in with
        the per-shape cost and is the noisiest number here.
        """
        jax = self._jax
        jnp = self._jnp
        n = int(self.prompt_ids.shape[0])
        # THE SHARED PREDICATE, not a second expression that means to agree with the
        # gate. ``None`` -> the last leg cannot run; the other three still do.
        shorter = second_prompt_length(n, self.page_size)

        ids = jnp.asarray(self.prompt_ids, dtype=jnp.int32)
        # BUILD THE MODEL FIRST, outside every counted region. This probe runs before
        # anything else touches ``self.model``, and constructing it is not free:
        # ``nnx`` initialises 99 parameters and ``apply_params`` casts each one, all
        # of which are eager ops that compile. Left inside the control leg they made
        # it read 33 instead of 1 -- a control that measures the model's CONSTRUCTION
        # cannot calibrate a count of its FORWARD.
        model = self.model

        # THE IMAGE, ONCE, FOR EVERY LEG BELOW. `Engine.prefill` takes it as
        # keywords and resolves the model's two hooks itself; the control leg
        # drives the model directly, so it resolves them here. Both are built
        # OUTSIDE every counted region -- see the next comment, and note that
        # `_built_positions` and `_prepare_model_inputs` are EAGER jnp work (a
        # tower run, for a multimodal port) that would otherwise land in the
        # control's count. Empty on a text-only family, where every call below
        # is byte for byte the one it has always been.
        image = self._image_inputs()

        # Leg 0: the CONTROL -- the same model on the same shapes, driven through the
        # forward THIS FILE jitted (``_jit_forward``). Everything but the call itself
        # is built OUTSIDE the counted region: the point is to measure one jitted
        # full-model forward and nothing else, so the metadata is assembled in numpy
        # and the cache allocated before the counter starts. Measured: exactly 1.
        pages = self._candidate_pages("compilation", self._default_pages())
        control_slots = np.asarray(
            [pages[p // self.page_size] * self.page_size + p % self.page_size
             for p in range(n)], dtype=np.int32)
        control_table = np.zeros(self.pages_per_seq, dtype=np.int32)
        control_table[: len(pages)] = np.asarray(pages, dtype=np.int32)
        jax.clear_caches()
        control_meta = self._metadata(
            positions=self._built_positions(model, ids, **image),
            block_tables=control_table,
            seq_lens=np.array([n], dtype=np.int32),
            query_start_loc=np.array([0, n], dtype=np.int32),
            request_distribution=np.array([0, 1, 0], dtype=np.int32),
        )
        control_slot = jnp.asarray(control_slots, dtype=jnp.int32)
        control_cache = self._empty_cache(self.native_dtype)
        control_kwargs = self._prepare_model_inputs(model, ids, **image)
        forward = self._jit_forward()
        with self._count_compilations() as counter:
            out = forward(model, ids, control_cache, control_meta,
                          control_slot, False, control_kwargs)
            jax.block_until_ready(out[0])
        control = counter.n
        control_names = list(counter.names)
        del control_cache, out
        gc.collect()

        # Legs 1-3: the CANDIDATE's own engine, on its own pool.
        jax.clear_caches()
        engine = self.engine_mod.Engine(model, self.run_config)
        with self._count_compilations() as counter:
            result = engine.prefill(ids, **image)
            jax.block_until_ready(result.logits)
        cold = counter.n
        # POINT OF USE: the engine allocated its own pages for every leg below.
        self._candidate_pages("compilation", result.page_ids)
        del result
        gc.collect()

        engine.reset()
        with self._count_compilations() as counter:
            repeat = engine.prefill(ids, **image)
            jax.block_until_ready(repeat.logits)
        same_shape = counter.n
        del repeat
        gc.collect()

        # Leg: THE DECODE LOOP, THE OTHER HALF OF THE CONTRACT. Every leg above
        # is a PREFILL, so without this leg "one compiled program per shape" covers
        # exactly half the contract -- instruction.md names both ("the graded loop
        # calls prefill once and then decodes step by step; EACH of those must reuse
        # a single compiled program rather than retracing per call"). Without this
        # leg, a decode that rebuilds its `nnx.jit` wrapper inside `decode_step` is
        # bit-identical to the reference and passes every other Spec check.
        #
        # ALWAYS RUNS: it needs no second prompt length, so it is computed here,
        # before the `shorter is None` branch below, and reported in both returns --
        # exactly the failure mode that made `same_shape` (above) a standalone check
        # instead of riding the geometry gate.
        #
        # Step 1 is the WARM-UP and is deliberately excluded from the count: the
        # decode shape (one new token against an already-primed cache) is new
        # relative to the prefill shape, so one compilation there is correct, not a
        # defect. Steps 2..N are steady state against an already-warm decode shape
        # and must cost zero.
        #
        # GUARDED BOTH WAYS -- see `steady_state_decode_tokens`, which owns the
        # predicate, the reason text and the minimum, and is where the two defects
        # this separates are written down. The guard is BELT AND BRACES: a golden
        # too short for this leg is refused by `_grade_precheck` before pytest
        # starts (`_harness_checks.check_scenario_carries_a_gradable_decode_run`),
        # so `reason is not None` is unreachable in a graded run and reachable from
        # a unit test.
        #
        # `None` is the unsupported shape, matching `decode_steps_probe`; and
        # `decode_steady_state_steps` is ALWAYS reported, so the check can assert
        # the measurement happened rather than assert on its absence.
        decode_tokens, decode_steady_state_reason = steady_state_decode_tokens(
            self.scenario)
        if decode_steady_state_reason is not None:
            decode_steady_state = None
            decode_steady_state_names = []
            decode_steady_state_steps = 0
        else:
            engine.reset()
            engine.prefill(ids, **image)
            warm = engine.decode_step(0, decode_tokens[0])
            jax.block_until_ready(warm.logits)
            counted = 0
            with self._count_compilations() as steady:
                for token in decode_tokens[1:4]:
                    step = engine.decode_step(0, token)
                    jax.block_until_ready(step.logits)
                    counted += 1
            decode_steady_state = int(steady.n)
            decode_steady_state_names = list(steady.names[:5])
            decode_steady_state_steps = counted
            del warm
            gc.collect()

        if shorter is None:
            # No second length in this geometry. The four legs above are the whole
            # answer; the check that grades the fifth (body) is deselected, so
            # returning a partial result here is a value the spec side reads, not a
            # failure it charges to the candidate.
            del engine
            gc.collect()
            return {
                "supported": True,
                "control": int(control), "control_names": control_names,
                "cold": int(cold), "same_shape": int(same_shape),
                "decode_steady_state": decode_steady_state,
                "decode_steady_state_names": decode_steady_state_names,
                "decode_steady_state_steps": decode_steady_state_steps,
                "decode_steady_state_reason": decode_steady_state_reason,
                "new_shape": None, "new_shape_names": [],
                "body": None, "body_names": [],
                "prompt_len": n, "other_prompt_len": None,
                "other_static_shapes": None,
            }

        # THE SECOND LENGTH KEEPS THE WHOLE IMAGE OR IT KEEPS NONE OF IT, and
        # that is a property of `second_prompt_length` plus the frozen prompt,
        # not something this leg can arrange. A truncation that landed INSIDE a
        # `<|image_pad|>` run would hand the position builder a placeholder
        # count its grid cannot account for, and the reference refuses that by
        # name ("grid ... merges to N tokens but the prompt reserves M
        # placeholders") rather than computing something wrong. On this
        # class's one multimodal instance, prompt 11, page_size 8, so
        # `shorter` is 7 and the four placeholders at offsets 1-4 are entirely
        # inside it. A future instance whose image sits at the tail would fail
        # here loudly; the fix would be the freeze's, not this leg's.
        short_ids = jnp.asarray(self.prompt_ids[:shorter], dtype=jnp.int32)
        engine.reset()
        with self._count_compilations() as counter:
            other = engine.prefill(short_ids, **image)
            jax.block_until_ready(other.logits)
        new_shape = counter.n
        new_shape_names = list(counter.names)
        # THE SECOND LENGTH'S METADATA SHAPES, READ HERE RATHER THAN DISCARDED.
        # `test_static_shape_bucketing` grades the bucket at the FROZEN prompt
        # length only, and on an instance where that prompt's page occupancy
        # happens to equal `pages_per_seq` the bucket and the occupancy are the
        # same number -- so a port that sizes its block table at `len(page_ids)`
        # produces the correct width there and three Spec checks agree with it.
        # On core_gpt_oss, for example, a prompt of 135 at page_size 32 occupies
        # 5 pages; where `pages_per_seq` also equals 5,
        # `test_static_shape_bucketing`, `test_block_table_contract` and
        # `test_attention_metadata_schema` all pass an occupancy-sized engine.
        # This leg catches that defect without depending on the instance geometry
        # avoiding the coincidence.
        #
        # This prefill ALREADY RAN, at a length whose occupancy is lower (4 of 6
        # here; 1 of 4 on the three page_size-8 instances), and its metadata was
        # thrown away one line below. Reading four integers off it costs no
        # forward, no compilation and no golden -- see
        # `test_spec.test_static_shape_recompilation`, which grades them.
        other_meta = other.attn_metadata
        short_shapes = {
            "prompt_len": int(shorter),
            "block_table_size": int(np.size(_to_np(other_meta.block_tables))),
            "seq_lens_size": int(np.size(_to_np(other_meta.seq_lens))),
            "query_start_loc_size": int(np.size(_to_np(other_meta.query_start_loc))),
            "padded_num_reqs": int(other_meta.padded_num_reqs),
        }
        del other, other_meta
        gc.collect()

        # THE SIGNAL, separated from the glue. ``new_shape`` above is dominated by
        # the engine's own eager bookkeeping at the new length: 15 of its
        # 16 compilations on a correct port are metadata primitives (iota, less, add,
        # select_n, gather) that a jitted and an unjitted engine both pay. Bounding
        # that total means bounding a number mostly made of noise.
        #
        # So the glue is warmed FIRST, at the new length, through the DRIVER's own
        # path (``_prefill_on_pages``, whose forward this file jitted): that compiles
        # the same ``BlockPool`` index programs, the same metadata arrays and the
        # same ``compute_logits`` at the same shapes, while the model BODY stays
        # inside one jitted program and its primitives are never compiled
        # individually. Whatever the engine then costs at that shape is the body.
        #
        # The residual is not exactly zero for a correct port -- the two paths build
        # positions slightly differently -- so this is a small measured bound, not a
        # claim of 1. What it is NOT is a bound on the glue.
        jax.clear_caches()
        engine_after = self.engine_mod.Engine(model, self.run_config)
        warm_logits, warm_cache, _wm, _ws = self._prefill_on_pages(
            self._candidate_pages("compilation", self._default_pages()),
            input_ids=self.prompt_ids[:shorter])
        jax.block_until_ready(warm_logits)
        del warm_logits, warm_cache
        gc.collect()
        with self._count_compilations() as counter:
            signal = engine_after.prefill(short_ids, **image)
            jax.block_until_ready(signal.logits)
        body = counter.n
        body_names = list(counter.names)
        del signal, engine_after, engine
        gc.collect()

        return {
            "supported": True,
            "control": int(control),
            "control_names": control_names,
            "cold": int(cold),
            "same_shape": int(same_shape),
            "decode_steady_state": decode_steady_state,
            "decode_steady_state_names": decode_steady_state_names,
            "decode_steady_state_steps": decode_steady_state_steps,
            "decode_steady_state_reason": decode_steady_state_reason,
            "new_shape": int(new_shape),
            "new_shape_names": new_shape_names,
            "body": int(body),
            "body_names": body_names,
            "prompt_len": n,
            "other_prompt_len": int(shorter),
            "other_static_shapes": short_shapes,
        }

    def traceability_probe(self) -> dict[str, Any]:
        """Does the frozen-scenario forward lower to a jaxpr with no host callback?

        THE FORWARD IT LOWERS IS THE ONE THE ENGINE JITS, which on a family with
        a host-side input hook means the already-prepared keywords go in too
        (:meth:`_prepare_model_inputs`). Lowering ``model(input_ids, ...)`` bare
        there would trace a graph the graded run never executes -- the embedding
        lookup instead of the merged table -- and this check's whole subject is
        what is INSIDE the traced region. The prepared arrays are closed over
        rather than passed as arguments, so they appear in the jaxpr as
        constants, which is what they are by the time the engine traces.
        """
        jax = self._jax
        meta = self._frozen_prefill_metadata("traceability")
        slot = self._jnp.asarray(self._frozen("traceability", "slot_mapping"),
                                 dtype=self._jnp.int32)
        cache = self._empty_cache(self.native_dtype)
        # `forward_ids`, not `prompt_ids`: this lowers THE FROZEN-SCENARIO
        # FORWARD, and `_frozen_prefill_metadata` above describes the whole
        # batch. Handing it sequence 0's tokens against a three-request metadata
        # would trace a shape the grade never runs.
        ids = self._jnp.asarray(self.forward_ids, dtype=self._jnp.int32)
        model = self.model
        # `_graded_image_inputs()`, NOT `_image_inputs()`: this lowers THE
        # GRADED FORWARD, and on an instance whose warm-up owns the picture the
        # graded ids carry no placeholders. See that method.
        model_kwargs = self._prepare_model_inputs(
            model, ids, **self._graded_image_inputs())

        def _forward(input_ids, kv_cache, attn_metadata, slot_mapping):
            return model(input_ids, kv_cache, attn_metadata, slot_mapping,
                         **model_kwargs)

        closed = jax.make_jaxpr(_forward)(ids, cache, meta, slot)
        callbacks = purity.jaxpr_uses_callbacks(closed)
        primitives = purity.count_jaxpr_primitives(closed)
        del cache
        gc.collect()
        return {
            "supported": True,
            "num_primitives": int(primitives),
            "callbacks": list(callbacks),
            "traced": bool(primitives > 0 and not callbacks),
        }

    # --- Numeric-tier probes ------------------------------------------------ #
    def prefill_probe(self, *, fp32: bool) -> dict[str, Any]:
        """``{logits, hidden_0..N}`` on the frozen scenario (frozen allocation)."""
        jnp = self._jnp
        model = self.fp32_model if fp32 else self.model
        dtype = jnp.float32 if fp32 else self.native_dtype
        want_hidden = self.compare_hidden_states
        # `kv_cache_state` rides this same forward (below), so it records through
        # the same read of the frozen arrays rather than declaring its policy
        # separately -- a probe that stopped honouring the frozen allocation must
        # not be able to leave a stale "frozen" behind for the artifact it wrote.
        probe = ("fp32_prefill",) if fp32 else ("prefill", "kv_cache_state")
        logits, hidden_states, cache, _meta, _slot = self._run_frozen_prefill(
            model, dtype, want_hidden=want_hidden, probe=probe)
        # THE DTYPE AND THE ARRAY TYPE ARE RECORDED BEFORE `_to_np`, WHICH DESTROYS
        # BOTH: it upcasts bf16 to fp32 for numpy and it coerces a `jax.Array` and an
        # `np.ndarray` identically. `test_prefill_logits` grades these two fields --
        # see `_to_np_with_provenance`.
        #
        # BOTH SIDES GO THROUGH `jnp.dtype(...).name`, exactly as
        # `paged_attention_unit_probe` does, and for the same reason: `dtype` here is
        # a jax scalar TYPE (``getattr(jnp, self.run_config.dtype)`` /
        # ``jnp.float32``), not a dtype instance, and `str(dtype)` on a type is
        # ``"<class 'jax.numpy.float32'>"`` -- never equal to the `str(x.dtype)` an
        # actual array reports. That mismatch already occurred in this
        # class's other dtype clause; this one does not repeat it.
        #
        # `logits_srctype` is UNEXERCISED, the same status its sibling
        # `out_srctype` carries in `paged_attention_unit_probe`, and for a
        # sharper reason: `test_prefill_logits` DOES assert on it, but no
        # mutation control makes that clause
        # fire. The numpy-leak mutation (`prefill_returns_numpy`) is caught by
        # `test_generate_runs` instead -- this probe never constructs an
        # `Engine`, so an `Engine.prefill` boundary leak is structurally
        # invisible here no matter what the clause says (see that row's note).
        # The clause is therefore a claim with no measured RED, not a proven
        # detector; a control that reaches this probe's own boundary would be
        # the thing that changes that.
        logits_np, logits_prov = _to_np_with_provenance(logits)
        out: dict[str, Any] = {
            "logits": logits_np,
            "logits_dtype": logits_prov["dtype"],
            "logits_srctype": logits_prov["type"],
            "requested_dtype": jnp.dtype(dtype).name,
        }
        if want_hidden and hidden_states is not None:
            # THE HIDDEN STATES CARRY PROVENANCE TOO, and the reason is a real
            # dtype escape. `logits_dtype` above compares two CANDIDATE-derived
            # values, so it detects a missing cast at the boundary, not the dtype
            # the forward actually RAN at. A port that drops layernorm's cast back
            # to the compute dtype -- promoting the entire forward to fp32 -- and
            # then narrows the returned logits at `compute_logits` passes every
            # check that reads only the boundary dtype. The A/B is unambiguous: the
            # leak ALONE fails `test_prefill_logits`; the leak plus one boundary
            # cast passes everything.
            #
            # The residual stream is where the leak is VISIBLE, because nothing
            # casts it: `PrefillResult.hidden` comes back float32 while
            # `RunConfig.dtype` says bfloat16. One boundary cast cannot hide a
            # leak that shows up on 25 arrays as well as one.
            #
            # `instruction.md` states the contract this grades in as many words:
            # "Do not hard-code a compute dtype. It comes from the run
            # configuration, and the model has to compute correctly in whatever
            # dtype it is handed." A forward that is fp32 REGARDLESS of the run
            # config has hard-coded it in substance, without writing a literal.
            hidden_dtypes: list[str] = []
            for i, layer in enumerate(hidden_states):
                layer_np, layer_prov = _to_np_with_provenance(layer)
                out[f"hidden_{i}"] = layer_np
                hidden_dtypes.append(layer_prov["dtype"])
            out["hidden_dtypes"] = hidden_dtypes
        if not fp32:
            # The cache after the frozen prefill write is a separate golden
            # (kv_cache_state.npz) but it is the SAME forward, so it is captured
            # here rather than paid for twice -- and it inherits that forward's
            # frozen allocation, which is what makes it comparable at all.
            # THE CONTAINER, group by group, in container order -- which is what
            # `kv_cache_state.npz` records (`cache` for one geometry,
            # `cache_group{i}` for several) and what `test_kv_cache_write`
            # compares member for member.
            self._kv_cache_state = self._cache_arrays(cache)
            self._native_prefill_cache = cache
        del logits, hidden_states
        gc.collect()
        return out

    def chunked_prefill_probe(self) -> dict[str, Any]:
        """N of the candidate's OWN ``engine.prefill`` calls, threading ONE cache.

        THE WHOLE PROBE, STATED PLAINLY: split ``self.prompt_ids`` -- sequence
        0's own full prompt, the array ``chunk_split_points`` is stated
        relative to -- at ``scenario.json::chunk_split_points``, and feed the
        pieces through ONE ``Engine`` instance in order, ``request_id`` fixed
        at 0 across every call. The FINAL chunk's logits are what
        ``test_chunked_prefill_logits`` grades against sequence 0's
        own rows of the one-shot golden -- a comparison a chunk boundary that
        dropped the preceding chunk's KV cannot survive.

        ``engine.reset()`` IS NEVER CALLED BETWEEN CHUNKS, and that omission is
        not an oversight to fix later -- threading one cache across the calls
        is the entire content of this probe. Calling ``reset()`` would turn the
        loop below into ``len(splits) + 1`` INDEPENDENT prefills, each
        attending to nothing before it, which is a different (and wrong)
        computation.

        ``USE_FROZEN_ALLOCATION["chunked_prefill"]`` is ``False``: like
        ``generation`` / ``fp32_generation``, this drives the candidate's OWN
        ``Engine``, whose ``BlockPool`` chooses the pages for every chunk.
        Handing it a frozen block table would mean re-implementing per-chunk
        allocation here, which would grade this file's arithmetic instead of
        the candidate's.

        UNSUPPORTED, WITHOUT SPENDING A FORWARD, in three cases nothing here
        could grade. **No split points**: every instance that shipped before
        ``chunked`` freezes ``chunk_split_points`` present and EMPTY,
        deliberately, so a reader cannot mistake "no chunking here" for "the
        freezer forgot" -- and this probe honours that distinction by costing
        those instances nothing rather than running a vacuous one-chunk
        "prefill" that nothing was ever going to grade (``test_chunked_
        prefill_logits`` is deselected on all of them by the mode gate
        regardless). **A split point outside ``forward_ids``**, and
        **a DESCENDING pair** (``splits[i] > splits[i + 1]``): both describe a
        golden that could not have been produced by chunking this array in
        order -- a descending pair in particular would make a later "chunk"
        overlap tokens an earlier one already forwarded, corrupting the
        sequence rather than merely wasting a call. All three return a
        ``reason`` a check can surface; none builds an ``Engine`` or a KV
        cache, which is what keeps the first two a true no-op -- in wall-clock,
        not only in the grade -- on every instance that has shipped so far.
        A FOURTH, added with the image plumbing below, sits in the same place
        for the same reason: a split that TEARS an image's placeholder run
        across two chunks.
        The harness asserts that no
        ``Engine`` is built for it, and that check caught
        this block one line BELOW the constructor, where it would have
        allocated a KV cache for a probe that returns nothing.

        A REPEATED SPLIT POINT (``splits[i] == splits[i + 1]``) IS NOT REFUSED.
        It describes a zero-width chunk -- a boundary declared twice -- and the
        loop below SKIPS it rather than issuing a call that would forward zero
        tokens. This is a deliberate design choice, not the only defensible one
        (refusing it outright would also be legitimate), and it is the one that
        makes ``n_chunks`` a claim capable of being WRONG: see the next
        paragraph.

        THE IMAGE IS FORWARDED HERE, AND WHICH CHUNK GETS IT IS COMPUTED
        RATHER THAN GUESSED. The offsets do not
        have to be FROZEN, because they are recoverable from the ids the probe
        already holds. :meth:`_image_placeholder_runs` finds them the same way
        the reference's own ``get_mrope_input_positions`` does -- each maximal
        run of ``image_token_id``, an attribute resolved BY NAME off the
        candidate's config, exactly as ``POSITIONS_HOOK`` and ``INPUTS_HOOK``
        are resolved by name elsewhere in this file.

        THE TWO KEYWORDS GO TO DIFFERENT PLACES, and that is the whole of the
        rule. On both reference frameworks:

          * ``image_grid_thw`` must reach EVERY chunk. The engine hands
            ``**model_kwargs`` to the POSITIONS builder as well as to the
            forward, and this family's builder needs the grid to place the run:
            withholding it from a continuation chunk raises "the prompt has 1
            run(s) of <|image_pad|> but 0 grid(s) were supplied".
          * ``pixel_values`` must reach EXACTLY ONE chunk -- the one whose slice
            contains the whole placeholder run. Passing it to a chunk with no
            placeholders, or to two chunks, raises "the prompt carries K
            <|image_pad|> ... but the tower produced R embeddings".

        A RUN TORN ACROSS TWO CHUNKS IS UNSUPPORTED, BY NAME, rather than
        forwarded and hoped for. Neither framework can answer it -- the tower
        emits R rows for the whole image and the chunk holds fewer placeholders
        than that -- so there is nothing for this probe to compare and saying so
        is the honest result. The freezer refuses to
        FREEZE such a split for the same reason, so a golden that would reach
        this branch cannot exist; the branch is here because a candidate's
        ``scenario.json`` is not this harness's to trust.

        ON A TEXT-ONLY INSTANCE NOTHING CHANGES, BYTE FOR BYTE.
        ``_image_inputs()`` returns ``{}`` there, so ``grid_only`` is ``{}``,
        ``owner`` is ``None``, and every ``prefill`` call below is spelled
        exactly as it was.

        ``n_chunks`` IS COUNTED FROM THE LOOP THAT RAN -- incremented once per
        chunk actually forwarded, never restated as ``len(splits) + 1`` a
        second time. This matters: on a strictly-increasing split list the
        loop always performs exactly ``len(splits) + 1`` iterations, so a
        mutant that computes ``n_chunks = len(splits) + 1`` after an otherwise
        untouched, correct loop is INDISTINGUISHABLE from the honest counter on
        every such input -- the two expressions denote the same integer by
        construction. The repeated-split-point skip above is what breaks that
        equivalence: ``chunk_split_points = [5, 5]`` makes the loop forward
        exactly 2 chunks while ``len(splits) + 1`` says 3, so it is the one
        input on which counting and restating can actually disagree, and it is
        what the harness exercises to prove the
        counter is real. A harness check refuses a probe whose ``n_chunks``
        does not equal ``len(splits) + 1``; that refusal is only evidence of
        something if the number it reads could actually have come out wrong.
        """
        jnp = self._jnp
        splits = [int(s) for s in (self.scenario.get("chunk_split_points") or [])]
        # `self.prompt_ids`, NOT `self.forward_ids`. `chunk_split_points` is
        # stated relative to SEQUENCE 0's OWN full prompt (instance.yaml's own
        # words: "chunk 1 fills slots 0-4... chunk 2 resumes at slot 5", always
        # counting from that request's token 0) -- `self.prompt_ids` is exactly
        # that array (the class docstring's own "TWO TOKEN ARRAYS" note, right
        # above `self.forward_ids`'s assignment). `self.forward_ids` is the
        # GRADED FORWARD's concatenated query tokens, which on ANY
        # `prefill_chunked` scenario is narrower than the full prompt (it
        # starts AFTER the final chunk boundary, by definition) and on a
        # scenario that also mixes in other requests (the
        # `shares_prefix_with` axis) is shorter still and holds tokens that are
        # not sequence 0's at all. Reading `self.forward_ids` would
        # split this instance's 11-token GRADED BATCH at `[0, 5, 11]` instead
        # of sequence 0's 12-token PROMPT at `[0, 5, 12]`, so the probe's
        # second "chunk" would be `forward_ids[5:11]` -- the tail two rows of
        # sequence 0's own chunk PLUS all of request 1's and request 2's rows
        # -- 6 tokens instead of sequence 0's honest 7, which reaches
        # `test_chunked_prefill_logits` as a plain shape mismatch (6 vs 7 rows).
        # UNEXERCISED ON EVERY INSTANCE BEFORE THIS ONE: every non-`chunked`
        # instance ships `chunk_split_points` empty, so the `if not splits`
        # guard below returns before either array is ever read for real.
        ids = np.asarray(self.prompt_ids).reshape(-1).astype(np.int32)
        n = int(ids.shape[0])
        if not splits:
            return {"supported": False,
                    "reason": "scenario.json's chunk_split_points is empty; "
                              "there is nothing to chunk"}
        out_of_range = [s for s in splits if not 0 < s < n]
        descending = any(splits[i] > splits[i + 1] for i in range(len(splits) - 1))
        if out_of_range or descending:
            return {"supported": False,
                    "reason": f"scenario.json's chunk_split_points {splits} "
                              f"must be non-decreasing and within (0, {n}); "
                              f"sequence 0's own prompt (prompt_ids) is {n} "
                              f"tokens long"}
        bounds = [0, *splits, n]
        # WHICH CHUNK OWNS THE IMAGE -- see the docstring. `image` is the whole
        # pair; `grid_only` is everything the POSITIONS builder needs and the
        # tower does not; `owner` is the index into `bounds` of the chunk whose
        # half-open slice contains every placeholder run, or `None` when there
        # is no image at all (every text-only instance).
        image = self._image_inputs()
        grid_only = {k: v for k, v in image.items() if k != "pixel_values"}
        owner = None
        if image:
            runs = self._image_placeholder_runs(ids)
            if not runs:
                return {"supported": False,
                        "reason": "the frozen scenario carries pixel_values but "
                                  "sequence 0's prompt has no image placeholder "
                                  "run to merge them onto, so no chunk can be "
                                  "handed the tower's rows"}
            span_lo = min(lo for lo, _ in runs)
            span_hi = max(hi for _, hi in runs)
            for i in range(len(bounds) - 1):
                if bounds[i] <= span_lo and span_hi <= bounds[i + 1]:
                    owner = i
                    break
            if owner is None:
                return {"supported": False,
                        "reason": f"chunk_split_points {splits} tears the "
                                  f"<|image_pad|> run [{span_lo}, {span_hi}) "
                                  f"across two chunks. The tower emits one row "
                                  f"per placeholder for the WHOLE image, so "
                                  f"neither chunk can be handed it -- the "
                                  f"reference refuses the forward by name and "
                                  f"there is nothing here to compare."}
        engine = self.engine_mod.Engine(self.model, self.run_config)
        # THE WHOLE RUN, HANDED TO EVERY CHUNK. `chunk` below is a SLICE; a
        # family whose positions are not `arange` builds them for the sequence
        # it is given, so a continuation chunk handed only itself would have its
        # run restart at 0 and lose the offset -- and the window taken out of
        # that too-short run comes back short or empty, reaching the forward as
        # a shape mismatch that names nothing. `engine.py`'s prefill refuses
        # that shape by name rather than computing it, so this is the call that
        # keeps the refusal unreachable.
        #
        # THE REFUSAL KEYS ON `continue_request`, NOT ON `cached > 0`.
        # `cached` is a prefix-cache OUTCOME, not a place in the run: a request's
        # FIRST chunk that hits an already-resident prefix has `cached > 0` while
        # its `input_ids` are the entire prompt, which is the case the engine's
        # own docstring calls "exactly right for a first chunk". Naming `cached`
        # here described a guard that would refuse that legitimate call.
        #
        # A NO-OP ON ALL 13 SHIPPED INSTANCES, and that is established rather than
        # assumed: only `core_qwen2_5_vl`'s model defines ``POSITIONS_HOOK``'s
        # method, and its frozen ``scenario.json`` carries
        # ``chunk_split_points: []`` (as do the other eight non-`chunked`
        # instances), so it returns unsupported above and never reaches this
        # loop -- the key is absent from ``instance.yaml``, and it is the
        # SCENARIO that this probe reads. Every instance that
        # DOES reach it is text-only, takes the engine's `arange` branch and
        # never reads this argument at all -- verified by driving the reference
        # engine both ways and comparing positions, logits, slot mappings and
        # cache state byte for byte.
        full_ids = jnp.asarray(ids, dtype=jnp.int32)
        result = None
        n_chunks = 0
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if lo == hi:
                # A repeated split point: zero tokens to forward, so there is
                # no call to make and nothing for `continue_request` to attach
                # to. Not counted -- see the docstring's `n_chunks` paragraph.
                continue
            chunk = jnp.asarray(ids[lo:hi], dtype=jnp.int32)
            # `request_id` fixed at 0 throughout. `continue_request` is keyed
            # on `n_chunks > 0` -- REAL chunks forwarded so far -- rather than
            # on the loop index `i`, so a skipped zero-width chunk earlier in
            # the loop cannot make the next real chunk wrongly claim to be the
            # first (by contract: `continue_request=False` on a live
            # request raises "already active").
            #
            # THE OWNING CHUNK GETS THE PIXELS; EVERY CHUNK GETS THE GRID. On a
            # text-only instance both dicts are empty and this is the call it
            # has always been.
            result = engine.prefill(chunk, request_id=0,
                                    continue_request=(n_chunks > 0),
                                    full_input_ids=full_ids,
                                    **(image if i == owner else grid_only))
            n_chunks += 1
        # POINT OF USE: `engine.prefill` allocated its own pages for every
        # chunk; nothing frozen entered this probe but the token ids.
        self._record_allocation("chunked_prefill", "candidate")
        logits_np, logits_prov = _to_np_with_provenance(result.logits)
        out = {
            "supported": True,
            "logits": logits_np,
            "logits_dtype": logits_prov["dtype"],
            "logits_srctype": logits_prov["type"],
            "chunk_split_points": splits,
            "n_chunks": n_chunks,
        }
        del engine, result
        gc.collect()
        return out

    def decode_steps_probe(self) -> dict[str, Any]:
        """Per-step logits from the frozen cache, at the frozen slots.

        The fed tokens come from ``scenario["decode_input_token_ids"]`` and NOT
        from the candidate's own argmax: the torch freezer fed its greedy tokens,
        so a candidate that diverges on step 0 would otherwise be compared on a
        different continuation and the mismatch would read as a decode bug at
        every later step. Without that key the probe reports unsupported and
        ``test_decode_step_logits`` fails with the reason, which is the loud
        version of the same fact.

        THE SLOT AND THE POSITION ARE TWO DIFFERENT NUMBERS, and this probe used
        to spell them as one. ``base_len + i`` is a TOKEN INDEX: it is where the
        step's KV lands in the flat cache and how long the sequence now is. The
        ROPE column is what the model rotates by, and the reference engine keeps
        it separately (``Engine._next_pos``, advanced by ``_advance``) for
        exactly this reason -- the two coincide for a text-only family and
        diverge for one with M-RoPE, where a VL prompt's last position is
        generally larger than its length. :meth:`_advanced_positions` is the
        rope column; ``pos`` below stays the token index.

        NO IMAGE KEYWORD APPEARS BELOW, and that is right rather than an
        oversight: the tower runs once at prefill and never in the decode loop
        (the reference's ``decode_step`` hands its jitted forward an empty
        ``model_kwargs``), so the only thing the image has to reach here is the
        position column -- and it reaches it through the FROZEN prefill
        positions, which is also what keeps this probe's ``use_frozen_
        allocation: True`` declaration honest.
        """
        jnp = self._jnp
        # Renamed at the boundary (fixtures.build_candidate): the ids are an INPUT
        # to this teacher-forced probe and are not the array any check grades.
        tokens = self.scenario.get("decode_input_token_ids")
        if not tokens:
            return {"supported": False,
                    "reason": "scenario.json carries no `decode_input_token_ids`; "
                              "the decode probe must be fed the reference's "
                              "tokens, not the candidate's own argmax"}
        tokens = [int(t) for t in tokens][: self.num_decode_steps]
        cache = getattr(self, "_native_prefill_cache", None)
        if cache is None:
            _l, _h, cache, _m, _s = self._run_frozen_prefill(
                self.model, self.native_dtype, want_hidden=False, probe="decode_steps")
        base_len = int(self.prompt_ids.shape[0])
        fwd = self._jit_forward()
        logits_per_step: list[np.ndarray] = []
        for i, token in enumerate(tokens):
            pos = base_len + i
            meta = self._metadata(
                positions=self._advanced_positions(i),
                block_tables=self._frozen("decode_steps", "block_tables"),
                seq_lens=np.array([pos + 1], dtype=np.int32),
                query_start_loc=np.array([0, 1], dtype=np.int32),
                request_distribution=np.array([1, 0, 0], dtype=np.int32),
            )
            slot = jnp.asarray([self._decode_slot(pos, "decode_steps")], dtype=jnp.int32)
            hidden, cache = fwd(self.model, jnp.asarray([token], dtype=jnp.int32),
                                cache, meta, slot, False, {})
            logits_per_step.append(_to_np(self.model.compute_logits(hidden))[0])
        del cache
        self._native_prefill_cache = None
        gc.collect()
        return {"supported": True, "num_steps": len(logits_per_step),
                "token_ids": tokens,
                **{f"logits_step_{i}": arr for i, arr in enumerate(logits_per_step)}}

    def paged_attention_unit_probe(self, *, fp32: bool) -> dict[str, Any]:
        """The isolated attention op on the frozen ``(q, k, v, cache, metadata)``.

        Deliberately EAGER: ``attention/paged.py``'s two preconditions read
        metadata values, so they are live here and skipped inside any trace. This
        is the one probe that exercises them.
        """
        jnp = self._jnp
        probe = "fp32_paged_attention_unit" if fp32 else "paged_attention_unit"
        units = self.scenario.get("units") or {}
        missing = [k for k in ("q", "k", "v") if k not in units]
        if missing:
            return {"supported": False,
                    "reason": f"golden/paged_attention_units.npz carries no "
                              f"{missing} input(s); the isolated attention probe "
                              "must run on the reference's own q/k/v"}
        dtype = jnp.float32 if fp32 else self.native_dtype
        q = jnp.asarray(np.asarray(units["q"], dtype=np.float32), dtype=dtype)
        k = jnp.asarray(np.asarray(units["k"], dtype=np.float32), dtype=dtype)
        v = jnp.asarray(np.asarray(units["v"], dtype=np.float32), dtype=dtype)
        num_kv_heads = int(k.shape[1])
        head_dim = int(q.shape[2])
        cache_layer = jnp.zeros(
            (self.num_pages, self.page_size, 2, num_kv_heads, head_dim), dtype=dtype)
        slot = jnp.asarray(self._frozen(probe, "slot_mapping"), dtype=jnp.int32)
        cache_layer = self.kv_cache_mod.write_kv(cache_layer, k, v, slot)
        meta = self._frozen_prefill_metadata(probe)
        # THE OP THE REFERENCE'S OWN LAYER RAN, keywords included. A family whose
        # attention takes `sinks` (a learned per-head logit in the softmax
        # denominator) or a `sliding_window` has them frozen into
        # `paged_attention_units.npz` beside q/k/v, because the frozen unit is
        # captured from a REAL layer and layer 0 of a windowed model is not the
        # default-argument op. Without this the two `paged_attention_op` checks
        # graded sink-free, window-free attention on a model that has both --
        # on core_gpt_oss, whose layer 0 is `sliding_attention` with
        # sinks. Absent keys mean the reference passed none, so `extra` is empty
        # and the call is exactly what it was for core_llama3.
        extra = {}
        if units.get("sinks") is not None:
            extra["sinks"] = jnp.asarray(
                np.asarray(units["sinks"], dtype=np.float32), dtype=dtype)
        if units.get("sliding_window") is not None:
            extra["sliding_window"] = int(np.asarray(units["sliding_window"]))
        out = self.paged_mod.paged_attention(
            q, cache_layer, meta, num_kv_heads=num_kv_heads,
            sm_scale=float(head_dim) ** -0.5,
            **extra,
        )
        # THE DTYPE IS RECORDED BEFORE `_to_np`, WHICH DESTROYS IT: that helper
        # upcasts bf16 to fp32 so numpy can hold it, so by the time the check sees
        # the array every tier reads `float32` and a native leg that ran at fp32 is
        # indistinguishable from one that did not. One line in
        # `attention/paged.py` (`return out` instead of `return out.astype(
        # query.dtype)`) makes the whole native forward run in fp32, and the suite
        # still passes because the relative tolerance admits the resulting error.
        # `test_numeric.test_paged_attention_op` grades this field.
        #
        # BOTH SIDES GO THROUGH `jnp.dtype(...).name`. `dtype` here is a jax scalar
        # TYPE, not a dtype -- `str(jnp.float32)` is "<class 'jax.numpy.float32'>",
        # which never equals "float32" and would have failed the reference.
        #
        # `out_srctype` added for consistency with `_to_np_with_provenance` (this
        # probe motivated that helper and is generalised through it below for the
        # array-type half too, not just the dtype half it already had). UNEXERCISED:
        # no test asserts it today -- this probe is EAGER (see the docstring) and is
        # the best-placed surface in the suite for a future `Engine`/probe boundary
        # numpy-leak check, but adding that check is outside this task's scope.
        out_np, out_prov = _to_np_with_provenance(out)
        result = {"supported": True, "out": out_np,
                  "out_dtype": jnp.dtype(out.dtype).name,
                  "out_srctype": out_prov["type"],
                  "requested_dtype": jnp.dtype(dtype).name,
                  "num_kv_heads": num_kv_heads, "head_dim": head_dim}
        del q, k, v, cache_layer, out
        gc.collect()
        return result

    def sampler_probe(self) -> dict[str, Any]:
        """The deterministic transform chain over the frozen ``(temp, k, p)`` grid.

        No RNG draw is taken or compared -- torch Philox and JAX threefry are
        different PRNG families. Grid entries are keyed by their
        STRING index so the shape survives a round trip through the golden's
        JSON, where integer keys do not exist.
        """
        jnp = self._jnp
        raw = self.scenario.get("sampler_logits")
        if raw is None:
            return {"supported": False,
                    "reason": "golden/sampler.npz carries no input `logits` (the "
                              "sampler golden is a json/npz pair -- the arrays are "
                              "in the npz half); the sampler transforms must be "
                              "graded on the reference's frozen logits, not on the "
                              "candidate's"}
        # The transforms are graded on the REFERENCE's frozen logits, never on
        # ones the candidate produced -- the same "handed, not derived" policy the
        # allocation-dependent probes follow, recorded the same way.
        self._record_allocation("sampler_outputs", "frozen")
        logits = jnp.asarray(np.asarray(raw, dtype=np.float32))
        if logits.ndim == 1:
            logits = logits[None, :]
        grid: dict[str, Any] = {}
        for index, entry in enumerate(self.sampler_grid):
            params = self.sampler_mod.SamplingParams(
                temperature=float(entry.get("temperature", 1.0)),
                top_k=int(entry.get("top_k", 0)),
                top_p=float(entry.get("top_p", 1.0)),
            )
            grid[str(index)] = _to_np(self.sampler_mod.process_logits(logits, params))
        values, indices = self.sampler_mod.topn_logprobs(logits, self.logprobs_top_n)
        return {
            "supported": True,
            "grid": grid,
            "grid_params": [dict(e) for e in self.sampler_grid],
            "topn_logprobs": _to_np(values),
            "topn_token_ids": _to_np(indices).astype(np.int64),
        }

    # --- Behavioral-tier probes --------------------------------------------- #
    def _generation_run_config(self, *, fp32: bool):
        """The ``RunConfig`` the generation leg runs under.

        Built the same way ``__init__`` builds the serving one -- through
        ``_run_config_kwargs`` out of ``repo_config`` -- rather than through
        ``dataclasses.replace``, so a candidate whose ``RunConfig`` is
        frozen, slotted or validating in ``__post_init__`` is constructed exactly
        as the graded path constructs it. The fp32 leg overrides ONE field.
        """
        fields = _run_config_kwargs(self.config_mod.RunConfig, self.repo_config)
        if fp32:
            fields["dtype"] = self.fp32_kv_cache_dtype
        return self.config_mod.RunConfig(**fields)

    def _expected_cache_shape(self) -> tuple[int, ...]:
        """The KV cache shape ``make_kv_cache`` is contracted to produce.

        ``[layers, pages, page_size, 2, kv_heads, head_dim]`` -- the class layout,
        already graded against ``golden/cache_spec.json`` by ``test_cache_spec``,
        so an engine holding an array of exactly this shape is holding a KV cache
        and nothing else in an ``Engine`` plausibly shares it.

        THE SHAPE OF THE ONE GROUP, not of the container. The cache is a tuple
        now, and every family shipped today puts one array in it whose shape is
        this. :meth:`_observed_cache_dtype`'s fallback scan already looks one
        level into a tuple, so it finds that array whether the engine holds the
        container, the array, or a list of both.
        """
        return (int(self.model_config.num_hidden_layers), self.num_pages,
                self.page_size, 2, int(self.model_config.num_key_value_heads),
                int(self.model_config.head_dim))

    def _observed_cache_dtype(self, engine) -> tuple[str | None, str | None]:
        """``(dtype, how it was found)`` for the cache the engine ACTUALLY holds.

        CALL THIS **AFTER** ``generate()``, never before. An engine that allocates
        its cache lazily on the first prefill -- a common and entirely legal design
        -- has nothing to report at construction time, and reading early reported
        ``None`` for it and failed a correct port.

        TWO LOOKUPS, IN ORDER, and neither invents a value:

        1. ``engine.kv_cache``, the reference's name and the one the failure
           message asks for.
        2. Failing that, a SHAPE-MATCHED scan of ``vars(engine)`` (one level into a
           list / tuple / dict value), for an array whose shape is exactly
           :meth:`_expected_cache_shape`. A port whose only deviation is calling
           its cache something else is a LEGAL port -- ``interface.json`` pins no
           ``Engine`` attribute -- and failing it would cost a port with
           bit-identical numerics its reward. That is a false negative against a
           correct implementation, which is worse than the hole it was guarding.

        IT STAYS FAIL-CLOSED, which is the whole point of the clause. ``None`` when
        nothing of that shape is found, and ``None`` when the matches disagree
        about dtype -- never the dtype this grader asked for. A fallback that
        answered with the requested value would walk straight back into the
        tautology this clause avoids: an engine that both
        hides its cache and allocates bf16 regardless of ``RunConfig.dtype`` still
        FAILS, because its one shape-matched array reports ``bfloat16``.
        """
        direct = None
        try:
            direct = engine.kv_cache
        except Exception:  # noqa: BLE001 - absence is a VALUE here, not a crash
            direct = None
        dtype = getattr(direct, "dtype", None)
        if dtype is None and isinstance(direct, (list, tuple)) and direct:
            # THE CACHE IS A CONTAINER, so the attribute holds a tuple of arrays
            # rather than one array. Its dtype is the dtype its groups AGREE on;
            # a container whose groups disagree, or that holds anything without a
            # dtype, answers None and falls through to the shape scan exactly as
            # an absent attribute does. Reading group 0 alone would report a
            # dtype the cache as a whole does not have.
            inner = {str(getattr(g, "dtype", None)) for g in direct}
            if len(inner) == 1 and "None" not in inner:
                return inner.pop(), "attribute"
        if dtype is not None:
            return str(dtype), "attribute"

        want = self._expected_cache_shape()

        def _candidates(value: Any):
            yield value
            if isinstance(value, (list, tuple)):
                yield from value
            elif isinstance(value, dict):
                yield from value.values()

        found: set[str] = set()
        try:
            members = list(vars(engine).values())
        except TypeError:  # an Engine with __slots__ and no __dict__
            members = []
        for member in members:
            for value in _candidates(member):
                shape = getattr(value, "shape", None)
                inner = getattr(value, "dtype", None)
                if inner is not None and shape is not None and tuple(shape) == want:
                    found.add(str(inner))
        # One answer or none. Two arrays of cache shape with DIFFERENT dtypes is
        # not something to pick from -- reporting either would be a guess.
        return (found.pop(), "shape_scan") if len(found) == 1 else (None, None)

    def generation_probe(self, *, fp32: bool) -> dict[str, Any]:
        """The candidate's own ``engine.generate`` loop, end to end.

        TWO LEGS, ONE LOOP. The native leg is the serving run. The fp32 leg is the
        same free-running decode on the upcast model with an **fp32 KV cache**, and
        it is what ``test_greedy_sequence_fp32`` / ``test_decode_trajectory`` grade
        against ``fp32_generation.npz`` -- an artifact the freezer produced under
        exactly that construction (``meta['fp32_kv_cache_dtype']``). See
        :attr:`_CandidateRuntime.fp32_kv_cache_dtype` for the cost this incurs when
        it is got wrong, and note that it is got wrong SILENTLY: the tokens still
        agree.

        ``kv_cache_dtype_observed`` comes from :meth:`_observed_cache_dtype`, read
        **after** ``generate()`` so a lazily-allocating engine has something to
        report. **It never falls back to what this grader asked for.** Falling back
        to ``run_config.dtype`` would be vacuous: this file builds
        ``run_config.dtype`` from the pin, so a consumer comparing it to the pin
        would compare the pin to itself. For example, an engine that renames its
        cache and hardcodes bf16 regardless of ``RunConfig.dtype`` would pass that
        comparison while diverging from the fp32 golden.

        ``kv_cache_dtype_requested`` is reported beside it -- the two answer
        different questions (what the grader asked for, what the engine did), and
        collapsing them into one field is what made the clause vacuous.
        ``kv_cache_dtype_source`` records WHICH lookup answered, so "the engine
        named it" and "we found it by shape" stay distinguishable.

        ``logits_per_step_srctypes`` is recorded PER STEP, not once, because step 0
        of ``GenerateResult.logits_per_step`` is not built the way the later steps
        are: ``Engine.generate`` seeds its loop with
        ``self.prefill(input_ids, request_id).logits[-1]`` and appends that value to
        ``per_step`` UNCONVERTED, before any cast; every later step instead comes
        from ``decode_step(...).logits[0]``. So a numpy leak confined to
        ``Engine.prefill``'s return crosses this API boundary at step 0 only, and
        ``tokens``/``step_logprobs`` cannot see it either way -- ``GenerateResult``
        rebuilds both of those with its own ``jnp.array(...)`` call, which launders
        any leak on the way in. ``logits_per_step`` is the one field ``generate()``
        returns unconverted, which is what makes it the reachable surface for this
        check -- see ``test_generate_runs``.

        ``**self._image_inputs()`` is what makes the serving run the SAME run
        the golden was frozen from on a multimodal family: ``Engine.generate``
        forwards its extra keywords verbatim to ``prefill`` and to nothing else,
        which is also the reason no image reaches the decode loop. Empty on a
        text-only family, where this call is byte for byte what it was.
        """
        jnp = self._jnp
        probe = "fp32_generation" if fp32 else "generation"
        model = self.fp32_model if fp32 else self.model
        run_config = self._generation_run_config(fp32=fp32)
        engine = self.engine_mod.Engine(model, run_config)
        result = engine.generate(jnp.asarray(self.prompt_ids, dtype=jnp.int32),
                                 num_steps=self.num_decode_steps,
                                 **self._image_inputs())
        # AFTER `generate`, deliberately -- see `_observed_cache_dtype`. An engine
        # that allocates its cache on the first prefill has nothing to show before
        # this line, and reading early failed it for being lazy.
        observed, source = self._observed_cache_dtype(engine)
        # POINT OF USE: `generate` allocated its own pages and drove its own
        # decode loop; nothing frozen entered it but the prompt.
        self._record_allocation(probe, "candidate")
        per_step = [_to_np_with_provenance(x) for x in (result.logits_per_step or [])]
        out = {
            "supported": True,
            "kv_cache_dtype_requested": str(run_config.dtype),
            "kv_cache_dtype_observed": observed,
            "kv_cache_dtype_source": source,
            "kv_cache_shape_looked_for": list(self._expected_cache_shape()),
            "tokens": _to_np(result.tokens).astype(np.int64),
            "step_logprobs": _to_np(result.step_logprobs).astype(np.float32),
            "logits_per_step": [arr for arr, _ in per_step],
            "logits_per_step_srctypes": [prov["type"] for _, prov in per_step],
        }
        del engine, result
        gc.collect()
        return out

    # --- perf: the candidate's own decode-timing probe ------------------------ #
    def _perf_round(
        self, k_values: list[int]
    ) -> tuple[list[tuple[int, float]], np.ndarray | None]:
        """One K sweep -- the raw ``(K, seconds)`` pairs one round of
        :meth:`perf_probe` reports.

        Split out of ``perf_probe`` so its round loop can call this repeatedly
        without duplicating the sweep. See ``perf_probe``'s docstring for the
        round loop and the throttle check.

        ONE ``Engine`` PER ROUND, BUILT AND WARMED **OUTSIDE** THE TIMED REGION.
        Building a fresh ``Engine`` for every TIMED call
        puts a full XLA compile inside the clock -- a large, K-INDEPENDENT
        cost that swamps the small K-span signal this probe exists to see.
        Carrying
        compiled executables across K is exactly what we want: a compile is
        K-INDEPENDENT, so it belongs in the fitted intercept, and cancelling a
        K-independent constant via the intercept is the entire reason
        ``fit_step_time`` fits a slope instead of timing one call. So: one
        ``Engine`` is built once per round, warmed with one untimed
        ``generate()`` (this is where the compile actually lands), and
        construction and tracing sit OUTSIDE every timed K leg below.

        ``engine.reset()`` BETWEEN LEGS, NOT A NEW ENGINE, closes the
        LEGITIMATE half of the old argument: a leg must not inherit the
        previous leg's warm KV pages, or leg N+1 is timing "decode N+1 steps on
        top of a live N-token history" instead of "prefill this prompt, then
        decode K steps" -- a different, growing amount of work each leg, which
        would put a spurious K-dependent term of its own into the slope.
        ``reset()`` is graded, pinned interface (``interface.json``'s
        ``engine.py::Engine.reset``, zero args) -- every candidate implements
        it, so this is safe to call on a port, not just the reference -- and it
        clears page/request state and zeros the KV cache without discarding the
        compiled executables ``self._jit_fwd`` (or a port's equivalent) holds.
        The call to ``reset()`` sits outside the timed region, matching
        construction, but its effect is NOT necessarily complete before the
        clock starts on the NEXT leg: ``jnp.zeros_like`` dispatches
        asynchronously and nothing here blocks on it, so a sub-millisecond
        K-independent tail from leg N's reset can land inside leg N+1's timed
        window. Harmless for the same reason a compile landing in round 0 is:
        it is constant across every leg, so it is absorbed by the intercept,
        not the slope -- the fit needs the *slope* to be free of it, not every
        nanosecond of the clock.

        ``tokens`` is the K == ``num_decode_steps`` leg's sampled ids, and it is
        the work gate's subject: without it the fastest port is the one that
        decodes nothing.
        """
        jax = self._jax
        jnp = self._jnp
        run_config = self._generation_run_config(fp32=False)
        prompt = jnp.asarray(self.prompt_ids, dtype=jnp.int32)
        # THE IMAGE IS BUILT ONCE, OUTSIDE EVERY TIMED LEG, for the same reason
        # the engine is: `_image_inputs` does eager device work (two casts) and
        # it is K-INDEPENDENT, so it belongs nowhere near the clock. What it
        # feeds -- a host-side tower run inside each `generate`'s prefill -- IS
        # inside the clock and is also K-independent, so it lands in the fitted
        # intercept exactly as the prefill it is part of already does. Empty on
        # a text-only family, where every call below is unchanged.
        image = self._image_inputs()
        calls: list[tuple[int, float]] = []
        tokens: np.ndarray | None = None
        engine = self.engine_mod.Engine(self.model, run_config)
        # The warm-up: a full, untimed generate() at the sweep's own smallest K.
        # This is where the prefill trace and the decode-step trace compile --
        # once, for the whole round -- because neither trace's shape depends on
        # num_steps (a Python loop calls the same compiled per-step function
        # `num_steps` times; it does not re-trace per K). `block_until_ready`
        # here too: the reset() immediately after must not race a still-running
        # dispatch from this call.
        warm = engine.generate(prompt, num_steps=int(k_values[0]), **image)
        jax.block_until_ready(warm.tokens)
        del warm
        engine.reset()
        for k in list(k_values):
            started = time.perf_counter()
            result = engine.generate(prompt, num_steps=k, **image)
            # Block before stopping the clock. jax dispatch is asynchronous, and a
            # timer that stops at dispatch measures the host loop, not the device.
            jax.block_until_ready(result.tokens)
            calls.append((int(k), time.perf_counter() - started))
            if int(k) == int(self.num_decode_steps):
                tokens = _to_np(result.tokens).astype(np.int64)
            del result
            # OUTSIDE the timed region, matching construction and the warm-up
            # above: the next leg must start from the same empty state this one
            # did, not from this leg's now-live pages.
            engine.reset()
        del engine
        gc.collect()
        return calls, tokens

    def perf_probe(self, *, k_values: list[int]) -> dict[str, Any]:
        """Time the candidate's own decode loop at several step counts, over
        ``self.perf_rounds`` rounds.

        RETURNS RAW ``(K, seconds)`` PAIRS AND NOTHING ELSE, GROUPED BY ROUND.
        Every reduction -- the fit, the median, the quorum -- happens
        parent-side, for the reason jaxopt's ``stats.py`` header
        gives: a candidate must not be able to compute any number reported
        about itself. This method is inside the sandbox with the port.

        OWNS ITS OWN ROUNDS AND ITS OWN THROTTLE CHECK, DELIBERATELY.
        ``Candidate._ensure_run`` spawns ONE subprocess that runs every probe
        and returns one batched result; there is no mechanism for the parent to
        invoke this probe once per round, inspect the result, and decide
        whether to run another. So the ``perf_rounds`` rounds
        all happen HERE, in one call, and so does the throttle check
        BETWEEN them: :func:`_t2j.perf.nvml.throttle_reasons` is
        read before and after each round's sweep, and a round that saw ANY
        throttle reason is marked ``discarded`` and retried -- up to
        :data:`PERF_ROUND_ATTEMPT_MULTIPLIER` times ``perf_rounds`` attempts --
        rather than counted toward the target. Every attempt is returned,
        discarded or not, so a discard is diagnosable from the artifact instead
        of silently vanishing.

        Absent K values report unsupported rather than running a zero-length
        sweep: an instance whose ``perf_k_values`` is empty has nothing to time,
        and a probe that silently returned an empty ``rounds`` list would read,
        downstream, as a round that ran and discarded nothing.
        """
        if not k_values:
            return {"supported": False,
                    "reason": "no perf_k_values on this instance"}
        max_attempts = max(int(self.perf_rounds), 1) * PERF_ROUND_ATTEMPT_MULTIPLIER
        rounds: list[dict[str, Any]] = []
        tokens: np.ndarray | None = None
        good = 0
        attempts = 0
        while good < self.perf_rounds and attempts < max_attempts:
            attempts += 1
            before = nvml.throttle_reasons()
            calls, round_tokens = self._perf_round(k_values)
            after = nvml.throttle_reasons()
            throttle = sorted(set(before) | set(after))
            discarded = bool(throttle)
            rounds.append({
                "round": len(rounds),
                "discarded": discarded,
                "throttle": throttle,
                "calls": calls,
            })
            if round_tokens is not None:
                tokens = round_tokens
            if not discarded:
                good += 1
        # POINT OF USE: every round's `Engine` allocates and resets its own
        # pages (see `_perf_round`); nothing frozen entered this probe but the
        # prompt -- see USE_FROZEN_ALLOCATION["perf"]'s entry for why there is
        # no seam.
        self._record_allocation("perf", "candidate")
        return {"supported": True, "rounds": rounds, "tokens": tokens}

    # --- the candidate's own entry point (serve.py) -------------------------- #
    def _serve_config(self, **overrides: Any) -> dict[str, Any]:
        """The config dict ``run_serve`` is driven by, at this instance's geometry.

        ONE BUILDER FOR BOTH PROBES, so the L5 and L6 legs cannot drift apart on
        anything but what L6 deliberately overrides (the two dtypes and the
        workload). Everything else -- the repo id, the paging geometry, the seed,
        the step count -- is the same dict.
        """
        config: dict[str, Any] = {
            "model_name_or_path": self.repo_config["model_name_or_path"],
            **_serve_run_fields(self.repo_config),
            # THE STEP COUNT THE PROBE ACTUALLY DRIVES, overriding repo_config's
            # copy of it. `self.num_decode_steps` is `meta['num_decode_steps']`
            # falling back to the RunConfig field, and `test_serve_end_to_end`
            # compares serve's token array to the by-hand one ELEMENTWISE -- so an
            # instance whose two sources of that number disagreed would fail the
            # check on a length mismatch that is entirely the golden's doing.
            "num_decode_steps": self.num_decode_steps,
        }
        config.update(overrides)
        return config

    def serve_probe(self) -> dict[str, Any]:
        """The candidate's own serve.py, driven ONLY by a config dict.

        SELF-CONSISTENCY, NOT GOLDEN PARITY. ``wired_tokens`` is this runtime
        driving ``Engine`` by hand from the same geometry, in the same process, at
        the same dtype, on the same device. Same code, same framework, so the
        comparison is deterministic and EXACT: any difference is a wiring
        difference, never a precision one. Do not import
        ``native_thresholds.decisive_margin`` in the check that reads this -- that
        margin excuses a CROSS-framework argmax flip in ``test_greedy_sequence``,
        and there is no cross-framework gap to excuse in this comparison.
        Importing it would let a genuinely mis-wired serve.py through.

        THE BY-HAND LEG USES ``self.model`` -- THE MODEL THIS RUNTIME ALREADY
        BUILT AND LOADED -- AND MUST NOT BE "SIMPLIFIED" INTO CALLING serve.py's
        OWN ``_build``. That asymmetry is the entire discriminating power of this
        probe. ``serve.py::_build`` is where ``apply_params(model,
        convert_checkpoint(...))`` lives, and a port that drops that one call
        builds a correctly-shaped model on RANDOM nnx-initialised weights. With
        the legs as written, serve's tokens come off random weights and the
        by-hand tokens come off the checkpoint, so they disagree and the check
        fails. Route both legs through ``_build`` and both hold the SAME random
        weights, both produce the same garbage tokens, and the mutation passes
        this check while every other tier stays green (nothing else in the suite
        ever calls serve.py).

        IMPORTED HERE, NOT IN ``__init__``, for the same reason: a candidate that
        never wrote ``serve.py`` must fail the ONE check that reads it, not the
        whole grade. An import at construction time is common-mode (see
        ``run_all``) and would charge every check for one absent module.
        (``cand.serve()`` has exactly one
        graded call site, ``test_behavioral.py``'s ``test_serve_end_to_end``.
        ``held_out_probe`` also drives ``serve.py``, but through its own import
        and its own probe key, so an absent module costs that check separately
        rather than making this one two.)

        AND ``run_serve``'s RETURN IS COERCED BEFORE IT IS STORED, for the third
        instance of that same argument -- this time about the PICKLE rather than
        about an import. ``run_all``'s per-probe capture cannot reach a value
        that only fails on the way across the pipe, and this was the one probe
        handing that pipe a verbatim candidate object graph. See
        :func:`_json_safe` for the measured collapse and for what the coercion
        does and does not repair.
        """
        import hashlib    # noqa: PLC0415 - worker-only
        import importlib  # noqa: PLC0415 - see the docstring: NOT at construction
        import tempfile   # noqa: PLC0415 - worker-only; the parent stays stdlib-thin

        jnp = self._jnp
        serve_mod = importlib.import_module("serve")
        # THE IMAGE IS HANDED OVER AS A PATH, and that is the whole point:
        # serve.py opens, decodes and patchifies it ITSELF, exactly as it
        # tokenizes a raw prompt string. Passing `pixel_values` in would move that
        # computation out of the port and leave the by-hand leg comparing against
        # an answer this grader supplied. `test_serve_end_to_end` requires the
        # path and its sha256 from the frozen `multimodal` fact, so a probe that
        # quietly stopped doing this reads as a failure rather than as coverage.
        #
        # `images` IS SET WHENEVER THE GOLDEN HAS A PNG, even if this port ships
        # no `image_processing` module. serve.py's own processor is what is under
        # test; a port without one then fails the by-hand leg below, which
        # attributes the defect to the port rather than to the harness.
        png = (self.scenario.get("image") or {}).get("png_bytes")
        loaded = self._image_processor() if png else None
        image_path = image_sha = None
        image_kwargs: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="t2j_serve_") as out_dir:
            # A REAL DIRECTORY, cleaned up after: `run_serve` writes
            # `serve_results.json` into it, and serialization is part of what this
            # probe puts on the graded path. The value compared is the RETURNED
            # dict, so nothing downstream depends on the file surviving.
            overrides: dict[str, Any] = {}
            if png:
                written = Path(out_dir) / "image_fixture.png"
                written.write_bytes(bytes(png))
                image_path, overrides["images"] = str(written), [str(written)]
                image_sha = hashlib.sha256(bytes(png)).hexdigest()
            if loaded is not None:
                module, config_ip = loaded
                pixel_values, grid = module.preprocess(bytes(png), config_ip)
                image_kwargs = {"pixel_values": pixel_values,
                                "image_grid_thw": grid}
            # INSIDE the `with`, and it has to be: the PNG lives in this same
            # temp dir and `run_serve` is what opens it.
            #
            # COERCED AT THE CALL SITE, not on the way into `out` below, and the
            # placement is the decision. See `_json_safe` for the measurement:
            # this is the ONE probe that would otherwise put an unfiltered
            # candidate object graph into the single cross-sandbox pickle, and
            # everything downstream of this line -- the `config` echo, the
            # `generations[0]` guard, the tokenizer decode -- then reads plain
            # types too, so no reader of this payload has to know which side of
            # the coercion it is on. `serve_results.json` is written by
            # `run_serve` itself, so nothing on disk changes.
            results = _json_safe(serve_mod.run_serve(self._serve_config(
                prompt_token_ids=[[int(t) for t in self.prompt_ids]],
                do_generate=True, output_dir=out_dir, **overrides)))
        gc.collect()

        # The by-hand leg: a FRESH Engine on the already-built model, so the only
        # thing under test is serve.py's own wiring. `_generation_run_config(
        # fp32=False)` is the same RunConfig the native `generation` probe runs
        # under, which is also what serve.py's `run_config(dtype=kv_cache_dtype)`
        # resolves to when the config names no `kv_cache_dtype` (it defaults to
        # the compute dtype in `get_serve_args`).
        engine = self.engine_mod.Engine(
            self.model, self._generation_run_config(fp32=False))
        # `**image_kwargs` is what makes the by-hand leg comparable on a
        # multimodal family: `Engine.generate` forwards them verbatim to
        # `prefill`. Empty on a text-only one, where this call is byte-for-byte
        # what it has always been. THE PIXELS COME FROM THE PORT'S OWN PROCESSOR,
        # not from `image_preprocess.npz`: both legs must run the same port, or
        # the comparison would credit serve.py for a processor this file supplied.
        ref = engine.generate(jnp.asarray(self.prompt_ids, dtype=jnp.int32),
                              num_steps=self.num_decode_steps, **image_kwargs)
        # POINT OF USE: both legs allocated their own pages -- serve.py through
        # its own Engine, the by-hand leg through this one. Nothing frozen entered
        # either; see USE_FROZEN_ALLOCATION's entry for why there is no seam.
        self._record_allocation("serve", "candidate")
        config = results.get("config") or {}

        # THE EXPECTED DECODE, computed here because the check has no tokenizer:
        # `tests/` never imports the candidate's modules, and this runtime already
        # holds both handles (`self.loader_mod`, `self.model_dir` -- the same pair
        # `loader_facts_probe` reports on).
        #
        # THE IDS FED IN ARE THE ONES SERVE ITSELF RETURNED, not the by-hand leg's,
        # and that keeps this clause INDEPENDENT of the token-equality clause
        # beside it. A serve.py with wrong tokens but an honest decode of them
        # fails on tokens alone; one with right tokens and a fabricated text fails
        # on text alone. Feeding `ref.tokens` in instead would make the text clause
        # a second reading of the token clause and it would stop grading the
        # tokenizer at all.
        #
        # A DECODE COMPARISON, NEVER A DECODE->ENCODE ROUND TRIP. BPE
        # normalisation makes `encode(decode(ids)) == ids` false for legitimate
        # inputs (whitespace and NFC folding both move ids), so asserting the round
        # trip would false-fail correct ports -- the defect class the `_RUN_FIELDS`
        # repair just removed. Both sides here are the SAME call on the SAME ids,
        # so the comparison cannot false-fail; it can only catch a `text` that did
        # not come from this checkpoint's tokenizer.
        tokenizer = self.loader_mod.load_tokenizer(self.model_dir)
        generations = results.get("generations")
        served_tokens = None
        if isinstance(generations, list) and generations and isinstance(
                generations[0], dict):
            served_tokens = generations[0].get("tokens")
        expected_text = None
        if tokenizer is not None and served_tokens is not None:
            expected_text = tokenizer.decode([int(t) for t in served_tokens])

        out = {
            "supported": True,
            "results": results,
            "wired_tokens": _to_np(ref.tokens).astype(np.int64),
            # See the block above. `has_tokenizer` separates "the decode
            # disagreed" from "there was nothing to decode with", which are
            # different failures and must not share a message.
            "expected_text": expected_text,
            "has_tokenizer": tokenizer is not None,
            # THE RESOLVED CONFIG, NOT THE ONE WE SENT -- the same read
            # the freezer makes off the reference. `run_serve`
            # echoes what `get_serve_args` settled on, so a serve.py that
            # substituted a dtype (or defaulted `kv_cache_dtype` to something
            # other than the compute dtype) is visible here rather than as
            # numbers nothing can explain.
            "resolved_dtype": str(config.get("dtype")),
            "resolved_kv_cache_dtype": str(config.get("kv_cache_dtype")),
            # THE IMAGE THIS PROBE ACTUALLY FED IN, reported so the check can
            # assert it rather than assume it. See `test_behavioral
            # ._SERVE_IMAGE_PATH_KEY` / `_SERVE_IMAGE_SHA_KEY`: both legs of the
            # token comparison above would agree with each other on ANY image, so
            # without this pair the multimodal leg would read as coverage of a
            # picture nothing else has seen. `None` on a text-only family, where
            # that clause is deselected.
            "serve_image_path": image_path,
            "serve_image_sha256": image_sha,
        }
        del engine, ref
        gc.collect()
        return out

    def held_out_probe(self) -> dict[str, Any]:
        """Teacher-forced logprobs over the held-out set, fp32 (L6).

        Driven through serve.py's SCORE mode, not through ``Engine`` directly:
        L6's whole distinction from L3 is that it grades the candidate's own
        top-level entry point over a set that moves every serving axis at once
        (eight prompt lengths straddling the page boundary from both sides, four
        continuation lengths, both entropy regimes) rather than the one canonical
        12-token scenario every other artifact is frozen from.

        BOTH DTYPES ARE FP32 -- the compute one AND the cache. Running an fp32
        model against a bf16 serving cache diverges sharply from the artifact
        with every token still agreeing and every other check still green.
        ``kv_cache_dtype`` is
        therefore passed EXPLICITLY from ``meta['fp32_kv_cache_dtype']`` -- the pin
        the freezer stamped off the config its own fp32 window built -- and never
        left to default to the compute dtype at this call site.

        NOTHING FROM ``ref_logprobs`` IS REACHABLE FROM HERE. The payload carries
        the four INPUT arrays only (:data:`HELD_OUT_INPUT_KEYS`); the frozen
        answer stays in the grader's process, where the comparison happens.

        THE RETURNED GRID IS LAID OUT LIKE THE ARTIFACT: ``(P, C_max)`` float32,
        right-padded with zeros, row ``i`` valid on ``[:cont_lens[i]]``. The check
        masks past the length before comparing, so the padding is never graded --
        but it must still be zeros and it must still be this shape, because a
        shape mismatch is the one thing that check reports before it can mask
        anything.
        """
        import importlib  # noqa: PLC0415 - see serve_probe: NOT at construction
        import tempfile   # noqa: PLC0415

        spec = self.held_out_spec or {}
        absent = [key for key in HELD_OUT_INPUT_KEYS if key not in spec]
        if absent:
            return {
                "supported": False,
                "reason": f"the grader supplied no held-out {absent} to score, so "
                          "there is nothing for L6 to measure. golden/held_out.npz "
                          "is a REQUIRED artifact (the freezer "
                          "writes it); this is a harness/golden gap, not a port "
                          "defect.",
            }
        prompt_rows = np.asarray(spec["prompt_ids"]).tolist()
        prompt_lens = [int(n) for n in np.asarray(spec["prompt_lens"]).reshape(-1)]
        cont_ids = np.asarray(spec["cont_ids"])
        cont_rows = cont_ids.tolist()
        cont_lens = [int(n) for n in np.asarray(spec["cont_lens"]).reshape(-1)]
        # THE LENGTHS ARE FROZEN BESIDE THE IDS BECAUSE 0 IS A LEGAL TOKEN ID:
        # the padding is invisible in `prompt_ids` / `cont_ids` and can only be
        # trimmed by reading the length arrays.
        prompts = [[int(t) for t in row[:n]] for row, n in zip(prompt_rows, prompt_lens)]
        conts = [[int(t) for t in row[:n]] for row, n in zip(cont_rows, cont_lens)]

        serve_mod = importlib.import_module("serve")
        with tempfile.TemporaryDirectory(prefix="t2j_heldout_") as out_dir:
            results = serve_mod.run_serve(self._serve_config(
                dtype="float32", kv_cache_dtype=self.fp32_kv_cache_dtype,
                prompt_token_ids=prompts, continuations=conts,
                do_generate=False, do_score=True, output_dir=out_dir))
        gc.collect()

        scores = results.get("scores") or []
        if len(scores) != len(prompts):
            return {
                "supported": False,
                "reason": f"serve.py returned {len(scores)} score record(s) for "
                          f"{len(prompts)} scored prompt(s); do_score pairs them "
                          "1:1 and in request order.",
            }
        width = int(cont_ids.shape[1])
        logprobs = np.zeros((len(prompts), width), np.float32)
        for i, score in enumerate(scores):
            # POSITIONAL, then CHECKED -- the same three reads
            # the freezer makes on the reference side, so a
            # reordered or mis-paired result set is reported as itself rather
            # than arriving as a diffuse numeric failure the check would attribute
            # to precision.
            if int(score.get("index", i)) != i:
                return {
                    "supported": False,
                    "reason": f"score record {i} reports index "
                              f"{score.get('index')}; the records must come back "
                              "in request order.",
                }
            values = [float(x) for x in score["logprobs"]]
            if len(values) != len(conts[i]):
                return {
                    "supported": False,
                    "reason": f"entry {i}: {len(values)} log-prob(s) for a "
                              f"{len(conts[i])}-token continuation; teacher-forced "
                              "scoring produces exactly one per token.",
                }
            logprobs[i, :len(values)] = values
        # POINT OF USE: serve.py allocated every page itself, for all P requests.
        self._record_allocation("held_out", "candidate")
        gc.collect()
        return {"supported": True, "logprobs": logprobs}

    def invariance_probes(self) -> dict[str, Any]:
        """Every golden-free self-consistency property, keyed by kind.

        Each kind is produced by its own method so a failure names the property
        that failed rather than "the invariance probe". A kind that this mode
        cannot express answers with an unsupported marker -- it is still a KEY,
        because a missing key is the silent failure this whole file is arranged
        against.

        AND THE CAPTURE IS PER KIND, which is what makes the sentence above true
        of the CODE and not only of the layout. ``run_all`` registers this whole
        method as ONE ``produce("invariance", ...)``, so its capture is per PROBE:
        before this loop caught anything, the first kind to raise took the other
        eight with it and ``Candidate.invariance`` reported "the candidate's
        'invariance' probe raised" for every one of them.

        Both shapes:

        * ``mode="core"`` with only ``_invariance_determinism`` raising -- the
          whole probe raised, so all four graded core invariance checks
          (``test_paging_invariance``, ``test_decode_prefill_consistency``,
          ``test_cache_reuse_invariance``, ``test_determinism``) failed on one
          kind's fault, and the five kinds they read collapsed to one traceback;
        * ``mode="ragged"`` with the real methods bound -- the whole probe raised
          ``NotImplementedError`` from ``_invariance_batching``, because the four
          I2 kinds are short-circuited ONLY by ``if self.mode == "core"``. On the
          first ragged instance that is all six applicable invariance checks
          failing for EVERY candidate, including the four core ones that are
          fully implemented, before any port is graded.

        The capture is the same value shape ``run_all.produce`` uses -- a dict
        carrying ``__probe_error__`` -- so ``Candidate.invariance`` converts it
        back into an exception at the point a check READS that kind, and nothing
        passes vacuously on a swallowed failure.

        ``BaseException``, matching ``produce`` and for its stated reason: this
        runs inside the candidate's own worker, where a ``SystemExit`` out of a
        probe is something the candidate's code did, and charging every kind for
        it would be the misattribution this whole method is about.
        """
        out: dict[str, Any] = {}
        for kind in sorted(INVARIANCE_KINDS):
            if kind in _MULTI_SEQUENCE_KINDS and self.mode == "core":
                out[kind] = {
                    "supported": False,
                    "reason": f"the `core` mode runs one sequence at a time, so "
                              f"the {kind!r} invariant has no second sequence to "
                              "state it over",
                }
                continue
            try:
                out[kind] = getattr(self, f"_invariance_{kind}")()
            except BaseException:               # noqa: BLE001 - re-raised on read
                out[kind] = {"__probe_error__": traceback.format_exc()}
            gc.collect()
        return out

    def _prefill_on_pages(self, page_ids: list[int], *, cache=None,
                          input_ids=None, dtype=None, image=None):
        """Prefill the frozen prompt onto ``page_ids``, DERIVING the allocation.

        The block table and slot mapping come from the candidate's own
        ``BlockPool`` index methods -- which is the point of the paging probes:
        the block-table gather and the slot arithmetic are exactly what an
        off-by-one breaks, and the canonical scenario's single contiguous
        allocation cannot distinguish a correct gather from an ignored one.

        THE POSITIONS AND THE EXTRA FORWARD KEYWORDS ARE THE MODEL'S, resolved
        through the same two hooks ``engine.Engine`` resolves
        (:meth:`_built_positions`, :meth:`_prepare_model_inputs`). This method
        must not fall back to ``np.arange(n)``, which is the shared DEFAULT
        builder's answer and not every model's: a family whose positions are a
        ``(t, h, w)`` triple would get a rank-1 column its own rope refuses by
        name, so the three paging/cache-reuse invariants would raise before they
        could compare anything.

        ``image`` NAMES WHICH PICTURE, and it is a parameter rather than a
        canonical read for one reason: :meth:`battery_probe` drives BOTH this
        method and :meth:`_handed_forward`, and that method deliberately reads
        the VARIANT's own mapping so a variant is never forwarded the canonical
        scenario's picture. Left to read ``self._image_inputs()`` here, the two
        battery legs would source their image differently -- the mixed-stage leg
        from the variant, the paging leg from the canonical scenario -- against
        the SAME variant's ids. Unreachable today (no instance ships a
        ``battery/``, and no multimodal one is planned to), and recorded here
        because it is exactly the drift a sibling helper exists to prevent.
        ``None`` means the canonical scenario's image, which is every other
        caller; ``{}`` means none at all and is what a text-only variant passes.

        Note that ``ids`` may be a PREFIX of the frozen prompt
        (``compilation_probe``'s body-warming leg, and a battery variant) while
        the image is the whole scenario's -- see the note at ``short_ids`` for
        the geometry that makes that sound.
        """
        jnp = self._jnp
        dtype = dtype or self.native_dtype
        ids = self.prompt_ids if input_ids is None else np.asarray(input_ids)
        n = int(np.asarray(ids).shape[0])
        pool = self.block_table_mod.BlockPool(self.num_pages, self.page_size)
        slot = pool.slot_mapping(page_ids, 0, n)
        block_tables = pool.block_table_row(page_ids, self.pages_per_seq)
        token_ids = jnp.asarray(ids, dtype=jnp.int32)
        image = self._image_inputs() if image is None else image
        meta = self._metadata(
            positions=self._built_positions(self.model, token_ids, **image),
            block_tables=block_tables,
            seq_lens=np.array([n], dtype=np.int32),
            query_start_loc=np.array([0, n], dtype=np.int32),
            request_distribution=np.array([0, 1, 0], dtype=np.int32),
        )
        if cache is None:
            cache = self._empty_cache(dtype)
        model_kwargs = self._prepare_model_inputs(self.model, token_ids, **image)
        hidden, cache = self._jit_forward()(
            self.model, token_ids, cache, meta, slot, False, model_kwargs)
        return self.model.compute_logits(hidden), cache, meta, slot

    def _default_pages(self) -> list[int]:
        n = int(self.prompt_ids.shape[0])
        need = max(1, (n + self.page_size - 1) // self.page_size)
        return list(range(need))

    def _permuted_pages(self) -> list[int]:
        """A DIFFERENT physical page set holding the same logical KV.

        Taken from the top of the pool downwards and reversed, so a table that is
        read in the wrong order (or ignored entirely) produces different logits.

        THE TWO SETS NEED NOT BE DISJOINT, and this docstring claimed they were
        ("so it cannot overlap the ascending default") until core_gpt_oss.
        Disjointness needs ``num_pages >= 2 * need``; the frozen prompt there
        occupies 5 of 8 pages, so the default ``[0,1,2,3,4]`` and the permuted
        ``[7,6,5,4,3]`` share ``{3, 4}``. (The three ``page_size``-8 instances
        need 2 pages of 8 or 12 and are disjoint, which is why it read as true.)

        THE PROBE IS UNAFFECTED, which is why this is a comment fix and not a
        geometry change. What it grades is that a DIFFERENT logical-to-physical
        map produces the same logits, and the two maps here are ``i -> i`` and
        ``i -> 7 - i``: they agree at no integer, so every logical index moves,
        including the two whose destinations happen to be in both sets. The two
        legs also run on separate freshly-zeroed caches, so a shared page id
        carries nothing between them.
        """
        need = len(self._default_pages())
        pages = list(range(self.num_pages - need, self.num_pages))[::-1]
        return pages

    def _invariance_paging_default(self) -> dict[str, Any]:
        pages = self._candidate_pages("invariance.paging_default",
                                      self._default_pages())
        logits, cache, _m, _s = self._prefill_on_pages(pages)
        out = {"supported": True, "logits": _to_np(logits), "page_ids": pages}
        del logits, cache
        gc.collect()
        return out

    def _invariance_paging_permuted(self) -> dict[str, Any]:
        pages = self._candidate_pages("invariance.paging_permuted",
                                      self._permuted_pages())
        logits, cache, _m, _s = self._prefill_on_pages(pages)
        out = {"supported": True, "logits": _to_np(logits), "page_ids": pages}
        del logits, cache
        gc.collect()
        return out

    def _far_decode_position(self, n: int) -> int | None:
        """The first position on the page AFTER the one the prompt ends on, or None.

        The near comparison (position ``n-1``) never leaves the pages the prompt
        already occupies, so no page-boundary defect can reach it. This is the
        position that does: ``page_size * ceil(n / page_size)`` is the first slot
        of the next page, 16 on this instance's 12-token prompt at ``page_size``
        8. ``None`` when the frozen scenario cannot get there -- not enough
        continuation tokens, or past ``max_model_len``.
        """
        far = self.page_size * ((n + self.page_size - 1) // self.page_size)
        if far < n:
            return None
        # Renamed at the boundary (fixtures.build_candidate): an INPUT to this
        # probe, not the array any check grades.
        extra = list(self.scenario.get("decode_input_token_ids") or [])
        if far - n + 1 > len(extra):
            return None
        if far + 1 > int(getattr(self.run_config, "max_model_len", far + 1)):
            return None
        return int(far)

    def _invariance_decode_prefill(self) -> dict[str, Any]:
        """Token t's decode logits == its logits from a prefill over the prefix.

        DRIVEN THROUGH THE CANDIDATE'S OWN ``Engine``, on the candidate's own
        allocation, and that is the whole content of this probe. It does not build
        its own ``AttentionMetadata`` and its own slot from the FROZEN page ids:
        doing so would let the grader supply the positions, the ``seq_lens``, the
        block table and the slot -- so no candidate-side defect in any of them could
        reach the comparison. A decode RoPE
        position off-by-one, a stale ``seq_lens`` and a clamped page index would all
        PASS it, and those are exactly the three mechanisms this probe
        guards against. Both legs go through ``Engine.prefill`` / ``Engine.decode_step``
        / ``Engine.reset`` -- all four in ``interface.json``'s method contract --
        so the candidate computes every one of those quantities itself.

        TWO COMPARISON POINTS, because one is not enough on this geometry:

        * NEAR, position ``n-1``: prefill the prompt's prefix, decode its last
          token, compare against row ``n-1`` of a single full prefill. Reaches
          only the pages the prompt already occupies.
        * FAR, the first position on the next page (:meth:`_far_decode_position`):
          keep decoding the frozen continuation tokens to that position and
          compare against the same row of a single prefill over the same prefix.
          This is the leg a page-boundary defect can reach -- the near leg cannot,
          and a clamped page gather passed the old probe for exactly that reason.

        ``None`` for the far position is reported as ``far_position: None`` rather
        than omitted, so a consumer can tell "this geometry cannot reach a second
        page" from "the probe stopped doing it".

        AND A THIRD, WHICH ONLY RUNS WHEN THE SECOND CANNOT. The far position is
        outside the prompt, so it needs continuation tokens and ``max_model_len``
        headroom. On core_gpt_oss (prompt 135, ``page_size`` 32) the next
        boundary is 160, which is 26 continuation tokens away against 8 frozen and
        one past ``max_model_len`` -- so ``far`` is ``None``, the boundary leg does
        not run, and NOTHING in the graded run makes ``Engine._grow`` extend an
        existing allocation. Prefill takes 5 pages; the deepest position any check
        reaches is 143, still 5 pages. A port whose ``_grow`` allocates once and
        then never grows scores 1.0 there.

        So when ``far`` is ``None`` the probe crosses the boundary from inside the
        prompt instead (:func:`interior_boundary_position`): prefill ``ids[:b]``,
        decode ``ids[b]``, and compare against row ``b`` of the FULL prefill leg 2
        already ran. Same self-comparison and same one-step decode as the near
        leg, at the position where ``_grow`` has to extend. ONE new prompt shape;
        no golden, no re-freeze. Reported as ``interior_position`` (``None`` when
        the far leg ran, or when the prompt spans less than two pages), so a
        consumer can tell which of the two boundary legs happened.

        AND THE ALLOCATION ITSELF IS REPORTED, not only its effect on the logits,
        because on this family the effect is inside the band. With
        ``_grow`` mutated to allocate once and never grow, core_gpt_oss's decode
        at position 128 clamps -- ``slot_mapping`` indexes a jnp array, and JAX
        out-of-bounds indexing CLAMPS rather than raising -- so the step writes
        its KV over position 96's slot and reads position 0's key as its own. That
        is one wrong key in 129 against a native tier deliberately widened to
        ``centered_cosine_floor: 0.95`` for this family's expert-routing flips,
        and the logit comparison alone does NOT reject it.

        So the probe also reports ``interior_prefill_pages`` (the short prefill's
        own ``page_ids``), the decode's ``block_tables`` and its ``slot`` -- all
        three already on the results ``interface.json`` pins -- and the check
        asserts the boundary page is one the prefill did not already hold and that
        the KV landed in it. That clause is exact and needs no tolerance.

        EVERY PREFILL BELOW CARRIES THE FROZEN IMAGE and no decode does, which
        is the split ``Engine.generate`` itself makes -- extra keywords reach
        ``prefill`` and stop there. Both legs are prefills of the SAME prompt
        (or a prefix, or that prompt plus frozen continuation tokens), so they
        see the same picture and the self-comparison stays a comparison of paths
        rather than of inputs. The near leg's prefix is ``ids[:n - 1]``: on a
        prompt whose last token is part of a ``<|image_pad|>`` run that would
        cut the run, and the reference refuses that by name rather than
        computing something wrong -- see the note at ``short_ids`` in
        ``compilation_probe`` for the same geometry constraint stated in full.
        """
        jnp = self._jnp
        n = int(self.prompt_ids.shape[0])
        if n < 2:
            return {"supported": False, "reason": "the frozen prompt has one token"}
        probe = "invariance.decode_prefill"
        Engine = self.engine_mod.Engine
        image = self._image_inputs()
        ids = [int(t) for t in np.asarray(self.prompt_ids).reshape(-1)]
        # Renamed at the boundary (fixtures.build_candidate): an INPUT to this
        # probe, not the array any check grades.
        extra = [int(t) for t in (self.scenario.get("decode_input_token_ids") or [])]
        far = self._far_decode_position(n)

        def _row(logits, index: int) -> np.ndarray:
            return _to_np(logits)[index]

        # LEG 1 -- the candidate's engine, prefilling the prefix and then DECODING.
        # POINT OF USE: `Engine.prefill` ran its own BlockPool and `Engine._metadata`
        # built every field the old probe handed in.
        self._record_allocation(probe, "candidate")
        decode_engine = Engine(self.model, self.run_config)
        decode_engine.prefill(jnp.asarray(ids[: n - 1], dtype=jnp.int32), **image)
        near_decode = _row(decode_engine.decode_step(0, ids[n - 1]).logits, 0)
        far_decode = None
        if far is not None:
            for token in extra[: far - n + 1]:
                far_decode = _row(decode_engine.decode_step(0, token).logits, 0)

        # LEG 2 -- the same rows out of single full prefills, same engine class,
        # `reset()` between them because `prefill` refuses a live request.
        #
        # EVERY ROW IS KEPT, not just the last: leg 3 below needs the row at the
        # interior page boundary and this is the prefill that has it, so taking it
        # here is the difference between that leg costing one new prompt shape and
        # costing two. `_row` materialised the whole array anyway.
        prefill_engine = Engine(self.model, self.run_config)
        full_rows = _to_np(prefill_engine.prefill(
            jnp.asarray(ids, dtype=jnp.int32), **image).logits)
        near_prefill = full_rows[n - 1]
        far_prefill = None
        if far is not None:
            prefill_engine.reset()
            far_prefill = _row(prefill_engine.prefill(jnp.asarray(
                ids + extra[: far - n + 1], dtype=jnp.int32), **image).logits,
                far)

        out: dict[str, Any] = {
            "supported": True,
            "near_position": n - 1,
            "decode_logits": near_decode,
            "prefill_last_row": near_prefill,
            "far_position": far,
        }
        if far is not None:
            out["far_decode_logits"] = far_decode
            out["far_prefill_row"] = far_prefill

        # LEG 3 -- THE BOUNDARY FROM INSIDE THE PROMPT, only where leg 2 could not
        # reach one. A fresh engine: the two above are past this position already.
        #
        # ONE NEW SHAPE, NOT TWO. The reference row is taken from the FULL prefill
        # leg 2 already ran (`full_rows`), so the only forward this leg adds is the
        # short prefill -- the comparison is the same property the near leg states
        # (row t of a prefill == the decode that produces position t), at a
        # position where the decode has to allocate. That matters on core_gpt_oss,
        # where each additional prompt length is another compiled program and
        # another CUDA graph in a process already holding a 20B MoE.
        interior = (interior_boundary_position(n, self.page_size)
                    if far is None else None)
        out["interior_position"] = interior
        if interior is not None:
            inner = Engine(self.model, self.run_config)
            first = inner.prefill(jnp.asarray(ids[:interior], dtype=jnp.int32),
                                  **image)
            # THE DECODE THAT MUST GROW: position `interior` is the first slot of
            # the next page, so `_grow` has to extend the allocation here.
            step = inner.decode_step(0, ids[interior])
            out["interior_prefill_pages"] = [int(p) for p in first.page_ids]
            out["interior_block_table"] = [
                int(x) for x in
                np.asarray(_to_np(step.attn_metadata.block_tables)).reshape(-1)]
            out["interior_slot"] = int(
                np.asarray(_to_np(step.slot_mapping)).reshape(-1)[0])
            out["interior_decode_logits"] = _row(step.logits, 0)
            out["interior_prefill_row"] = full_rows[interior]
            del first, step, inner
            gc.collect()

        del decode_engine, prefill_engine, full_rows
        gc.collect()
        return out

    def _invariance_cache_reuse(self) -> dict[str, Any]:
        """Freeing and reallocating pages must not corrupt a live sequence.

        Derives its own allocation (``use_frozen_allocation=False``): the whole
        property is about what the candidate's ``BlockPool`` hands out and what
        that does to pages a live request already owns.

        TWO OBSERVABLES, AND THE SECOND ONE IS WHY THIS PROBE WAS REBUILT. The
        logits leg alone was measurably incapable of seeing an allocator defect,
        and the reason is structural rather than a matter of this geometry:
        ``modeling.py`` writes the KV cache BEFORE it reads it, inside the same
        forward, so the closing forward below rewrites every slot it then gathers
        and ``after_reuse`` is a pure function of
        ``(model, prompt_ids, meta, slot)`` -- INDEPENDENT OF THE INCOMING CACHE.
        An allocator whose ``free`` forgets which pages are live hands pages 0 and 1
        back out, random KV is scribbled over all twelve of the live sequence's
        slots, and the logits leg still returns bit-identical arrays -- caught by
        none of the other graded checks.

        So the live sequence's own KV is snapshotted and compared directly. That
        clause is what grades cache-reuse safety; it needs no forward pass and
        cannot be rewritten out from under itself.

        THE LOGITS LEG STAYS, and is not redundant. A gather that reads ``bt + 1``
        moves the READ window off the write window and onto a scratch page, so the
        scribbled noise reaches the output and the logits leg fails. The two
        clauses see different faults -- the snapshot sees a bad ALLOCATOR, the
        logits see a bad GATHER -- and neither subsumes the other.
        """
        jnp = self._jnp
        pool = self.block_table_mod.BlockPool(self.num_pages, self.page_size)
        n = int(self.prompt_ids.shape[0])
        live_pages = self._candidate_pages("invariance.cache_reuse", pool.allocate(n))
        logits_before, cache, meta, slot = self._prefill_on_pages(live_pages)
        before = _to_np(logits_before)
        # The live request's own pages, before anything is churned around them.
        # Every layer, both planes: a defect that reaches only layer 7's V plane
        # is as fatal as one that reaches layer 0's K plane.
        live_index = jnp.asarray(live_pages, dtype=jnp.int32)
        # EVERY GROUP, in container order. A defect that reaches only the group
        # holding the full-attention layers is as fatal as one that reaches the
        # sliding group, and on a one-group family this is the same single array
        # it has always been, wrapped in a tuple.
        live_kv_before = tuple(_to_np(group[:, live_index]) for group in cache)

        # Churn the rest of the pool around the live request: allocate, scribble
        # random KV over the pages, free them, take them again and scribble
        # different KV. A correct implementation never lets any of that reach the
        # live sequence, whose own pages were never freed.
        scratch = pool.allocate(self.page_size * max(1, pool.num_free // 2))
        rng = np.random.default_rng(self.seed + 17)
        for round_index in range(2):
            if round_index:
                pool.free(scratch)
                scratch = pool.allocate(self.page_size * len(scratch))
            slots = np.concatenate([
                np.arange(p * self.page_size, (p + 1) * self.page_size)
                for p in scratch
            ]).astype(np.int32)
            # THROUGH THE CONTAINER, GROUP BY GROUP AND SLOT BY SLOT -- the same
            # arithmetic, reached the way the model reaches it. The noise is drawn
            # at EACH GROUP'S OWN (num_kv_heads, head_dim) rather than at the
            # config's scalar pair: on a family whose geometries differ that pair
            # describes one group of several, and `write_kv`'s reshape would
            # refuse the other. `n_slots`, not `num_hidden_layers`, for the same
            # reason -- a family that SHARES a slot allocates fewer of them than
            # it has layers.
            scribbled = []
            for group in cache:
                kv_heads, head_dim = int(group.shape[4]), int(group.shape[5])
                noise = rng.normal(
                    size=(2, slots.size, kv_heads, head_dim)).astype(np.float32)
                key = jnp.asarray(noise[0], dtype=self.native_dtype)
                value = jnp.asarray(noise[1], dtype=self.native_dtype)
                # `cache_slot`, NOT `slot`. THE OBVIOUS NAME SHADOWS THE
                # SLOT MAPPING `_prefill_on_pages` returned above, which the
                # CLOSING FORWARD at the end of this method still needs -- and
                # after the loop it would be a bare int, which would raise
                # `ValueError: Cannot broadcast to shape with fewer dimensions:
                # arr_shape=(124, 1, 256) shape=(1, 256)` out of `write_kv`,
                # because the forward would be handed `11` where it wanted 124
                # slots. A loop variable named `slot` would shadow the slot
                # mapping, so this loop uses `cache_slot` instead.
                for cache_slot in range(int(group.shape[0])):
                    group = group.at[cache_slot].set(self.kv_cache_mod.write_kv(
                        group[cache_slot], key, value, jnp.asarray(slots)))
                scribbled.append(group)
            cache = tuple(scribbled)

        # READ BEFORE THE CLOSING FORWARD, not after: that forward writes the live
        # slots on its way through, so a snapshot taken afterwards would compare
        # the churn against its own repair and could never differ.
        live_kv_after = tuple(_to_np(group[:, live_index]) for group in cache)

        # THE SAME COMPOSITION `_prefill_on_pages` RAN, re-run on the churned
        # cache: the same ids, the same `meta` and `slot` it returned, and the
        # same prepared keywords. Rebuilt here rather than returned alongside
        # `meta` because they are a function of the ids and the frozen image and
        # nothing this method did can have moved them -- and because a probe
        # that closed over a stale merged embedding table would be comparing two
        # different forwards while claiming to compare two caches.
        closing_ids = jnp.asarray(self.prompt_ids, dtype=jnp.int32)
        hidden, _cache = self._jit_forward()(
            self.model, closing_ids, cache, meta, slot, False,
            self._prepare_model_inputs(self.model, closing_ids,
                                       **self._image_inputs()))
        after = _to_np(self.model.compute_logits(hidden))
        del cache, logits_before, hidden
        gc.collect()
        return {"supported": True, "before_reuse": before, "after_reuse": after,
                "live_kv_before": live_kv_before, "live_kv_after": live_kv_after,
                "live_pages": [int(p) for p in live_pages],
                "reused_pages": [int(p) for p in scratch]}

    def _invariance_determinism(self) -> dict[str, Any]:
        probe = "invariance.determinism"
        first, _hs, cache_a, _m, _s = self._run_frozen_prefill(
            self.model, self.native_dtype, want_hidden=False, probe=probe)
        first = _to_np(first)
        second, _hs2, cache_b, _m2, _s2 = self._run_frozen_prefill(
            self.model, self.native_dtype, want_hidden=False, probe=probe)
        second = _to_np(second)
        del cache_a, cache_b
        gc.collect()
        return {"supported": True, "first": first, "second": second}

    # --- the ragged invariants ----------------------------------------------- #
    def _ragged_requests(self) -> list[dict[str, Any]]:
        """The frozen batch, one entry per request: ``{stage, tokens, page_ids}``.

        Read off ``scenario["requests"]``, which a multi-sequence freeze fills in
        with each request's own stage and tokens. A `core` scenario carries one
        request and neither field, so the defaults reconstruct exactly what it
        has always meant -- one full prefill of ``prompt_token_ids``.

        ``shares_prefix_with`` / ``shared_prefix_len`` are ADDITIVE, ``chunked``
        -only fields: which earlier request (by index into this same list) a
        request's leading page is lent from, and how many of its own leading
        tokens are the shared ones (instance.yaml's own fields, restated
        verbatim by the freezer once it resolves the shared page id). ``None``
        on every request of every mode that does not declare
        one, which is every mode before `chunked`. Both existing callers
        (``_invariance_batching``, ``_invariance_cross_sequence_leakage``)
        destructure only ``stage``/``tokens``/``page_ids`` and are unaffected.
        """
        out: list[dict[str, Any]] = []
        for entry in (self.scenario.get("requests") or []):
            tokens = entry.get("prompt_token_ids")
            if tokens is None:
                tokens = self.scenario.get("prompt_token_ids") or self.prompt_ids
            out.append({
                "stage": str(entry.get("stage") or "prefill"),
                "tokens": [int(t) for t in np.asarray(tokens).reshape(-1)],
                "page_ids": [int(p) for p in (entry.get("page_ids") or self.page_ids)],
                "shares_prefix_with": (
                    int(entry["shares_prefix_with"])
                    if entry.get("shares_prefix_with") is not None else None),
                "shared_prefix_len": (
                    int(entry["shared_prefix_len"])
                    if entry.get("shared_prefix_len") is not None else None),
                # THE PER-REQUEST IMAGE MARKER (Task B), restated verbatim from
                # the frozen scenario: the instance holds ONE fixture and the
                # plan says whose placeholders it belongs to.
                # ``_model_kwargs_for`` reads it. False on every request of
                # every text-only instance and on every `core` scenario, which
                # carries one request and no such key.
                "image": bool(entry.get("image")),
            })
        return out

    def _model_kwargs_for(self, request: dict[str, Any]) -> dict[str, Any]:
        """This request's multimodal inputs, from the FROZEN golden, or ``{}``.

        Read off ``inputs.npz`` through :meth:`_image_inputs` -- the one place
        this file decides what an engine call is handed -- and NOT off the
        candidate's own ``image_preprocess`` probe. The grader compares this
        forward against a frozen row, so it must consume the same bytes the
        freezer drove the reference with; the port of the image processor agrees
        with the reference to 2.4e-07 rather than exactly, and re-deriving the
        pixels here would fold that error into every logits comparison, where it
        would read as a defect in the decoder.

        WHICH request has the image is the frozen scenario's to say
        (``_ragged_requests`` carries the marker through), because the instance
        holds ONE fixture and the plan says whose placeholders it belongs to.
        """
        return self._image_inputs() if request.get("image") else {}

    def _positions_for(self, request: dict[str, Any]) -> "np.ndarray":
        """One request's FULL positions, at the model's own rope rank.

        :meth:`_built_positions` per request, so this shares the three-step hook
        resolution that method's docstring records rather than restating it --
        a bare ``getattr(model, "get_input_positions")`` here would hardcode the
        hook NAME this file is not supposed to know and would drop the
        ``default_input_positions`` fallback whose absence takes four checks red
        on `chunked_llama3`.

        The request's own multimodal keywords are forwarded uninspected, which
        is what makes an M-RoPE family's ``(R, num_tokens)`` answer this
        request's and not another's.
        """
        tokens = self._jnp.asarray(request["tokens"], dtype=self._jnp.int32)
        return _to_np(self._built_positions(
            self.model, tokens, **self._model_kwargs_for(request))).astype(np.int32)

    def _ragged_forward(self, requests: list[dict[str, Any]], probe: str):
        """Run ``requests`` as ONE mixed batch. Returns ``(logits, query_start_loc)``.

        Composed the way the reference composes it: a `decode` request
        contributes one query token over cached history, so the history is
        written by a preceding warm-up forward on the same cache. Both forwards
        go through the driver's own ``_jit_forward`` and ``_metadata``, and the
        block table and slot mapping come from the candidate's own ``BlockPool``
        index methods -- the same seam ``_prefill_on_pages`` uses, and for the
        same reason: the gather and the slot arithmetic are exactly what these
        two invariants exist to catch, so they must be the CANDIDATE's.

        WHAT THIS DOES NOT REACH, stated rather than left to be assumed: the
        candidate's SCHEDULER, because there is no scheduler in this class's
        contract -- ``interface.json`` pins ``prefill``/``decode_step``/
        ``generate``/``reset``, all single-request. So these two probes grade the
        ragged ATTENTION path (block-table gather, per-request masking,
        cu_seqlens arithmetic) and not a batching policy. The allocator's own
        failure mode -- one page handed to two live requests -- is
        ``test_cache_reuse_safety``'s, which does drive ``BlockPool.allocate``.

        PER-REQUEST POSITIONS AND PER-REQUEST IMAGES. Each request's positions
        come from the model's own builder when it defines one and from
        ``arange`` when it does not (:meth:`_positions_for`, which is
        :meth:`_built_positions` per request), and the blocks are concatenated
        on the LAST axis -- not appended to a flat list -- because an M-RoPE
        family's positions are ``(R, num_tokens)`` and the requests join along
        the token axis. A rank-1 family produces the identical flat vector it
        always did, which is what keeps every text-only instance's goldens
        byte-identical.

        WHICH FORWARD IS HANDED WHICH IMAGE MIRRORS THE FREEZER, because the
        frozen row it is compared against was produced that way: a request gives
        its image to whichever forward covers its ``<|image_pad|>`` rows -- the
        WARM-UP when its graded query starts past them (``start`` non-zero,
        which is every decode), and the GRADED forward when its query starts at
        0. That is ``_build_batch``'s ``warm_mm`` / ``graded_mm`` pair, clause
        for clause. It is deliberately NOT "the graded forward gets request 0's":
        request 0 is a decode on both of this instance's battery compositions,
        and hardcoding it would hand the same picture to both forwards on one
        geometry and to neither on another.

        AND BOTH FORWARDS' KEYWORDS ARE PREPARED ON THE HOST
        (:meth:`_prepare_model_inputs`), before the trace. ``_jit_forward``
        traces its ``model_kwargs`` argument -- by design, because what it is
        meant to carry is a merged embedding table, which is data -- so an
        ``image_grid_thw`` handed in raw reaches the vision tower as a TRACER and
        the tower's Python control flow over it raises
        ``ConcretizationTypeError`` at ``jnp.arange(grid_h)``. The reference's
        own engine refuses this arrangement by name
        (``engine.Engine._model_inputs``: "anything the model has to read as a
        CONCRETE value must be consumed on the host, inside INPUTS_HOOK"), and
        every other image-bearing call site in this file already prepares first.
        This matters once an instance's ragged batch carries a picture: both
        call sites must prepare first rather than pass ``{}``.
        """
        jnp = self._jnp
        page_size = self.page_size
        pool = self.block_table_mod.BlockPool(self.num_pages, self.page_size)

        graded: dict[str, list] = {"ids": [], "pos": [], "slots": [], "q": [],
                                   "len": [], "mm": []}
        warm: dict[str, list] = {"ids": [], "pos": [], "slots": [], "q": [], "mm": []}
        rows = []
        for request in requests:
            tokens, pages = request["tokens"], request["page_ids"]
            n = len(tokens)
            q_len = n if request["stage"] == "prefill" else 1
            start = n - q_len
            # SLICED OUT OF THE WHOLE REQUEST'S POSITIONS, never rebuilt from
            # the slice -- the freezer's own rule (`_build_batch`), and for its
            # reason: an image collapses several tokens onto fewer positions, so
            # recomputing from ``tokens[start:]`` would restart the run and lose
            # the offset. Computed ONCE per request and windowed twice.
            full = self._positions_for(request)
            # A REQUEST CONTRIBUTES ITS IMAGE TO WHICHEVER FORWARD COVERS ITS
            # PLACEHOLDERS -- `_build_batch`'s `warm_mm` / `graded_mm` rule, and
            # the two branches are complementary because a request's `start` is
            # either 0 or it is not.
            request_mm = self._model_kwargs_for(request)
            if request_mm and start:
                warm["mm"].append(request_mm)
            elif request_mm:
                graded["mm"].append(request_mm)
            graded["ids"] += tokens[start:]
            graded["pos"].append(full[..., start:n])
            graded["slots"] += [int(s) for s in _to_np(
                pool.slot_mapping(pages, start, q_len)).reshape(-1)]
            graded["q"].append(q_len)
            graded["len"].append(n)
            warm["ids"] += tokens[:start]
            warm["pos"].append(full[..., :start])
            if start:
                warm["slots"] += [int(s) for s in _to_np(
                    pool.slot_mapping(pages, 0, start)).reshape(-1)]
            warm["q"].append(start)
            rows.append(_to_np(pool.block_table_row(pages, self.pages_per_seq)))
        table = np.concatenate(rows).astype(np.int32)
        graded_positions = np.concatenate(graded["pos"], axis=-1).astype(np.int32)
        warm_positions = np.concatenate(warm["pos"], axis=-1).astype(np.int32)
        warm_mm = {key: value for one in warm["mm"] for key, value in one.items()}
        graded_mm = {key: value for one in graded["mm"] for key, value in one.items()}

        def cu(lengths):
            out = [0]
            for value in lengths:
                out.append(out[-1] + value)
            return np.asarray(out, dtype=np.int32)

        n_decode = sum(1 for r in requests if r["stage"] == "decode")
        cache = self._empty_cache(self.native_dtype)
        fwd = self._jit_forward()
        if any(warm["q"]):
            warm_meta = self._metadata(
                positions=warm_positions,
                block_tables=table,
                seq_lens=np.asarray(warm["q"], dtype=np.int32),
                query_start_loc=cu(warm["q"]),
                request_distribution=np.asarray(
                    [0, sum(1 for c in warm["q"] if c), 0], dtype=np.int32),
            )
            warm_tokens = jnp.asarray(warm["ids"], dtype=jnp.int32)
            _h, cache = fwd(self.model, warm_tokens, cache,
                            warm_meta,
                            jnp.asarray(warm["slots"], dtype=jnp.int32), False,
                            self._prepare_model_inputs(
                                self.model, warm_tokens, **warm_mm))
        meta = self._metadata(
            positions=graded_positions,
            block_tables=table,
            seq_lens=np.asarray(graded["len"], dtype=np.int32),
            query_start_loc=cu(graded["q"]),
            request_distribution=np.asarray(
                [n_decode, len(requests) - n_decode, 0], dtype=np.int32),
        )
        graded_tokens = jnp.asarray(graded["ids"], dtype=jnp.int32)
        hidden, cache = fwd(self.model, graded_tokens,
                            cache, meta, jnp.asarray(graded["slots"], dtype=jnp.int32),
                            False, self._prepare_model_inputs(
                                self.model, graded_tokens, **graded_mm))
        logits = _to_np(self.model.compute_logits(hidden))
        del cache, hidden
        gc.collect()
        # POINT OF USE: the page lists came from the frozen scenario, the block
        # table and the slot mapping from the candidate's own BlockPool.
        self._record_allocation(probe, "frozen")
        return logits, [int(x) for x in cu(graded["q"])]

    @staticmethod
    def _graded_index(requests: list[dict[str, Any]]) -> int:
        """Which request the batching invariant grades: the LAST `decode` one.

        Not request 0, and the difference is the whole content of the check. A
        request at ``query_start_loc[0] == 0`` makes the ``- query_start_loc[s]``
        term in ``q_pos`` a no-op and makes sequence 0 the correct answer to
        ``seq_of_tok``, so a port that hardcodes either is INERT there and live
        everywhere else -- the same argument
        ``test_a_ragged_batch_matches_running_each_sequence_alone`` makes in the
        reference's own unit suite. A `decode` request additionally has
        ``seq_len > q_len``, so its ``q_pos`` is not its token index either.
        Falls back to the last request, then to 0.
        """
        decodes = [i for i, r in enumerate(requests) if r["stage"] == "decode"]
        return decodes[-1] if decodes else max(len(requests) - 1, 0)

    def _invariance_batching(self) -> dict[str, Any]:
        """Request i alone == request i inside a ragged batch.

        Golden-free and self-comparing. Note what that does NOT buy: over the
        seven I1 mutations that cleared Spec 17/17 and Numeric 9/9, the
        golden-free invariants caught 0/7, precisely because both legs run the
        candidate. It is worth having anyway -- batching is this mode's defining
        axis -- but it is not a substitute for the frozen comparison, and a
        mutation that is wrong the same way in both legs is invisible here.

        WHAT MAKES IT DISCRIMINATE AT ALL, since both legs are the same function:
        the two legs differ in the batch the request sits in. The lone leg has
        ``query_start_loc == [0, q_len]`` and one block-table row, so a port that
        hardcodes sequence 0, drops the ``- query_start_loc[s]`` term, or reads
        the block table one row too wide is CORRECT in the lone leg and wrong in
        the batched one. A defect that is wrong in both is not.

        Toleranced by the check at the calibrated native band, because the two
        legs reduce over different shapes.
        """
        requests = self._ragged_requests()
        if len(requests) < 2:
            return {"supported": False,
                    "reason": "the frozen scenario declares one request, so there "
                              "is no batch to be invariant against"}
        index = self._graded_index(requests)
        batched, qsl = self._ragged_forward(requests, "invariance.batching")
        lo, hi = qsl[index], qsl[index + 1]
        alone, _ = self._ragged_forward([requests[index]], "invariance.batching")
        out = {"supported": True,
               "alone": np.asarray(alone),
               "batched": np.asarray(batched[lo:hi]),
               "graded_request": index}
        gc.collect()
        return out

    def _invariance_cross_sequence_leakage(self) -> dict[str, Any]:
        """Perturb request j; every OTHER request's logits must be BIT-unchanged.

        The classic paged-attention bug hit directly: a block table read one row
        too wide, a mask that spans requests, a page handed to two live
        sequences. Both legs keep every other request's tokens, its position in
        the batch and the batch's shape identical -- only request j's TOKENS move
        -- so any difference at all is leakage, and the check compares with
        ``assert_array_equal`` rather than a tolerance.

        EVERY other request is returned, not one. Returning a single victim would
        make the check blind to leakage that reaches only the third row of a
        three-request batch, and the concatenation costs nothing because the
        comparison is exact rather than aggregate.
        """
        requests = self._ragged_requests()
        if len(requests) < 2:
            return {"supported": False,
                    "reason": "cross-sequence leakage needs a second sequence; the "
                              "frozen scenario declares one request"}
        source = self._graded_index(requests)
        before, qsl = self._ragged_forward(requests, "invariance.cross_sequence_leakage")

        perturbed = [dict(r) for r in requests]
        tokens = list(perturbed[source]["tokens"])
        # Length-preserving and in-vocab, so the batch's SHAPE cannot move -- and
        # in the MIDDLE of the request, which on a `decode` request is inside the
        # cached prefix rather than in its query token. That is the ragged-specific
        # surface: the perturbed KV reaches the output only through the warm-up
        # write and the block-table gather, which is where the leakage this check
        # is named for would live. (Position 0 would also work and is a worse
        # choice: swapping BOS is a degenerate input, and BOS is the one position
        # whose KV every implementation treats specially.)
        vocab = int(self.model_config.vocab_size)
        moved = len(tokens) // 2
        tokens[moved] = (tokens[moved] + 1) % vocab
        perturbed[source] = {**perturbed[source], "tokens": tokens}
        after, qsl_after = self._ragged_forward(
            perturbed, "invariance.cross_sequence_leakage")
        if qsl_after != qsl:
            return {"supported": False,
                    "reason": f"the perturbation changed the batch SHAPE "
                              f"({qsl} -> {qsl_after}), so a difference would no "
                              f"longer be evidence of leakage"}

        # A REQUEST THAT SHARES PHYSICAL PAGES WITH THE PERTURBED ONE IS NOT
        # INDEPENDENT OF IT, AND COMPARING IT HERE IS A FALSE POSITIVE.
        #
        # On `chunked_qwen2_5_vl` this shows up on the REFERENCE itself. That
        # instance's r2 declares
        # `shares_prefix_with: 1`, so `validate_sequences` PREPENDS r1's own
        # first page to r2's block table and the two read the same physical
        # slots for positions [0, 8) BY CONSTRUCTION -- that is what a prefix
        # cache IS. `_graded_index` then picks "the last decode request", which
        # on that batch is r1 -- the LENDER -- and the perturbation lands at
        # `len(tokens) // 2 == 4`, inside the shared range. r2's answer moves,
        # correctly, and the check called it leakage.
        #
        # WHY NO INSTANCE HIT THIS BEFORE, which is the part worth writing down:
        # on `chunked_llama3`, `chunked_qwen2` and `chunked_qwen3` the lender is
        # r0 (a `prefill_chunked`) and the only decode is r2, which lends
        # nothing -- so `_graded_index` never selects a lender and the defect is
        # INERT. It is not multimodal in nature; any text-only chunked instance
        # whose sharing request borrowed from its decode request would fail
        # identically. `chunked_qwen2_5_vl` is forced into that shape because
        # its sequence 0 is the image-bearing one and a shared prefix taken from
        # IT would copy the `<|image_pad|>` placeholders into a second request
        # that cannot declare the fixture.
        #
        # PAGE OVERLAP, NOT `shares_prefix_with`, because overlap is the
        # PHYSICAL condition and the declaration is only one way to arrive at
        # it. Two requests that read the same slot cannot be independent
        # whatever the scenario calls the relationship.
        source_pages = set(int(p) for p in (requests[source].get("page_ids") or []))
        shared_with_source = [
            i for i in range(len(requests))
            if i != source and source_pages
            & set(int(p) for p in (requests[i].get("page_ids") or []))]
        keep = [i for i in range(len(requests))
                if i != source and i not in shared_with_source]
        # ...AND IF THAT LEAVES NOTHING, SAY SO RATHER THAN PASS. An empty
        # comparison is bit-equal for free, which is the vacuous-coverage shape
        # this file refuses everywhere else.
        if not keep:
            return {"supported": False,
                    "reason": f"every other request shares physical pages with "
                              f"request {source} ({shared_with_source}), so "
                              f"there is no independent sequence left whose "
                              f"logits a leak could move"}
        rows_before = np.concatenate([before[qsl[i]:qsl[i + 1]] for i in keep])
        rows_after = np.concatenate([after[qsl[i]:qsl[i + 1]] for i in keep])
        lo, hi = qsl[source], qsl[source + 1]
        out = {
            "supported": True,
            "before_perturbation": rows_before,
            "after_perturbation": rows_after,
            "perturbed_request": source,
            "compared_requests": keep,
            # Reported so a reader can see WHY a request is absent from the
            # comparison rather than having to infer it from the scenario.
            "excluded_as_page_sharers": shared_with_source,
            # GUARD THE GUARD: if the perturbation moved nothing at all, the
            # bit-equality above is satisfied by a probe that perturbed nothing,
            # and the check would read as coverage while grading air.
            "perturbation_moved_its_own_request": bool(
                not np.array_equal(before[lo:hi], after[lo:hi])),
        }
        gc.collect()
        return out

    def _invariance_chunking(self) -> dict[str, Any]:
        """The same prompt, chunked vs. single-shot, over the SAME final rows.

        Golden-free and self-comparing, like every probe in this family: over
        the seven I1 mutations that cleared Spec 17/17 and Numeric 9/9, the
        golden-free invariants caught 0/7, because both legs run the
        candidate. Worth having anyway -- chunking is this mode's defining
        axis -- but it is not a substitute for
        ``test_chunked_prefill_logits``'s frozen comparison: a defect that is
        wrong the SAME way with or without a chunk boundary (there is none
        here -- the single-shot leg never crosses one) would be invisible to
        this probe alone.

        THE CHUNKED LEG REUSES ``chunked_prefill_probe`` -- the exact call
        sequence ``test_chunked_prefill_logits`` grades -- rather than a
        second copy of its loop, so this probe can never disagree with that
        check about what "chunked" means. It is therefore unsupported for
        the same three reasons that probe is (no split points, a split
        outside the prompt, a descending pair): there is nothing here to
        compare against either.

        THE SINGLE-SHOT LEG is a FRESH ``Engine``. A fresh engine has no
        prior ``prefill`` call, which is the same starting state ``.reset()``
        would produce on the chunked leg's engine -- and that engine is
        already discarded inside ``chunked_prefill_probe``, so there is
        nothing here to call ``.reset()`` on. It forwards THE WHOLE IMAGE,
        which is the only composition that can be compared against the chunked
        leg: that leg hands the pixels to whichever chunk holds the
        placeholders, and one un-split call IS that chunk. This paragraph used
        to say the leg "forwards NO IMAGE, and it must not start forwarding one
        on its own" -- correct while the chunked leg forwarded none either, and
        wrong the moment it did. The two are still composed together and the
        gate is still the same one, so they cannot get out of step: this probe
        is unsupported whenever that one is, INCLUDING when the run is torn
        across a boundary.

        ON A TEXT-ONLY INSTANCE ``**self._image_inputs()`` EXPANDS TO NOTHING
        and this call is byte for byte the one it has always been.

        BOTH LEGS ARE SLICED TO THE SAME TAIL, not just their last row:
        ``chunked_prefill_probe``'s own ``logits`` IS already only the final
        chunk, and slicing the single-shot leg to match compares every row a
        chunk boundary could have gotten wrong, not only the very last one.
        The boundary arithmetic mirrors (rather than imports)
        ``chunked_prefill_probe``'s own -- a repeated interior split point is
        a zero-width no-op there too -- because this needs only the LAST
        real boundary, not the whole per-chunk loop.
        """
        chunked = self.chunked_prefill_probe()
        if not chunked.get("supported"):
            return {"supported": False, "reason": chunked.get("reason")}
        jnp = self._jnp
        # `self.prompt_ids`, NOT `self.forward_ids` -- see
        # `chunked_prefill_probe`'s own note on the same substitution. This
        # leg re-forwards sequence 0's FULL prompt in one shot, so it needs
        # that request's own token array, not the (possibly multi-request,
        # possibly narrower) graded-batch concatenation.
        ids = np.asarray(self.prompt_ids).reshape(-1).astype(np.int32)
        n = int(ids.shape[0])
        bounds = [0, *[int(s) for s in chunked["chunk_split_points"]], n]
        bounds = [b for i, b in enumerate(bounds) if i == 0 or b != bounds[i - 1]]
        start = bounds[-2]
        engine = self.engine_mod.Engine(self.model, self.run_config)
        single = engine.prefill(jnp.asarray(ids, dtype=jnp.int32), request_id=0,
                                **self._image_inputs())
        single_tail = _to_np(single.logits)[start:]
        self._record_allocation("invariance.chunking", "candidate")
        out = {
            "supported": True,
            "chunked": np.asarray(chunked["logits"]),
            "single_shot": single_tail,
        }
        del engine, single
        gc.collect()
        return out

    def _invariance_prefix_cache(self) -> dict[str, Any]:
        """A request served via an APC hit == the same request recomputed cold.

        Golden-free and self-comparing -- see ``_invariance_chunking``'s
        docstring for the 0/7 caveat this whole family carries.

        GUARD THE GUARD, TWICE OVER. Both findings are real properties of this
        class, not merely anticipated:

        1. ``PrefixCache.stats()["hits"]`` is a LIFETIME counter (``reset()``
           deliberately preserves it across a fresh pool), so an absolute
           read would see hits from whatever ran before this probe. This
           probe reads it as a DELTA across the sharing request's own
           ``prefill`` call.
        2. THE DELTA STILL DOES NOT PROVE THE CACHE WAS *USED*.
           ``PrefixCache.acquire`` increments ``_hits`` from its own hash
           lookup, independent of whether the engine actually SKIPS
           recomputing the shared tokens: an engine that
           calls ``acquire()`` (so the counter moves) and then discards
           ``cached`` for ``forward_ids``/``positions``/``slot_mapping``
           passes every check that reads only the hit counter or the
           returned page list, because both come from the SAME ``acquire()``
           call in the correct engine and in that mutant alike. What
           differs is what the forward actually COMPUTED:
           ``attn_metadata.input_positions`` has ``shared_prefix_len`` fewer
           entries and starts at ``shared_prefix_len`` in a correct
           implementation, and the full request starting at 0 in one that
           recomputed everything anyway. ``cache_hit`` below asserts on
           THAT, not on the counter -- the counter is kept only as
           corroborating evidence, ANDed in rather than trusted alone.

        THE "RECOMPUTED" LEG is a fresh ``Engine``/pool: the sharing
        request, prefilled alone, with nothing before it to share a page
        with -- a pure cold miss, and the logits an engine that ignored the
        cache entirely would also produce. The comparison the GRADED check
        makes (``cached`` == ``recomputed``) is exactly the property a cache
        is supposed to have: what you get must not depend on whether it was
        served from a shared page or computed fresh.

        NO IMAGE REACHES THE THREE ``Engine.prefill`` CALLS BELOW, and that is
        a declared boundary rather than an omission -- it belongs with
        ``chunked_prefill_probe``'s and ``_ragged_forward``'s. This probe reads
        its requests out of ``_ragged_requests()``, whose entries carry tokens
        and pages and no picture, and ``prefix_cache`` is in
        ``_MULTI_SEQUENCE_KINDS`` so it answers unsupported on every `core`
        instance -- which is every multimodal instance this class has. A
        multimodal `ragged` freeze would have to put each request's own image
        beside its own tokens before this could ask for one.
        """
        requests = self._ragged_requests()
        sharing = next((i for i, r in enumerate(requests)
                        if r.get("shares_prefix_with") is not None), None)
        if sharing is None:
            return {"supported": False,
                    "reason": "no request in the frozen scenario declares "
                              "shares_prefix_with; there is no prefix-cache "
                              "hit to compare"}
        source = int(requests[sharing]["shares_prefix_with"])
        shared_len = int(requests[sharing]["shared_prefix_len"])
        own_tokens = requests[sharing]["tokens"]
        n_own = len(own_tokens)
        jnp = self._jnp

        engine = self.engine_mod.Engine(self.model, self.run_config)
        source_ids = jnp.asarray(requests[source]["tokens"], dtype=jnp.int32)
        engine.prefill(source_ids, request_id=source)
        hits_before = int(engine.prefix_cache.stats()["hits"])
        own_ids = jnp.asarray(own_tokens, dtype=jnp.int32)
        cached_result = engine.prefill(own_ids, request_id=sharing)
        hits_after = int(engine.prefix_cache.stats()["hits"])
        cached_logits = _to_np(cached_result.logits)
        positions = [int(p) for p in
                     np.asarray(cached_result.attn_metadata.input_positions).reshape(-1)]

        fresh = self.engine_mod.Engine(self.model, self.run_config)
        recomputed_result = fresh.prefill(own_ids, request_id=sharing)
        # SLICED TO MATCH THE CACHED LEG'S SHAPE, and this is not optional.
        # `Engine.prefill` truncates the FORWARD on a hit, not only the
        # positions (`forward_ids = input_ids[cached:]`), so `cached_result
        # .logits` has only `n_own - shared_len` rows while this fresh,
        # cold engine returns all `n_own` -- the same tail-slicing
        # `_invariance_chunking` already does for the identical reason, one
        # method above. Comparing the two leg-wide would fail
        # `_assert_logit_parity`'s shape assertion for EVERY candidate,
        # including a correct one, precisely when the cache is working
        # (`shared_len > 0`, which `cache_hit` requires to ever read True) --
        # a check that cannot pass rather than one that cannot fail.
        recomputed_logits = _to_np(recomputed_result.logits)[shared_len:]

        self._record_allocation("invariance.prefix_cache", "candidate")
        out = {
            "supported": True,
            "cached": cached_logits,
            "recomputed": recomputed_logits,
            "hits_delta": hits_after - hits_before,
            "input_positions": positions,
            # GUARD THE GUARD: what the forward actually computed, not what
            # the cache observed. See the docstring's finding 2 -- the hit
            # counter alone cannot tell a real hit from one that was found
            # and then ignored, so this ANDs the counter moving with the
            # forward's own query actually being only the uncached tail.
            # RANK-AGNOSTIC, and the rank is the whole of what was wrong here.
            # `positions` above is FLATTENED for reporting, so on an M-RoPE
            # family it is `rank * q_len` long and `== list(range(...))` is
            # False for a perfectly correct engine -- `input_positions` is
            # `(3, q_len)` there and passed through at whatever rank it arrives
            # (`engine.py::_metadata`, which refuses to normalise it for exactly
            # this reason). On `chunked_qwen2_5_vl` the guard read
            # `[8, 9, 10, 8, 9, 10, 8, 9, 10]` against `[8, 9, 10]`, reported
            # `cache_hit: False` with `hits_delta: 1`, and failed
            # `test_prefix_cache_invariance` on the reference itself.
            #
            # THE PROPERTY IS UNCHANGED AND IS STILL THE STRONG ONE: every rope
            # row of the forward's own query must be exactly the uncached tail,
            # which is what says the engine SKIPPED the shared prefix rather
            # than finding the hit and recomputing anyway. Checking the column
            # COUNT alone would be the weaker check and would pass an engine
            # that forwarded the right number of the wrong tokens.
            "cache_hit": bool(hits_after > hits_before) and _rows_are(
                cached_result.attn_metadata.input_positions,
                list(range(shared_len, n_own))),
        }
        del engine, fresh, cached_result, recomputed_result
        gc.collect()
        return out

    # --- L3 battery ---------------------------------------------------------- #
    def battery_probe(self) -> dict[str, Any]:
        """Re-forward every frozen battery variant.

        ``scenario["battery"]`` is ``{axis: {variant: {...}}}``. A variant takes
        one of two shapes, and which one it is decided by what its own frozen
        ``inputs.npz`` carries rather than by the axis name:

        * ``{"input_ids": [...], "page_ids": [...]}`` -- a SINGLE-SEQUENCE
          variant, re-forwarded on that page assignment. The prompt-length and
          paging axes are this shape: the layout or the prompt varies while the
          composition does not.
        * a full handed metadata (``query_start_loc``, ``seq_lens``,
          ``block_tables``, ``slot_mapping``, and ``warmup_*`` where the variant
          has decode requests) -- a MIXED-STAGE variant, run through exactly the
          composition the canonical scenario runs through
          (:meth:`_handed_forward`). ``batch_composition`` is this shape, because
          the thing it varies IS the metadata: a variant that only handed over
          `input_ids` could not express "two decodes and one prefill" at all.

        Absent (nothing frozen yet) -> unsupported, and the battery tests are
        presence-deselected anyway.
        """
        battery = self.scenario.get("battery") or {}
        if not battery:
            return {"supported": False, "reason": "the golden froze no battery"}
        out: dict[str, Any] = {"supported": True, "axes": {}}
        for axis, variants in sorted(battery.items()):
            axis_out: dict[str, Any] = {}
            for name, variant in sorted(variants.items()):
                ids = np.asarray(variant["input_ids"]).reshape(-1).astype(np.int32)
                if "query_start_loc" in variant:
                    logits, _hs, cache, _m, _s = self._handed_forward(
                        variant, self.model, self.native_dtype,
                        want_hidden=False, probe="battery")
                    entry = {"logits": _to_np(logits),
                             "num_tokens": int(ids.shape[0]),
                             "query_start_loc": [
                                 int(x) for x in np.asarray(
                                     variant["query_start_loc"]).reshape(-1)],
                             "request_distribution": [
                                 int(x) for x in np.asarray(
                                     variant["request_distribution"]).reshape(-1)]}
                else:
                    # The variant's OWN frozen page assignment where it has one --
                    # that is what the paging axis varies. A variant that pins none
                    # falls back to the candidate's default layout, and the probe
                    # records that honestly rather than claiming the whole battery
                    # was handed its allocation.
                    pages = (self._frozen_pages("battery", variant["page_ids"])
                             if variant.get("page_ids")
                             else self._candidate_pages("battery",
                                                        self._default_pages()))
                    # `image=` FROM THE VARIANT, matching the mixed-stage leg
                    # above -- `_handed_forward` reads `self._image_inputs(
                    # variant)`, and the two legs of one battery must not source
                    # their picture differently against the same variant's ids.
                    logits, cache, _m, _s = self._prefill_on_pages(
                        pages, input_ids=ids, image=self._image_inputs(variant))
                    entry = {"logits": _to_np(logits),
                             "page_ids": pages,
                             "num_tokens": int(ids.shape[0])}
                axis_out[name] = entry
                del logits, cache
                gc.collect()
            out["axes"][axis] = axis_out
        return out

    def reference_units_probe(self) -> dict[str, Any]:
        """Per-submodule I/O replay (``test_reference_units``).

        The candidate's OWN submodule, run on the REFERENCE's recorded input, at
        fp32. Being handed the input is what makes the comparison LOCAL: a
        candidate whose layer 3 is wrong is still gradeable on layer 0's
        ``qkv_proj``, which it would not be if the activation came from its own
        upstream. Same "handed, not derived" discipline as for the
        allocation, applied to the activation stream.

        UNSUPPORTED ONLY WHEN THE GOLDEN IS ABSENT. ``reference_units.{json,npz}``
        is OPTIONAL and has its own producer
        (the freezer); without it
        ``conftest`` presence-deselects the check and this returns the marker.
        With it, every unit is replayed.

        A UNIT THAT RAISES IS RECORDED, NOT PROPAGATED, and that is a deliberate
        exception to this file's "a probe whose computation fails raises" rule.
        Every other probe drives the candidate's own composition, so an exception
        means the run is broken. This one calls NINE independent submodules with
        foreign inputs, and one of them refusing a tensor is a verdict about THAT
        submodule -- the exact thing the tier exists to localise. Letting it kill
        the worker would turn one wrong projection into an error on every check in
        all three tiers. The error text is carried out as a value and
        ``test_reference_units`` fails on it by name; nothing is swallowed.

        ``Exception``, NOT ``BaseException``, and the difference is not cosmetic:
        the broader form also catches ``SystemExit``, ``KeyboardInterrupt`` and
        ``GeneratorExit``, so a worker being torn down -- by the grade's own hard
        kill, by an OOM handler, by anything that raises outside the exception
        hierarchy -- would be reported to the reader as "your ``qkv_proj``
        raised". Localisation is only worth having if what it localises is the
        candidate. Everything a candidate submodule can legitimately do wrong is
        an ``Exception``.
        """
        spec = self.scenario.get("reference_units") or {}
        units = spec.get("units") or {}
        if not units:
            return {"supported": False,
                    "reason": "golden/reference_units.{json,npz} carries no frozen "
                              "submodule inputs; the per-submodule replay must run "
                              "on the reference's own recorded activations"}
        jnp = self._jnp
        # fp32: the units are frozen fp32 (fp32-robust tier). Uses the
        # cached fp32 model, so this probe must run inside the fp32 block of
        # `run_all` rather than build a second copy of the weights.
        model = self.fp32_model
        out: dict[str, Any] = {"supported": True, "units": {},
                               "layer_index": spec.get("layer_index")}
        for name, entry in sorted(units.items()):
            path, call = str(entry["torch_path"]), str(entry.get("call") or "__call__")
            target = resolve_module_path(model, path)
            fn = target if call == "__call__" else getattr(target, call, None)
            if target is None or not callable(fn):
                out["units"][name] = {
                    "supported": False, "path": path, "call": call,
                    "reason": f"your model exposes no callable at {path!r}"
                              + ("" if call == "__call__" else f".{call}")}
                continue
            args = []
            for value in entry["inputs"]:
                arr = np.asarray(value)
                args.append(jnp.asarray(
                    arr, dtype=(jnp.int32 if np.issubdtype(arr.dtype, np.integer)
                                else jnp.float32)))
            try:
                result = fn(*args)
            except Exception as exc:  # noqa: BLE001 - a verdict about ONE unit
                out["units"][name] = {
                    "supported": False, "path": path, "call": call,
                    "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
                continue
            values = result if isinstance(result, tuple) else (result,)
            out["units"][name] = {
                "supported": True, "path": path, "call": call,
                "outputs": [_to_np(v) for v in values],
            }
        gc.collect()
        return out

    # -- the image processor (multimodal families only) ------------------------ #
    def _optional_image_processing(self):
        """The ``image_processing`` module, or ``None`` if this family ships none.

        NOT IMPORTED IN ``__init__`` beside the nine every family has. Six of the
        seven instances in this class are text-only and ship no
        ``image_processing``; hoisting it up there would turn "this family has no
        image processor" into a worker that never starts, which scores every check
        in all four tiers at zero for a port that is complete.

        "ABSENT" AND "BROKEN" ARE DIFFERENT ANSWERS, which is why this is not a
        bare ``try: import``. ``find_spec`` asks only whether the module EXISTS;
        the import that follows is unguarded, so a module that is present and
        raises on import propagates that exception to ``run_all``'s per-probe
        capture and is reported as the port's fault. Swallowing both would let a
        multimodal port with a syntax error in one file read as "text-only" and
        deselect nothing -- the probe would answer the unsupported marker and the
        graded check would report a missing module that is right there.

        DELIBERATELY NOT PARAMETERISED over the module name, and the obvious
        refactor to ``_optional_module(name)`` is the bug this shape exists to
        prevent. ``test_no_upstream_imports.py`` walks the ast of everything under
        ``tasks/`` and treats ``importlib.import_module`` with a
        non-literal argument as an offender on sight -- not because a variable is
        suspicious, but because it cannot be CLEARED statically, so one such call
        blinds the guard to whatever it resolves to at runtime. A literal is the
        whole requirement. If a second optional module ever appears, give it its
        own method with its own literal rather than a shared parameter.
        """
        import importlib
        import importlib.util

        try:
            found = importlib.util.find_spec("image_processing") is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return None
        return importlib.import_module("image_processing")

    def _image_processor(self) -> tuple[Any, Any] | None:
        """``(image_processing module, ImageProcessorConfig)``, or ``None``.

        The config is read from the CHECKPOINT by the port's own
        ``ImageProcessorConfig.from_hf`` -- ``preprocessor_config.json``, beside
        the ``config.json`` that ``ModelConfig.from_hf`` reads -- rather than
        being handed over here. That is the whole point: ``min_pixels`` and
        ``max_pixels`` are what ``smart_resize``'s two clamps are computed
        against, this checkpoint's are NOT ``smart_resize``'s own signature
        defaults (3136 / 12845056 against 3136 / 1003520), and supplying them
        from the grader would grade the arithmetic while giving away the input
        the arithmetic is wrong without.
        """
        module = self._optional_image_processing()
        if module is None:
            return None
        return module, module.ImageProcessorConfig.from_hf(self.model_dir)

    #: What both image probes say when the family is text-only. One string, so the
    #: two cannot drift into disagreeing about the same condition.
    _NO_IMAGE_PROCESSOR = (
        "this repo has no top-level `image_processing` module. Expected on a "
        "text-only family, where the two image checks are deselected; on a "
        "multimodal one it is the module `serve.py` needs to turn an image into "
        "pixel_values and image_grid_thw")

    def image_preprocess_probe(self) -> dict[str, Any]:
        """The port's own image processor, on the frozen fixture's PNG BYTES.

        THE ENCODED PNG IS THE INPUT, not a decoded array, because the decode is
        part of what is graded: ``PIL`` is on
        ``check_inference_structure.ARCH_ALLOWED_IMPORTS`` for both reference
        trees precisely so the port owns this step, the way it owns tokenizing a
        raw prompt string. ``fixtures._image_inputs`` is what withholds the
        recording's own ``pixel_values`` from this payload.

        ``_to_np_with_provenance`` rather than ``_to_np``: the processor's output
        is float32 on both reference sides by construction (upstream's is
        dtype-blind and the tower casts on entry), and ``np.asarray`` cannot say
        afterwards that a port narrowed it to bf16 -- which would be a real
        divergence from the recording, visible in the value clause but much
        cheaper to read off the dtype.
        """
        spec = self.scenario.get("image") or {}
        png = spec.get("png_bytes")
        if not png:
            return {"supported": False,
                    "reason": "the frozen golden carries no image_fixture.png, so "
                              "there is nothing for the image processor to run on"}
        loaded = self._image_processor()
        if loaded is None:
            return {"supported": False, "reason": self._NO_IMAGE_PROCESSOR}
        module, config = loaded
        pixel_values, grid = module.preprocess(bytes(png), config)
        values, provenance = _to_np_with_provenance(pixel_values)
        return {
            "supported": True,
            "pixel_values": values,
            "pixel_values_dtype": provenance["dtype"],
            "pixel_values_srctype": provenance["type"],
            "image_grid_thw": _to_np(grid),
        }

    def smart_resize_probe(self) -> dict[str, Any]:
        """``smart_resize`` over the frozen table of NON-CONFORMING sizes.

        THE WHOLE TABLE IN ONE PROBE, not one call per case, because the probe
        surface is computed once in the worker and read many times in the parent
        -- there is no live candidate at check time to call again.

        A CASE THAT RAISES IS RECORDED, NOT PROPAGATED, the same exception
        ``reference_units_probe`` makes and for the same reason: this calls one
        pure function five times on deliberately awkward inputs (a 1x1 image, a
        4000x4000 one), and one of them refusing is a verdict about THAT input,
        which is exactly what the table exists to localise. ``Exception``, not
        ``BaseException`` -- a worker being torn down is not the port getting
        ``(1, 1)`` wrong.
        """
        spec = self.scenario.get("image") or {}
        cases = [[int(h), int(w)] for h, w in (spec.get("smart_resize_cases") or [])]
        if not cases:
            return {"supported": False,
                    "reason": "the frozen golden records no smart_resize cases, so "
                              "there is no table to reproduce"}
        loaded = self._image_processor()
        if loaded is None:
            return {"supported": False, "reason": self._NO_IMAGE_PROCESSOR}
        module, config = loaded
        targets: list[list[int] | None] = []
        errors: dict[int, str] = {}
        for index, (height, width) in enumerate(cases):
            try:
                target_h, target_w = module.smart_resize(height, width, config)
            except Exception as exc:  # noqa: BLE001 - a verdict about ONE case
                targets.append(None)
                errors[index] = f"{type(exc).__name__}: {str(exc)[:400]}"
                continue
            targets.append([int(target_h), int(target_w)])
        return {"supported": True, "cases": cases, "targets": targets,
                "errors": errors}

    # -- the multimodal MODEL path (multimodal families only) ------------------ #
    #: What the three probes below say when the frozen scenario carries no image.
    #: One string, so they cannot drift into disagreeing about the same condition.
    _NO_FROZEN_IMAGE = (
        "the frozen golden's inputs.npz carries no pixel_values/image_grid_thw "
        "pair, so there is no image for the tower or the merge to run on. "
        "Expected on a text-only family, where the multimodal checks are "
        "deselected; on a multimodal one it is a broken freeze.")

    def _frozen_image_inputs(self) -> tuple[Any, Any] | None:
        """``(pixel_values, image_grid_thw)`` off ``inputs.npz``, or ``None``.

        float32 pixel_values REGARDLESS OF THE SERVING DTYPE, which is the
        processor's own contract and the dtype the freezer drove the reference
        with: the tower casts
        on entry, so narrowing here would quantise the input twice and the native
        and fp32 windows would then see different pixels.

        int32 for the grid, matching every other integer array this file hands a
        candidate -- ``jax_enable_x64`` is off for the whole class, so an int64
        would be silently downcast with a warning.

        NOT ROUTED THROUGH ``_frozen``. That helper records an ALLOCATION policy,
        and these two arrays are model inputs, not a page assignment; recording
        "frozen" for them would put a claim about paging into
        ``allocation_source`` that the probe never made.

        THE PAIR VIEW OF :meth:`_image_inputs`, and it reads that method rather
        than the npz so the two cannot drift: the twenty-odd probes that forward
        the image and the three that consume it directly must be looking at the
        same two arrays at the same two dtypes, or a merge probe would be
        grading a picture no forward ever saw. This shape exists because its
        three callers want the arrays POSITIONALLY and want the ``None`` --
        `supported: False` with a reason is what they answer on a text-only
        family, where a bare ``{}`` would read as "no image supplied" instead.
        """
        image = self._image_inputs()
        if not image:
            return None
        return image["pixel_values"], image["image_grid_thw"]

    def multimodal_prefill_probe(self) -> dict[str, Any]:
        """The candidate's own ``Engine.prefill`` WITH the image, in one call.

        TWO CHECKS READ THIS ONE PROBE -- ``test_mrope_positions`` reads
        ``positions``, ``test_multimodal_prefill_logits`` reads ``logits`` -- and
        that coupling is deliberate rather than a shortcut. Both are properties of
        the same forward, and the alternative (a positions-only probe resolving
        ``Engine.POSITIONS_HOOK`` by name) would grade a hook a port is not
        obliged to have: a candidate whose ``__call__`` builds M-RoPE positions
        inline is correct and would report "unsupported" there. What the engine
        put in ``attn_metadata.input_positions`` is what the model ACTUALLY
        consumed, on every design. That field is ``input_positions`` and not
        ``positions``: ``inputs.npz``'s KEY is ``positions`` while
        ``AttentionMetadata``'s FIELD is ``input_positions``, and this docstring
        named the npz key for a dataclass attribute -- the same slip already
        fixed in the probe and in the shape-hint message. A port that believes
        this sentence adds a field the frozen
        schema forbids.

        THE ENGINE'S OWN ALLOCATOR, not the frozen one, and the golden is
        nevertheless the frozen forward's. The same model
        run through a HANDED forward -- frozen ``positions``, frozen
        ``block_tables``, frozen ``slot_mapping``, and the tower's output merged
        by ``prepare_model_inputs`` -- reproduces this probe's logits BIT FOR BIT
        (identical ``cosine_similarity``, ``token_kl``, ``max_abs_diff`` and
        per-token centered cosine to the last digit, and the identical argmax on
        all eleven rows). Page assignment cannot move a logit: attention gathers
        through the block table, and which physical page a key lives on does not
        change the order it is reduced in. Handing a frozen block table in here
        instead would mean reimplementing ``Engine.prefill``'s allocation in this
        file, the same argument ``generation_probe``'s ``USE_FROZEN_ALLOCATION``
        entry makes.

        WHAT IT ADDS OVER ``test_prefill_logits``, which grades the same golden.
        That check drives :meth:`_handed_forward`, which passes ``input_ids`` and
        the FROZEN positions and no image at all; this one hands the engine the
        pixels and lets it derive its own positions, so it is the only graded path
        on which the M-RoPE builder, the host-side tower hook and the embedding
        scatter are all live at once. A port can pass one and fail the other, in
        either direction, and the pair is what says which half is wrong.

        ``prompt_ids`` -- SEQUENCE 0 -- AND NOT ``forward_ids``. This probe drives
        ``Engine.prefill``, which is SINGLE-REQUEST by contract
        (``golden/interface.json`` pins ``prefill``/``decode_step``/``generate``/
        ``reset``, all of them one request), and the frozen image it is handed is
        sequence 0's. On a
        `core` instance the two id arrays are the same array and nothing about
        this call moves. On a `ragged` or `chunked` one ``forward_ids`` is the
        MIXED BATCH's query tokens concatenated, and feeding that concatenation
        to a single-sequence prefill runs three unrelated requests as one
        sequence: on ``ragged_qwen2_5_vl`` the M-RoPE positions then
        come back ``[... 8, 9, 10]`` where the frozen batch holds ``[... 8, 4,
        8]`` -- the two trailing entries are sequences 1 and 2's own decode
        positions, not a continuation of sequence 0's run -- and the logits
        then diverge from the frozen row.

        SO THE CALLER COMPARES A WINDOW, and this probe returns the WHOLE of
        sequence 0's run rather than pre-slicing it. That is the freezer's own
        rule read back (``_build_batch``: positions are "SLICED OUT OF THE WHOLE
        SEQUENCE'S, never rebuilt from the slice", because an image collapses
        several tokens onto fewer positions and recomputing from ``tokens[start:]``
        restarts the run and loses the offset), and it keeps the choice of window
        in the grading process -- ``Golden.graded_window``, derived from
        ``query_start_loc`` and ``seq_lens``, which the candidate does not pick.
        The alternative considered and rejected was a single-sequence golden of
        this probe's own: it would freeze rows that ``prefill_forward.npz``
        already holds, add an artifact to every multimodal instance, and let the
        two copies drift -- the same argument ``Golden.decode_margins`` makes for
        deriving sequence 0's prefill row instead of freezing it.
        """
        jnp = self._jnp
        image = self._frozen_image_inputs()
        if image is None:
            return {"supported": False, "reason": self._NO_FROZEN_IMAGE}
        pixel_values, grid = image
        engine = self.engine_mod.Engine(
            self.model, self._generation_run_config(fp32=False))
        result = engine.prefill(
            jnp.asarray(self.prompt_ids, dtype=jnp.int32),
            pixel_values=pixel_values, image_grid_thw=grid)
        # POINT OF USE: `prefill` allocated its own pages; nothing frozen entered
        # it but the prompt and the image.
        self._record_allocation("multimodal_prefill", "candidate")
        # `input_positions` IS THE FIELD NAME, and it is not the same string as
        # `inputs.npz`'s `positions` key. `AttentionMetadata`'s frozen dataclass
        # schema (golden/interface.json) spells it `input_positions` and
        # `test_attention_metadata_schema` already grades that spelling, so this
        # reads the contracted name and nothing else -- a `getattr` chain that
        # also accepted `positions` would silently grade a port that renamed the
        # field, which is exactly what that other check exists to refuse.
        positions = getattr(result.attn_metadata, "input_positions", None)
        if positions is None:
            raise ValueError(
                "your PrefillResult.attn_metadata carries no `input_positions` "
                "field. It is part of AttentionMetadata's frozen dataclass "
                "schema (golden/interface.json), and it is where the M-RoPE "
                "position ids the model consumed are observable.")
        positions_np, positions_prov = _to_np_with_provenance(positions)
        logits_np, logits_prov = _to_np_with_provenance(result.logits)
        out = {
            "supported": True,
            # int64 on the way out, like every other id array this file returns:
            # the VALUES are what is compared and numpy's default integer width
            # keeps a comparison from failing on a dtype nobody chose.
            "positions": np.asarray(positions_np).astype(np.int64),
            # ...and the PRE-conversion dtype beside it, because `np.asarray`
            # cannot say afterwards that a port returned float positions. A rope
            # index is an integer; a float one that happens to hold whole numbers
            # compares equal and is a different contract.
            "positions_dtype": positions_prov["dtype"],
            "positions_shape": [int(d) for d in np.shape(positions_np)],
            "logits": logits_np,
            "logits_dtype": logits_prov["dtype"],
            "requested_dtype": jnp.dtype(self.native_dtype).name,
            # SEQUENCE 0'S OWN LENGTH, which is what this probe forwarded. The
            # caller windows both arrays with it, so a probe that drove some
            # other id array is a length mismatch the check reports rather than
            # a silent misalignment.
            "num_tokens": int(self.prompt_ids.shape[0]),
        }
        del engine, result
        gc.collect()
        return out

    def image_embedding_merge_probe(self) -> dict[str, Any]:
        """The tower's rows, the text embeddings, and the merge of the two.

        ALL THREE FROM THE CANDIDATE, and the check that reads them compares them
        AGAINST EACH OTHER rather than against a frozen table. That is the only
        way to state the property that matters: the merged sequence must carry the
        TOWER's rows at exactly the ``<|image_pad|>`` offsets and the TEXT
        embedding's rows everywhere else. A port that APPENDS the image rows
        instead of scattering them -- the single most common way to get this wrong,
        and one that leaves every shape plausible when the placeholder run happens
        to sit at the start or end -- produces a merged table that matches neither
        source at the offsets, and no frozen array is needed to see it.

        THE THREE CALLS ARE THE REFERENCE'S OWN PUBLIC METHODS, all three named in
        ``golden/interface.json`` and therefore already required of a candidate by
        ``test_interface``: ``get_image_embeds(pixel_values, image_grid_thw)``,
        ``get_input_embeddings(input_ids)`` for the text-only table, and
        ``get_input_embeddings(input_ids, image_embeds)`` for the merged one.

        THE PLACEHOLDER OFFSETS ARE DERIVED FROM THE PORT'S OWN CONFIG, not handed
        in: ``config.image_token_id`` is what the model's own scatter keys on, so
        reading it here grades the same value the merge used. A port that resolved
        the wrong id scatters into the wrong rows AND reports those rows, which
        would cancel -- so the check cross-checks the offsets against the frozen
        prompt ids, which this file does not let the candidate choose.
        """
        image = self._frozen_image_inputs()
        if image is None:
            return {"supported": False, "reason": self._NO_FROZEN_IMAGE}
        pixel_values, grid = image
        jnp = self._jnp
        model = self.model
        ids = jnp.asarray(self.forward_ids, dtype=jnp.int32)
        for name in ("get_image_embeds", "get_input_embeddings"):
            if not callable(getattr(model, name, None)):
                return {"supported": False,
                        "reason": f"your model exposes no callable {name!r}; it "
                                  "is in golden/interface.json and is what turns "
                                  "an image into rows and writes them over the "
                                  "<|image_pad|> placeholders"}
        # THE IDS AND THE IMAGE MUST COME FROM THE SAME FORWARD. This probe's
        # subject is the SCATTER -- the tower's rows landing on exactly the
        # `<|image_pad|>` offsets -- so it has to run on the ids that HAVE those
        # offsets. `self.forward_ids` is the GRADED forward's, and on an
        # instance whose warm-up owns the picture (a multimodal sequence 0 that
        # is also a `prefill_chunked`) those ids carry no placeholders at all
        # and `get_input_embeddings` refuses. See `_image_owner_forward`.
        ids = jnp.asarray(self._image_owner_forward()[0], dtype=jnp.int32)
        image_embeds = model.get_image_embeds(pixel_values, grid)
        text_embeds = model.get_input_embeddings(ids)
        merged = model.get_input_embeddings(ids, image_embeds)
        token_id = getattr(self.model_config, "image_token_id", None)
        return {
            "supported": True,
            "image_embeds": _to_np(image_embeds),
            "text_embeds": _to_np(text_embeds),
            "merged": _to_np(merged),
            # The rows the CANDIDATE's own config says are placeholders. Reported
            # rather than asserted here: the check compares them against the
            # frozen prompt's own runs of that id, which is where the comparison
            # has content.
            "image_token_id": None if token_id is None else int(token_id),
            # THE OWNING FORWARD'S ids, matching `merged` above -- reporting the
            # graded forward's rows beside a merge computed on the warm-up's
            # would make the check compare two different sequences.
            "placeholder_rows": (
                [] if token_id is None else
                [int(i) for i in np.flatnonzero(
                    np.asarray(self._image_owner_forward()[0])
                    == int(token_id)).tolist()]),
        }

    def vision_tower_units_probe(self) -> dict[str, Any]:
        """Per-tower-submodule I/O replay, plus the window geometry.

        ``reference_units_probe`` one tier over, and it copies that method's two
        load-bearing properties verbatim:

        * the candidate's OWN submodule is run on the REFERENCE's recorded input,
          so a broken ``patch_embed`` cannot contaminate the verdict on ``merger``;
        * a unit that RAISES is recorded, not propagated (``Exception``, never
          ``BaseException`` -- a worker being torn down is not the port getting
          window attention wrong), so one faulty submodule localises instead of
          killing every check in the run.

        THE GEOMETRY IS THE ADDITION, and it is why this probe exists at all
        rather than being three more rows in the decoder's unit table.
        ``encoder_metadata(grid_thw)`` is pure geometry -- it touches no parameter
        -- and at the graded prompt's own grid its answer is DEGENERATE: the
        window permutation is the identity and ``cu_window_seqlens`` equals
        ``cu_seqlens``, so a tower with no window attention reproduces it exactly.
        The recording is therefore made at a 140x140 fixture whose grid (1, 10, 10)
        gives four windows, and the frozen geometry is compared exactly on the
        integers. The freezer REFUSES to write a
        degenerate one, so the check cannot go quietly vacuous behind a re-freeze.

        A 1-D INTEGER ARRAY IS WHAT ``cu_seqlens`` CROSSES AS, and the two block
        units are handed it exactly as recorded. The torch tower calls
        ``cu_seqlens.tolist()`` and the JAX one ``[int(b) for b in cu_seqlens]``;
        an array satisfies both, a Python tuple only the second, and a torch
        tensor only the first -- so the array is the one spelling that replays on
        either side.
        """
        spec = self.scenario.get("vision_tower_units") or {}
        units = spec.get("units") or {}
        if not units:
            return {"supported": False,
                    "reason": "golden/vision_tower_units.{json,npz} carries no "
                              "frozen tower submodule inputs; the per-submodule "
                              "replay must run on the reference's own recorded "
                              "activations"}
        jnp = self._jnp
        # fp32: the tower units are frozen fp32, the same tier and the same
        # reason `reference_units_probe` gives, so this probe reads the cached
        # fp32 model and must run inside `run_all`'s fp32 block.
        model = self.fp32_model
        out: dict[str, Any] = {"supported": True, "units": {}, "geometry": None}
        for name, entry in sorted(units.items()):
            path, call = str(entry["torch_path"]), str(entry.get("call") or "__call__")
            target = resolve_module_path(model, path)
            fn = target if call == "__call__" else getattr(target, call, None)
            if target is None or not callable(fn):
                out["units"][name] = {
                    "supported": False, "path": path, "call": call,
                    "reason": f"your model exposes no callable at {path!r}"
                              + ("" if call == "__call__" else f".{call}")}
                continue
            args = []
            for value in entry["inputs"]:
                arr = np.asarray(value)
                args.append(jnp.asarray(
                    arr, dtype=(jnp.int32 if np.issubdtype(arr.dtype, np.integer)
                                else jnp.float32)))
            try:
                result = fn(*args)
            except Exception as exc:  # noqa: BLE001 - a verdict about ONE unit
                out["units"][name] = {
                    "supported": False, "path": path, "call": call,
                    "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
                continue
            values = result if isinstance(result, tuple) else (result,)
            out["units"][name] = {
                "supported": True, "path": path, "call": call,
                "outputs": [_to_np(v) for v in values]}

        # THE GEOMETRY, from the tower the units were resolved against. Recorded
        # per key rather than positionally so a failure names `cu_window_seqlens`
        # instead of "output 3", and captured under its own try for the same
        # localisation reason the unit loop has.
        method = str(spec.get("geometry_method") or "encoder_metadata")
        tower = resolve_module_path(model, "visual")
        fn = getattr(tower, method, None) if tower is not None else None
        if not callable(fn):
            out["geometry"] = {
                "supported": False,
                "reason": f"your vision tower exposes no callable {method!r} at "
                          "`visual`; it is in golden/interface.json and it is "
                          "where the window permutation is computed"}
        else:
            grid = [[int(v) for v in row] for row in (spec.get("grid_thw") or [])]
            try:
                values = fn(grid)
                out["geometry"] = {
                    "supported": True,
                    # `np.asarray` on each value, because a tuple of Python ints
                    # (`cu_seqlens` on the JAX side) and a device array
                    # (`window_index`) both have to arrive as one comparable
                    # thing; `_to_np` handles the second and not the first.
                    "values": {str(k): np.asarray(_to_np(v) if hasattr(v, "shape")
                                                  else v)
                               for k, v in dict(values).items()},
                }
            except Exception as exc:  # noqa: BLE001 - a verdict about the geometry
                out["geometry"] = {
                    "supported": False,
                    "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
        gc.collect()
        return out

    def loader_facts_probe(self) -> dict[str, Any]:
        """What the loader resolved, so a spec check can assert the /data contract.

        ``model_name_or_path`` is ECHOED from the payload and is labelled as such:
        the grader hands the repo id in and the candidate never resolves one, so a
        check comparing it to ``meta['repo_config']`` compares a value to itself.
        It stays because a failure message needs to say which model was asked for;
        the key that can actually FAIL is ``model_dir`` (what
        ``resolve_model_dir`` made of it) and ``rejects_unknown_model``.

        NOT IN :data:`COMMON_MODE_PROBES`, deliberately, and see that table for
        the measurement. Nothing downstream reads this probe's result: it is a
        REPORT, not an input, so a raise in here is captured by ``run_all`` like
        any other probe's and costs exactly the one graded check that reads the
        accessor -- ``test_spec.test_loader_spec``. Do not add a second reader
        without moving that claim with it.
        """
        return {
            "data_root": str(self.loader_mod.DATA_ROOT),
            "model_dir": str(self.model_dir),
            # Echoed, not resolved -- see the docstring. Do not assert on it.
            "requested_model_name_or_path": str(self.repo_config["model_name_or_path"]),
            "has_tokenizer": bool(
                callable(getattr(self.loader_mod, "load_tokenizer", None))
                and self.loader_mod.load_tokenizer(self.model_dir) is not None
            ),
            # COERCED FOR THE SAME REASON `serve_probe`'s payload is, and found
            # while enumerating that surface: the ROWS are the candidate's
            # module-level constant, so their leaves are whatever it put there,
            # and a leaf that will not pickle takes the whole grade down from
            # OUTSIDE any probe body where `produce` could catch it. A no-op on
            # every reference (`interface.json` freezes `[str, str, int]` rows,
            # and `test_loader_spec` compares against that JSON), and on a
            # candidate that put something else there it fails that one check by
            # name instead of collapsing the run.
            "stacked_params": _json_safe([list(row) for row in
                                          getattr(self.loader_mod,
                                                  "STACKED_PARAMS", ())]),
            # THE OFFLINE HALF OF THE /data CONTRACT, and the one loader fact that
            # is not an echo. `resolve_model_dir` must REFUSE a repo id that is not
            # baked into the image rather than reaching for the hub or handing back
            # a path nothing checked -- grading has no network, so a loader that
            # falls through returns a directory whose weights do not exist and the
            # failure surfaces hundreds of lines later. The exception TYPE is
            # reported, not just a boolean, so "refused" and "crashed for some other
            # reason" stay distinguishable.
            "rejects_unknown_model": self._rejects_unknown_model(),
        }

    def _rejects_unknown_model(self) -> dict[str, Any]:
        """What ``resolve_model_dir`` does with a repo id nothing bakes.

        ``{"raised": None}`` means it RETURNED, which is the failure. Otherwise the
        exception's type name and message are both reported: "it raised something"
        is a weaker fact than the check wants, because a loader that is simply broken
        also raises. The type is what separates "refused this lookup" from "crashed",
        and only a reported type lets the spec check say so. The id is obviously
        synthetic, so it cannot collide with a real checkpoint in any image.

        ``Exception`` and NOT ``BaseException``: a ``SystemExit`` or a
        ``KeyboardInterrupt`` out of ``resolve_model_dir`` is the worker being torn
        down, and reporting it here would tell ``test_loader_spec`` "your loader
        refused" -- a PASS -- on a run that was killed. Those propagate to the
        worker's own top-level handler, which reports them to the parent as a
        ``CandidateError``. Same fix, and the same reason, as the per-unit handler
        in ``reference_units_probe``.
        """
        try:
            self.loader_mod.resolve_model_dir("t2j-not-a-real-org/t2j-not-a-real-model")
        except Exception as exc:  # noqa: BLE001 - the TYPE is the evidence
            return {"raised": type(exc).__name__, "message": str(exc)[:400]}
        return {"raised": None, "message": "resolve_model_dir returned a path"}

    def resolved_config_probe(self) -> dict[str, Any]:
        """The candidate's OWN ``RunConfig``, read back AFTER ``__post_init__``.

        Not an echo of the payload, and the difference is measurable: the grader
        builds ``RunConfig(**repo_config)``, but what the object then HOLDS is
        whatever the candidate's own ``__post_init__`` left there. A port that
        clamps, defaults or silently overrides a knob shows up here as a value that
        is not the one it was handed -- for example a mutation that does exactly
        that (``object.__setattr__(self, 'num_pages', 16)``).

        The FIELD LIST is the candidate's too (``__dataclass_fields__``), not the
        contract's, so a ``RunConfig`` that dropped a field reports a short map and
        the spec check names the missing knob -- rather than the driver dying three
        frames down on ``int(self.run_config.<field>)``.

        Values are coerced to int / float / str / bool so the map survives the pickle
        back across the sandbox boundary whatever a candidate annotated them as.
        """
        def _plain(value: Any) -> Any:
            if isinstance(value, bool) or value is None:
                return value
            if isinstance(value, (int, float, str)):
                return value
            return str(value)

        fields = tuple(getattr(self.run_config, "__dataclass_fields__", {}) or {})
        return {
            "fields": list(fields),
            "values": {name: _plain(getattr(self.run_config, name, None))
                       for name in fields},
        }

    # --- orchestration ------------------------------------------------------ #
    def run_all(self) -> dict[str, Any]:
        """Produce every probe, in an order chosen for peak device memory.

        ``produce`` is PER-PROBE, NOT WHOLE-RUN. A bare call would let a swallowed
        exception plus a ``None`` pass vacuously as a numeric test -- but it does
        not follow that the whole run must die with it: the cost would be that ANY
        probe's exception scored the floor, the same a candidate that never
        imports earns, while ``instruction.md`` promises that a partial port earns
        partial credit.

        A captured failure is a value that RAISES ON READ (see
        :meth:`Candidate._probe`), so every check that touches the failed probe
        still fails with the traceback and nothing passes vacuously; the checks
        that do not touch it are graded normally. :data:`COMMON_MODE_PROBES` is
        the deliberate exception AMONG THE PROBES -- those failures still take the
        whole run down, because isolating them would not produce partial credit,
        only 29 identical tracebacks.

        IT IS NOT THE ONLY COMMON-MODE SURFACE IN THE WORKER, and reading it as
        one is wrong in the direction that matters. Everything in
        :func:`_worker_entry` that is not a probe is equally global: building
        :class:`_CandidateRuntime` (the import and the conversion, which is what
        :data:`COMMON_MODE_PROBES` is really a proxy for), the
        ``relative_golden_hits`` walk, and the assembly of ``sandbox_report``
        itself. Each of those raising aborts the worker exactly as a common-mode
        probe does. What the class DOES guarantee is that the two diagnostics
        most exposed to an environmental failure -- ``_fresh_import_would_find``
        and the transitive ``subprocess.run`` -- are wrapped where they stand and
        report a sentinel instead.

        AND THE PIPE ITSELF IS A COMMON-MODE SURFACE, which this enumeration
        must not omit -- omitting it would let ``serve_probe`` ship a verbatim
        candidate object graph. Every result
        below crosses to the parent in ONE ``conn.send``/``recv()`` pair, so a
        value that raises during (un)pickling aborts the grade from OUTSIDE any
        probe body, where no ``try`` in this function can see it.

        THE RULE THAT CLOSES IT IS AT THE POINT OF PRODUCTION, NOT HERE: a probe
        must return numpy / int / float / str / bool / None and containers of
        those. :func:`_json_safe` is how a payload whose leaves belong to the
        CANDIDATE is made to -- the two that do are ``serve_probe``'s
        ``run_serve`` return and ``loader_facts``' ``stacked_params`` rows, and
        both go through it.

        IT IS A RULE THE CLASS FOLLOWS, NOT AN INVARIANT ANYTHING ENFORCES, and
        stating it the other way would be this file's fifth mechanism-asserting
        sentence with no mechanism. A KNOWN LIVE EXCEPTION, unfixed and recorded
        rather than papered over: ``vision_tower_units_probe``'s geometry leg
        does ``np.asarray(_to_np(v) if hasattr(v, "shape") else v)`` over the
        candidate's own returned mapping, and ``np.asarray`` of a
        non-array object is a 0-d OBJECT-dtype array still holding it. That
        satisfies "numpy" literally and puts a candidate class into the pickle
        anyway -- the same collapse, on the VL surface, plus the ``np.asarray``
        hang :func:`_json_safe`'s own array clause is narrowed against. Do not
        read the paragraph above as covering it.

        Probes whose FROZEN INPUT is legitimately absent return an unsupported
        marker from inside themselves; that is a value, and exactly one test
        asserts on it.
        """
        results: dict[str, Any] = {}

        def produce(name: str, fn, **kwargs) -> None:
            # `BaseException` HERE AND `Exception` ON THE PARENT'S RECEIVE SIDE
            # (`_recv_within`'s caller), DELIBERATELY, and the asymmetry is the
            # decision rather than an oversight. THIS side runs INSIDE the
            # candidate's own worker, where a `SystemExit` is something the
            # candidate's code did -- a port that calls `sys.exit()` out of a
            # probe has produced a probe fault, and capturing it as one is the
            # accurate reading; letting it through would exit the worker and
            # charge all 29 checks for a single probe. THE PARENT side is the
            # grader's own process: a `KeyboardInterrupt` or `SystemExit` there
            # is the OPERATOR or the harness, never the candidate, so it is
            # narrowed to `Exception` so an interrupted grade stops instead
            # of being recorded as a candidate failure. Same words, opposite
            # owners.
            try:
                results[name] = fn(**kwargs)
            except BaseException:                       # noqa: BLE001 - re-raised on read
                if name in COMMON_MODE_PROBES:
                    raise
                results[name] = {"__probe_error__": traceback.format_exc()}
            gc.collect()

        # Cheap, conversion-only.
        produce("params_spec", self.params_spec_probe)
        produce("param_norms", self.param_norms_probe)
        produce("cache_spec", self.cache_spec_probe)
        produce("loader_facts", self.loader_facts_probe)
        produce("resolved_config", self.resolved_config_probe)

        # BEFORE anything else compiles. This probe calls ``jax.clear_caches()``, so
        # running it later would evict programs the earlier probes had already paid
        # for; running it here means everything after it warms from what it left
        # behind. The one cost is that the engine below re-traces (a fresh
        # ``nnx.jit`` wrapper), which is one compilation.
        produce("compilation", self.compilation_probe)

        # The candidate's own allocator (the three grading probes).
        # NOT routed through `produce`: it fans one call into FOUR required
        # results (plus `derived_page_ids`), so an uncaught exception here would
        # otherwise kill the whole run even though nothing else reads the candidate's own
        # allocator -- every allocation-DEPENDENT probe below is handed the
        # FROZEN block table / slot mapping instead, so a broken
        # allocator cannot cascade into them. Captured the same way `produce`
        # captures: all four (five) downstream keys inherit ONE traceback, so a
        # read of any of them raises naming itself, and nothing else is affected.
        #
        # `BaseException`, matching `produce` above rather than the parent's
        # receive-side `Exception`: this is worker-side capture of the
        # CANDIDATE's own allocator, so a `sys.exit()` from inside it is a
        # probe fault and is recorded as one. See `produce`'s comment for the
        # full statement of the asymmetry and why it is intentional.
        try:
            allocator = self.allocator_probe()
        except BaseException:                            # noqa: BLE001 - re-raised on read
            _allocator_error = {"__probe_error__": traceback.format_exc()}
            results["metadata_schema"] = dict(_allocator_error)
            results["block_table"] = dict(_allocator_error)
            results["slot_mapping"] = dict(_allocator_error)
            results["static_shapes"] = dict(_allocator_error)
            results["derived_page_ids"] = dict(_allocator_error)
        else:
            results["metadata_schema"] = allocator["metadata_schema"]
            results["block_table"] = allocator["block_table"]
            results["slot_mapping"] = allocator["slot_mapping"]
            results["static_shapes"] = allocator["static_shapes"]
            results["derived_page_ids"] = allocator["page_ids"]
        gc.collect()

        produce("traceability", self.traceability_probe)

        # THE IMAGE PROCESSOR, and it is here for two independent reasons. It
        # touches neither the model nor the engine -- it is a PNG and some float32
        # arithmetic -- so it costs nothing wherever it sits; and it must be AFTER
        # `compilation` above, which calls `jax.clear_caches()` and would evict
        # whatever these traced. Both answer `supported: False` on a text-only
        # family, where the checks that read them are deselected anyway.
        produce("image_preprocess", self.image_preprocess_probe)
        produce("smart_resize", self.smart_resize_probe)

        # THE MULTIMODAL MODEL PATH, native precision. Beside `generation` in
        # spirit -- both build an `Engine` of their own -- but placed here, before
        # the native prefill, for one deliberate reason: `multimodal_prefill` runs
        # the vision tower ON THE HOST (the reference resolves the tower through
        # `Engine.INPUTS_HOOK` because on jax 0.11 every array born inside a
        # `jit` trace is a tracer and `vision.py`'s window permutation cannot
        # execute under one), so it compiles a DECODER-ONLY graph whose shapes are
        # the same ones every probe below re-uses. `image_embedding_merge` runs
        # the tower eagerly and compiles nothing at all. Both answer
        # `supported: False` on a text-only family, where the checks reading them
        # are deselected.
        produce("multimodal_prefill", self.multimodal_prefill_probe)
        produce("image_embedding_merge", self.image_embedding_merge_probe)

        # Native-precision numeric tier. `prefill` caches the post-prefill cache
        # for `kv_cache_state` and `decode_steps`, so the three share one forward.
        produce("prefill", self.prefill_probe, fp32=False)
        if _is_probe_error(results["prefill"]):
            # `kv_cache_state` is a SIDE EFFECT of a successful `prefill_probe`
            # (it stashes `self._kv_cache_state` just before returning); a failed
            # prefill never reaches that line, so the plain `getattr(..., None)`
            # below would otherwise hand back a bare `None`. `_validate_results`'
            # `empty` guard treats a required probe's `None` as a swallowed
            # exception and aborts the WHOLE run -- exactly the global collapse
            # this isolation exists to prevent. Inheriting prefill's own captured
            # failure keeps the abort scoped to `kv_cache_state`'s own readers.
            results["kv_cache_state"] = results["prefill"]
        else:
            results["kv_cache_state"] = getattr(self, "_kv_cache_state", None)
        produce("decode_steps", self.decode_steps_probe)
        produce("paged_attention_unit", self.paged_attention_unit_probe, fp32=False)
        produce("sampler_outputs", self.sampler_probe)
        produce("generation", self.generation_probe, fp32=False)
        # Beside `generation`, not beside `prefill` above: both drive a fresh
        # `Engine` of their own at native dtype, and `chunked_prefill_probe`'s
        # own docstring is what explains why (`USE_FROZEN_ALLOCATION` False,
        # candidate's own allocator). Cheap on every instance shipped so far --
        # `core`/`ragged`/`core_gpt_oss` freeze an empty `chunk_split_points`,
        # so this returns `supported: False` without building an `Engine`.
        produce("chunked_prefill", self.chunked_prefill_probe)
        produce("invariance", self.invariance_probes)
        produce("battery", self.battery_probe)

        # fp32 last: it holds a second copy of the weights, and freeing it before
        # anything else runs keeps the peak at one model plus one upcast rather
        # than two models plus a decode cache.
        produce("fp32_prefill", self.prefill_probe, fp32=True)
        produce("fp32_paged_attention_unit", self.paged_attention_unit_probe, fp32=True)
        # The SECOND full N-step decode loop, and it is inside the fp32 block for
        # the same reason `reference_units` is: it reads `self.fp32_model`, so
        # running it beside the native `generation` above would hold a second copy
        # of the weights alive across every probe in between.
        produce("fp32_generation", self.generation_probe, fp32=True)
        # INSIDE the fp32 block, and that is the whole reason it is here rather
        # than beside `battery` above: the units are frozen fp32, so the probe
        # reads `self.fp32_model`, and running it earlier would keep a second copy
        # of the weights alive across the generation and invariance probes -- the
        # peak this ordering exists to hold down.
        produce("reference_units", self.reference_units_probe)
        # INSIDE the fp32 block, beside `reference_units` and for the identical
        # reason: the tower units are frozen fp32, so this probe reads
        # `self.fp32_model`, and running it earlier would keep a second copy of
        # the weights alive across every probe in between.
        produce("vision_tower_units", self.vision_tower_units_probe)
        self._release_fp32()

        # AFTER `_release_fp32()` -- not inside the fp32 block above, which is
        # where the fp32 label would suggest `held_out` belongs. Neither probe
        # reads `self.fp32_model`: each calls `serve.run_serve`, which builds its
        # OWN model internally from its own config (that is the point -- they
        # grade the candidate's entry point, not a model this file hands it).
        # Registering them in the fp32 block would therefore hold THREE copies of
        # the weights resident at once -- the native one, this file's upcast one,
        # and serve.py's -- which is exactly the peak this whole ordering exists to
        # hold down. `serve_probe`'s by-hand leg reads `self.model`, the NATIVE
        # model, which is still alive after the release above. NOT the last
        # produce() call any more -- see `perf` immediately below.
        produce("serve", self.serve_probe)
        produce("held_out", self.held_out_probe)

        # ACTUALLY LAST: timing must not perturb anything a
        # numeric or behavioral check reads, so `perf` runs after every other
        # probe, correctness or otherwise. `k_values` comes off the payload
        # (`instance.yaml::perf_k_values`, mirrored into `meta.json`); an
        # instance that declares none gets `perf_probe`'s own
        # `supported: False` branch rather than a crash, and `produce`'s capture
        # would isolate a raise here to this one probe regardless.
        produce("perf", self.perf_probe, k_values=self.perf_k_values)

        # OBSERVED, not declared. Built from what the probes recorded at their
        # points of use above -- iterating USE_FROZEN_ALLOCATION here instead
        # would make every assertion on this map an assertion about the table.
        results["allocation_source"] = dict(sorted(self._allocation_source.items()))
        return results


def _F():
    """``_formulas``, imported lazily so the parent never pulls jax in."""
    from . import _formulas

    return _formulas


# =========================================================================== #
# Subprocess plumbing.
# =========================================================================== #
def _worker_entry(conn, payload: dict[str, Any], banned: list[str]) -> None:
    """The one child. Blocks torch FIRST, then imports anything.

    Order is the whole point: ``install_torch_blocker`` goes in before the
    candidate's ``sys.path`` entry and before ``import jax``, so a candidate that
    reaches for torch cannot be served by an already-loaded module. The blocker
    is also PROVEN live rather than assumed -- ``sandbox_report`` records what
    ``import torch`` actually raised, and its exception TYPE is what a test keys
    off: in an environment that has no torch at all, a missing blocker and a
    working blocker both "fail to import torch", and only the type tells them
    apart.

    IT ALSO MOVES ITS CWD, AND THAT IS A SECURITY FIX RATHER THAN HYGIENE.
    Without the chdir, from the cwd this child would
    inherit, ``Path('.').rglob('golden/fp32_generation.npz')`` returns the
    frozen answer -- the graded token ids and ``step_logprobs`` -- and a candidate
    that returned those from its probes would score against the goldens with no
    model at all. ``tests/Dockerfile`` ends ``WORKDIR /tests`` and bakes ``golden/``
    there, and this is a plain ``spawn`` child with the parent's cwd, the same uid
    and unrestricted file I/O. Starting in an empty scratch directory removes the
    zero-knowledge relative path.

    WHAT THIS DOES **NOT** DO, stated here so nobody reads it as isolation: the
    goldens are still on the filesystem and still readable by absolute path. This
    raises the cost of cheating; it does not make it impossible, and it cannot --
    executing untrusted code in-process is not isolable without a filesystem
    sandbox (a bind mount the worker cannot name, or a uid with no read on
    ``/tests``). That is a known limitation, not claimed here.

    THE EVIDENCE IS CARRIED BACK, not assumed, exactly as ``torch_import_error_type``
    is: ``sandbox_report`` records the cwd this child actually ran in and the result
    of the relative probe above, and ``test_spec.test_independence`` asserts the
    probe came back empty. Proven falsifiable: with the ``chdir`` removed the probe
    returns the golden and that check FAILS.
    """
    import importlib.machinery
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    scratch_cwd = None
    sitecustomize_dir = None
    try:
        os.environ[SANDBOX_ENV_FLAG] = "1"
        # BEFORE the candidate's sys.path entry and before anything is imported, for
        # the same reason the blocker goes in first: a relative path resolved during
        # the candidate's own import is resolved against whatever the cwd is then.
        scratch_cwd = tempfile.mkdtemp(prefix="t2j-candidate-cwd-")
        os.chdir(scratch_cwd)

        # THE SPAWN CHILD ALSO INHERITED THE GRADER'S sys.path, A SEPARATE HOLE
        # FROM THE CWD ABOVE. The grader's own tests directory --
        # this module's grandparent -- appears on this child's inherited path
        # (twice), so `import fixtures; fixtures.load_golden(...)` would return the
        # entire frozen Golden, no model involved at all. This closes THAT
        # route: the tests directory comes off sys.path before anything else
        # is imported, and any grader module that snuck into the cache before
        # this point (`fixtures`, if some earlier probe imported it) is
        # evicted so a later `import fixtures` cannot be served from it.
        #
        # WHAT THIS DOES **NOT** DO. `_t2j.candidate` -- this very module --
        # was already imported before this line could even run: multiprocessing
        # had to import it to find `_worker_entry` in the first place. It stays
        # in `sys.modules` no matter what happens to `sys.path`, and
        # `sys.modules['_t2j.candidate'].__file__` still names this file, from
        # which the golden directory is two `.parent`s away. That is the
        # ABSOLUTE-PATH route, not the module route, and no `sys.path` /
        # `sys.modules` edit can close it -- the goldens are still on disk and
        # still readable by this process's own uid (root, in the verifier
        # image; no Dockerfile here sets USER). Real isolation needs a mount
        # namespace or a uid drop on the golden directory, and neither is
        # implemented. This narrows the hole; it does not close it.
        _grader_dir = Path(__file__).resolve().parent.parent
        sys.path[:] = [p for p in sys.path
                       if not p or Path(p).resolve() != _grader_dir]
        for _name in [n for n in sys.modules
                      if n == "fixtures" or n.startswith("fixtures.")]:
            del sys.modules[_name]

        # THE EVIDENCE, computed by RE-SCANNING sys.path AFTER the mutation
        # above rather than by reporting a before-the-fact snapshot of what got
        # removed. A snapshot-of-what-was-removed is non-empty exactly when the
        # fix WORKED, which would make the "== []" assertion below fail on a
        # correct implementation. A fresh post-mutation scan is empty when the
        # fix worked and -- provably, see the falsifiability note below --
        # non-empty again the moment the `sys.path[:] = [...]` line above is
        # reverted or bypassed.
        tests_dir_on_sys_path = [p for p in sys.path
                                 if p and Path(p).resolve() == _grader_dir]

        # THE EVIDENCE IS CARRIED BACK, not assumed. `importlib.util.find_spec`
        # is the obvious tool here and it is the WRONG one: it checks
        # `sys.modules` first, and `_t2j.candidate`'s entry survives the prune
        # above for the reason just explained -- it would report "importable"
        # forever, fix or no fix. `PathFinder.find_spec` walks `sys.path`
        # directly and never consults that cache, so it answers the question
        # this probe actually asks: would a FRESH import find this name right
        # now. On the unpruned path this
        # returns True for both names below; after the prune, False for both,
        # `_t2j.candidate` included -- which only demonstrates that a SECOND,
        # independent worker could no longer import it either, not that the
        # first one (this process) forgot it.
        def _fresh_import_would_find(fullname: str) -> bool | str:
            """``True``/``False``, or an ``"error: ..."`` sentinel.

            WRAPPED, because this runs inside ``_worker_entry``'s own ``try``,
            whose ``except BaseException`` is what reports a genuine candidate
            failure to the parent (see the module docstring). Before this wrap,
            any exception from ``PathFinder.find_spec`` -- a corrupt ``sys.path``
            entry, a misbehaving import hook -- would be indistinguishable from
            the candidate itself failing, aborting the WHOLE worker over a
            diagnostic probe that has nothing to do with the candidate's own
            code. FAILS CLOSED: the sentinel is truthy, so the caller below
            treats "the probe could not answer" the same as "yes, importable"
            rather than silently reporting the module route as closed when it
            was never actually checked.
            """
            try:
                parts = fullname.split(".")
                search_path = None
                for i, part in enumerate(parts):
                    spec = importlib.machinery.PathFinder.find_spec(part, search_path)
                    if spec is None:
                        return False
                    if i < len(parts) - 1:
                        if not spec.submodule_search_locations:
                            return False
                        search_path = list(spec.submodule_search_locations)
                return True
            except BaseException as exc:  # noqa: BLE001 - reported, never raised
                return f"error: {type(exc).__name__}: {exc}"

        grader_modules_importable = []
        for _name in ("fixtures", "_t2j.candidate"):
            _found = _fresh_import_would_find(_name)
            if _found:
                grader_modules_importable.append(
                    _name if _found is True else f"{_name} ({_found})")

        # The probe that verifies the fix, run from where the candidate will run.
        # Capped: this is an empty directory, so the walk is O(1) here, and a cap
        # means a candidate that fills its own cwd cannot make the grade quadratic.
        relative_golden_hits = [str(p) for _n, p in zip(
            range(5), Path(".").rglob("golden/*.npz"))]
        # fp32 means fp32: without this XLA:GPU lowers an fp32 matmul to TF32
        # (10-bit mantissa) and the fp32 exactness tier measures TF32 noise
        # instead of a porting bug. Set before jax is imported anywhere, and
        # ASSIGNED rather than defaulted: it is a Global Constraint of the class,
        # so a value inherited from whoever launched the grade must not win. The
        # effective setting is reported back in `sandbox_report` so a test can
        # assert what the child actually ran under, not what the parent intended.
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        blocker = sandbox.install_torch_blocker(banned)

        # THE BLOCKER MUST SURVIVE A subprocess, AND THE ENV MARKER DOES NOT DO
        # IT. `SANDBOX_ENV_FLAG` is not read anywhere, so it does not propagate
        # to a nested spawn, and a candidate that
        # did its real work in `subprocess.run([sys.executable, ...])` would run
        # that child with NO meta-path blocker at all -- `install_torch_blocker`
        # above only ever touches THIS process's `sys.meta_path`, and
        # `banned_modules_in_sys` only ever inspects THIS process's
        # `sys.modules`.
        #
        # `sitecustomize` is the mechanism CPython already has for this: it is
        # imported by the `site` module before any user code, in every
        # interpreter that finds it on `sys.path` -- which a subprocess builds
        # from `PYTHONPATH` before `site` ever runs.
        #
        # IT DOES NOT LIVE IN THE SCRATCH CWD, AND THE SECOND TEMPDIR EXISTS
        # FOR EXACTLY THAT REASON. `scratch_cwd` is the directory this worker
        # just chdir'd into, so it is the CANDIDATE's own working directory:
        # same process, same uid, writable. A candidate does not have to defeat
        # the blocker there, only to empty the file that installs it --
        # with the file in the cwd and the cwd on PYTHONPATH,
        # truncating `sitecustomize.py` to zero bytes before spawning gets an
        # unblocked child (`IMPORTED`) instead of `TorchBlockedError`.
        #
        # WHAT THE SEPARATE DIRECTORY BUYS, EXACTLY -- IT IS A NARROWING AND NOT
        # A CLOSURE. It defeats SHADOWING: a candidate-planted `sitecustomize.py`
        # in the cwd does not shadow this one and the child still reports
        # `TorchBlockedError`. The mechanism is startup ORDER, not precedence
        # -- `site` imports `sitecustomize` during interpreter init, when
        # `sys.path` already carries the PYTHONPATH entries but NOT the `''`
        # that `-c` prepends afterwards, so a cwd copy is never even looked at
        # (a third check: a `sitecustomize.py` in the cwd with no
        # PYTHONPATH at all does not run).
        #
        # It does NOT defeat OVERWRITING. Shadowing and emptying are different
        # attacks and only the first is closed: the candidate runs in this
        # process at this uid, so it can read `PYTHONPATH`, find this file and
        # truncate it. Against the shim text below and the real
        # `sandbox.py`: baseline (shim intact, separate PYTHONPATH dir) ->
        # `TorchBlockedError`; after truncating `$PYTHONPATH[0]/sitecustomize.py`
        # -> `IMPORTED`. What the move buys is that the attack now has to READ
        # the environment to find the file, where before a blind write into the
        # cwd sufficed -- worth having, and not a defence.
        #
        # Both directories are `tempfile.mkdtemp` and both are removed in the
        # `finally` below, so neither outlives this one worker.
        #
        # PYTHONPATH gets exactly ONE new entry: `sitecustomize_dir`. Not
        # `scratch_cwd` (writable by the candidate -- see above), not
        # `_grader_dir` (this module's grandparent, pruned off `sys.path`
        # above) and not `sys.path` itself -- a subprocess starts a fresh
        # interpreter and never sees this process's `sys.path`, only its
        # environment. Appending whatever PYTHONPATH this worker itself
        # inherited (normally empty here; see `tests/Dockerfile`, which sets
        # none) rather than overwriting it so a legitimate caller's entries
        # are not silently dropped.
        sitecustomize_dir = tempfile.mkdtemp(prefix="t2j-candidate-sitecustomize-")
        (Path(sitecustomize_dir) / "sitecustomize.py").write_text(
            "import os, sys\n"
            "if os.environ.get('T2J_TORCH_SANDBOX') == '1':\n"
            "    sys.path.insert(0, os.environ['T2J_SANDBOX_HELPERS'])\n"
            "    import sandbox as _s\n"
            "    _s.install_torch_blocker(os.environ['T2J_BANNED'].split(','))\n")
        # `_t2j/`, the directory THIS file (`_t2j/candidate.py`) lives in --
        # NOT `tests/`, this module's grandparent, which was just pruned off
        # `sys.path` above precisely so a fresh import could not reach
        # `fixtures` or the goldens. `sandbox.py` is a sibling of this file,
        # one level inside `tests/`, so exposing only this directory hands a
        # subprocess `import sandbox` without reopening the route
        # `tests_dir_on_sys_path` / `grader_modules_importable` exist to prove
        # closed.
        os.environ["T2J_SANDBOX_HELPERS"] = str(Path(__file__).resolve().parent)
        os.environ["T2J_BANNED"] = ",".join(blocker.banned)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [sitecustomize_dir, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)

        # WHAT THIS DOES **NOT** DO, stated here so nobody reads it as closed.
        # THIS HOLE CANNOT BE CLOSED FROM INSIDE: the candidate's code runs in
        # this process, at this uid, after this block, so every mechanism here
        # is a file it can rewrite or an environment variable it can unset. The
        # honest deliverable is the list, not a defence. Five routes, all
        # checked against the shim text above and the real `sandbox.py`:
        #
        #   1. `python -S` skips `site` entirely, so `sitecustomize` is never
        #      imported.
        #   2. A stripped `env=` on the candidate's own spawn (one that drops
        #      `PYTHONPATH` / `T2J_TORCH_SANDBOX`).
        #   3. Truncating `$PYTHONPATH[0]/sitecustomize.py` -> `IMPORTED`. The
        #      second tempdir moved this from a blind cwd write to one that has
        #      to read `PYTHONPATH` first; it did not remove it.
        #   4. `os.environ['T2J_BANNED'] = ''` -- the shim installs a blocker
        #      that bans nothing.
        #   5. `del os.environ['T2J_TORCH_SANDBOX']`, or
        #      `os.environ['T2J_SANDBOX_HELPERS'] = '/nonexistent'` (`site`
        #      swallows the ImportError). Both -> `IMPORTED`.
        #
        # Routes 3-5 need no `env=` at all: the shim reads state the candidate
        # owns. So what this mechanism narrows the hole to is "a subprocess
        # spawned by a candidate that has not thought about any of the five",
        # which is a real improvement over "any subprocess at all" and is the
        # same kind of bound this file already states for the absolute-path
        # route around the scratch cwd.

        try:
            import torch  # noqa: F401
        except BaseException as exc:  # noqa: BLE001 - the TYPE is the evidence
            torch_error = type(exc).__name__
            torch_message = str(exc)
        else:
            torch_error = None
            torch_message = "torch imported successfully inside the sandbox"

        # THE TRANSITIVE PROBE, beside the in-process one above and answering a
        # different question: not "is the blocker installed HERE" but "does a
        # CHILD this worker spawns inherit it". A real `subprocess.run` rather
        # than a mock, because the only thing that proves `sitecustomize` is
        # genuinely importable by a fresh interpreter is a fresh interpreter
        # actually starting and importing it -- `site` only runs
        # `execsitecustomize` on process bootstrap, and nothing shy of a real
        # child exercises that path. The child prints the exception TYPE, for
        # the same reason the in-process probe reports a type rather than a
        # boolean: the verifier image has no torch installed at all, so a
        # missing blocker (`ModuleNotFoundError`) and a working one
        # (`TorchBlockedError`) both "fail to import torch", and only the type
        # tells them apart.
        #
        # WRAPPED, for exactly the reason `_fresh_import_would_find` above is
        # (see its docstring): this runs inside `_worker_entry`'s own `try`,
        # whose `except BaseException` is what reports a genuine CANDIDATE
        # failure to the parent. Unwrapped, a `subprocess.TimeoutExpired` (a
        # hung child), an `OSError` (fork or pipe exhaustion under memory
        # pressure) or a `PermissionError` on `sys.executable` would abort the
        # WHOLE worker over a diagnostic -- all 29 graded checks reporting one
        # `CandidateError`, reward 0.0645, the non-importing floor -- and this
        # probe runs BEFORE `sys.path.insert(repo_dir)` below, so nothing it
        # touches is the candidate's code at all. `TimeoutExpired` is covered:
        # it subclasses `SubprocessError`.
        #
        # FAILS CLOSED, in the sense that matters here: the sentinel is not
        # "TorchBlockedError", so `test_spec.test_independence` -- the one
        # check that reads this field -- still fails rather than passing on an
        # unanswered question. It fails naming `probe-failed:`, which that
        # check's failure dict spells out as a HARNESS fault. Failing one check
        # beats failing twenty-nine.
        try:
            probe = subprocess.run(
                [sys.executable, "-c",
                 "import sys\n"
                 "try:\n    import torch\n    print('IMPORTED')\n"
                 "except BaseException as e:\n    print(type(e).__name__)\n"],
                capture_output=True, text=True, timeout=120)
            subprocess_torch_error = probe.stdout.strip() or None
        except (OSError, subprocess.SubprocessError) as exc:  # noqa: BLE001
            subprocess_torch_error = f"probe-failed: {type(exc).__name__}"

        sys.path.insert(0, str(Path(payload["repo_dir"]).resolve()))

        # Deliberately NO jax.config.update("jax_enable_x64", True). See the
        # module docstring: x64 is off class-wide and nothing here needs float64.
        runtime = _CandidateRuntime(payload)
        results = runtime.run_all()
        import jax  # already resident: the runtime imported it

        results["sandbox_report"] = {
            "blocker_installed": any(
                isinstance(f, sandbox.BannedModuleFinder) for f in sys.meta_path),
            "banned_roots": list(blocker.banned),
            "torch_import_error_type": torch_error,
            "torch_import_error": torch_message,
            # THE TRANSITIVE EVIDENCE: what a FRESH interpreter, spawned by
            # this worker and inheriting only its environment (not its
            # `sys.meta_path`), reported when it tried `import torch`. See
            # `test_spec.test_independence` and
            # `test_the_torch_blocker_survives_into_a_subprocess`.
            "subprocess_torch_import_error_type": subprocess_torch_error,
            # The EFFECTIVE precision, read out of jax rather than out of the
            # environment: the env var is the lever, this is the thing it moved.
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "matmul_precision_env": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
            # THE NO-NETWORK PIN, READ IN THE PROCESS THAT WOULD DO THE
            # DOWNLOADING. The grader's conftest ASSIGNS both before anything
            # imports, and this child inherits `os.environ` across the `spawn`
            # -- so reading them back in the PARENT proves only that the parent
            # assigned them, which is not a claim about anything. Here it is a
            # claim: it covers the spawn actually carrying the environment
            # across, and it covers the candidate's own code, which runs (in
            # `run_all`, above) BEFORE this dict is assembled and can therefore
            # be seen to have cleared the variable in order to reach the hub.
            # `test_spec.test_loader_spec` grades this.
            "offline_env": {name: os.environ.get(name)
                            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
            # WHERE THIS CHILD RAN, and what a relative walk from there could see.
            # The second is the graded one -- see the chdir above and
            # `test_spec.test_independence`. An empty list is the whole claim: no
            # golden artifact is reachable from the worker's cwd by a relative path.
            "cwd": os.getcwd(),
            "relative_golden_hits": relative_golden_hits,
            # THE MODULE ROUTE, closed above; NOT the absolute-path route, which
            # is not and cannot be closed here. See the comment at the prune
            # site for what each of these does and does not prove.
            "tests_dir_on_sys_path": tests_dir_on_sys_path,
            "grader_modules_importable": grader_modules_importable,
        }
        leaked = sandbox.banned_modules_in_sys(banned)
        conn.send({"ok": True, "results": results, "banned_modules_in_sys": leaked})
    except BaseException as exc:  # report everything to the parent
        conn.send({
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        conn.close()
        # Leave the scratch cwd before removing it, or the next relative path in
        # this process resolves against a deleted directory. Best-effort: the child
        # is about to exit either way, and a failed cleanup must not turn into a
        # second exception on top of whatever brought us here.
        if scratch_cwd is not None:
            try:
                os.chdir(tempfile.gettempdir())
            except OSError:
                pass
            shutil.rmtree(scratch_cwd, ignore_errors=True)
        # The sitecustomize directory is NOT the cwd (that is the point of it
        # being a second tempdir), so it needs no chdir dance -- just the same
        # best-effort removal, so the blocker shim does not outlive the worker
        # that installed it.
        if sitecustomize_dir is not None:
            shutil.rmtree(sitecustomize_dir, ignore_errors=True)


#: Floor on the bounded receive, so a budget that happens to expire in the
#: microsecond between ``poll()`` and ``recv()`` reads as "the message is here"
#: rather than as a spurious timeout. Bounds the overshoot at one second.
_RECV_FLOOR_S = 1.0


def _recv_within(conn, timeout: float):
    """``conn.recv()`` with a deadline; ``None`` means the deadline passed.

    ``poll()`` only reports that a message STARTED arriving. A worker that writes
    part of a pickle and then wedges -- an XLA hang mid-serialisation, a child the
    OOM killer takes between chunks -- leaves a bare ``recv()`` blocked forever,
    and the grade then runs past the very ``generate_timeout_s`` that exists to
    stop it. The read happens on a daemon thread the parent can simply abandon;
    ``_hard_kill`` follows immediately, which closes the pipe and lets the thread
    die with it.
    """
    box: queue.Queue = queue.Queue(maxsize=1)

    def _pull() -> None:
        try:
            box.put(("ok", conn.recv()))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the parent
            box.put(("error", exc))

    threading.Thread(target=_pull, daemon=True, name="t2j-candidate-recv").start()
    try:
        kind, value = box.get(timeout=max(float(timeout), _RECV_FLOOR_S))
    except queue.Empty:
        return None
    if kind == "error":
        raise value
    return value


# =========================================================================== #
# Parent-side handle.
# =========================================================================== #
class Candidate:
    """Parent-side handle: spawns the torch-blocked worker once and caches results.

    ``__init__`` stores configuration and NOTHING runs. The worker spawns on the
    first probe access, so a missing or broken repo surfaces inside the test that
    needed it rather than at collection time.
    """

    def __init__(
        self,
        repo_dir: str,
        *,
        scenario: dict[str, Any],
        repo_config: dict[str, Any],
        mode: str = "core",
        seed: int = 0,
        sampler_grid: list[dict[str, Any]] | None = None,
        logprobs_top_n: int = 5,
        num_decode_steps: int | None = None,
        compare_hidden_states: bool = False,
        fp32_kv_cache_dtype: str = "float32",
        held_out: dict[str, Any] | None = None,
        generate_timeout_s: float = 900.0,
        perf_k_values: list[int] | None = None,
        perf_rounds: int = DEFAULT_PERF_ROUNDS,
        device: str | None = None,
        data_root: str | None = None,
        banned: list[str] | None = None,
    ) -> None:
        self.repo_dir = str(Path(repo_dir).resolve())
        self.mode = str(mode)
        self._banned = list(banned) if banned is not None else list(
            sandbox.DEFAULT_BANNED_MODULES)
        # No weights are handed over. `repo_config["model_name_or_path"]` is the
        # HF repo id; the candidate's own loader resolves it to /data/{repo_id}
        # and converts the checkpoint itself.
        self._payload: dict[str, Any] = {
            "repo_dir": self.repo_dir,
            "repo_config": dict(repo_config),
            "mode": self.mode,
            "seed": int(seed),
            "scenario": dict(scenario),
            "sampler_grid": [dict(e) for e in (sampler_grid or [])],
            "logprobs_top_n": int(logprobs_top_n),
            "num_decode_steps": num_decode_steps,
            "compare_hidden_states": bool(compare_hidden_states),
            # A CONTRACT OF THE fp32 GOLDEN, not a knob: the freezer stamps it off
            # the config its own fp32 window built, and the fp32 generation leg
            # must allocate its KV cache at the same dtype or its logprobs sit 180x
            # further from the artifact it is graded against -- with every token
            # still agreeing. Handed in rather than defaulted in the worker so a
            # golden frozen under a different choice moves the probe with it.
            "fp32_kv_cache_dtype": str(fp32_kv_cache_dtype),
            # THE L6 WORKLOAD, FILTERED AT THIS BOUNDARY. `held_out.npz` carries
            # five arrays and only four of them are inputs; the fifth,
            # `ref_logprobs`, is the answer `test_held_out_logprobs` compares
            # against. Selecting by :data:`HELD_OUT_INPUT_KEYS` HERE rather than in
            # the caller is what makes "the candidate never receives the answer" a
            # property of the class instead of a convention every caller has to
            # remember -- the same argument `fixtures._SCENARIO_INPUT_KEYS` makes
            # for the scenario, at the only other boundary a frozen output could
            # cross. An absent key is left absent rather than raised on: the probe
            # reports it as unsupported, which fails L6 with a reason instead of
            # failing every check with a fixture error.
            "held_out": ({k: held_out[k] for k in HELD_OUT_INPUT_KEYS
                          if k in held_out} if held_out else None),
            # The perf stage's own geometry, read by
            # `_CandidateRuntime.perf_probe`. `perf_k_values` is
            # `instance.yaml::perf_k_values` mirrored into `meta.json`; an
            # instance that declares none leaves this `[]` and `perf_probe`
            # reports `supported: False` rather than sweeping nothing.
            # `perf_rounds` is R, a global instrument parameter --
            # see :data:`DEFAULT_PERF_ROUNDS` for why it defaults here rather
            # than in `instance.yaml`.
            "perf_k_values": [int(k) for k in (perf_k_values or [])],
            "perf_rounds": int(perf_rounds),
            "device": device,
            "data_root": data_root,
        }
        self._timeout = float(generate_timeout_s)
        self._results: dict[str, Any] | None = None
        self._error: dict[str, Any] | None = None

    # --- execution ---------------------------------------------------------- #
    def _ensure_run(self) -> dict[str, Any]:
        if self._results is not None:
            return self._results
        if self._error is not None:
            self._raise_error()
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_worker_entry,
                              args=(child_conn, self._payload, self._banned))
        process.start()
        child_conn.close()
        payload: dict[str, Any] | None = None
        # ONE budget covers both halves. `poll` bounds the wait for the message
        # to START; `_recv_within` bounds the wait for it to FINISH, out of what
        # is left. A worker that begins sending and then stalls must not be able
        # to hold the grade open past its own timeout.
        deadline = time.monotonic() + self._timeout
        try:
            if parent_conn.poll(self._timeout):
                try:
                    payload = _recv_within(parent_conn, deadline - time.monotonic())
                except EOFError:
                    # NAME THE SIGNAL, because "closed without sending a result"
                    # is the same sentence for two very different events and the
                    # reader cannot tell them apart. A worker that RETURNED
                    # nothing is a candidate defect; a worker the kernel KILLED
                    # is usually the machine -- an OOM under `task.toml`'s
                    # `memory_mb`, or a CUDA/XLA fault delivering SIGSEGV.
                    #
                    # DELIBERATELY NOT AN ABORT PATH, unlike the timeout beside
                    # it, and the asymmetry is argued rather than accidental.
                    # Wall clock is CONTENDED and container memory is NOT: the
                    # timeout fires on a correct port when the box is loaded
                    # (it can exceed the 900 s budget), whereas
                    # `--memory` is a per-container hard limit a noisy neighbour
                    # cannot move. A genuinely bloated port does not even land
                    # here -- a JAX device OOM raises `XlaRuntimeError` inside
                    # the worker and is scored cleanly with a real message. What
                    # is left is mostly a candidate that kills itself, where
                    # reward 0.0 is the CORRECT answer; giving it an abort would
                    # hand every failing port a free, instant, undetectable
                    # self-destruct. So this stays scored, and only becomes
                    # legible.
                    # REAP FIRST. `_hard_kill` runs in the `finally` BELOW, i.e.
                    # after this handler, so `process.exitcode` is still None
                    # here unless we join. The pipe is already closed, so the
                    # child is on its way out and this returns at once.
                    process.join(timeout=5)
                    code = process.exitcode
                    if code is not None and code < 0:
                        try:
                            name = signal.Signals(-code).name
                        except ValueError:                # pragma: no cover
                            name = f"signal {-code}"
                        detail = (
                            f"the candidate sandbox was KILLED by {name} before "
                            f"it sent a result. This is usually the machine "
                            f"rather than the port: the container's `memory_mb` "
                            f"limit, or a CUDA/XLA fault. A JAX device OOM does "
                            f"NOT arrive here -- it is raised inside the worker "
                            f"and reported with its own message -- so if you see "
                            f"this, check the host's kernel log before reading it "
                            f"as a porting defect.")
                    else:
                        detail = (
                            f"the candidate sandbox closed without sending a "
                            f"result (worker exit code {code!r}, i.e. it returned "
                            f"rather than being killed)")
                    payload = {"ok": False, "error": detail,
                               "worker_exitcode": code}
                except Exception as exc:      # noqa: BLE001
                    # ANY receive-side failure, memoised below before it can
                    # escape. CPython raises EOFError for a zero-byte read but
                    # OSError("got end of file during message") for a child
                    # killed part-way through the pickle -- precisely the case
                    # `_recv_within`'s own docstring exists for ("an XLA hang
                    # mid-serialisation, a child the OOM killer takes between
                    # chunks"). `pickle.UnpicklingError`, `ConnectionResetError`
                    # and `MemoryError` land here too. Letting any of these
                    # escape left `self._error` unset and `_hard_kill`
                    # unreached: every one of the 31 session-scoped checks that
                    # share this `Candidate` respawned an unreaped worker still
                    # holding the converted weights (31 x ~68s > the 1800s
                    # verifier budget), and the container died before
                    # `summary_to_reward` ever ran -- a void grade, not a low
                    # one.
                    #
                    # `Exception`, not `BaseException`: every failure mode
                    # named above is an `Exception` subclass, `_hard_kill`
                    # below is already unconditional via `finally` (so
                    # cleanup does not depend on this clause at all), and a
                    # `KeyboardInterrupt`/`SystemExit` delivered while blocked
                    # in `_recv_within`'s `box.get()` must stay uncaught --
                    # catching it here would memoise it as an ordinary
                    # candidate failure and every one of the 31 checks would
                    # then raise the same swallowed interrupt instead of the
                    # run actually stopping.
                    payload = {"ok": False,
                               "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()}
        finally:
            # ALWAYS reap, unconditionally: on timeout to stop a runaway decode
            # loop, on a receive-side failure so the doomed worker does not
            # outlive the exception that reports it, on success to give the GPU
            # memory back before the next test needs it -- not only on the path
            # that falls through the `try` above.
            self._hard_kill(process)
        if payload is None:
            waited = time.monotonic() - (deadline - self._timeout)
            payload = {"ok": False,
                       "timeout": True,
                       "error": f"candidate sandbox exceeded its WHOLE-SANDBOX "
                                f"budget generate_timeout_s={self._timeout:.0f}s "
                                f"after {waited:.0f}s (it either never answered "
                                f"or stalled part-way through sending its "
                                f"result). NOTE: this budget bounds the ONE "
                                f"worker spawn that runs every probe in "
                                f"REQUIRED_PROBES -- both prefills, decode, both "
                                f"generations, serve.py::_build, held-out "
                                f"scoring, the batteries and the invariances -- "
                                f"NOT the candidate's generate() loop alone. The "
                                f"honest reference needs a large fraction of it, "
                                f"so on a loaded host this fires on a CORRECT "
                                f"port. Re-run on a quiet box before reading it "
                                f"as a defect."}
        if payload.get("ok") and payload.get("banned_modules_in_sys"):
            payload = {
                "ok": False,
                "error": "candidate imported banned modules inside the sandbox: "
                         f"{payload['banned_modules_in_sys']}",
            }
        if not payload.get("ok"):
            self._error = payload
            self._raise_error()
        self._results = _validate_results(payload["results"])
        self._leaks = list(payload.get("banned_modules_in_sys") or [])
        return self._results

    def _raise_error(self) -> None:
        assert self._error is not None
        detail = (self._error.get("traceback") or self._error.get("error")
                  or "unknown candidate failure")
        # MEMOISED, so the distinction survives. `_error` is replayed to every
        # one of the session-scoped checks that share this Candidate; if the
        # subclass were chosen only at the raise site of the FIRST one, the other
        # thirty would report a plain CandidateError and conftest would score the
        # run it is supposed to refuse.
        cls = CandidateTimeout if self._error.get("timeout") else CandidateError
        raise cls(f"candidate sandbox failed:\n{detail}")

    def _probe(self, name: str) -> Any:
        """One probe, or ``CandidateError`` naming it.

        The counterpart of ``run_all``'s per-probe capture: the failure is
        carried as a value across the sandbox boundary and converted back into
        an exception HERE, at the point a check reads it, so it fails that check
        and no other. Every accessor below that names exactly one probe key goes
        through this rather than indexing ``self._ensure_run()`` directly.
        """
        value = self._ensure_run()[name]
        if _is_probe_error(value):
            raise CandidateError(
                f"the candidate's {name!r} probe raised. Only the checks that "
                f"read it are affected.\n{value['__probe_error__']}")
        return value

    @staticmethod
    def _hard_kill(process) -> None:
        """SIGTERM, then SIGKILL if it will not die within the grace period.

        A runaway XLA compile sits in native code that ignores SIGTERM;
        escalating keeps a timed-out worker -- and the several GB of device
        memory it holds -- from being orphaned into the next test.
        """
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)

    # --- parent-only static scan --------------------------------------------- #
    def purity_scan(self) -> purity.PurityReport:
        """Static AST scan of the whole candidate repo (stdlib only, no import)."""
        return purity.scan_repo_for_torch(self.repo_dir, allow_dirs=())

    # --- cached probe accessors ----------------------------------------------- #
    @property
    def banned_leaks(self) -> list[str]:
        """Banned modules that were loaded in the sandbox when it finished."""
        self._ensure_run()
        return list(self._leaks)

    @property
    def sandbox_report(self) -> dict[str, Any]:
        return self._probe("sandbox_report")

    @property
    def allocation_source(self) -> dict[str, str]:
        """``{probe: "frozen" | "candidate" | "mixed"}`` -- the frozen-allocation policy AS EXECUTED.

        Every entry was written by the probe that ran, at the line where it took
        the allocation, and probes that did not run are simply absent. This is
        deliberately NOT a rendering of :data:`USE_FROZEN_ALLOCATION`: a map built
        by iterating that table would make every assertion about it a tautology,
        and the table drives nothing.
        """
        return self._probe("allocation_source")

    @property
    def params_spec(self) -> dict[str, Any]:
        return self._probe("params_spec")

    @property
    def param_norms(self) -> dict[str, Any]:
        return self._probe("param_norms")

    @property
    def cache_spec(self) -> dict[str, Any]:
        return self._probe("cache_spec")

    @property
    def loader_facts(self) -> dict[str, Any]:
        return self._probe("loader_facts")

    @property
    def resolved_config(self) -> dict[str, Any]:
        """The candidate's own ``RunConfig`` field list + values, post-``__post_init__``."""
        return self._probe("resolved_config")

    @property
    def compilation(self) -> dict[str, Any]:
        """XLA compilation counts on the control / cold / same-shape / new-shape legs."""
        return self._probe("compilation")

    @property
    def metadata_schema(self) -> dict[str, Any]:
        return self._probe("metadata_schema")

    @property
    def block_table(self) -> np.ndarray:
        return self._probe("block_table")

    @property
    def slot_mapping(self) -> np.ndarray:
        return self._probe("slot_mapping")

    @property
    def static_shapes(self) -> dict[str, Any]:
        return self._probe("static_shapes")

    @property
    def traceability(self) -> dict[str, Any]:
        return self._probe("traceability")

    def prefill(self, fp32: bool = False) -> dict[str, Any]:
        """``{logits, hidden_0..N}`` on the frozen scenario."""
        return self._probe("fp32_prefill" if fp32 else "prefill")

    @property
    def chunked_prefill(self) -> dict[str, Any]:
        """``{supported, logits, chunk_split_points, n_chunks}``.

        A PROPERTY, not a method like :meth:`prefill`: there is one leg, not
        two -- no fp32 counterpart exists for this probe. See
        :meth:`_CandidateRuntime.chunked_prefill_probe` for what it runs and why
        ``n_chunks`` is the field the harness's check reads to refuse a probe that
        silently ran one chunk.
        """
        return self._probe("chunked_prefill")

    @property
    def kv_cache_state(self) -> tuple:
        """The post-prefill cache, one numpy array PER GROUP in container order.

        A tuple for every family, including the one-group case: the frozen
        artifact spells the two cases differently (``cache`` XOR
        ``cache_group{i}``) and ``test_kv_cache_write`` compares the counts, so a
        probe value that collapsed one group to a bare array would make the count
        clause need a special case to say the one thing it exists to say.
        """
        return self._probe("kv_cache_state")

    @property
    def decode_steps(self) -> dict[str, Any]:
        return self._probe("decode_steps")

    def paged_attention_unit(self, fp32: bool = False) -> dict[str, Any]:
        return self._probe(
            "fp32_paged_attention_unit" if fp32 else "paged_attention_unit")

    @property
    def sampler_outputs(self) -> dict[str, Any]:
        return self._probe("sampler_outputs")

    def generation(self, fp32: bool = False) -> dict[str, Any]:
        """The free-running ``engine.generate`` result, at one of two precisions.

        A METHOD, mirroring :meth:`prefill` and :meth:`paged_attention_unit`,
        because there are now two legs and the fp32 one is the benchmark's headline
        equivalence check. ``fp32=True`` is the leg built on an **fp32 KV cache**
        (``meta['fp32_kv_cache_dtype']``); see
        :meth:`_CandidateRuntime.generation_probe`.
        """
        return self._probe("fp32_generation" if fp32 else "generation")

    def generation_step_logits(self, fp32: bool = False) -> list[np.ndarray]:
        """Per-step PRE-sampling logits, parallel to ``.generation()["tokens"]``.

        A DIAGNOSTIC AND A STRUCTURAL INPUT -- **not** the source of any tolerance.
        ``test_greedy_sequence``'s tie-tolerant branch does NOT read the top-2
        margin here: a margin read
        off the CANDIDATE lets a port with flat logits manufacture its own
        exemption, so the margin comes from the frozen golden
        (``fixtures.Golden.decode_margins``) and nothing in this file feeds it.

        What it serves instead is ``test_generate_runs``'s structural clause: the
        engine must have emitted one pre-sampling row per generated token, and
        row ``i``'s argmax must be token ``i``. Under the default
        ``SamplingParams`` (temperature 1.0, top_k 0, top_p 1.0) the sampler's
        transform is the identity and greedy is exactly that argmax, so the
        relation is integer equality with no tolerance. This is falsifiable:
        an engine that appends the
        row AFTER its decode step instead of before -- the natural off-by-one, and
        one that leaves the tokens, the logprobs and every golden comparison
        untouched -- fails ``test_generate_runs`` at generated step 0 while the
        Spec, Numeric and the other behavioral checks all pass.

        PRESENT OR FAILS, like every other accessor here. Degrading to ``[]``
        would hand that clause an empty sequence, and "no row disagreed" is
        indistinguishable from "no row was examined" -- the exact vacuous pass the
        ``REQUIRED_PROBES`` guard exists to rule out.
        """
        key = "fp32_generation" if fp32 else "generation"
        rows = self._probe(key).get("logits_per_step") or []
        if not rows:
            raise CandidateError(
                f"the candidate's engine.generate() returned no per-step logits on "
                f"the {key!r} leg, so GenerateResult.logits_per_step is empty. "
                "test_generate_runs needs them to check that each emitted token is "
                "the argmax of the row the engine says it sampled from; an empty "
                "list would make that clause pass without examining anything."
            )
        return list(rows)

    def serve(self) -> dict[str, Any]:
        """``serve.py`` driven from a config, plus the same engine wired by hand.

        A METHOD, not a property, so it reads as the call it is at both ends
        (``cand.serve()``) and so a later leg switch -- an fp32 one, say -- is an
        argument rather than a second accessor.

        ``{supported, results, wired_tokens, expected_text, has_tokenizer,
        resolved_dtype, resolved_kv_cache_dtype}``; see
        :meth:`_CandidateRuntime.serve_probe` for why the two legs are built the
        way they are, and in particular why the by-hand one must NOT go through
        serve.py's own ``_build``.
        """
        return self._probe("serve")

    def held_out(self) -> dict[str, Any]:
        """``{supported, logprobs}`` -- teacher-forced fp32 logprobs, L6.

        ``logprobs`` is ``(P, C_max)`` float32 laid out exactly like
        ``golden/held_out.npz``'s ``ref_logprobs``: right-padded with zeros, row
        ``i`` valid on ``[:cont_lens[i]]``. The frozen counterpart never crossed
        into the sandbox (:data:`HELD_OUT_INPUT_KEYS`), so this grid was computed,
        not echoed.
        """
        return self._probe("held_out")

    @property
    def battery(self) -> dict[str, Any]:
        return self._probe("battery")

    @property
    def perf(self) -> dict[str, Any]:
        """``{supported, rounds, tokens}`` -- raw decode-timing pairs.

        ``rounds`` is a list of ``{round, discarded, throttle, calls}``, one per
        attempt :meth:`_CandidateRuntime.perf_probe` made (the target
        round count, plus any throttled retries). ``calls`` is
        ``[(K, seconds), ...]`` for that attempt's K sweep, raw and unreduced --
        the fit, the median and the quorum are `_t2j/perf/runner.py`'s job, not
        this class's. ``tokens`` is the K == ``num_decode_steps`` leg's sampled
        ids, the work-gate subject.
        """
        return self._probe("perf")

    @property
    def reference_units(self) -> dict[str, Any]:
        return self._probe("reference_units")

    @property
    def image_preprocess(self) -> dict[str, Any]:
        """``pixel_values`` + ``image_grid_thw`` from the port's own processor.

        ``supported: False`` on a text-only family; the check that reads it is
        deselected there on the ``multimodal`` gate, never on this marker.
        """
        return self._probe("image_preprocess")

    @property
    def smart_resize(self) -> dict[str, Any]:
        """The port's ``smart_resize`` over the frozen non-conforming table."""
        return self._probe("smart_resize")

    @property
    def multimodal_prefill(self) -> dict[str, Any]:
        """``{positions, logits, ...}`` from ONE image-conditioned
        ``Engine.prefill``.

        Read by ``test_mrope_positions`` and ``test_multimodal_prefill_logits``,
        which is a coupling the probe's own docstring argues for rather than
        hides: both are properties of the same forward, so a tower that raises
        fails both -- exactly as a broken forward fails ``test_prefill_logits``
        and ``test_hidden_states`` together.

        ``supported: False`` on a text-only family; both checks are deselected
        there on the ``multimodal`` gate, never on this marker.
        """
        return self._probe("multimodal_prefill")

    @property
    def image_embedding_merge(self) -> dict[str, Any]:
        """The tower rows, the text embeddings and the merged table, all three
        from the candidate -- see the probe for why the comparison is between
        them rather than against a frozen array."""
        return self._probe("image_embedding_merge")

    @property
    def vision_tower_units(self) -> dict[str, Any]:
        """Per-tower-submodule replay plus ``encoder_metadata``'s own return."""
        return self._probe("vision_tower_units")

    def invariance(self, kind: str) -> dict[str, Any]:
        """One golden-free self-consistency probe, by name.

        An unrecognised name RAISES. A typo in a behavioral test module must not
        read as "the candidate does not support that property" -- that is how a
        check silently stops grading anything at all -- so the name is validated
        against the static :data:`INVARIANCE_KINDS` before the results are even
        consulted.

        TWO LEVELS OF CAPTURE ARE UNWRAPPED HERE, in the order they can occur.

        ``self._probe("invariance")`` raises FIRST, and names the whole probe, if
        the loop's own scaffolding raised -- ``run_all`` runs
        ``invariance_probes()`` as one ``produce`` call, so anything outside the
        per-kind ``try`` (the ``INVARIANCE_KINDS`` walk itself) is still
        common-mode. That is now a narrow surface rather than the default: it no
        longer covers a KIND raising.

        Then the KIND's own value, which ``invariance_probes`` captures
        individually. A kind that raised is a dict carrying ``__probe_error__``,
        and it is converted back into a ``CandidateError`` NAMING THAT KIND right
        here, at the point a check reads it -- the same parent-side conversion
        ``Candidate._probe`` performs for a whole probe, and the reason the
        per-kind capture does not turn a failure into a vacuous pass. Returning
        the raw dict instead would hand ``test_determinism`` a payload whose
        ``.get("supported")`` is falsy, i.e. "the candidate does not support
        determinism", which is a different (and false) statement about the port.
        """
        if kind not in INVARIANCE_KINDS:
            raise RuntimeError(
                f"unknown probe: {kind!r}; the self-consistency probes are "
                f"{sorted(INVARIANCE_KINDS)}"
            )
        probes = self._probe("invariance")
        if kind not in probes:
            raise CandidateError(
                f"the sandbox returned no result for the {kind!r} invariance "
                "probe; every registered probe must be present"
            )
        value = probes[kind]
        if _is_probe_error(value):
            raise CandidateError(
                f"the candidate's {kind!r} invariance probe raised. Only the "
                f"checks that read {kind!r} are affected; the other "
                f"{len(INVARIANCE_KINDS) - 1} kinds were produced normally.\n"
                f"{value['__probe_error__']}")
        return value


def _validate_results(results: dict[str, Any]) -> dict[str, Any]:
    """Every required probe present and non-``None``, or raise.

    This is the guard against the one failure mode that costs the most: a probe
    that swallows its own exception and hands back ``None``, which a downstream
    numeric check then compares against a golden and -- depending on the
    comparator -- passes. Absence and ``None`` are reported separately because
    they have different causes (a dropped ``produce`` call versus a swallowed
    exception).
    """
    missing = [name for name in REQUIRED_PROBES if name not in results]
    empty = [name for name in REQUIRED_PROBES
             if name in results and results[name] is None]
    if missing or empty:
        raise CandidateError(
            "the candidate sandbox returned an incomplete probe set -- "
            f"absent: {missing}; None: {empty}. Every probe must produce a value "
            "or raise; a silently absent artifact is a check that passes vacuously."
        )

    # INFORMATIONAL, not a gate: each name here still fails only the checks that
    # read it (Candidate._probe raises on that read). Printed so a graded run's
    # log says which probes were isolated, rather than leaving that only
    # inferable from which checks happened to fail.
    failed = sorted(name for name in REQUIRED_PROBES if _is_probe_error(results.get(name)))
    if failed:
        print(f"[t2j] probes that raised (each fails only its own checks): {failed}")

    # `invariance` itself may BE a captured probe error: ``run_all`` runs
    # ``invariance_probes()`` as one ``produce`` call, so the loop's own
    # scaffolding is still common-mode even though each KIND is now captured
    # individually inside it. In that case `results["invariance"]` is
    # ``{"__probe_error__": ...}``, and ITS keys (just the one) are not the
    # invariance kind registry at all. Reading through that shape here would
    # report every registered kind as "missing" and raise a global error for what
    # is actually a probe-level fault that `Candidate.invariance` already isolates
    # on read (see its docstring). A GENUINELY incomplete (non-error) invariance
    # dict -- a real kind dropped -- still raises exactly as before, and a kind
    # that RAISED is not incomplete: it is present, carrying its own
    # ``__probe_error__``, and fails only the checks that read it.
    invariance = results["invariance"]
    if not _is_probe_error(invariance):
        unknown = sorted(set(INVARIANCE_KINDS) - set(invariance))
        if unknown:
            raise CandidateError(
                f"the sandbox returned no result for invariance probe(s) {unknown}"
            )
    if not results["allocation_source"]:
        raise CandidateError(
            "no probe recorded which allocation it honoured, so "
            "`allocation_source` is empty. It is written at each point of use; "
            "an empty map means the frozen-allocation policy is once again a declaration "
            "nothing enforces."
        )
    return results
