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

"""Fixtures for the DRIVER's own unit tests (not for the graded tiers).

The graded tiers grade a candidate against frozen reference numbers. These tests
grade the DRIVER: that every probe produces a value, that a failing probe raises
instead of returning ``None``, that the torch blocker is really installed, and
that the frozen-allocation policy is really per-probe. None of that needs a
frozen reference number -- it needs a runnable CONFIGURATION.

So: use the real ``tests/golden``, and SYNTHESIZE from ``tests/instance.yaml``
(the same per-instance source of truth the freezer mirrors into ``meta.json``)
whatever the real one does not have yet. The synthetic parts carry real inputs and
zeroed outputs, and every test here is written so that a zeroed output cannot make
it pass. The synthesis is announced loudly on stderr so it can never be mistaken
for a real golden -- and the GRADED suite does the opposite thing on purpose: its
``load_golden`` raises ``FileNotFoundError`` naming the freezer (its hint is
``fixtures._MISSING_GOLDEN_HINT``; the goldens ship via ``git lfs pull``) rather
than inventing anything.

WHY IT IS AN OVERLAY AND NOT AN ALL-OR-NOTHING CHOICE. The golden catalog has SIX
producers and they land across several tasks: the canonical freezer
writes eleven artifacts, while
``interface.json``, ``param_norms.json``, ``paged_attention_units.npz``,
``reference_units.*`` and the batteries each have their own. So there is a long
window in which ``tests/golden`` genuinely exists and ``load_golden`` still cannot
open it, and an all-or-nothing predicate turns that window into 26 setup errors
whose cause is "a producer that has not been written yet". Overlaying uses every
real artifact the moment it appears -- which is the point: after the canonical
freeze these tests run against the REAL non-identity page assignment, which is
strictly better coverage than the synthetic ascending one they had before -- and
falls back only where it must.

THE SYNTHETIC HALF RAN IN NO DEVELOPER PATH AND ROTTED, which is the reason the
:func:`_synthesize` below now mirrors ``instance.yaml`` instead of hand-listing a
dozen meta keys, and the reason
``test_candidate_contracts.py::test_the_synthetic_golden_overlay_is_a_contract_
complete_golden`` exists. Once ``tests/golden`` is fully frozen -- the state every
authoring host and every container has been in since the canonical freeze landed
-- ``golden_dir`` returns the real directory and NOT ONE LINE of the synthesis
executes. It drifted three keys behind the meta contract without a single red
test.

In BOTH documented states (an authoring host with the checkpoint and a
git-ignored ``tests/golden/meta.json`` not yet fetched, and a bare tree with no
golden at all), a hand-listed synthetic meta breaks the overlay:
``fixtures.build_candidate`` raises ``KeyError:
'fp32_kv_cache_dtype'`` -- the meta key ``fixtures.py`` hard-indexes because the
fp32 KV dtype moves that artifact by 180x -- out of the session-scoped ``cand`` /
``misallocated_cand`` / ``desynced_cand`` fixtures, erroring the driver tests
at setup, including every per-probe-allocation and freezer-desync test.
The fully-synthetic tree additionally fails ``_harness_checks``:
``golden_manifest`` (the placeholder npz files carried an array called
``placeholder`` and none of the ones the graded modules read by name),
``threshold_extras_name_something_a_check_reads`` (no ``fp32_thresholds`` /
``native_thresholds`` band at all), ``meta_mirrors_instance_yaml`` (fifteen yaml
keys absent from the synthetic meta), and ``scenario_keys_are_classified`` (a
``synthetic: true`` marker in ``scenario.json`` that is in neither
``fixtures._SCENARIO_INPUT_KEYS`` nor ``_SCENARIO_OUTPUT_KEYS``).
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[3]
TESTS = TASK / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

# The frozen probe prompt: "The capital of France is Paris, and the capital of",
# 12 tokens including BOS -- instance.yaml's ``prompt_len``. Restated from
# ``tests_parity/_spec.py`` rather than imported, because that module belongs to
# the cross-framework parity gate and the grader must not depend on it.
PROMPT_IDS = [128000, 791, 6864, 315, 9822, 374, 12366, 11, 323, 279, 6864, 315]

# A small stand-in vocabulary for the sampler probe's frozen logits. The sampler
# transforms are shape-agnostic, so grading them on 512 columns exercises exactly
# the same code as 128256 would while keeping the fixture a few KB.
_SYNTHETIC_VOCAB = 512


def _instance() -> dict:
    import yaml

    return yaml.safe_load((TESTS / "instance.yaml").read_text())


#: What ``fixtures.load_golden`` opens. A name absent from ``tests/golden`` is
#: synthesized; a name present is used as frozen. Ordered as the catalog is.
_LOADED_BY_FIXTURES: tuple[str, ...] = (
    "meta.json", "scenario.json", "inputs.npz", "cache_spec.json",
    "prefill_forward.npz", "fp32_prefill_forward.npz", "kv_cache_state.npz",
    "decode_step.npz", "paged_attention_units.npz", "sampler.json", "sampler.npz",
    "generation.npz", "fp32_generation.npz", "params_spec.json", "param_norms.json",
    "interface.json", "held_out.npz",
)


def _synthesize(out: Path, have: frozenset[str] = frozenset()) -> Path:
    """Write a minimal, self-consistent golden tree. Real inputs, zeroed outputs.

    ``have`` names artifacts already present in ``out`` (copied from a real, partly
    frozen ``tests/golden``); those are left exactly as frozen.
    """
    inst = _instance()
    page_size = int(inst["page_size"])
    num_pages = int(inst["num_pages"])
    pages_per_seq = int(inst["pages_per_seq"])
    prompt_ids = [int(t) for t in (inst.get("prompt_token_ids") or PROMPT_IDS)]
    prompt = np.array(prompt_ids[: int(inst["prompt_len"])], dtype=np.int32)
    n = int(prompt.shape[0])
    steps = int(inst["num_decode_steps"])

    # The frozen allocation. instance.yaml pins a NON-IDENTITY page
    # assignment -- see its comment -- and this mirrors it rather than inventing
    # the ascending one the free-list allocator would hand out, because an
    # ascending assignment makes ``slot == pos`` and a probe that ignored the
    # block table would look correct here.
    page_ids = [int(p) for p in (inst.get("page_ids")
                                 or range(pages_per_seq))]
    positions = np.arange(n, dtype=np.int32)
    slot = np.array(
        [page_ids[p // page_size] * page_size + (p % page_size) for p in range(n)],
        dtype=np.int32)
    block_tables = np.zeros(pages_per_seq, dtype=np.int32)
    block_tables[: len(page_ids)] = page_ids

    out.mkdir(parents=True, exist_ok=True)

    def npz(name: str, **arrays) -> None:
        """Write ``name`` unless the real golden already froze it."""
        if name not in have:
            np.savez(out / name, **arrays)

    def js(name: str, payload) -> None:
        if name not in have:
            (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True))

    npz(
        "inputs.npz",
        input_ids=prompt,
        positions=positions,
        seq_lens=np.array([n], dtype=np.int32),
        query_start_loc=np.array([0, n], dtype=np.int32),
        request_distribution=np.array([0, 1, 0], dtype=np.int32),
        block_tables=block_tables,
        slot_mapping=slot,
    )

    repo_config = {
        "model_name_or_path": inst["model_name_or_path"],
        "page_size": page_size,
        "num_pages": num_pages,
        "max_num_seqs": int(inst["max_num_seqs"]),
        "pages_per_seq": pages_per_seq,
        "max_model_len": int(inst["max_model_len"]),
        "dtype": inst["dtype"],
        "seed": int(inst["seed"]),
        "num_decode_steps": steps,
    }
    # THE META IS MIRRORED FROM instance.yaml, NOT TRANSCRIBED FROM IT.
    #
    # A hand-written dict of keys falls behind the contract with nothing red:
    # `fp32_kv_cache_dtype`, which
    # `fixtures.build_candidate` hard-indexes (the fp32 KV dtype moves
    # `fp32_generation.npz` by 180x, so there is no honest default), and the
    # `fp32_thresholds` / `native_thresholds` bands, which
    # `_harness_checks.check_threshold_extras_name_something_a_check_reads` demands
    # by NAME because an absent band and a misspelled one must not look the same.
    # More yaml keys are simply not there, which
    # `check_meta_mirrors_instance_yaml` reads as a golden frozen against a
    # different contract -- correctly.
    #
    # A transcription is a second copy of instance.yaml's key set, and this file
    # is the one place in the tree where nothing notices when a second copy stops
    # agreeing. So it applies the freezer's OWN rule instead:
    # the freezer's instance-mirroring step is "every yaml key except the
    # authoring-only ones", and `_harness_checks._INSTANCE_KEYS_NOT_MIRRORED` is
    # that exemption list already restated for the check that enforces it. Reused
    # rather than restated a third time -- a new yaml key now lands here on the day
    # it is added, which is exactly what the check compares against.
    #
    # The four computed keys below are the freezer's `step_meta` minus what only a
    # real freeze can know (`hashes`, `artifacts`, the true provenance). Each is a
    # value the graded path reads and the yaml does not state.
    from dataclasses import asdict

    import _harness_checks
    from _t2j import compare

    meta = {k: v for k, v in inst.items()
            if k not in _harness_checks._INSTANCE_KEYS_NOT_MIRRORED}
    meta.update({
        "repo_config": repo_config,
        # The dtype the fp32 GENERATION leg must allocate its KV cache at. The
        # freezer records what its fp32 window actually built; this states the
        # class contract, which is the only honest answer when no fp32 window ran.
        "fp32_kv_cache_dtype": "float32",
        # The class profile band, computed the way the freezer computes it, not a
        # loose `abs_tol: 1e-1` stand-in -- `Golden.
        # native_thresholds` uses this band as the base its overrides apply to, so
        # a made-up profile silently changes what the driver's own tests measure.
        "thresholds": asdict(
            compare.thresholds_for_profile(inst["precision_profile"])),
        "fp32_thresholds": {**asdict(compare.FP32_EXACT_THRESHOLDS),
                            **(inst.get("fp32_thresholds") or {})},
        "native_thresholds": dict(inst.get("native_thresholds") or {}),
        # SYNTHETIC IN BOTH FIELDS THAT NAME A MACHINE, so `_assert_pinned_device`
        # and any reader see at once that no device produced these numbers.
        "provenance": {"gpu_model": "SYNTHETIC", "freeze_device": "SYNTHETIC"},
        # The marker stays HERE, in meta, and is gone from `scenario.json`: meta
        # legitimately carries keys the yaml never had, while every scenario key is
        # classified as a sandbox INPUT or a frozen OUTPUT by name
        # (`_harness_checks.check_scenario_keys_are_classified`), and an
        # unclassified one is a task-authoring failure by that check's definition.
        # It was failing on this exact key.
        "synthetic": True,
    })
    js("meta.json", meta)

    # Decode tokens are PINNED by the scenario rather than taken from the
    # candidate's own argmax -- see decode_steps_probe. Arbitrary but in-vocab
    # here; the freezer writes the reference's greedy continuation.
    js("scenario.json", {
        "mode": inst["mode"],
        "page_ids": page_ids,
        "page_size": page_size,
        "pages_per_seq": pages_per_seq,
        "prompt_len": n,
        "num_decode_steps": steps,
        "decode_token_ids": [int(t) for t in prompt_ids[:steps]],
        "requests": [{"request_id": 0, "prompt_len": n, "page_ids": page_ids}],
    })

    # Frozen (q, k, v) for the isolated attention probe. Deterministic, at the
    # real head geometry of Llama-3.2-1B.
    rng = np.random.default_rng(0)
    heads, kv_heads, head_dim = 32, 8, 64
    npz(
        "paged_attention_units.npz",
        q=rng.normal(size=(n, heads, head_dim)).astype(np.float32),
        k=rng.normal(size=(n, kv_heads, head_dim)).astype(np.float32),
        v=rng.normal(size=(n, kv_heads, head_dim)).astype(np.float32),
        out_native=np.zeros((n, heads, head_dim), dtype=np.float32),
        out_fp32=np.zeros((n, heads, head_dim), dtype=np.float32),
    )
    # The sampler golden is a json/npz pair (the arrays are five vocab-width rows;
    # see fixtures._load_sampler). Synthesized as a pair too, so a partly frozen
    # tree that has one half and not the other still loads -- and the two halves
    # are made to AGREE: `grid_params` advertises N entries, so the npz carries N
    # `grid_{i}` arrays. Without that, `_load_sampler` returns `grid == {}` beside a
    # four-entry `grid_params`, which is the shape of a half-frozen golden and
    # would be indistinguishable from one.
    sampler_grid = list(inst["sampler_grid"])
    top_n = int(inst["logprobs_top_n"])
    js("sampler.json", {"grid_params": sampler_grid,
                        "logprobs_top_n": top_n,
                        "arrays": "sampler.npz",
                        "synthetic": True})
    npz("sampler.npz",
        logits=rng.normal(size=(1, _SYNTHETIC_VOCAB)).astype(np.float32),
        topn_logprobs=np.zeros((1, top_n), dtype=np.float32),
        topn_token_ids=np.zeros((1, top_n), dtype=np.int32),
        **{f"grid_{i}": np.zeros((1, _SYNTHETIC_VOCAB), dtype=np.float32)
           for i in range(len(sampler_grid))})

    # The L6 held-out set (produced by the freezer), which is REQUIRED by
    # `fixtures.load_golden` -- so without a stand-in a partly-frozen tree fails to
    # load at all. Real inputs, zeroed output, same rule as everything below: ONE
    # entry, the frozen prompt with a two-token continuation taken from it, so the
    # arrays are self-consistent (prompt_len + cont_len <= max_model_len) and the
    # probe that reads them has something real to score. `ref_logprobs` is the
    # OUTPUT half and is zeroed; nothing in this package compares against it.
    npz("held_out.npz",
        prompt_ids=prompt.reshape(1, -1).astype(np.int32),
        prompt_lens=np.array([n], dtype=np.int32),
        cont_ids=np.asarray(prompt_ids[:2], dtype=np.int32).reshape(1, -1),
        cont_lens=np.array([2], dtype=np.int32),
        ref_logprobs=np.zeros((1, 2), dtype=np.float32))

    # Zeroed outputs, UNDER THE ARRAY NAMES THE GRADED MODULES READ. Never
    # compared against by any test in this package -- but "the file exists" is not
    # what the golden manifest requires of them: `_t2j/golden_manifest._COMMON`
    # names every array a graded module reads by name, so a single
    # `placeholder=zeros(1)` array per file (which is what these six were) fails
    # `check_golden_manifest` on a fully synthetic tree with three missing tokens
    # and would fail with twenty-one once the manifest was completed. Writing the
    # real names costs the same bytes and keeps the synthetic tree contract-shaped,
    # which is the whole claim this overlay rests on.
    #
    # THE SHAPES ARE NOMINAL, not the model's. A vocab-width row here is
    # `_SYNTHETIC_VOCAB`, not 128 256, for the reason the sampler fixture gives:
    # nothing in this package compares these against anything, and the transforms
    # that DO run are shape-agnostic. The one shape that has to be right is the
    # STEP COUNT, because `Golden.decode_margins` walks `logits_step_{i}` against
    # `meta['num_decode_steps']` and a short run would misalign it.
    logits = np.zeros((n, _SYNTHETIC_VOCAB), dtype=np.float32)
    npz("prefill_forward.npz", logits=logits)
    npz("fp32_prefill_forward.npz", logits=logits)
    # One flat zeroed tensor. The real cache is
    # (layers, 2, num_pages, page_size, kv_heads, head_dim); nothing here reads it,
    # and inventing the head geometry a second time is how the two copies disagree.
    npz("kv_cache_state.npz", cache=np.zeros(1, dtype=np.float32))
    npz("decode_step.npz",
        **{f"logits_step_{i}": np.zeros((1, _SYNTHETIC_VOCAB), dtype=np.float32)
           for i in range(steps)})
    for name in ("generation.npz", "fp32_generation.npz"):
        npz(name,
            tokens=np.zeros(steps, dtype=np.int32),
            step_logprobs=np.zeros(steps, dtype=np.float32))
    for name in ("cache_spec", "params_spec", "param_norms", "interface"):
        js(f"{name}.json", {})
    return out


def missing_from_golden(real: Path) -> list[str]:
    """Artifacts ``fixtures.load_golden`` opens that ``real`` does not have."""
    have = {p.name for p in real.iterdir() if p.is_file()} if real.is_dir() else set()
    return [name for name in _LOADED_BY_FIXTURES if name not in have]


@pytest.fixture(scope="session")
def golden_dir(tmp_path_factory) -> Path:
    """The real ``tests/golden``, with the not-yet-produced artifacts filled in.

    Three states, and the middle one is the whole reason this is an overlay:

    * nothing frozen             -> a fully synthetic tree;
    * PARTLY frozen              -> the real artifacts, plus a synthetic stand-in
                                    for each one whose producer has not run yet
                                    (the catalog is split across six producers);
    * fully frozen               -> the real directory itself, untouched.
    """
    real = TESTS / "golden"
    missing = missing_from_golden(real)
    if not missing:
        return real

    out = tmp_path_factory.mktemp("synthetic-golden")
    have = frozenset()
    if real.is_dir():
        for item in sorted(real.iterdir()):
            if item.is_file():
                shutil.copy2(item, out / item.name)
        have = frozenset(p.name for p in out.iterdir() if p.is_file())
    what = ("tests/golden is not frozen yet" if not have
            else f"tests/golden is missing {len(missing)} artifact(s) "
                 f"({', '.join(missing)}) whose producers have not run yet")
    print(
        f"\n[t2j] {what} -- the driver's unit tests are running against a golden "
        f"at {out} with SYNTHETIC stand-ins for those. Real inputs, zeroed "
        "outputs; no numeric parity is being claimed for the synthetic parts.",
        file=sys.stderr,
    )
    return _synthesize(out, have)


@pytest.fixture(scope="session")
def golden(golden_dir):
    from fixtures import load_golden

    return load_golden(golden_dir)


@pytest.fixture(scope="session")
def cand(golden):
    from fixtures import build_candidate

    # This suite's entire purpose is validating the reference driver, never a
    # port -- exactly the case `allow_reference_default` exists for.
    return build_candidate(TESTS, golden, allow_reference_default=True)


@dataclass
class Reallocated:
    """A driver plus the allocation its golden pins, so a test can name both."""

    cand: object
    page_ids: list[int]
    slot_mapping: np.ndarray
    page_size: int

    @property
    def written_pages(self) -> set[int]:
        """The pages this golden's frozen slot mapping should touch."""
        return {int(s) // self.page_size
                for s in np.asarray(self.slot_mapping).reshape(-1)}


def _reallocated_golden(golden_dir: Path, out: Path, *, shift_block_tables: bool):
    """Copy ``golden_dir`` to ``out`` with the sequence moved to OTHER pages.

    ``shift_block_tables=False`` moves ``scenario.json``'s ``page_ids`` and leaves
    ``inputs.npz``'s ``block_tables`` behind -- the freezer desync
    ``assert_frozen_allocation_agrees`` exists to refuse.
    """
    for item in Path(golden_dir).iterdir():
        if item.is_file():
            shutil.copy2(item, out / item.name)

    scenario = json.loads((out / "scenario.json").read_text())
    inputs = {k: v for k, v in np.load(out / "inputs.npz").items()}
    page_size = int(scenario["page_size"])
    pages_per_seq = int(scenario["pages_per_seq"])
    shifted = [p + pages_per_seq for p in scenario["page_ids"]]
    scenario["page_ids"] = shifted
    (out / "scenario.json").write_text(json.dumps(scenario, indent=2, sort_keys=True))

    n = int(np.asarray(inputs["input_ids"]).shape[0])
    slot = np.array(
        [shifted[p // page_size] * page_size + (p % page_size) for p in range(n)],
        dtype=np.int32)
    if shift_block_tables:
        inputs["slot_mapping"] = slot
        bt = np.zeros(pages_per_seq, dtype=np.int32)
        bt[: len(shifted)] = shifted
        inputs["block_tables"] = bt
    np.savez(out / "inputs.npz", **inputs)
    return shifted, slot, page_size


@pytest.fixture(scope="session")
def misallocated_cand(golden_dir, tmp_path_factory):
    """A second driver whose FROZEN allocation is deliberately elsewhere.

    Its ``scenario.json`` / ``inputs.npz`` place the sequence on pages the
    reference allocator would never hand out. That makes the frozen-allocation boundary
    observable: a probe that is HANDED the allocation must move (its cache tensor
    lands on those other pages), and a probe that DERIVES it must not (the
    candidate's own free list is unchanged). A single global flag in either
    direction breaks one half of that and is caught by
    ``test_the_frozen_allocation_is_per_probe_not_global``.
    """
    from fixtures import build_candidate, load_golden

    out = tmp_path_factory.mktemp("misallocated-golden")
    pages, slot, page_size = _reallocated_golden(golden_dir, out,
                                                 shift_block_tables=True)
    # This suite's entire purpose is validating the reference driver, never a
    # port -- exactly the case `allow_reference_default` exists for.
    return Reallocated(cand=build_candidate(TESTS, load_golden(out),
                                            allow_reference_default=True),
                       page_ids=pages, slot_mapping=slot, page_size=page_size)


@pytest.fixture(scope="session")
def desynced_cand(golden_dir, tmp_path_factory):
    """A driver over a golden that pins ONE allocation in two contradictory ways.

    ``scenario.json`` says the sequence lives on the shifted pages while
    ``inputs.npz``'s ``block_tables`` still says the original ones. Nothing in the
    probe surface compares those two, so without a guard the decode path would
    read pages the prefill never wrote and the candidate would be blamed.
    """
    from fixtures import build_candidate, load_golden

    out = tmp_path_factory.mktemp("desynced-golden")
    pages, slot, page_size = _reallocated_golden(golden_dir, out,
                                                 shift_block_tables=False)
    # This suite's entire purpose is validating the reference driver, never a
    # port -- exactly the case `allow_reference_default` exists for.
    return Reallocated(cand=build_candidate(TESTS, load_golden(out),
                                            allow_reference_default=True),
                       page_ids=pages, slot_mapping=slot, page_size=page_size)
