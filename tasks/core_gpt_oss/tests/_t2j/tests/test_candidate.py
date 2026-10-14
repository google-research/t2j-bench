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

"""The driver, exercised against the golden reference as candidate.

These are the grader's own unit tests, not graded checks. They fail loudly if the
driver silently returns None for a probe -- which would show up downstream as a
numeric test that passes vacuously.

NAME THE MODULES to run them::

    pytest tasks/core_llama3/tests/_t2j/tests/test_*.py

The graded conftest one level up carries ``collect_ignore_glob =
["_t2j/tests/*"]``, so ``pytest .../tests`` -- the GRADED invocation -- does not
recurse in here and collect the grader's own suite into the candidate's score.
pytest does not apply that ignore to a path named directly, which is why the
line above still works.

They need the baked checkpoint, and the CONTRACT path for it is
``/data/{repo_id}`` inside both images. On an authoring host with no ``/data``,
``$T2J_INFERENCE_DATA_ROOT`` names an extracted copy and the driver rebinds
``model.loader.DATA_ROOT`` -- the same seam ``tests_parity/_run_jax.py`` and
``test_resolve_maps_a_repo_id_under_data`` already use. Nothing about the graded
contract moves: ``resolve_model_dir`` still hardcodes ``/data``, and the resolved
root is reported back through ``cand.loader_facts`` so a spec check can assert it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[3]
TESTS = TASK / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

_REPO_ID = "meta-llama/Llama-3.2-1B-Instruct"
_ROOTS = [Path("/data")]
if os.environ.get("T2J_INFERENCE_DATA_ROOT"):
    _ROOTS.append(Path(os.environ["T2J_INFERENCE_DATA_ROOT"]))

pytestmark = pytest.mark.skipif(
    not any((root / _REPO_ID / "config.json").exists() for root in _ROOTS),
    reason="needs the baked checkpoint at /data/{repo_id} "
           "(or $T2J_INFERENCE_DATA_ROOT/{repo_id} on an authoring host)",
)


# --------------------------------------------------------------------------- #
# The four checks the task brief pins.
# --------------------------------------------------------------------------- #
def test_the_sandbox_blocks_torch(cand):
    assert cand.banned_leaks == []


def test_prefill_returns_logits_of_the_frozen_shape(cand):
    out = cand.prefill(fp32=False)
    assert out["logits"].ndim == 2
    assert np.isfinite(out["logits"]).all()


def test_a_probe_that_raises_surfaces_as_an_error_not_a_none(cand):
    with pytest.raises(RuntimeError, match="unknown probe"):
        cand.invariance("not_a_real_invariant")


def test_generation_returns_the_configured_number_of_steps(cand, golden):
    for fp32 in (False, True):
        g = cand.generation(fp32=fp32)
        assert g["tokens"].shape[0] == int(golden.meta["num_decode_steps"]) == 8


def test_the_fp32_generation_leg_allocates_an_fp32_kv_cache(cand, golden):
    """The 180x trap, pinned on the CANDIDATE side.

    ``fp32_generation.npz`` was frozen with the KV cache at fp32
    (``meta['fp32_kv_cache_dtype']``). The same fp32 model on the bf16 SERVING
    cache sits roughly 180x further from that artifact -- and every token still
    agrees, so no equality clause anywhere would notice. The probe therefore has
    to be built at the pinned dtype, and the dtype it ACTUALLY allocated has to
    be observable.

    OBSERVED, not declared: ``kv_cache_dtype_observed`` is read off the engine's
    own cache after construction, with NO fallback to the dtype the grader asked
    for. The native leg is asserted alongside it, because "both legs report
    float32" is the way this assertion goes vacuous.
    """
    want = str(golden.meta["fp32_kv_cache_dtype"])
    serving = str(golden.meta["repo_config"]["dtype"])
    assert want != serving, "the pin equals the serving dtype; this proves nothing"
    fp32, native = cand.generation(fp32=True), cand.generation(fp32=False)
    assert fp32["kv_cache_dtype_observed"] == want, fp32
    assert fp32["kv_cache_dtype_requested"] == want, fp32
    assert native["kv_cache_dtype_observed"] == serving, native
    # On the reference the FIRST lookup answers, so the shape scan below is not
    # what is carrying this assertion.
    assert fp32["kv_cache_dtype_source"] == "attribute", fp32
    # The fallback is GONE, and its absence is the fix: `kv_cache_dtype` used to
    # be `observed or requested`, so an engine that hid its cache handed the
    # graded clause the grader's own pin. Nothing may reintroduce a field that
    # answers the dtype question without having observed it.
    assert "kv_cache_dtype" not in fp32, (
        "a collapsed dtype field is back; the graded clause must read "
        "`kv_cache_dtype_observed`, which is None when nothing was observed"
    )


# --------------------------------------------------------------------------- #
# ...and the discrimination the four cannot supply on their own.
# --------------------------------------------------------------------------- #
def test_the_torch_blocker_is_actually_installed_in_the_worker(cand):
    """``banned_leaks == []`` does NOT prove the blocker ran.

    Neither does "torch failed to import": the verifier image has no torch at
    all, so a missing blocker and a working blocker produce the same
    ``ModuleNotFoundError``. The exception TYPE is the only thing that separates
    them, so that is what is asserted -- and it holds in BOTH environments,
    because ``BannedModuleFinder`` raises before any real finder is consulted.
    """
    report = cand.sandbox_report
    assert report["blocker_installed"] is True
    assert report["torch_import_error_type"] == "TorchBlockedError", report
    assert "torch" in report["banned_roots"]


def test_the_worker_does_not_run_where_the_goldens_are(cand):
    """The filesystem half of the same "prove it, don't assume it" rule.

    ``_worker_entry`` moves its cwd for a reason: from the cwd the worker would
    otherwise inherit, ``rglob('golden/fp32_generation.npz')`` returns the frozen
    answer tokens and ``step_logprobs``. The child is a plain ``spawn`` child with
    the parent's cwd and unrestricted I/O, so the payload split
    ``fixtures.build_candidate`` describes was true of the payload and false of the
    process.

    Two assertions, and the second is the one that means something: the cwd is not
    the tests directory, and the relative probe the child ran from it came back
    empty. Neither claims the goldens are UNREACHABLE -- an absolute path still
    reads them, which is an I2 filesystem-sandbox item.
    """
    from pathlib import Path

    report = cand.sandbox_report
    cwd = Path(report["cwd"]).resolve()
    tests_dir = Path(__file__).resolve().parents[2]
    assert cwd != tests_dir and tests_dir not in cwd.parents, report
    assert report["relative_golden_hits"] == [], report


def test_every_probe_produced_a_value(cand):
    """The contract ``_validate_results`` enforces, observed end to end."""
    from _t2j import candidate as C

    results = cand._ensure_run()
    for name in C.REQUIRED_PROBES:
        assert name in results, f"{name} is absent from the worker's result"
        assert results[name] is not None, f"{name} came back None"
    for kind in C.INVARIANCE_KINDS:
        assert kind in results["invariance"], kind


def _written_pages(kv_cache_state) -> set[int]:
    """The page indices a cache CONTAINER actually holds data on.

    ``kv_cache_state`` is a tuple with one ``(n_slots, num_pages, page_size, 2,
    kv_heads, head_dim)`` tensor per distinct KV geometry, over a cache that
    started at zero, so "this page was written" is exactly "some element on this
    page is nonzero" in ANY group. WHERE data landed, not what its bits are --
    which is the property that survives autotune noise.

    The union across groups is the right reduction because every allocating layer
    scatters to the same ``slot_mapping`` regardless of which group holds its slot,
    so the page set is a property of the scenario and not of the grouping.
    """
    groups = (kv_cache_state if isinstance(kv_cache_state, (tuple, list))
              else [kv_cache_state])
    pages: set[int] = set()
    for group in groups:
        arr = np.asarray(group)
        nonzero = np.any(arr != 0, axis=tuple(i for i in range(arr.ndim) if i != 1))
        pages |= {int(i) for i in np.flatnonzero(nonzero)}
    return pages


def test_the_frozen_allocation_is_per_probe_not_global(cand, golden, misallocated_cand):
    """The frozen-allocation boundary, observed rather than asserted from a table.

    ``misallocated_cand`` is the same repo driven by a scenario that pins a
    DIFFERENT page assignment. Two things must follow, and no global flag can
    produce both:

      * the probes that are HANDED the allocation move -- the cache tensor is
        written to the pages the golden names, and to no others;
      * the probes that DERIVE it do not -- the candidate's own free-list
        allocator is unchanged, so ``block_table`` and ``slot_mapping`` are
        identical.

    Force every probe to the frozen allocation and the second pair breaks (and
    ``test_block_table_contract`` starts comparing the golden to itself); force
    every probe off it and the first breaks (and a correct candidate with a
    different allocator fails ``test_kv_cache_write``).

    THE HANDED DIRECTION IS ASSERTED POSITIONALLY, and that is the whole reason
    this test can fail. The obvious form -- ``not array_equal(cache_a, cache_b)``
    -- decides nothing here, because the two drivers run in two SPAWNED processes
    and whether two bf16 forwards agree bit-for-bit across a process boundary is
    a property of the CARD's occupancy, not of this code.
    Concurrent bf16 forwards on a shared card are nondeterministic: XLA's
    timing-based GEMM autotuner can pick different kernels under load, so the
    same forward run concurrently yields several distinct digests (a few bf16
    ULP apart, enough to flip the argmax on a couple of rows), while sequential
    forwards on an idle card are bit-identical; disabling autotuning collapses
    the concurrent case back to one digest, which pins the autotuner as the
    cause. These cards are shared. So a byte-inequality between two spawned
    workers passes on noise whenever anything else is running, and its
    discrimination is decided by who else is on the GPU -- which is not a test.
    Page IDENTITY is: it is a property of where the writes landed, and it holds
    in both configurations.

    Within ONE process the same comparison IS bit-exact, which is what
    ``test_the_paging_invariants_hold`` and
    ``test_two_identical_calls_are_bit_identical`` assert -- and the invariance
    probes are deliberately arranged to run both legs in the same worker for
    exactly that reason. The cross-process prefill comparison below is by COSINE
    and nothing tighter for the same reason.
    """
    from _t2j import _formulas as F

    np.testing.assert_array_equal(cand.block_table, misallocated_cand.cand.block_table)
    np.testing.assert_array_equal(cand.slot_mapping, misallocated_cand.cand.slot_mapping)

    page_size = int(golden.scenario["page_size"])
    frozen_pages = {int(s) // page_size
                    for s in np.asarray(golden.inputs["slot_mapping"]).reshape(-1)}
    assert _written_pages(cand.kv_cache_state) == frozen_pages, (
        "the cache probe did not write exactly the pages the frozen slot mapping "
        "names"
    )
    assert (_written_pages(misallocated_cand.cand.kv_cache_state)
            == misallocated_cand.written_pages), (
        "the cache probe ignored the frozen slot mapping it was handed: it wrote "
        "the same pages for a scenario that pins different ones"
    )
    # ...and the two really are different pages, or the pair above says nothing.
    assert frozen_pages != misallocated_cand.written_pages

    a, b = cand.prefill()["logits"], misallocated_cand.cand.prefill()["logits"]
    assert F.centered_cosine(a, b) > 0.9995


def test_the_allocation_policy_the_probes_EXECUTED_matches_the_declared_table(cand):
    """``allocation_source`` is observed behaviour; the table is a declaration.

    Comparing them is what makes ``USE_FROZEN_ALLOCATION`` a control. Before this,
    ``allocation_source`` was BUILT by iterating that table, so every assertion on
    it was the table asserting itself and a code-only change to which allocation a
    probe honours moved neither side. Each entry below was written by the probe
    that ran, at the line where it took its allocation.
    """
    from _t2j import candidate as C

    def ran(probe: str) -> bool:
        """Did this probe execute? -- read off the result, not off a list here.

        A probe whose frozen input is absent answers ``{"supported": False}`` and
        honours no allocation at all, so it records nothing. Everything else ran.
        """
        if probe.startswith("invariance."):
            result = cand.invariance(probe.split(".", 1)[1])
        else:
            result = cand._ensure_run().get(probe)
        return not (isinstance(result, dict) and result.get("supported") is False)

    observed = cand.allocation_source
    assert observed, "no probe recorded its allocation policy"

    # COVERAGE, first: a probe that quietly stopped recording would otherwise
    # satisfy the value check below by not being in it.
    expected = {probe for probe in C.USE_FROZEN_ALLOCATION if ran(probe)}
    assert set(observed) == expected, (
        f"recorded but not declared: {sorted(set(observed) - expected)}; "
        f"ran but recorded nothing: {sorted(expected - set(observed))}"
    )

    for probe, source in sorted(observed.items()):
        want = "frozen" if C.USE_FROZEN_ALLOCATION[probe] else "candidate"
        assert source == want, (
            f"{probe} DECLARES {want!r} in USE_FROZEN_ALLOCATION but EXECUTED "
            f"{source!r}"
        )
    # The boundary itself: both halves were exercised, so a global flag in either
    # direction is visible here and not only in the table.
    assert observed["block_table"] == "candidate"
    assert observed["kv_cache_state"] == "frozen"
    assert set(observed.values()) == {"frozen", "candidate"}


def test_the_isolated_attention_probe_matches_the_numpy_reference(cand, golden):
    """A third implementation, sharing no code with the candidate's.

    ``_formulas.reference_paged_attention_numpy`` is a numpy transcription of the
    same formulation. Running it over the same frozen ``(q, k, v)``, the same
    frozen slot mapping and the same frozen block table is what turns "the probe
    returned an array" into "the probe returned the right array".
    """
    from _t2j import _formulas as F

    unit = cand.paged_attention_unit(fp32=True)
    assert unit["supported"] is True, unit
    got = unit["out"]

    units = golden.paged_attention_units
    q = np.asarray(units["q"], dtype=np.float32)
    k = np.asarray(units["k"], dtype=np.float32)
    v = np.asarray(units["v"], dtype=np.float32)
    inputs = golden.inputs
    slots = np.asarray(inputs["slot_mapping"]).astype(np.int64)
    spec = cand.cache_spec
    cache = np.zeros((spec["num_pages"], spec["page_size"], 2, k.shape[1], q.shape[2]),
                     dtype=np.float32)
    flat = cache.reshape(-1, 2, k.shape[1], q.shape[2])
    flat[slots, 0] = k
    flat[slots, 1] = v

    want = F.reference_paged_attention_numpy(
        q, flat.reshape(cache.shape),
        block_tables=np.asarray(inputs["block_tables"]),
        seq_lens=np.asarray(inputs["seq_lens"]),
        query_start_loc=np.asarray(inputs["query_start_loc"]),
        num_kv_heads=int(k.shape[1]),
        sm_scale=float(q.shape[2]) ** -0.5,
    )
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def test_the_decode_probe_feeds_the_frozen_tokens_from_the_frozen_slots(cand, golden):
    """Not the candidate's own argmax.

    The torch freezer fed ITS greedy continuation; a candidate that diverges at
    step 0 must still be compared against the reference's step-1 input, or every
    later step reads as a decode bug.
    """
    steps = cand.decode_steps
    assert steps["supported"] is True, steps
    want = [int(t) for t in golden.scenario["decode_token_ids"]]
    assert steps["token_ids"] == want
    assert steps["num_steps"] == len(want)
    for i in range(steps["num_steps"]):
        row = steps[f"logits_step_{i}"]
        assert row.ndim == 1 and np.isfinite(row).all()


def test_the_paging_invariants_hold_and_really_used_different_pages(cand):
    """Same logical KV on different physical pages -> identical logits.

    The second assertion is what stops this passing vacuously: if the two probes
    were handed the same page ids, the equality would say nothing at all.
    """
    a = cand.invariance("paging_default")
    b = cand.invariance("paging_permuted")
    assert a["page_ids"] != b["page_ids"], "the two legs used the same pages"
    assert not set(a["page_ids"]) & set(b["page_ids"])
    np.testing.assert_array_equal(a["logits"], b["logits"])


def test_decode_and_prefill_agree_on_the_same_prefix(cand):
    """Graded as a DISTRIBUTION, not elementwise -- measured, not preferred.

    A naive form of ``test_decode_prefill_consistency`` compares raw logits
    with ``assert_allclose(rtol=2e-3, atol=2e-3)``. The reference itself fails
    that: at the serving dtype the two paths run different reduction shapes (a
    12-token prefill matmul versus a 1-token decode matmul), so 7.4% of the
    128256 logits land one to four bf16 ULP apart -- max abs 0.125 on logits of
    magnitude ~3.5, and a max RELATIVE difference of 228x on the near-zero tail,
    which is exactly why abs/rel comparison on raw logits is forbidden.

    Cosine, KL and argmax are the calibrated measures, and the reference's own
    self-noise sets the floor for each. The bounds below sit an order above the
    KL floor and leave the argmax exact,
    which is still nowhere near vacuous -- the pre-RoPE-K bug this invariant
    exists to catch moved the same comparison by 5.64 at the I1-9 vLLM gate,
    three orders past the noise, and it moves the argmax too.
    """
    from _t2j import _formulas as F

    r = cand.invariance("decode_prefill")
    assert r["supported"] is True, r
    decode, prefill = r["decode_logits"], r["prefill_last_row"]
    cosine = F.centered_cosine(decode, prefill)
    kl = F.softmax_kl(prefill, decode)
    top1 = F.top1_agreement(prefill, decode)
    detail = (f"cosine={cosine:.8f} kl={kl:.3e} top1={top1} "
              f"max_abs={float(np.max(np.abs(decode - prefill))):.4f}")
    assert top1 == 1.0, detail
    assert cosine > 0.9995, detail
    assert kl < 1e-2, detail


def test_freeing_and_reallocating_pages_leaves_a_live_sequence_alone(cand):
    r = cand.invariance("cache_reuse")
    assert r["supported"] is True, r
    assert r["reused_pages"], "nothing was actually freed and reallocated"
    assert not set(r["live_pages"]) & set(r["reused_pages"])
    # The live pages' own KV first: the logits leg cannot see an allocator defect
    # (the closing forward rewrites every slot it reads), so this is the clause
    # that grades the property. See _invariance_cache_reuse's docstring.
    np.testing.assert_array_equal(r["live_kv_before"], r["live_kv_after"])
    np.testing.assert_array_equal(r["before_reuse"], r["after_reuse"])


def test_two_identical_calls_are_bit_identical(cand):
    r = cand.invariance("determinism")
    np.testing.assert_array_equal(r["first"], r["second"])


def test_the_multi_sequence_invariants_report_unsupported_on_core_rather_than_lying(cand):
    """`core` runs one sequence, so these have no second sequence to speak of.

    Their tests are DESELECTED on `core`, so this marker is never graded here --
    but it must exist, because when I2 opens the mode gate an unimplemented probe
    has to FAIL the check rather than quietly pass.
    """
    for kind in ("batching", "cross_sequence_leakage", "chunking", "prefix_cache"):
        r = cand.invariance(kind)
        assert r["supported"] is False
        assert "core" in r["reason"] or "sequence" in r["reason"]


def test_the_spec_probes_describe_the_real_converted_checkpoint(cand):
    """Post-fusion keys, both halves of the tie, and a live cache geometry."""
    spec = cand.params_spec
    assert any(k.endswith("self_attn.qkv_proj.weight") for k in spec)
    assert any(k.endswith("mlp.gate_up_proj.weight") for k in spec)
    # The tie-dedup trap: flax's pytree drops one of these names and torch's
    # named_parameters() drops the other. convert_checkpoint keeps both, which is
    # what makes the set equality in test_param_correspondence honest.
    assert "lm_head.weight" in spec and "model.embed_tokens.weight" in spec

    norms = cand.param_norms
    assert set(norms) == set(spec)
    assert all(entry["finite"] for entry in norms.values())
    assert all(entry["norm"] > 0 for entry in norms.values()), "an unloaded tensor"

    cache = cand.cache_spec
    assert cache["shape"] == [
        16, cache["num_pages"], cache["page_size"], 2,
        cache["num_kv_heads"], cache["head_dim"],
    ]


def test_the_metadata_schema_is_read_by_field_never_by_instance_equality(cand):
    """``AttentionMetadata(...) == AttentionMetadata(...)`` RAISES on both sides.

    (``ValueError`` in JAX, ``RuntimeError`` in torch: a frozen dataclass's
    generated ``__eq__`` compares field tuples and ``bool()`` of a multi-element
    array is ambiguous.) ``hash()`` raises here too. So the probe must report
    FIELDS, and it must have survived producing them.
    """
    schema = cand.metadata_schema
    assert schema["fields"] == [
        "input_positions", "block_tables", "seq_lens", "query_start_loc",
        "request_distribution", "padded_num_reqs",
    ]
    assert schema["schema"]["padded_num_reqs"]["kind"] == "static"
    for name in ("input_positions", "block_tables", "seq_lens", "query_start_loc",
                 "request_distribution"):
        entry = schema["schema"][name]
        assert entry["kind"] == "array"
        assert entry["dtype"] == "int32", f"{name} must be int32 (metadata contract)"
    assert cand.static_shapes["padded_num_reqs"] == cand.static_shapes["max_num_seqs"]
    # ...and ONE field is reported by VALUE, from both stages the engine has.
    # `test_attention_metadata_schema` grades those two lists; this asserts the
    # probe still produces them, because the check reads them through `.get()`
    # and a probe that stopped reporting the key must not read as "no defect".
    assert cand.metadata_schema["request_distribution"] == {
        "prefill": [0, 1, 0], "decode": [1, 0, 0]}


def test_the_forward_lowers_to_a_jaxpr_with_no_host_callback(cand):
    t = cand.traceability
    assert t["supported"] is True and t["traced"] is True
    assert t["callbacks"] == []
    assert t["num_primitives"] > 8


def test_the_sampler_grid_is_graded_and_the_identity_row_is_a_no_op(cand, golden):
    """Transforms only -- no RNG draw is taken, so nothing here is PRNG-family
    dependent. The identity row must return the logits unchanged, and
    a masking row must actually mask, or the probe is grading nothing."""
    out = cand.sampler_outputs
    assert out["supported"] is True, out
    grid = out["grid"]
    assert set(grid) == {str(i) for i in range(len(golden.meta["sampler_grid"]))}

    raw = np.asarray(golden.sampler["logits"], dtype=np.float32)
    np.testing.assert_allclose(grid["0"], raw, rtol=0, atol=0)   # temp 1, no k, no p
    assert np.isneginf(grid["1"]).any(), "the top-k row masked nothing"
    assert np.isneginf(grid["2"]).any(), "the top-p row masked nothing"

    n = int(golden.meta["logprobs_top_n"])
    assert out["topn_logprobs"].shape[-1] == n
    assert (np.diff(out["topn_logprobs"], axis=-1) <= 0).all(), "not descending"


def test_the_fp32_prefill_is_a_second_forward_not_the_native_one_relabelled(cand):
    native = cand.prefill(fp32=False)["logits"]
    fp32 = cand.prefill(fp32=True)["logits"]
    assert native.shape == fp32.shape
    # bf16 and fp32 forwards of the same weights agree in direction and disagree
    # in the last digits; identical arrays would mean one was reused for both.
    assert not np.array_equal(native, fp32)
    from _t2j import _formulas as F
    assert F.centered_cosine(fp32, native) > 0.99
    assert F.top1_agreement(fp32, native) > 0.8


def test_hidden_states_are_frozen_when_the_instance_asks_for_them(cand, golden):
    if not golden.meta["compare_hidden_states"]:
        pytest.fail("this instance sets compare_hidden_states: false -- update the test")
    out = cand.prefill(fp32=False)
    keys = sorted((k for k in out if k.startswith("hidden_")),
                  key=lambda k: int(k.split("_", 1)[1]))
    # HF / ref_jax_repo convention: embedding output first, post-final-norm last.
    assert len(keys) == cand.cache_spec["num_layers"] + 1
    assert not np.array_equal(out[keys[0]], out[keys[-1]])


def test_the_pending_probes_say_so_instead_of_passing_quietly(cand):
    """``battery`` has nothing frozen, and says so rather than passing quietly.

    It is presence-DESELECTED today. The marker is what makes the day its golden
    appears a loud failure rather than a silent pass against a probe nobody wrote
    -- which is exactly what happened to ``reference_units``: its freezer
    landed, the marker turned into a real
    replay, and the check below is what asserts the replay is real.
    """
    assert cand.battery["supported"] is False


def test_the_reference_unit_replay_runs_the_candidates_own_submodules(cand, golden):
    """Every frozen unit resolves on the candidate and returns the right SHAPES.

    Shapes and resolution only -- the VALUES are the graded tier's business
    (``test_numeric.test_reference_units``), and this package grades the driver.
    What it rules out is the failure this probe is most exposed to: a path that
    silently does not resolve, or a unit set that quietly shrank, either of which
    would leave the graded check comparing a handful of units and reporting a
    clean pass.

    Skipped-by-deselection upstream when the golden is absent, so it asserts the
    unsupported marker in that case instead of inventing a reason to pass.
    """
    probe = cand.reference_units
    if not golden.reference_units:
        assert probe["supported"] is False
        assert "reference_units" in probe["reason"]
        return

    meta = golden.reference_units["meta"]
    want = list(meta["modeling_units"])
    assert sorted(probe["units"]) == sorted(want), {
        "frozen": sorted(want), "replayed": sorted(probe["units"])}
    for name in want:
        got = probe["units"][name]
        assert got["supported"] is True, got
        frozen_outputs = meta["units"][name]["outputs"]
        assert len(got["outputs"]) == len(frozen_outputs), (name, got["path"])
        for value, spec in zip(got["outputs"], frozen_outputs):
            assert list(np.shape(value)) == list(spec["shape"]), (name, spec["key"])


def test_the_jitted_forward_is_compiled_once_and_reused():
    """One wrapper, because a fresh one per call recompiles the whole model.

    A fresh ``nnx.jit`` per call carries a fresh trace cache, so every probe
    re-traces and re-compiles all 16 layers, while one shared wrapper compiles the
    model once and reuses it across calls.

    The assertion below checks that COMPARISON -- one wrapper against one per call
    -- by wrapper identity rather than by counting compilations, so it does not
    depend on any particular compile count (``compilation_probe`` calls
    ``jax.clear_caches()`` and constructs more engines, so raw counts do not
    describe a single worker anyway).

    NOT a determinism fix. Separate compilations of one graph INSIDE ONE PROCESS
    agree: XLA caches autotune results per process, keyed by the HLO fingerprint,
    so the second compile reuses the first one's winner. The bf16 divergence this
    class does have is CROSS-PROCESS under contention, which caching inside one
    worker cannot touch.

    Asserted on the runtime's own accessor rather than by counting compiles:
    identity of the returned wrapper is what "one trace cache" means. Constructed
    without ``__init__`` on purpose -- the cache is the ONLY state this method
    reads, so the assertion costs no checkpoint and no device.
    """
    from _t2j import candidate as C

    runtime = object.__new__(C._CandidateRuntime)
    runtime._fwd = None
    first = runtime._jit_forward()
    assert runtime._jit_forward() is first, (
        "each call returns a new nnx.jit wrapper, so every probe recompiles"
    )
    assert runtime._fwd is first


def test_the_worker_ran_under_the_mandated_matmul_precision(cand):
    """``JAX_DEFAULT_MATMUL_PRECISION=highest`` is a Global Constraint.

    Reported by the child out of ``jax.config``, not asserted about the parent's
    environment: the child is where the fp32 tier's matmuls actually run, and a
    ``setdefault`` in either process would let an inherited ``default`` win and
    grade the 1e-6 fp32 tier at TF32's 10-bit mantissa.
    """
    report = cand.sandbox_report
    assert report["matmul_precision"] == "highest", report
    assert report["matmul_precision_env"] == "highest", report


def test_the_per_step_generation_logits_are_present_and_parallel_to_the_tokens(cand):
    """C4's plumbing, and the hole it used to have.

    ``generation_step_logits`` degraded to ``[]`` when the engine returned none,
    which would hand ``test_generate_runs``'s argmax clause nothing to walk --
    "no row disagreed" and "no row examined" are the same empty list.

    Both legs are checked: the fp32 leg is a second full decode loop and a
    worker that produced rows for one and not the other would otherwise be
    invisible here.
    """
    width = cand.prefill()["logits"].shape[-1]
    for fp32 in (False, True):
        rows = cand.generation_step_logits(fp32=fp32)
        tokens = cand.generation(fp32=fp32)["tokens"]
        assert len(rows) == tokens.shape[0] > 0
        for row in rows:
            assert np.asarray(row).reshape(-1).shape[0] == width
            assert np.isfinite(np.asarray(row)).all()


def test_a_golden_whose_two_allocation_pins_disagree_is_refused(desynced_cand):
    """``scenario.json:page_ids`` vs ``inputs.npz:block_tables``, end to end.

    The desynced golden is a freezer bug, not a candidate bug, and it has to read
    as one: without the guard the decode path gathers pages the prefill never
    wrote and the numbers land on the candidate's scoreboard.
    """
    from _t2j import candidate as C

    with pytest.raises(C.CandidateError, match="contradicts itself"):
        desynced_cand.cand.prefill()


def test_the_loader_resolved_the_checkpoint_under_the_data_contract(cand):
    facts = cand.loader_facts
    assert facts["model_dir"].endswith(_REPO_ID)
    assert facts["has_tokenizer"] is True
    assert [row[0] for row in facts["stacked_params"]] == [
        "qkv_proj", "qkv_proj", "qkv_proj", "gate_up_proj", "gate_up_proj"]
    if not os.environ.get("T2J_INFERENCE_DATA_ROOT"):
        assert facts["data_root"] == "/data", (
            "in a graded run the loader must resolve under /data and nowhere else"
        )
