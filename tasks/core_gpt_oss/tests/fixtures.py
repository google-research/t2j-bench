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

"""Golden + candidate loaders for the standalone inference grader.

Hermetic: imports only the vendored ``_t2j`` helpers, ``numpy`` and the standard
library. Every per-instance specific comes from ``golden/`` (data) and
``golden/meta.json`` (mode, thresholds, ``repo_config``, sampler grid), so this
file is identical across the 18 instances of the class and nothing per-instance
is hardcoded in the grader.

``_guard`` is kept VERBATIM from ``tasks/dpo_bloom/tests/fixtures.py``. It is the
one thing that turns a bad fetch into a legible error rather than a comparison
against a 130-byte text stub, and re-deriving it here would be re-deriving the
one piece of this file that has already been paid for.

WHAT IS DIFFERENT FROM THE TRAINING CLASS, and why: the artifact list is the
inference catalog (scenario, cache spec, prefill/decode/paged-attention units,
sampler grid, generation) instead of the training one, and there is no
``gradients`` / ``loss_artifact`` / ``lr_schedule`` / ``train_trajectory`` /
``grad_authenticity`` -- inference has no backward pass, so those have no
analogue and their absence is asserted by the harness rather than assumed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _t2j import compare
from _t2j.candidate import DEFAULT_PERF_ROUNDS, Candidate

# A Git LFS pointer file starts with this; detect a forgotten `git lfs pull`
# (or, in the no-LFS setup, a golden that was never regenerated) and fail loudly
# instead of comparing against a 130-byte text stub.
_LFS_POINTER_PREFIX = b"version https://git-lfs"


#: What to tell someone whose golden tree is missing a file. This is the message a
#: STUCK PERSON actually reads, so it has to be true: the goldens ship via Git LFS
#: under the unified `tasks/` tree, so `git lfs pull` is the cheap path and is
#: named first; regeneration is the fallback, and is deterministic (~20 s).
_MISSING_GOLDEN_HINT = (
    "Run `git lfs pull` to fetch it (the goldens ship via Git LFS under "
    "tasks/). To regenerate it instead, re-freeze from the reference "
    "implementation (~20 s, deterministic, and "
    "byte-reproducible under --check); that path must run inside the "
    "model image, which bakes the checkpoint at /data."
)


def _guard(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"missing golden artifact: {path}. {_MISSING_GOLDEN_HINT}"
        )
    with path.open("rb") as handle:
        if handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
            raise RuntimeError(f"{path} is a Git LFS pointer, not real data — run `git lfs pull` or regenerate.")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(_guard(path).read_text())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return {k: v for k, v in np.load(_guard(path)).items()}


#: ``NumericThresholds``'s own field names, read off the dataclass rather than
#: transcribed: a field added there must not need an edit here to be honoured.
_THRESHOLD_FIELDS: frozenset[str] = frozenset(
    compare.NumericThresholds.__dataclass_fields__)


def _threshold_fields(values: dict | None) -> dict:
    """The subset of ``values`` that ``NumericThresholds(**...)`` accepts."""
    return {k: v for k, v in (values or {}).items() if k in _THRESHOLD_FIELDS}


def _threshold_extras(values: dict | None) -> dict:
    """The rest: calibrated scalars that belong to a CHECK, not to the comparator.

    ``NumericThresholds`` is the shared band (abs/rel/cosine/KL/argmax margin).
    A per-instance number that only one check reads -- a per-token logprob
    tolerance, a top1-top2 decisiveness margin -- is still a calibrated
    threshold and still belongs in ``instance.yaml`` beside the others, with its
    measured floor and its ratio. It just cannot be a constructor argument.
    Splitting them here is what lets a later task ADD one without touching this
    file or breaking every check in the tier on a ``TypeError``.
    """
    return {k: v for k, v in (values or {}).items() if k not in _THRESHOLD_FIELDS}


@dataclass
class Golden:
    """Frozen reference artifacts + materialized thresholds for one instance."""

    dir: Path
    meta: dict
    # The frozen scenario and the arrays derived from it. Every
    # allocation-dependent probe is HANDED these rather than deriving them.
    scenario: dict
    inputs: dict
    cache_spec: dict
    # Numeric tier.
    prefill_forward: dict
    fp32_prefill_forward: dict
    kv_cache_state: dict
    decode_step: dict
    paged_attention_units: dict
    sampler: dict
    # Behavioral tier. Two precisions, and they are not interchangeable:
    # ``generation`` is the bf16 SERVING run (frozen before the freezer's fp32
    # window, which is what makes a lost fp32 restore visible as a cross-artifact
    # disagreement), while ``fp32_generation`` is the same prompt and the same
    # number of decode steps on the upcast model. The equivalence claim -- same
    # sampled sequence, same likelihood -- is only meaningful against the fp32 one:
    # The cross-port per-token logprob gap is 3.8e-05 at fp32 against
    # 6.4e-02 ... 1.4e-01 at native, and at native that gap is the same size as one
    # correct implementation's own cross-process spread (5.0e-02), so no native
    # bound can separate a port error from noise. ``instance.yaml``'s
    # ``fp32_thresholds.logprob_atol`` carries the calibrated value.
    generation: dict
    fp32_generation: dict
    # Spec tier.
    params_spec: dict
    param_norms: dict
    interface: dict
    # L6. REQUIRED, not optional: L6 is an
    # always-on graded check, so a missing artifact is a hard failure rather
    # than a quiet deselect. The predecessor -- held_out_nll.json, presence-
    # gated on a producer that was never written and an env var nothing set --
    # never executed on any instance, in either task class.
    held_out: dict
    # Optional (OPTIONAL_GOLDENS): each has its own producer and its
    # check is presence-deselected until that producer has run. ``None`` here is
    # a coverage gap the conftest reports, never a silent pass.
    reference_units: dict | None = None
    battery: dict | None = None
    # THE MULTIMODAL PAIR, and its gate is NOT presence -- it is the ``multimodal``
    # family fact (``conftest._APPLICABLE``). The distinction matters: the six
    # text-only instances have no image at all, so their two image checks are
    # DESELECTED and never read these; a family that DECLARES ``multimodal: true``
    # and whose golden tree is nevertheless missing the recording gets a loud
    # failure from the check rather than a quiet deselect, which is the right way
    # round for an artifact the instance says it has. ``image_preprocess`` holds
    # the image-preprocess producer's arrays; ``image_fixture_png`` is the
    # PNG those arrays were computed FROM, and it is the only thing here that
    # crosses into the sandbox (see :func:`build_candidate`).
    image_preprocess: dict | None = None
    image_fixture_png: bytes | None = None
    # THE TOWER RECORDING, same gate and same rationale as the pair above.
    # ``vision_tower_units`` is ``{"meta": <the json payload>, "arrays": {key:
    # array}}`` -- the shape :func:`_load_reference_units` already established for
    # ``reference_units``, so the two read alike. Its INPUT arrays cross into the
    # sandbox (``_vision_tower_unit_inputs``); its outputs and its geometry stay
    # here, where the comparison happens.
    vision_tower_units: dict | None = None

    @property
    def mode(self) -> str:
        return self.meta["mode"]

    @property
    def perf_k_values(self) -> list[int]:
        """The perf sweep's K values (`instance.yaml::perf_k_values`, mirrored
        into `meta.json`). `.get`, not `[...]`: unlike `mode`, this
        key is not required on every golden today -- an instance whose
        goldens have not yet been re-frozen for the perf stage carries none,
        and `[]` here reads the same way `Candidate.perf_k_values`'s
        empty-list default does, rather than raising for every caller.

        NO `perf_page_ids` COUNTERPART, DELIBERATELY. The sweep's largest K
        might seem to need its own frozen page assignment the way the graded
        run's `page_ids` does, but `perf_probe` drives the candidate's own
        allocator (`USE_FROZEN_ALLOCATION["perf"] = False`), so nothing
        anywhere reads a frozen perf page list, and a `Golden` accessor for a
        key nothing produces would be exactly the same defect one layer up.
        """
        return [int(k) for k in (self.meta.get("perf_k_values") or [])]

    @property
    def thresholds(self) -> compare.NumericThresholds:
        """The PROFILE band (``precision_profile`` -> ``thresholds_for_profile``).

        NO GRADED CHECK READS THIS, and it should stay that way. It is the
        class-wide band before any per-instance calibration, so grading against it
        means grading at a tolerance nothing measured -- which on the native tier
        is a 10x loosening (``cosine_floor`` 0.98 against the calibrated centered
        floor of 0.998) and on ``argmax_margin`` is the value the
        reference-as-candidate FAILS. Its one live consumer is
        :attr:`native_thresholds`, which uses it as the base the overrides are
        applied to.

        **Grade against** :attr:`native_thresholds` **or** :attr:`fp32_thresholds`.
        Kept public because the profile is a real, separately meaningful quantity
        (a failure message that shows both tells a reader how far the instance
        departed from the class), not because it is a band to compare at.
        """
        return compare.NumericThresholds(**_threshold_fields(self.meta["thresholds"]))

    @property
    def fp32_thresholds(self) -> compare.NumericThresholds:
        """Tight fp32 exactness gate; per-instance via
        meta, MERGED over the shared constant -- symmetric with
        :attr:`native_thresholds`, not a replacement of it.

        ``NumericThresholds`` gives a dataclass default to ``argmax_margin``
        only; ``abs_tol``, ``rel_tol``, ``cosine_floor`` and ``max_kl`` are all
        required. Constructing straight from ``meta["fp32_thresholds"]`` alone
        would let a PARTIAL band -- missing any of those four -- raise
        ``TypeError`` from the constructor, inside every numeric and behavioral
        check that reads this property: reward 0, charged to the candidate, for
        an instance-authoring omission that has nothing to do with the port.
        ``meta.json`` always carries a whole band because the freezer
        merges ``instance.yaml``'s ``fp32_thresholds`` over
        ``compare.FP32_EXACT_THRESHOLDS`` at freeze time. The merge belongs here
        too, in the reader, so any other producer of ``meta.json`` (or a
        hand-edited one) gets the same floor instead of a crash.

        Keys that are NOT ``NumericThresholds`` fields are filtered out here and
        served by :attr:`fp32_extras` instead -- a calibrated scalar that belongs
        to one CHECK rather than to the comparator (``logprob_atol``)
        has nowhere else to live, and without the filter adding one would raise
        ``TypeError`` from the constructor for every check in the tier.
        """
        # `asdict`, not `__dict__`: the two agree for a plain dataclass instance,
        # but `__dict__` is an implementation detail (empty under `__slots__`,
        # and it does not see a field served by a descriptor), and this line's
        # whole claim is symmetry with `native_thresholds` below, which merges
        # two real dicts. The mechanism now matches the claim.
        base = _threshold_fields(asdict(compare.FP32_EXACT_THRESHOLDS))
        over = _threshold_fields(self.meta.get("fp32_thresholds"))
        return compare.NumericThresholds(**{**base, **over})

    @property
    def fp32_extras(self) -> dict:
        """``fp32_thresholds`` keys that are not ``NumericThresholds`` fields."""
        return _threshold_extras(self.meta.get("fp32_thresholds"))

    @property
    def native_thresholds(self) -> compare.NumericThresholds:
        """The PROFILE band, overridden per instance by ``meta['native_thresholds']``.

        Two layers, because they answer different questions.
        ``thresholds_for_profile('bf16_native')`` is a CLASS constant: what two
        independent bf16 kernel stacks may differ by, in general. A per-instance
        override is a MEASUREMENT: what this instance's reference-as-candidate
        actually differs by, recorded with its floor and its ratio in
        ``instance.yaml``. An instance that measures tighter than the profile
        should grade tighter, and an instance that legitimately does not is
        obliged to say so in one place a reader can find.

        Empty ``native_thresholds`` -> exactly :attr:`thresholds`, so an instance
        that has not been calibrated is graded at the profile and nothing silently
        loosens.
        """
        base = _threshold_fields(self.meta["thresholds"])
        over = _threshold_fields(self.meta.get("native_thresholds"))
        return compare.NumericThresholds(**{**base, **over})

    @property
    def native_extras(self) -> dict:
        """``native_thresholds`` keys that are not ``NumericThresholds`` fields.

        Where a NATIVE-tier scalar belonging to one check lives
        (``decisive_margin``, a two-sided bound on the greedy-sequence check that
        no comparator knows about).
        """
        return _threshold_extras(self.meta.get("native_thresholds"))

    @property
    def profile(self) -> str:
        return self.meta["precision_profile"]

    # The frozen allocation. Named accessors rather than raw dict
    # indexing so a scenario-schema change breaks in ONE place, not in every
    # test that reads it.
    @property
    def scenario_block_table(self) -> np.ndarray:
        return np.asarray(self.inputs["block_tables"])

    @property
    def scenario_slot_mapping(self) -> np.ndarray:
        return np.asarray(self.inputs["slot_mapping"])

    @property
    def scenario_page_ids(self) -> list[int]:
        return list(self.scenario["page_ids"])

    @property
    def decode_margins(self) -> np.ndarray:
        """Per-GENERATED-STEP top1-top2 logit gap, native tier. DERIVED, not frozen.

        ``test_greedy_sequence`` grades the FIRST divergent generated step against
        this, so the array has to be indexed the way ``generate`` indexes its
        output -- and the frozen decode artifact is NOT indexed that way.

        THE ALIGNMENT, because getting it wrong is a silent off-by-one in the
        headline check rather than an error. ``Engine.generate`` emits token ``0``
        from the LAST PREFILL ROW and token ``i>0`` from decode step ``i-1``; the
        final decode step's logits are discarded (they would predict a ninth token
        nothing emits). ``decode_step.npz``, by contrast, holds every decode step
        including that last one, so its step ``i`` predicts token ``i+1``. Reading
        it positionally therefore compares generated step ``i`` against the margin
        of step ``i+1``.

        On ``core_llama3`` the two lists are visibly a shift of each
        other::

            decode_step.npz, positionally  5.125 5.375 1.625 0.5 2.125 5.375 0.5 4.625
            generation-aligned (this)      0.375 5.125 5.375 1.625 0.5 2.125 5.375 0.5

        The trailing 4.625 has no generated counterpart and is dropped; the leading
        0.375 is the prefill row, which no decode artifact carries at all.

        AND "THE LAST PREFILL ROW" IS NOT ROW ``-1`` once the graded forward holds
        more than one request. At ``core`` the batch is one sequence, so row ``-1``
        IS sequence 0's last prompt row and the distinction is invisible. At
        ``ragged`` and ``chunked`` the forward is a mixed batch and row ``-1`` is the
        LAST REQUEST's single decode token -- a different sequence entirely, which
        emits no token in ``generation.npz`` at all.

        On ``ragged_llama3`` (``query_start_loc`` ``[0, 12, 13, 14]``),
        the generation golden settles which row is meant::

            row -1  (sequence 2)   argmax  5679   margin 4.0000
            row 11  (sequence 0)   argmax 10057   margin 0.3750
            generation.npz tokens[0] == 10057

        Row ``-1``'s argmax appears nowhere in the generated sequence. Taking it
        made ``decode_margins[0]`` 4.0000 instead of 0.3750, and since
        ``test_greedy_sequence`` excuses a divergence only when the golden's own
        margin is BELOW the decisive threshold, a candidate diverging at step 0 for
        a real bf16 tie was reported as "a real divergence, not a tie". Too strict,
        on the step most likely to tie, on every ragged and chunked instance.

        Only index 0 was ever affected: ``decode_step.npz``'s entries are
        single-row and already sequence 0's.

        NO NEW GOLDEN. Both halves already exist -- ``prefill_forward.npz::logits``
        and ``decode_step.npz`` -- and freezing a derived quantity would let it
        disagree with the artifacts it is derived from.
        """
        logits = np.asarray(self.prefill_forward["logits"])
        # Sequence 0's last row, which is row -1 exactly when there is one request.
        starts = self.inputs.get("query_start_loc")
        first_row = -1
        if starts is not None:
            starts = np.asarray(starts).reshape(-1)
            if starts.size >= 2:
                first_row = int(starts[1]) - 1
        rows = [logits[first_row]]
        steps = sorted((k for k in self.decode_step if k.startswith("logits_step_")),
                       key=lambda k: int(k.rsplit("_", 1)[1]))
        want = int(self.meta.get("num_decode_steps") or (len(steps) + 1))
        rows += [np.asarray(self.decode_step[k]) for k in steps[: want - 1]]
        margins = []
        for row in rows:
            top2 = np.partition(np.asarray(row, dtype=np.float64).reshape(-1), -2)[-2:]
            margins.append(float(top2[1] - top2[0]))
        return np.asarray(margins, dtype=np.float64)

    @property
    def smart_resize_table(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """The frozen ``smart_resize`` cases, ``[((h, w), (target_h, target_w))]``.

        A LIST OF PAIRS, not a dict keyed by the input, and the reason is that the
        recording is two parallel arrays (``smart_resize_input_hw`` /
        ``smart_resize_output_hw``) whose ORDER is part of what it says. A dict
        would silently collapse a duplicated input to one entry, which is exactly
        the freeze defect ``test_smart_resize_targets``' coverage clause is there
        to see; a list cannot.

        Empty when the instance froze no table -- the check that reads it asserts
        it is not, so an empty table is a loud artifact failure rather than a loop
        that runs zero times and reports a pass.
        """
        frozen = self.image_preprocess or {}
        inputs = frozen.get("smart_resize_input_hw")
        outputs = frozen.get("smart_resize_output_hw")
        if inputs is None or outputs is None:
            return []
        pairs = zip(np.asarray(inputs).reshape(-1, 2).tolist(),
                    np.asarray(outputs).reshape(-1, 2).tolist())
        return [((int(h), int(w)), (int(th), int(tw))) for (h, w), (th, tw) in pairs]

    @property
    def mrope_positions(self) -> np.ndarray | None:
        """The frozen ``positions`` the graded prefill actually consumed.

        ``inputs.npz::positions``, which is rank 1 on every text-only family and
        ``(rope_sections, num_tokens)`` on one with M-RoPE. Named rather than read
        inline so ``test_mrope_positions`` states WHICH array it grades, and so
        the "there is no positions array at all" case is a ``None`` a check can
        report as an artifact defect instead of a ``KeyError``.

        IT IS ALSO AN INPUT TO THE GRADED FORWARD, and that is stated here rather
        than left for a reader to notice: ``build_candidate`` copies every
        ``inputs.npz`` key into the sandbox, so this exact array is in the
        candidate's payload. The check that reads it therefore grades whether the
        port's own position BUILDER agrees with the reference -- a localiser, the
        same standing ``test_kv_cache_write``'s occupancy clause has against a
        ``slot_mapping`` the probe is likewise handed -- and is not an anti-cheat
        gate. Nothing in this class claims to be one; see ``build_candidate``'s
        own note on what the worker's scratch cwd does and does not buy.
        """
        value = self.inputs.get("positions")
        return None if value is None else np.asarray(value)

    @property
    def graded_window(self) -> tuple[int, int, int, int]:
        """Sequence 0's window, on BOTH sides: ``(lo, hi, own_lo, own_hi)``.

        ``[lo, hi)`` indexes the GRADED FORWARD's own token axis -- the rows of
        ``prefill_forward.npz::logits`` and the columns of ``inputs.npz::
        positions`` that belong to sequence 0. ``[own_lo, own_hi)`` indexes
        SEQUENCE 0'S OWN RUN, which is what a single-request probe forwards, and
        the two ranges have the same length by construction.

        WHY BOTH, AND WHY DERIVED RATHER THAN FROZEN. A mixed-stage batch
        concatenates several requests' query tokens, so on a `ragged` or
        `chunked` instance sequence 0 owns a PREFIX of the graded forward and not
        the whole of it; and a sequence whose stage is `decode` contributes one
        query token that sits at the END of its own run, not at the start. A
        probe that drives ``Engine.prefill`` -- single-request by contract -- can
        only produce sequence 0's whole run, so the comparison is a window on
        each side. Both numbers follow from arrays already frozen
        (``query_start_loc``, ``seq_lens``), and deriving them keeps this from
        becoming a third artifact that can disagree with the two it summarises --
        the same argument :meth:`decode_margins` makes.

        ON `core` IT IS THE IDENTITY. One request, ``query_start_loc == [0, n]``,
        ``seq_lens == [n]``: ``(0, n, 0, n)``, so every caller slices the whole
        array and no `core` comparison changes shape or content.

        A golden carrying neither key falls back to the full ``input_ids``
        length, which is the only thing a single-forward scenario can mean.
        """
        ids = np.asarray(self.inputs.get("input_ids", [])).reshape(-1)
        total = int(ids.shape[0])
        starts = self.inputs.get("query_start_loc")
        lo, hi = 0, total
        if starts is not None:
            starts = np.asarray(starts).reshape(-1)
            if starts.size >= 2:
                lo, hi = int(starts[0]), int(starts[1])
        # `seq_lens[0]` is sequence 0's FULL length -- the history the graded
        # step attends over, which for a prefill is its query and for a decode is
        # its query plus the prefix a warm-up forward already wrote. Its own
        # window therefore ENDS at that length and starts however many tokens
        # back its query is long. `_build_batch`'s `start = n - q_len`, read back.
        seq_lens = self.inputs.get("seq_lens")
        own_hi = hi - lo
        if seq_lens is not None:
            seq_lens = np.asarray(seq_lens).reshape(-1)
            if seq_lens.size >= 1:
                own_hi = int(seq_lens[0])
        return lo, hi, own_hi - (hi - lo), own_hi

    def hidden_states(self) -> list:
        """Ordered per-layer hidden states from ``prefill_forward.npz``.

        Empty when the instance froze none. Index 0 is the embedding output and
        the final index is the post-final-norm state (HF / ref_jax_repo
        convention: length ``num_hidden_layers + 1``).
        """
        keys = sorted((k for k in self.prefill_forward if k.startswith("hidden_")),
                      key=lambda k: int(k.split("_", 1)[1]))
        return [self.prefill_forward[k] for k in keys]


def _load_sampler(golden_dir: Path) -> dict:
    """``sampler.json`` + ``sampler.npz`` as the single dict the probes expect.

    The sampler golden is TWO files because its arrays are five full vocab-width
    rows: as JSON they were 11.6 MB of decimal text -- a third of the golden tree --
    with 256 431 ``-Infinity`` tokens, which is a Python extension to JSON that
    non-Python readers reject. The json half keeps ``grid_params`` and
    ``logprobs_top_n``; the npz half keeps ``logits``, ``grid_{i}``,
    ``topn_logprobs`` and ``topn_token_ids``.

    They are rejoined HERE, once, so the split is invisible to every consumer:
    ``Golden.sampler`` has exactly the keys the candidate's ``sampler_probe``
    returns, ``grid`` included, keyed by the same string indices.
    """
    sampler = dict(_load_json(golden_dir / "sampler.json"))
    arrays = _load_npz(golden_dir / "sampler.npz")
    sampler["grid"] = {name.split("_", 1)[1]: value
                       for name, value in arrays.items()
                       if name.startswith("grid_")}
    for name in ("logits", "topn_logprobs", "topn_token_ids"):
        if name in arrays:
            sampler[name] = arrays[name]
    return sampler


def _load_reference_units(golden_dir: Path) -> dict | None:
    """``reference_units.{json,npz}`` as ``{"meta": ..., "arrays": ...}``, or None.

    Both halves are required together -- the json names the submodules and the
    npz holds their I/O, so half a pair is a broken freeze rather than a partial
    feature.
    """
    js, npz = golden_dir / "reference_units.json", golden_dir / "reference_units.npz"
    if not (js.exists() and npz.exists()):
        return None
    return {"meta": _load_json(js), "arrays": _load_npz(npz)}


def _load_vision_tower_units(golden_dir: Path) -> dict | None:
    """``vision_tower_units.{json,npz}`` as ``{"meta": ..., "arrays": ...}``.

    The same both-halves-or-neither rule and the same shape as
    :func:`_load_reference_units` -- deliberately, so the two tower/decoder unit
    tiers read alike in ``test_numeric.py`` -- over the vision-tower producer's
    pair.

    ABSENT IS NOT AN ERROR HERE, and unlike ``reference_units`` it is not a
    deselection either: the check that reads this is gated on the ``multimodal``
    FAMILY fact, so a text-only instance never looks and a multimodal one whose
    freeze is incomplete fails by name. See ``conftest._APPLICABLE`` for why the
    two optional-artifact tiers are gated differently.
    """
    js = golden_dir / "vision_tower_units.json"
    npz = golden_dir / "vision_tower_units.npz"
    if not (js.exists() and npz.exists()):
        return None
    return {"meta": _load_json(js), "arrays": _load_npz(npz)}


def _load_battery(golden_dir: Path) -> dict | None:
    """The L3 battery: ``{axis: {variant: {stem: data}}}``, or None if absent.

    Layout is ``golden/battery/{axis}/{variant}/*`` -- an AXIS level
    the training class does not have, because inference generalises along four
    independent axes (prompt length, batch composition, paging, chunk boundary)
    rather than one. Each variant carries ``inputs.npz`` plus whatever forward
    artifacts its axis freezes.
    """
    root = golden_dir / "battery"
    if not root.is_dir():
        return None
    battery: dict = {}
    for axis_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        variants: dict = {}
        for variant_dir in sorted(p for p in axis_dir.iterdir() if p.is_dir()):
            if not (variant_dir / "inputs.npz").exists():
                continue
            entry: dict = {}
            for path in sorted(variant_dir.iterdir()):
                if path.suffix == ".npz":
                    entry[path.stem] = _load_npz(path)
                elif path.suffix == ".json":
                    entry[path.stem] = _load_json(path)
            variants[variant_dir.name] = entry
        if variants:
            battery[axis_dir.name] = variants
    return battery or None


#: The two multimodal artifacts, by filename. Named here so the loader, the
#: manifest's ``OPTIONAL_GOLDENS`` and the image-preprocess producer all
#: spell them the same way.
_IMAGE_PREPROCESS_NPZ = "image_preprocess.npz"
_IMAGE_FIXTURE_PNG = "image_fixture.png"


def _load_image_preprocess(golden_dir: Path) -> tuple[dict | None, bytes | None]:
    """``(arrays, png bytes)`` for the multimodal recording, or ``(None, None)``.

    BOTH HALVES OR NEITHER, the same rule :func:`_load_reference_units` applies to
    its json/npz pair and for a sharper reason here: the npz records
    ``source_png_sha256``, so half a pair is a recording that cannot be bound to
    the image it describes. A tree carrying one and not the other loads as absent
    and the check that reads it says so by name.

    ABSENT IS NOT AN ERROR HERE. Six of the seven instances are text-only and have
    no image; the applicability gate is the ``multimodal`` FAMILY FACT rather than
    the presence of these files, so an instance that declares itself multimodal
    and is missing them fails loudly in ``test_image_preprocess`` -- with the
    filename -- instead of being quietly excused by its own incomplete freeze.
    """
    npz, png = golden_dir / _IMAGE_PREPROCESS_NPZ, golden_dir / _IMAGE_FIXTURE_PNG
    if not (npz.exists() and png.exists()):
        return None, None
    return _load_npz(npz), _guard(png).read_bytes()


def load_golden(golden_dir: str | Path) -> Golden:
    d = Path(golden_dir)
    image_arrays, image_png = _load_image_preprocess(d)
    return Golden(
        dir=d,
        meta=_load_json(d / "meta.json"),
        scenario=_load_json(d / "scenario.json"),
        inputs=_load_npz(d / "inputs.npz"),
        cache_spec=_load_json(d / "cache_spec.json"),
        prefill_forward=_load_npz(d / "prefill_forward.npz"),
        fp32_prefill_forward=_load_npz(d / "fp32_prefill_forward.npz"),
        kv_cache_state=_load_npz(d / "kv_cache_state.npz"),
        decode_step=_load_npz(d / "decode_step.npz"),
        paged_attention_units=_load_npz(d / "paged_attention_units.npz"),
        sampler=_load_sampler(d),
        generation=_load_npz(d / "generation.npz"),
        fp32_generation=_load_npz(d / "fp32_generation.npz"),
        params_spec=_load_json(d / "params_spec.json"),
        param_norms=_load_json(d / "param_norms.json"),
        interface=_load_json(d / "interface.json"),
        # UNCONDITIONAL, like every other required artifact above and unlike the
        # `held_out_nll.json` it replaces: an absent L6 golden raises here, naming
        # the file, instead of arriving as a `None` that deselects a graded check.
        held_out=_load_npz(d / "held_out.npz"),
        reference_units=_load_reference_units(d),
        battery=_load_battery(d),
        image_preprocess=image_arrays,
        image_fixture_png=image_png,
        vision_tower_units=_load_vision_tower_units(d),
    )


#: Battery-variant ``inputs.npz`` keys that cross into the sandbox as ARRAYS.
#:
#: An ALLOW-LIST, in the same spirit as :data:`_SCENARIO_INPUT_KEYS` one level
#: up: the battery's whole point is that its variants are re-forwarded against a
#: frozen answer, so a key added to a variant npz must be classified as an input
#: deliberately rather than swept across by a ``dict(inputs)``.
#:
#: The metadata group (everything after ``positions``) is what a MIXED-STAGE
#: variant needs and a single-sequence one does not carry: `batch_composition`
#: varies the request distribution, which cannot be expressed by ``input_ids``
#: alone. Passing them is the same "handed, not derived" discipline the
#: frozen-allocation policy states for the canonical scenario's allocation,
#: applied per variant.
#:
#: THE TWO IMAGE PAIRS ARE PER-FORWARD, and their prefixes say which forward.
#: A variant re-stages the batch, so the sequence that owns the instance's image
#: can be a `decode` here and a `prefill` in the canonical scenario -- and its
#: ``<|image_pad|>`` rows then live in ``warmup_input_ids`` rather than in
#: ``input_ids``. Both spellings are allowed through because both are things a
#: forward was HANDED; which one a variant carries is the freezer's statement of
#: the attribution, and `_handed_forward` reads it back rather than assuming.
_BATTERY_INPUT_KEYS: tuple[str, ...] = (
    "positions", "seq_lens", "query_start_loc", "request_distribution",
    "block_tables", "slot_mapping",
    "warmup_input_ids", "warmup_positions", "warmup_seq_lens",
    "warmup_query_start_loc", "warmup_request_distribution",
    "warmup_slot_mapping",
    "pixel_values", "image_grid_thw",
    "warmup_pixel_values", "warmup_image_grid_thw",
)


def _battery_inputs(golden: Golden) -> dict | None:
    """``{axis: {variant: {input_ids, page_ids, ...}}}`` for the candidate.

    Only the INPUT side crosses into the sandbox; the frozen outputs stay here
    and are compared in the parent. ``page_ids`` rides along because the paging
    axis varies exactly that, and it is part of the variant's frozen allocation;
    :data:`_BATTERY_INPUT_KEYS` rides along because a mixed-stage variant's
    composition IS its metadata.
    """
    if not golden.battery:
        return None
    out: dict = {}
    for axis, variants in golden.battery.items():
        axis_out = {}
        for name, entry in variants.items():
            inputs = entry.get("inputs") or {}
            if "input_ids" not in inputs:
                continue
            variant = {"input_ids": np.asarray(inputs["input_ids"])}
            if "page_ids" in inputs:
                variant["page_ids"] = [int(p) for p in np.asarray(inputs["page_ids"]).reshape(-1)]
            for key in _BATTERY_INPUT_KEYS:
                if key in inputs:
                    variant[key] = np.asarray(inputs[key])
            axis_out[name] = variant
        if axis_out:
            out[axis] = axis_out
    return out or None


def _reference_unit_inputs(golden: Golden) -> dict | None:
    """The frozen per-submodule INPUTS, for the candidate to replay (L1).

    Only the INPUT side crosses into the sandbox -- the same split
    :func:`_battery_inputs` makes, and for the same reason: the frozen OUTPUTS
    stay in the parent, where the comparison happens, so the answer is not part of
    what the candidate is HANDED. It is not out of its REACH -- see
    :func:`build_candidate` for the measurement and for what the worker's scratch
    cwd does and does not buy.

    A unit whose json entry names an array the npz does not carry is DROPPED
    rather than passed half-built: the probe would otherwise call a module with a
    missing argument and report the resulting ``TypeError`` as a candidate fault.
    ``test_reference_units`` asserts the unit set is complete against the json, so
    a dropped unit is a loud failure about the golden rather than a silent one.
    """
    if not golden.reference_units:
        return None
    meta = golden.reference_units.get("meta") or {}
    arrays = golden.reference_units.get("arrays") or {}
    units: dict = {}
    for name in (meta.get("modeling_units") or []):
        spec = (meta.get("units") or {}).get(name)
        if not spec:
            continue
        keys = [item["key"] for item in (spec.get("inputs") or [])]
        if not keys or any(key not in arrays for key in keys):
            continue
        units[name] = {
            "torch_path": spec["torch_path"],
            "call": spec.get("call", "__call__"),
            "inputs": [np.asarray(arrays[key]) for key in keys],
        }
    if not units:
        return None
    return {"layer_index": meta.get("layer_index"), "dtype": meta.get("dtype"),
            "units": units}


def _image_inputs(golden: Golden) -> dict | None:
    """``{"png_bytes": ..., "smart_resize_cases": [[h, w], ...]}``, or None.

    The INPUT half of ``image_preprocess.npz`` plus the tracked PNG, i.e. exactly
    what a port needs to reproduce the recording and nothing it could answer from.
    Same split :func:`_battery_inputs` and :func:`_reference_unit_inputs` make.

    ``png_bytes`` IS THE ENCODED FILE, not the recorded ``source_image_rgb``. The
    recording carries both, deliberately: the sha256 binds the
    recording to the bytes a human reviews in the diff, and the decoded array is
    the cross-check that splits "your decoder differs from Pillow's" from "your
    processor differs from the reference's". Only the encoded form crosses,
    because the decode is part of what is graded.

    ``smart_resize_cases`` is a list of plain ``[h, w]`` int pairs rather than an
    array: it is the argument list for a pure-integer function, it crosses a
    pickle, and an ``np.int64`` arriving where a port expects an ``int`` is a
    difference this boundary has no reason to introduce.
    """
    frozen = golden.image_preprocess
    if not frozen or golden.image_fixture_png is None:
        return None
    cases = np.asarray(frozen.get("smart_resize_input_hw", [])).reshape(-1, 2)
    return {
        "png_bytes": bytes(golden.image_fixture_png),
        "smart_resize_cases": [[int(h), int(w)] for h, w in cases.tolist()],
    }


def _vision_tower_unit_inputs(golden: Golden) -> dict | None:
    """The frozen per-TOWER-submodule INPUTS, for the candidate to replay.

    :func:`_reference_unit_inputs` one tier over, with the same input/output
    split and the same drop-a-half-built-unit rule, plus one thing that tier does
    not have: the ``grid_thw`` the geometry method is called on. That is an input
    -- it is the SHAPE of the image, which the processor hands out beside
    ``pixel_values`` -- and it is what makes ``encoder_metadata`` reproducible on
    the candidate's side at all.

    THE GEOMETRY'S OWN ARRAYS DO NOT CROSS. ``window_index``,
    ``cu_window_seqlens`` and the two rope tables are the frozen ANSWER
    ``test_vision_tower_units`` compares against; the ``cu_seqlens`` vectors that
    DO cross are the ones recorded as positional INPUTS to the two block units,
    which is the same "handed, not derived" discipline that lets a wrong window
    permutation still be localised to the block rather than contaminating it.
    """
    if not golden.vision_tower_units:
        return None
    meta = golden.vision_tower_units.get("meta") or {}
    arrays = golden.vision_tower_units.get("arrays") or {}
    units: dict = {}
    for name in (meta.get("tower_units") or []):
        spec = (meta.get("units") or {}).get(name)
        if not spec:
            continue
        keys = [item["key"] for item in (spec.get("inputs") or [])]
        if not keys or any(key not in arrays for key in keys):
            continue
        units[name] = {
            "torch_path": spec["torch_path"],
            "call": spec.get("call", "__call__"),
            "inputs": [np.asarray(arrays[key]) for key in keys],
        }
    if not units:
        return None
    grid = ((meta.get("fixture") or {}).get("grid_thw")
            or [[int(v) for v in row]
                for row in np.asarray(arrays.get("tower.in1", [])).reshape(-1, 3)])
    return {"dtype": meta.get("dtype"), "units": units,
            "geometry_method": meta.get("geometry_method"),
            "grid_thw": [[int(v) for v in row] for row in grid]}


#: THE PAYLOAD IS INPUTS ONLY, AND THIS PAIR IS WHERE THAT IS MADE TRUE RATHER THAN
#: CLAIMED. ``scenario.json::decode_token_ids`` is the reference's continuation --
#: byte-identical to ``fp32_generation.npz::tokens``, which ``test_greedy_sequence_fp32``
#: grades with ``np.array_equal`` and no tolerance. Shipping it under that name put the
#: graded answer in the sandbox four lines below a docstring saying nothing of the sort
#: crossed.
#:
#: The numeric decode probe still NEEDS those ids: it is teacher-forced on the
#: reference's continuation so that a step-0 divergence does not contaminate steps
#: 1..N (``test_numeric.test_decode_step_logits``). So they cross under a name that says
#: what they are FOR, and every other key is allow-listed rather than copied wholesale
#: -- a scenario that grows an output-shaped field in a future freeze must be added here
#: deliberately.
#:
#: MODULE LEVEL, not locals of :func:`build_candidate`, because the partition is
#: asserted from outside: ``_harness_checks.check_scenario_keys_are_classified`` reads
#: both tuples and refuses to grade a golden carrying a key in neither. See that check
#: for why the assertion moved out of this function.
_SCENARIO_INPUT_KEYS = (
    "mode", "prompt_len", "prompt_token_ids", "num_decode_steps",
    "page_size", "num_pages", "max_num_seqs", "pages_per_seq", "page_ids",
    "requests", "stage_schedule", "chunk_split_points", "arrays",
)

#: The frozen OUTPUT keys in ``scenario.json``: reference answers that must NOT cross
#: into the sandbox under any name. ``decode_token_ids`` crosses only renamed, as the
#: teacher-forcing INPUT above, and only because its value is unavoidably the
#: reference's own greedy continuation (see above).
_SCENARIO_OUTPUT_KEYS = ("decode_token_ids",)


def build_candidate(test_dir: str | Path, golden: Golden, *,
                     allow_reference_default: bool = False) -> Candidate:
    """Construct the candidate driver against the repo under test.

    Reads the candidate repo from ``T2J_JAX_REPO``. Under Harbor the verifier's
    ``test.sh`` sets it to the agent's produced repo (``/workspace/jax_repo``).
    Unset, it falls back to the committed ``../solution/ref_jax_repo`` only if
    the CALLER opts in -- either by passing ``allow_reference_default=True``
    (for a caller, such as the driver's own unit-test suite, whose entire
    purpose is validating the reference and never grades a port) or by the
    environment carrying ``T2J_GRADE_THE_REFERENCE=1`` (for an author running
    a bare ``pytest`` who has to type that name deliberately). Unset with
    neither, it raises rather than grading the reference silently.

    The keyword exists SPECIFICALLY so that opt-in cannot leak: a caller that
    needs the reference states so in its own call, not by mutating
    ``os.environ`` at import time, which would silently disable this guard for
    every other caller sharing the process.

    No weights are handed over: the candidate's own
    ``model/loader.py::convert_checkpoint`` reads the HF checkpoint that
    ``repo_config["model_name_or_path"]`` names, which every loader resolves to
    ``/data/{repo_id}`` -- baked into the verifier image, identical to the
    agent's.

    The ``scenario`` payload is the frozen scenario plus the frozen arrays every
    allocation-dependent probe is HANDED: ``inputs.npz`` under
    ``inputs``, the isolated attention op's own ``(q, k, v)`` under ``units``,
    the sampler's frozen input logits, the battery's per-variant inputs, and the
    per-submodule input activations under ``reference_units``.

    EVERY ONE OF THOSE IS AN INPUT, with ONE documented exception:
    ``scenario["decode_input_token_ids"]`` carries the reference's continuation.
    Against ``generation.npz::tokens`` (native) the match is STRUCTURAL: the
    freezer's ``step_generation`` reuses the exact ``generate()`` result object
    ``step_scenario`` already derived these ids from, so they cannot differ.
    Against ``fp32_generation.npz::tokens`` -- the array ``test_greedy_sequence_fp32``
    grades with tolerance-free ``np.array_equal`` -- the match is EMPIRICAL FOR
    THIS GOLDEN ONLY: the freezer runs a separate fp32 ``generate()`` call on an
    upcast model with its own KV cache, nothing in the freezer asserts token
    agreement between the two, and they merely happen to agree on all 8 tokens
    here; another checkpoint could diverge. See the allow-list comment below for
    why this is bounded and named rather than closed. The graded
    ``step_logprobs`` do NOT cross, so ``test_decode_trajectory`` cannot be
    answered from this payload. Apart from that one key, no frozen OUTPUT crosses
    into the sandbox -- the comparison happens in this process, against artifacts
    that are not part of the payload the candidate is handed.

    THAT IS TRUE OF THE PAYLOAD BUT NOT OF THE PROCESS, and the distinction
    matters more than the claim. Keeping the frozen answers out of the payload
    does not stop a probe from reading them off disk: from the worker's inherited
    cwd, ``Path('.').rglob('golden/fp32_generation.npz')`` would return the frozen
    answer -- the graded token ids and the ``step_logprobs`` -- so a probe COULD
    return it. The worker is a plain ``spawn`` child of this process with the same
    uid and unrestricted file I/O; ``sandbox.BannedModuleFinder`` blocks five
    module roots and ``purity.scan_repo_for_torch`` reads source, and nothing
    looks at file I/O.

    WHAT IS TRUE NOW, exactly. ``_worker_entry`` starts the child in an empty
    scratch directory, so no relative path finds a golden, and it carries the probe
    result back in ``sandbox_report['relative_golden_hits']`` where
    ``test_spec.test_independence`` asserts it is empty. That raises the cost of
    cheating from "one relative glob" to "construct the absolute path". **It does
    not close it.** The goldens are still on the filesystem and still readable, and
    that is inherent to executing untrusted code in this process: real isolation
    needs a filesystem sandbox -- a bind mount the worker cannot name, or a uid
    without read on the tests tree -- which is a known limitation and is not claimed here.
    """
    test_dir = Path(test_dir).resolve()
    meta = golden.meta
    # A POSITIVE GUARD, NOT A DEFAULT. A silent default would fall back to
    # `../solution/ref_jax_repo` -- the answer key -- whenever T2J_JAX_REPO was
    # unset, and score reward 1.0 without saying so. Under Harbor this specific
    # path is not reachable: `.dockerignore` excludes `tasks/*/solution`
    # from the verifier image, and `build_harbor_images.sh --prove-guard` is a
    # real positive/negative control for that exclusion. What makes rejecting a
    # silent default the right call anyway is everything OUTSIDE that one
    # container: `tests/test.sh`'s own exit-4 guard cannot be the backstop for
    # those paths either, because harbor discards the verifier's exit code
    # entirely (test.sh's own comment records this), and a bare `pytest` would
    # take this fallback silently, with no container and no .dockerignore involved.
    #
    # Self-validating the reference is still the right default for an author, so it
    # is kept -- behind a name that has to be typed, OR behind a caller that says so
    # in its own call (`allow_reference_default=True`; see the docstring for why that
    # is a parameter and not another environment variable). The control tooling
    # sets T2J_JAX_REPO already and is unaffected. The GRADED `cand` fixture
    # passes neither and must not: an unset T2J_JAX_REPO there means the
    # verifier resolved no candidate, and that is exactly the case this guard exists
    # to catch.
    repo_dir = os.environ.get("T2J_JAX_REPO")
    if not repo_dir:
        if not (allow_reference_default or
                os.environ.get("T2J_GRADE_THE_REFERENCE") == "1"):
            raise RuntimeError(
                "T2J_JAX_REPO is unset, so there is no candidate to grade. "
                "Defaulting to solution/ref_jax_repo -- the answer key -- would "
                "score 1.0. Set T2J_JAX_REPO=/path/to/jax_repo to grade a port, or "
                "T2J_GRADE_THE_REFERENCE=1 to deliberately self-validate the "
                "reference.")
        repo_dir = str((test_dir / ".." / "solution" / "ref_jax_repo").resolve())

    # THE ALLOW-LIST IS THE STATED BOUND ON "NO FROZEN OUTPUT CROSSES", AND THE
    # PARTITION IS ASSERTED -- BUT NOT HERE. A key the freezer grows and nobody
    # classified is a TASK-authoring defect, and this function is reached through
    # the SESSION-SCOPED `cand` fixture: raising here errors all 29 checks that
    # request it (only `test_interface` and `test_repo_structure` are static), for
    # reward 2/31 = 0.0645 -- the score a candidate that does not exist gets. That
    # is the same misrouting the T2J_JAX_REPO guard above was written to avoid, with
    # the sign flipped, so the assertion lives in
    # `_harness_checks.check_scenario_keys_are_classified`: fatal, before pytest,
    # NO reward.json, and a message that names the freeze rather than the port.
    #
    # WHAT REMAINS HERE IS THE COPY, AND ITS DIRECTION IS THE SAFE ONE. An
    # unclassified key is DROPPED, so it cannot carry a frozen answer into the
    # sandbox; what is lost without the precheck is loudness about an INPUT that
    # never arrives, which is precisely what that check restores.
    scenario = {k: golden.scenario[k] for k in _SCENARIO_INPUT_KEYS
                if k in golden.scenario}
    # `.get`, not `[...]`: an instance whose golden carries no decode run must
    # reach `decode_steps_probe`'s documented "reports unsupported" branch
    # (`_t2j/candidate.py`) instead of raising a `KeyError` out here, before the
    # worker is even spawned -- a hard index made that branch, and the docstring
    # describing it, unreachable.
    _decode_ids = golden.scenario.get("decode_token_ids")
    if _decode_ids is not None:
        scenario["decode_input_token_ids"] = list(_decode_ids)
    scenario["inputs"] = {k: np.asarray(v) for k, v in golden.inputs.items()}
    # INPUTS ONLY -- never `out_native` / `out_fp32`, which are what the check
    # compares against and stay in the parent. `sinks` and `sliding_window` are
    # inputs too: they are the keywords the REFERENCE's own layer passed to
    # `paged_attention`, frozen by the paged-attention producer so the
    # isolated probe runs the op the model runs. A family whose layer passes
    # neither has neither key here and the probe calls the two-keyword form, so
    # core_llama3 is unaffected.
    units = {k: np.asarray(v) for k, v in golden.paged_attention_units.items()
             if k in ("q", "k", "v", "sinks", "sliding_window")}
    if units:
        scenario["units"] = units
    if golden.sampler.get("logits") is not None:
        scenario["sampler_logits"] = np.asarray(golden.sampler["logits"], dtype=np.float32)
    battery = _battery_inputs(golden)
    if battery:
        scenario["battery"] = battery
    reference_units = _reference_unit_inputs(golden)
    if reference_units:
        scenario["reference_units"] = reference_units
    # THE IMAGE, AND ONLY ITS INPUT SIDE. The ENCODED PNG crosses -- not the
    # decoded `source_image_rgb`, and certainly not `pixel_values` -- because the
    # decode is part of what is graded: `PIL` is on the structure gate's import
    # allowlist for both reference trees precisely so `serve.py` turns an image
    # into pixels inside the port, the way it turns a raw prompt string into ids.
    # Handing over pre-decoded pixels would make "this family processes an image"
    # a property of this file.
    #
    # `smart_resize_cases` is likewise the INPUT column of the frozen table. The
    # OUTPUT column stays in this process, where the comparison happens.
    image = _image_inputs(golden)
    if image:
        scenario["image"] = image
    # THE TOWER'S RECORDED INPUTS, on the 140x140 fixture. Under its own key
    # rather than merged into `image` above, because the two describe DIFFERENT
    # IMAGES on purpose: `image` is the graded prompt's 56x56 input, this is the
    # off-prompt geometry that makes window attention observable at all. Merging
    # them would put two `pixel_values` under one name.
    tower_units = _vision_tower_unit_inputs(golden)
    if tower_units:
        scenario["vision_tower_units"] = tower_units

    return Candidate(
        str(Path(repo_dir).resolve()),
        scenario=scenario,
        repo_config=dict(meta["repo_config"]),
        mode=str(meta["mode"]),
        seed=int(meta["seed"]),
        sampler_grid=list(meta.get("sampler_grid") or []),
        logprobs_top_n=int(meta.get("logprobs_top_n", 5)),
        num_decode_steps=int(meta.get("num_decode_steps") or 0) or None,
        compare_hidden_states=bool(meta.get("compare_hidden_states", False)),
        # The dtype the fp32 GENERATION LEG must allocate its KV cache at, read
        # from the golden that leg is graded against. KeyError-loud: an fp32
        # artifact whose producer did not record this cannot be graded, because
        # the wrong choice costs 180x with every token still agreeing.
        fp32_kv_cache_dtype=str(meta["fp32_kv_cache_dtype"]),
        # THE L6 WORKLOAD, INPUTS ONLY. `Candidate.__init__` selects the four
        # INPUT arrays out of this by name (`candidate.HELD_OUT_INPUT_KEYS`) and
        # `ref_logprobs` -- the frozen answer the check compares against -- is
        # left in this process. The selection is done at that boundary rather
        # than here so it cannot be bypassed by a second caller, and the whole
        # dict is handed over rather than pre-filtered so the boundary sees (and
        # can refuse) a golden that is missing one of the four.
        held_out=dict(golden.held_out),
        # Hard wall-clock budget for the candidate's own generate() loop
        # (instance.yaml -> meta). Sized to catch a port that never terminates,
        # not to micro-time.
        generate_timeout_s=float(meta.get("generate_timeout_s") or 900.0),
        # The perf stage's own geometry, read off `meta.json` --
        # never off `instance.yaml` directly, the same rule every other geometry
        # key on this call obeys (`check_meta_mirrors_instance_yaml` is the gate
        # that keeps the two from drifting). `perf_k_values` is the K sweep;
        # absent, `Candidate.__init__` defaults it to `[]` and `perf_probe`
        # reports `supported: False` instead of sweeping nothing. `perf_rounds`
        # is R -- a global instrument parameter, not per-instance
        # geometry, so it is not required in `instance.yaml` and falls back to
        # `Candidate.__init__`'s own default when no instance overrides it.
        perf_k_values=[int(k) for k in (meta.get("perf_k_values") or [])],
        perf_rounds=int(meta.get("perf_rounds") or DEFAULT_PERF_ROUNDS),
        # AUTHORING-HOST ESCAPE HATCH, unset in every graded run: a host with no
        # /data names an extracted checkpoint here and the driver rebinds
        # model.loader.DATA_ROOT. resolve_model_dir still hardcodes /data, and
        # cand.loader_facts reports which root actually applied.
        data_root=os.environ.get("T2J_INFERENCE_DATA_ROOT") or None,
    )
