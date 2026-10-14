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

"""Static, MODE-keyed expected-golden manifest.

Applicability deselection (conftest) is driven by golden-file presence: a check whose
golden the freeze never produced is deselected (kept out of numerator AND denominator).
That is correct for a golden a mode genuinely has no analogue for -- but it means a
freeze BUG that silently drops a required golden reads as "not applicable" instead of
"coverage regression", and the continuous pass rate never notices.

This manifest is the ground truth for what a CORRECT freeze must emit per mode. It is
static (no torch, no candidate run, no jax): pure stdlib, so it is importable both in
the torch-free verifier container and from the freezer on the authoring host.

Difference from the training class's manifest, and the reason it is not a copy: the
table is keyed by **mode** (``core`` / ``ragged`` / ``chunked``), not by training
method, and the modes **nest** -- ``core`` is a strict subset of ``ragged`` is a strict
subset of ``chunked``. That nesting is asserted by a test, because a table that quietly
stops nesting means a mode lost coverage. Inference has no backward pass, so none of
the training artifacts (``gradients.json``, ``loss_artifact.json``,
``train_trajectory.json``, ``lr_schedule.json``, ``grad_authenticity.json``) have an
analogue here; their absence is likewise asserted rather than assumed.

Token grammar (inherited, plus one extension):
  * ``prefill_forward.npz``            -- a filename
  * ``fp32_prefill_forward:logits``    -- ``<npz-stem>:<array>``
  * ``sampler:grid_params``            -- ``<json-stem>:<key>``
  * ``battery/batch_composition``      -- **new:** a token containing ``/`` is a
    DIRECTORY-presence check. Batteries freeze as ``battery/{axis}/{variant}/*``, so
    the required unit is the axis directory, not a file.

ONE REQUIREMENT IS NOT STATIC, AND IT IS THE CACHE STATE. ``kv_cache_state.npz``
holds ``cache`` for a container with ONE KV geometry and ``cache_group{i}`` for a
container with several (:func:`cache_state_member_names`), so its required token
set is a function of the family rather than of the mode. :func:`resolve_required`
reads that ONE fact off the directory's own ``cache_spec.json`` -- ``groups`` XOR
``shape``, which the freezer writes from ``len(engine.kv_cache)`` -- and
substitutes the matching tokens into the static table.

That is deliberately NOT the ``meta_dependent_tokens`` treatment
``decode_step:logits_step_{i}`` gets, and the difference is which gate sees it.
Those tokens are kept OUT of ``missing_required`` because they need a
``meta.json`` this module refuses to assume; the cache kind needs only a file
that already sits in the same directory the question is being asked about, so it
can stay inside ``missing_required`` -- and it must, because ``missing_required``
is the FREEZER's own integrity gate and the whole point is that the first freeze
of a heterogeneous family passes it.

It also makes the two records of the cache KIND agree at GRADE time, which
nothing checked before: ``cache_spec.json`` and ``kv_cache_state.npz`` each
declare it independently, so a golden whose spec says ``groups`` and whose state
says ``cache`` would grade one group of a heterogeneous cache against a contract
describing all of them, at full credit, with no key missing from either file. The
freezer refuses that pairing at freeze time (``_gate_cache_kind``); requiring the
tokens the SPEC implies makes the same disagreement a missing required golden
here. The requirement is read from the DECLARATION and checked against the DATA,
which is what keeps it from being circular -- deriving it from the npz's own
member names would require whatever happens to be there.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

# Present for every mode (17 files + 26 within-file tokens = 43).
# (`len(_COMMON)` and `coverage_vector(...)['required_total']` are the live
# numbers; check against those rather than this comment.)
#
# npz key names are PINNED here and read by name in the later test modules:
#   kv_cache_state.npz        -> "cache" for a container with ONE KV geometry, and
#                                "cache_group0" .. "cache_group{n-1}" for one with n.
#                                EXCLUSIVE, never both, and never a "cache" that
#                                means group 0 -- the member NAMES are the
#                                declaration, exactly as `shape` XOR `groups` is in
#                                cache_spec.json. See `cache_state_member_names`.
#   paged_attention_units.npz -> "out_native", "out_fp32" (+ inputs "q", "k", "v")
#   decode_step.npz           -> "logits_step_{i}"
#   generation.npz            -> "tokens" (int32 -- jax_enable_x64 is off class-wide,
#                                so int64 is a dtype the graded side cannot produce),
#                                "step_logprobs"
#   fp32_generation.npz       -> the same two arrays, from the same prompt and the
#                                same num_decode_steps, computed by the UPCAST model
#                                inside the freezer's single fp32 window
#   sampler.npz               -> "logits", "grid_{i}" (parallel to sampler.json's
#                                "grid_params"), "topn_logprobs", "topn_token_ids"
#   held_out.npz              -> "prompt_ids", "prompt_lens", "cont_ids",
#                                "cont_lens" (int32) and "ref_logprobs" (float32),
#                                five parallel rows per held-out entry, right-
#                                padded with zeros. The lengths are frozen
#                                ALONGSIDE the ids because 0 is a legal token id:
#                                a consumer slices by "*_lens" and must never read
#                                past them.
#
# THE SAMPLER IS TWO FILES, and the split is not cosmetic. Its arrays are five full
# vocab-width rows; as JSON they were 11.6 MB of decimal text -- a third of the whole
# golden tree -- and the two masked grid points serialized 256 431 `-Infinity`
# tokens, which is a PYTHON extension to JSON that `jq`, Go and Rust `serde_json` all
# reject. npz stores -inf exactly and costs ~2.6 MB. `sampler.json` keeps the scalar
# facts (`grid_params`, `logprobs_top_n`), so it stays a real requirement rather than
# a stub, and `fixtures.load_golden` reassembles the two into one dict.
_COMMON: frozenset[str] = frozenset({
    "meta.json",
    "scenario.json",
    "inputs.npz",
    "cache_spec.json",
    "prefill_forward.npz",
    "fp32_prefill_forward.npz",
    "kv_cache_state.npz",
    "decode_step.npz",
    "paged_attention_units.npz",
    "sampler.json",
    "sampler.npz",
    "generation.npz",
    "fp32_generation.npz",
    "interface.json",
    "params_spec.json",
    "param_norms.json",
    # REQUIRED, NOT OPTIONAL, and that is the one place this token differs from the
    # `held_out_nll.json` it replaces. L6 is always-on here (teacher-forced
    # scoring of the held-out set costs P prefills and ZERO decode steps, less
    # than the two generations the suite already runs in-grade), so a missing
    # artifact has to be a hard coverage FAILURE rather
    # than the quiet presence-deselect that left `test_held_out_nll` dark on every
    # instance in both classes for its whole life.
    "held_out.npz",
    # within-file: the fp32 prefill always carries logits. It gates the fp32-robust
    # parity tier, which applies to every mode.
    "fp32_prefill_forward:logits",
    # within-file: BOTH graded precisions of the isolated attention op. The FILE
    # token above cannot see this, and the gap is not hypothetical -- the npz also
    # carries the op's INPUTS (`q`, `k`, `v`), so a producer that wrote the inputs
    # and dropped an output leaves a file that exists, loads, and hands the
    # candidate everything it needs while `test_paged_attention_op` has nothing
    # left to compare against. Naming the two outputs is what turns that into a
    # coverage FAIL (`_harness_checks.check_golden_manifest`) rather than a check that quietly
    # grades one tier instead of two.
    "paged_attention_units:out_fp32",
    "paged_attention_units:out_native",
    # within-file: BOTH arrays of the fp32 generation, for the same reason the two
    # attention outputs are named. The fp32 equivalence claim is "same sampled
    # sequence AND same likelihood", so a producer that wrote `tokens` and dropped
    # `step_logprobs` leaves a file that exists, loads, and silently reduces that
    # claim to half of itself.
    "fp32_generation:tokens",
    "fp32_generation:step_logprobs",

    # ------------------------------------------------------------------ #
    # EVERY OTHER ARRAY A GRADED MODULE READS BY NAME.
    #
    # The five tokens above this block were the whole within-file catalog, and
    # `_has_token` degrades everything else to FILE presence (`if ":" not in
    # token: return (golden_dir / token).exists()`). `fixtures.load_golden` opens
    # each file eagerly, so a missing FILE is caught -- but an npz that loads with
    # an array missing from inside it is invisible here and raises `KeyError` in
    # whichever graded check reads it, which the reward reduction counts as an
    # `error` and charges to the candidate.
    #
    # If `tokens` were dropped from `generation.npz` or `ref_logprobs` from
    # `held_out.npz`, `missing_required` would return `[]`, `check_golden_manifest`
    # would PASS and all six `_harness_checks` would PASS -- yet
    # `golden.generation["tokens"]` (test_behavioral, the headline
    # native-equivalence check) and `golden.held_out["ref_logprobs"]` (L6) would
    # both raise `KeyError`.
    #
    # THE LIST IS DERIVED FROM THE READERS, not from the freezer's output: an AST
    # walk of `test_spec.py` / `test_numeric.py` / `test_behavioral.py` for hard
    # subscripts of a `golden.<artifact>` dict, plus the `fixtures.Golden`
    # accessors and `fixtures.build_candidate` reads that stand between the
    # artifact and those modules (a failure there errors every graded check, not
    # one). Declaring an array the freezer happens to write but nobody reads would
    # make this a freeze-completeness assertion wearing a coverage-gate costume,
    # which is the confusion the token grammar exists to avoid.
    #
    # inputs.npz -- the frozen scenario. `block_tables` and `slot_mapping` are read
    # through `Golden.scenario_block_table` / `.scenario_slot_mapping` (the
    # named accessors) and `input_ids` by `_CandidateRuntime.__init__`, which
    # raises `ValueError` naming the artifact when it is empty; the other three are
    # hard-subscripted in `test_spec.test_static_shapes_are_static` and
    # `test_numeric`'s ragged/battery helpers.
    "inputs:input_ids",
    "inputs:block_tables",
    "inputs:slot_mapping",
    "inputs:seq_lens",
    "inputs:query_start_loc",
    "inputs:request_distribution",
    # The NATIVE prefill logits: the reference side of `test_prefill_logits`, and
    # the first row of `Golden.decode_margins`, which `test_greedy_sequence` grades
    # its first divergent step against.
    "prefill_forward:logits",
    # The post-prefill cache tensor -- the entire content of `test_kv_cache_write`.
    # THE RECTANGULAR SPELLING, and the only one this static table can hold. A
    # container with more than one KV geometry freezes `cache_group{i}` instead,
    # and `resolve_required` swaps this token for those after reading the
    # directory's own `cache_spec.json`. Left here rather than removed because it
    # is what fourteen of fifteen instances require and because a table that named
    # no cache state at all would let a freeze drop the artifact entirely.
    "kv_cache_state:cache",
    # The isolated attention op's INPUTS. The outputs are named above; without
    # these three the file still loads and `test_paged_attention_op` has nothing to
    # feed the candidate with.
    "paged_attention_units:q",
    "paged_attention_units:k",
    "paged_attention_units:v",
    # The sampler pair. `logits` is the frozen row `build_candidate` hands the
    # sampler probe (absent it, the probe has no input and the transform checks
    # fail against a candidate that was never asked anything); `topn_*` are
    # `test_logprobs`' two reference arrays; `grid_params` is the json-side scalar
    # fact that says how many `grid_{i}` arrays should exist.
    "sampler:logits",
    "sampler:topn_logprobs",
    "sampler:topn_token_ids",
    "sampler:grid_params",
    # The native generated sequence -- `test_greedy_sequence`'s reference.
    "generation:tokens",
    # The L6 held-out set, all five rows. The four INPUT arrays cross into the
    # sandbox by name (`_t2j.candidate.HELD_OUT_INPUT_KEYS`); `ref_logprobs` is the
    # frozen answer `test_held_out_logprobs` compares against and deliberately does
    # NOT cross.
    "held_out:prompt_ids",
    "held_out:prompt_lens",
    "held_out:cont_ids",
    "held_out:cont_lens",
    "held_out:ref_logprobs",

    # DELIBERATELY NOT DECLARED, each checked against the source rather than
    # assumed, because "the freezer writes it" is not the question this table asks:
    #
    #   generation:step_logprobs  -- a plausible candidate for this list, and it
    #       is NOT read. `golden.generation` is subscripted exactly once, at
    #       `test_behavioral.py:886`, for `tokens`; the only `step_logprobs` the
    #       grader compares against a golden is `fp32_generation`'s
    #       (`test_decode_trajectory`), and the native equivalence claim is
    #       token-identity only -- there is no native likelihood check, because at
    #       bf16 the cross-process spread on a summed logprob is the size of the
    #       effect. The freezer writes the array; nothing grades it. Declaring it
    #       would assert freeze completeness, not coverage.
    #   inputs:positions -- handed to the sandbox inside `scenario["inputs"]` and
    #       never subscripted by name on either side; the candidate builds its own
    #       positions, which is part of what `test_static_shapes` grades.
    #   sampler:grid_{i}, decode_step:logits_step_{i} -- their NAMES depend on the
    #       instance's geometry, so they cannot be static tokens. Both already have
    #       an explicit, named presence assertion at their reader
    #       (`test_sampler_transforms` compares the array count to `grid_params`;
    #       `test_decode_step_logits` lists the absent steps), and the decode side
    #       additionally goes through `_meta_dependent_tokens` below, which puts it
    #       in front of the precheck where the rest of this table lives.
    #   the *_spec / param_norms / interface JSON payloads -- their keys are
    #       per-checkpoint parameter names, not a fixed vocabulary, and
    #       `_has_token`'s json branch reads a FALSY value as absent, so a
    #       legitimately-zero norm would report as a missing golden.
})

# Mode-keyed required goldens. Nested exactly as the modes nest, so a
# `chunked` task is a strict superset of a `ragged` one -- a test asserts that,
# because a table that quietly stops nesting means a mode lost coverage.
REQUIRED_GOLDENS: dict[str, frozenset[str]] = {
    "core": _COMMON,
    "ragged": _COMMON | frozenset({"battery/batch_composition"}),
    "chunked": _COMMON | frozenset({"battery/batch_composition",
                                    "battery/chunk_boundary"}),
}

# Presence-gated. `prompt_length` and `paging` apply to every mode but degrade to
# a visible coverage gap rather than a hard failure when not yet frozen; a mode's
# DEFINING axis is required for that mode instead.
#
# `perf.json` is PRESENCE-GATED rather than REQUIRED on every mode, and that is a
# deliberate choice. The perf producer cannot always resolve a step-time slope on
# the smallest models: their step times do not track model size, so they carry the
# least signal against the same fixed per-round dispatch overhead, and the reducer
# can refuse to resolve a slope at the required confidence. Requiring `perf.json`
# on every mode would make that a HARD FAILURE: `check_golden_manifest`
# (fatal, in `_grade_precheck`) would refuse the whole session, so an instance the
# producer cannot resolve for would write no `reward.json` at all -- every check
# gone, not just the perf one, over an artifact none of the other 43 required
# goldens have anything to do with.
# PRESENCE-GATED instead, same shape as `reference_units.*` two lines up: a
# producer exists and runs correctly on most instances, so an instance it
# has not (yet, or ever) resolved for degrades to a visible coverage gap
# (`conftest._golden_facts`'s `has_perf_golden`, `_DESELECTION_KIND`'s
# `no_golden_here`) rather than an all-or-nothing session abort. Still listed in
# `PRESERVED` below -- that fact (the canonical CPU freezer does not write it) is
# unrelated to whether an instance is REQUIRED to have it.
#
# THE TWO MULTIMODAL ARTIFACTS JOIN THIS SET, and they are the first entries here
# whose absence is a FAMILY property rather than a producer gap. Six of the seven
# instances have no image in their scenario at all, so `image_preprocess.npz` /
# `image_fixture.png` are not a coverage hole there -- there is nothing to record
# -- and the two checks that read them are gated on `conftest._golden_facts
# ['multimodal']` rather than on presence (see `conftest._APPLICABLE` for why that
# way round: a multimodal instance with a missing recording must FAIL, not
# deselect). They are listed here so `coverage_vector`'s `optional_present`
# reports them where they exist, and in PRESERVED below because the canonical
# freezer does not write them.
#
# THE TOWER RECORDING AND ITS OWN FIXTURE JOIN THEM, for the same family reason
# and with one addition worth stating: `image_fixture_tower.png` is a 140x140
# image and NOT a second copy of the graded prompt's input. The graded prompt
# keeps the 56x56 one; the tower recording needs a grid whose window permutation
# is not the identity, which no size the prompt can afford reaches. So this pair is
# absent on the six text-only instances because they have no tower, and present
# on the multimodal one alongside -- not instead of -- the primary fixture.
OPTIONAL_GOLDENS: frozenset[str] = frozenset({
    "reference_units.json",
    "reference_units.npz",
    "battery/prompt_length",
    "battery/paging",
    "perf.json",
    "image_preprocess.npz",
    "image_fixture.png",
    "image_fixture_tower.png",
    "vision_tower_units.json",
    "vision_tower_units.npz",
})

# Required goldens the canonical freezer does NOT write -- each has its own
# producer, so they are already in the committed golden tree when the freezer runs
# and are absent from its output dir by construction. `missing_required` drops them
# ("after dropping PRESERVED-intersecting tokens, any remainder raises"); without
# that filter the freezer would fail its own integrity gate on its very first run.
#
#   param_norms.json           - its own producer
#   interface.json             - its own producer (static AST, no torch)
#   paged_attention_units.npz  - its own producer
#   reference_units.*          - its own producer
#   held_out.npz               - its own producer (L6, fp32 tier)
#   battery/*                  - its own producer (additive)
#   perf.json                  - its own producer (GPU, pinned SKU)
#
# The battery producer covers only ONE of the four battery axes: it implements
# `batch_composition` and nothing else, so:
#
#   battery/batch_composition  PRESENT on `ragged_llama3`, and REQUIRED there by
#                              the mode table below. Its check runs.
#   battery/{prompt_length, paging, chunk_boundary}
#                              absent from every golden tree; their checks are
#                              DESELECTED, not skipped (conftest's
#                              `_DESELECTION_KIND`, kind `no_producer`).
#
# The `no_producer` claim is asserted by the harness rather than from inside the
# graded container, because nothing inside it can: at grade time the tests tree
# cannot see the producers at all.
#
# `held_out_nll.json` IS GONE FROM THIS FILE ENTIRELY, and not merely renamed. It was
# the OPTIONAL, out-of-band L6 artifact of a check (`test_held_out_nll`) that was
# presence-gated on a golden with no producer AND on an environment variable nothing
# set, so it never executed on any instance in either class. `held_out.npz` replaces
# it in `_COMMON` as a REQUIRED token with an in-band consumer; the two are not the
# same artifact and nothing should preserve the old name.
PRESERVED: frozenset[str] = frozenset({
    "param_norms.json",
    "interface.json",
    "paged_attention_units.npz",
    "reference_units.json",
    "reference_units.npz",
    "held_out.npz",
    "battery/prompt_length",
    "battery/batch_composition",
    "battery/paging",
    "battery/chunk_boundary",
    # GPU-only, pinned-SKU producer -- absent from the freezer's CPU/torch_orig run
    # by construction, the same treatment param_norms.json and interface.json get,
    # for the same reason: without this, the canonical freezer fails
    # its own integrity gate on its very first run. UNLIKE those two (and like
    # reference_units.* above), it is also in OPTIONAL_GOLDENS now, not
    # REQUIRED_GOLDENS -- PRESERVED only states "the canonical freezer does not
    # write this", which stayed true the whole time; it never claimed "every
    # instance must have one".
    "perf.json",
    # Produced by its own image-preprocess producer (it needs transformers 5.4.0
    # and the checkpoint). LISTED HERE FOR THE ORPHAN GATE,
    # not for `missing_required`: neither is in REQUIRED_GOLDENS, so nothing
    # filters them out of a requirement -- but the freezer's `diff_goldens` step
    # walks every FILE in the golden directory and reports
    # anything it did not itself write and that is not PRESERVED as an
    # "ORPHAN". Without these two rows, a `--check` of the multimodal instance
    # reports the multimodal recording as stale on a tree
    # where it is exactly right.
    "image_preprocess.npz",
    "image_fixture.png",
    # The image-preprocess producer writes the second fixture in the same
    # run as the first, and the vision-tower producer reads it back to
    # record the tower's per-submodule I/O. Both are here for the ORPHAN GATE,
    # exactly as the two rows above are: neither is in REQUIRED_GOLDENS, so
    # nothing filters them out of a requirement -- but the freezer's
    # `diff_goldens` step walks every FILE in the golden directory and reports
    # anything it did not itself write and that is not PRESERVED as an "ORPHAN".
    "image_fixture_tower.png",
    "vision_tower_units.json",
    "vision_tower_units.npz",
})


def _npz_arrays(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            return {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    except (OSError, zipfile.BadZipFile):
        return set()


def _has_token(golden_dir: Path, token: str) -> bool:
    """True if a manifest token is present in ``golden_dir``.

    Forms: a filename, a ``dir/`` path (directory-presence -- the battery axes), or
    ``<npz-stem>:<array>`` / ``<json-stem>:<key>`` for a within-file requirement.

    A within-file token searches BOTH halves of a stem, and that is not a nicety.
    The original tried ``{stem}.npz`` and *returned* on it, so the ``{stem}.json``
    branch was unreachable for any stem carrying both -- which was no stem at all
    until ``sampler`` became a json/npz pair, at which point ``sampler:grid_params``
    and ``sampler:logprobs_top_n`` both resolved False despite being right there in
    the JSON. Nothing requires such a token today; the grammar invites exactly one,
    and the failure mode is a permanent phantom entry in ``missing_required`` on a
    CORRECT tree -- a coverage error that reads as a missing golden.

    Inherited semantics kept deliberately: on the json side a key whose value is
    FALSY (``0``, ``[]``, ``""``) reads as absent. That is right for the way this
    grammar is used -- ``scenario:chunk_split_points`` should not report "present"
    on a ``core`` instance that froze an empty list -- but it is a trap if a future
    token ever names a legitimately-zero scalar, so it is written down rather than
    left to be rediscovered.
    """
    golden_dir = Path(golden_dir)
    if "/" in token and ":" not in token:
        # A battery axis is a DIRECTORY of frozen variants; a stray file with that
        # name is not a frozen battery, so `.exists()` would be the wrong probe.
        return (golden_dir / token).is_dir()
    if ":" not in token:
        return (golden_dir / token).exists()
    stem, key = token.split(":", 1)
    npz = golden_dir / f"{stem}.npz"
    if npz.exists() and key in _npz_arrays(npz):
        return True
    js = golden_dir / f"{stem}.json"
    if js.exists():
        try:
            data = json.loads(js.read_text())
        except (OSError, ValueError):
            return False
        return bool(data.get(key))
    return False


def _token_files(token: str) -> set[str]:
    """The golden path(s) a token lives in (``stem:key`` -> {stem.json, stem.npz})."""
    if ":" not in token:
        return {token}
    stem = token.split(":", 1)[0]
    return {f"{stem}.json", f"{stem}.npz"}


#: The ``kv_cache_state.npz`` member a SINGLE-geometry container freezes, and the
#: static token in ``_COMMON`` that requires it. Named rather than spelled twice so
#: the substitution in :func:`resolve_required` cannot drift from the table.
CACHE_STATE_MEMBER: str = "cache"
CACHE_STATE_STEM: str = "kv_cache_state"
_CACHE_STATE_STATIC_TOKENS: frozenset[str] = frozenset(
    {f"{CACHE_STATE_STEM}:{CACHE_STATE_MEMBER}"})


def cache_state_member_names(num_groups: int) -> tuple[str, ...]:
    """The ``kv_cache_state.npz`` member names for a container of ``num_groups``.

    ONE GROUP -> ``("cache",)``. n > 1 -> ``("cache_group0", ..., "cache_group{n-1}")``,
    in CONTAINER ORDER, which is the order ``modeling.py``'s ``layer_slot``
    addresses and therefore the only order the artifact may be written in.

    THE KEY SET IS THE DECLARATION. Never both spellings, and never a ``cache``
    that means "group 0" -- which is the one form that would grade part of a
    heterogeneous cache against a contract describing all of it, at full credit,
    with nothing missing from either file. The single definition lives here rather
    than in the freezer because three surfaces spell it: the freezer that writes
    the members, this manifest that requires them, and
    ``test_numeric.test_kv_cache_write`` that reads them back.

    ZERO IS REFUSED BY NAME. ``make_kv_cache`` already refuses an empty ``groups``,
    for the reason this mirrors: a cache with no tensors threads through every
    layer without error and holds nothing, so "no groups" must not be spelled the
    same way as "one group".
    """
    count = int(num_groups)
    if count < 1:
        raise ValueError(
            f"a KV cache container holds at least one group; {num_groups!r} "
            "describes no cache at all, and 'no groups' must not be spelled the "
            "same way as 'one group'.")
    if count == 1:
        return (CACHE_STATE_MEMBER,)
    return tuple(f"cache_group{index}" for index in range(count))


def declared_cache_groups_in_spec(spec: dict) -> int | None:
    """How many KV geometries a LOADED ``cache_spec.json`` declares, or ``None``.

    ``groups`` (a non-empty list of per-group shape rows) -> its length; ``shape``
    (the rectangular spelling) -> 1. Anything else -- not a mapping, or a spec
    carrying NEITHER key -- is ``None``, which callers read as "undeclared" and NOT
    as "one".

    The distinction matters at exactly one place and it is the freezer's first run
    on a new family: ``None`` must fall back to the rectangular requirement (the
    conservative answer, and what every instance shipped today needs) rather than
    to "require nothing", because a requirement that evaporates when its own
    declaration is missing is a coverage hole that reads as a clean tree.

    Takes the loaded mapping rather than a path because ``test_numeric`` asks the
    same question of ``golden.cache_spec``, which ``fixtures.load_golden`` already
    read; two readers of one rule, one implementation of it.
    """
    if not isinstance(spec, dict):
        return None
    groups = spec.get("groups")
    if isinstance(groups, list) and groups:
        return len(groups)
    if spec.get("shape"):
        return 1
    return None


def declared_cache_groups(golden_dir: Path) -> int | None:
    """:func:`declared_cache_groups_in_spec` of ``golden_dir/cache_spec.json``.

    An absent or unparseable file is ``None``, the same answer a spec that
    declares neither key gives -- both mean "this tree does not say", and both
    fall back to the rectangular requirement at the one call site that cares.
    """
    try:
        spec = json.loads((Path(golden_dir) / "cache_spec.json").read_text())
    except (OSError, ValueError):
        return None
    return declared_cache_groups_in_spec(spec)


def cache_state_tokens(golden_dir: Path) -> frozenset[str]:
    """The ``kv_cache_state`` tokens this directory's OWN cache contract requires."""
    declared = declared_cache_groups(golden_dir)
    names = cache_state_member_names(1 if declared is None else declared)
    return frozenset(f"{CACHE_STATE_STEM}:{name}" for name in names)


def resolve_required(golden_dir: Path, mode: str) -> frozenset[str]:
    """``REQUIRED_GOLDENS[mode]`` with the cache-state tokens resolved for this tree.

    The static table can only name the rectangular spelling; this substitutes the
    per-group one when the directory's ``cache_spec.json`` declares ``groups``. An
    unknown mode answers the empty set, exactly as ``missing_required`` does --
    "nothing asserted", not "everything satisfied".

    IDENTITY ON EVERY RECTANGULAR TREE, which is what keeps the fourteen shipped
    instances' gate exactly where it was: ``declared_cache_groups`` answers 1 for
    them, ``cache_state_member_names(1)`` is ``("cache",)``, and the set that comes
    out is the table that went in.
    """
    required = REQUIRED_GOLDENS.get(str(mode).lower())
    if not required:
        return frozenset()
    return (required - _CACHE_STATE_STATIC_TOKENS) | cache_state_tokens(golden_dir)


def missing_required(golden_dir: Path, mode: str,
                     include_preserved: bool = False) -> list[str]:
    """Required manifest tokens (for ``mode``) absent from ``golden_dir``, sorted.

    PRESERVED tokens are excluded by default: they are required goldens with their own
    producers, so the freezer's integrity gate must not count them.
    Pass ``include_preserved=True`` at grade time, where every producer has already
    run and a missing ``interface.json`` really is a broken golden set.

    Unknown mode -> empty list (nothing asserted): the manifest only speaks for the
    three benchmark modes, and an unrecognised mode is a config error the mode-name
    gate elsewhere already surfaces.

    THE CACHE-STATE TOKENS COME FROM ``resolve_required``, not from the static
    table, because a heterogeneous container's frozen state is a per-group SET and
    no static token can name it. See this module's docstring for why that
    resolution stays inside this function while ``meta_dependent_tokens`` stays
    outside it.
    """
    required = resolve_required(golden_dir, mode)
    if not required:
        return []
    if not include_preserved:
        required = frozenset(tok for tok in required
                             if not (_token_files(tok) & PRESERVED))
    return sorted(tok for tok in required if not _has_token(golden_dir, tok))


def meta_dependent_tokens(meta: dict) -> frozenset[str]:
    """Required tokens whose NAMES come from the instance's geometry, not the mode.

    ``REQUIRED_GOLDENS`` above is static by design -- it is importable with nothing
    but the stdlib, from the verifier container and from the freezer alike, and it
    must stay answerable without a ``meta.json``. One family of required arrays is
    not expressible there: ``decode_step.npz`` holds one ``logits_step_{i}`` per
    ``meta['num_decode_steps']``, and ``test_numeric.test_decode_step_logits``
    reads every one of them by that constructed name.

    So it is a separate function with the meta as its argument, called by
    ``_harness_checks.check_golden_manifest`` (which has a loaded golden) and NOT
    by ``missing_required``: that one is also the freezer's own integrity gate and
    a directory-level completeness input, both of which speak for a
    directory rather than for an instance. The split is the same one
    ``include_preserved`` already makes -- one manifest, two questions.

    NO ``num_decode_steps`` -> NO TOKENS, rather than a guess. A meta that omits it
    is refused two gates earlier (``conftest._require_complete_geometry`` treats a
    missing geometry key as a usage error precisely because a default would
    silently deselect), and inventing a count here would report phantom missing
    arrays on a golden that is merely undescribed.
    """
    try:
        steps = int(meta.get("num_decode_steps") or 0)
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(f"decode_step:logits_step_{i}" for i in range(max(steps, 0)))


def coverage_vector(golden_dir: Path, mode: str) -> dict:
    """Per-instance coverage summary for the eval summary (the coverage vector).

    ``required_missing`` is the freeze-gate view (PRESERVED dropped);
    ``preserved_missing`` keeps the other producers' artifacts visible rather than
    silently uncounted.

    ``required_total`` COUNTS THE RESOLVED SET, so a heterogeneous family's extra
    cache-state members are in the denominator the coverage vector reports. Reading
    it off the static table instead would report 43 for a tree whose gate is
    actually checking 44 -- a denominator that disagrees with the numerator's own
    gate, which is the shape this vector exists to make visible.
    """
    required = resolve_required(golden_dir, mode)
    all_missing = missing_required(golden_dir, mode, include_preserved=True)
    gated = set(missing_required(golden_dir, mode))
    return {
        "mode": mode,
        "required_total": len(required),
        "required_missing": sorted(gated),
        "preserved_missing": sorted(t for t in all_missing if t not in gated),
        "optional_present": sorted(t for t in OPTIONAL_GOLDENS if _has_token(golden_dir, t)),
    }
