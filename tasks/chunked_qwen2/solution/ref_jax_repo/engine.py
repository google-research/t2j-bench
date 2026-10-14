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

"""The prefill -> decode driver, mirroring environment/torch_orig/engine.py.

Same classes, same fields, same method signatures, same order of operations.
Four differences, all forced by JAX:

  1. THE CACHE IS THREADED, NOT MUTATED. torch_orig scatters into
     `self.kv_cache` in place and the engine never has to think about it. Here
     the model returns a new cache and the engine reassigns it
     (`hidden, self.kv_cache = ...`). A port that keeps the torch signature
     silently drops every cache write and every decode attends to zeros.
  2. THE MODEL CALL IS JITTED ONCE, at construction, and reused. A port that
     re-lowers per step is functionally right and practically useless -- the
     probe geometry hides it, a 128k-vocab decode loop does not. `nnx.jit`
     rather than `jax.jit` with the config in `static_argnums`: a real
     `rope_scaling` is a dict, `ModelConfig` is a frozen dataclass holding it,
     and hashing that raises `TypeError: unhashable type: 'dict'`. So the
     config must never become part of a jit cache key. nnx keys off the module
     graph instead, which is why the same model whose graphdef is unhashable
     still jits.
  3. `validate_metadata` is called where the metadata is BUILT. paged_attention's
     two preconditions are eager-only, so a fully jitted forward -- which this
     engine's is -- never sees them. Neither branch can fire for THIS engine's
     own `_metadata`; it is a contract for a candidate's, and its docstring says
     so plainly.
  4. EXTRA MODEL INPUTS ARE PREPARED ON THE HOST, before the trace, if the model
     asks for it (:meth:`Engine._model_inputs`). torch_orig hands its extra
     keywords straight into the forward; here a subsystem that reads its own
     intermediates as CONCRETE values has to run outside the trace, and it hands
     the trace an already-merged embedding table instead. This is a property of
     how such a subsystem is WRITTEN, not a limit of jit -- see that method for
     the details and for what it would take to lift. It is the SAME
     `Engine.INPUTS_HOOK` mechanism `core` and `ragged` have, at the same
     spelling, resolved the same way; the class pays for one host-side input
     hook and a second spelling of it would be a defect.

`reset()` zeroes the cache by rebinding `jnp.zeros_like` OVER EVERY GROUP, the
functional analogue of torch's `zero_()`; there is no other kind of zeroing
here.

POSITIONS ARE A WINDOW ONTO THE RUN THE WHOLE PROMPT DEFINES -- the torch
twin's docstring argues it in full and this file behaves identically: a chunk's
positions are the slice `[cached, seq_len)` of a run built from the request's
COMPLETE ids, which :meth:`prefill` receives as `full_input_ids` and hands to
the model's own builder through `Engine.POSITIONS_HOOK`. A model that defines no
such builder gets `arange(cached, seq_len)`, which is the identical array that
window would have returned, so every text-only family -- which is every
`chunked` instance shipped today -- is bit-for-bit unaffected.

`default_input_positions` and `_input_positions` are the two pieces of the other
modes' delegation machinery that are STILL absent here, and their absence is a
consequence rather than a leftover: this fork has to know whether the model has
a builder AT ALL (`getattr(..., None)`), because the un-plumbed continuation it
refuses by name is a shape only a hook family can be in, and a `getattr` with a
real fallback cannot answer that. `_advance`, `INPUTS_HOOK`, `_model_inputs` and
`default_model_inputs` ARE here, at the other modes' spellings, and
:meth:`Engine.generate` carries `**model_kwargs` through to the prefill the way
the other two modes' does -- because that is the only entry point the golden
freezer uses, so a hook nothing could call through it would be a hook no golden
could ever exercise.

WHAT THAT DOES AND DOES NOT MEAN, stated narrowly on purpose, because the
opposite claim is what this class keeps paying for. THE ENGINE no longer refuses
a multimodal family: a keyword can reach the traced forward, a chunk's positions
are a window onto the run its full ids define, and the decode cursor is a column
at the model's own rope rank. THE CELL is a different question and this file
cannot answer it -- nothing here has been exercised against a real tower, no
`chunked` instance defines either model hook, and the harness has its own
blockers. Check those blockers before pricing a cell; do not read this
paragraph as saying a cell is buildable.

There is still no branch on a family, a model name, an instance or a
`multimodal` fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from attention.metadata import AttentionMetadata
from cache.block_table import BlockPool
from cache.kv_cache import make_kv_cache
from cache.prefix_cache import PrefixCache
from config import RunConfig
from sample.sampler import SamplingParams, greedy, process_logits


def default_model_inputs(input_ids, **model_kwargs) -> dict:
    """The model's extra forward keywords, unchanged -- the shared default.

    A text-only family passes none at all, so this is the identity on an empty
    dict. It exists so that :meth:`Engine._model_inputs` has something to
    resolve to without asking whether the model is multimodal.
    """
    return model_kwargs


@dataclass
class PrefillResult:
    hidden: jax.Array
    logits: jax.Array
    slot_mapping: jax.Array
    attn_metadata: AttentionMetadata
    page_ids: list[int]
    hidden_states: list[jax.Array] | None = None


@dataclass
class DecodeResult:
    hidden: jax.Array
    logits: jax.Array
    slot_mapping: jax.Array
    attn_metadata: AttentionMetadata


@dataclass
class GenerateResult:
    tokens: jax.Array
    step_logprobs: jax.Array
    logits_per_step: list[jax.Array] = field(default_factory=list)


def validate_metadata(attn_metadata: AttentionMetadata, num_query_tokens: int,
                      page_size: int) -> None:
    """Re-check paged_attention's two preconditions ON THE HOST, before tracing.

    A CONTRACT FOR CANDIDATE `_metadata` IMPLEMENTATIONS, NOT AN ACTIVE GUARD ON
    THIS ENGINE. Neither branch can fire for the reference:
    `Engine._metadata` builds `query_start_loc=[0, q_len]` from the same `q_len`
    it passes as `num_query_tokens`, so the counts are equal by construction;
    and `RunConfig.__post_init__` already refuses
    `pages_per_seq * page_size < max_model_len` while `_grow` refuses the
    overrun, so the block table always reaches far enough. Both branches are
    tautological as called from this file, and that is fine -- they are the
    checked half of an interface a candidate has to reimplement.

    What makes the contract worth stating at all is that `attention/paged.py`'s
    own versions of these two checks are EAGER-ONLY and cannot be otherwise:
    under jit `query_start_loc[-1]` is a tracer whether the metadata is passed
    in, closed over, or built inside the traced function, so they are skipped
    for every jitted forward -- and this engine's forward is jitted.
    attention/paged.py names that gap in its module docstring; this function is
    the other half of it, at the one point where the values are still concrete.

    The gap matters more on this side than on torch_orig's, for a specific
    reason: torch raises IndexError on an out-of-range gather, while JAX
    CLAMPS. A padded query token therefore gets `seq_of_tok == num_seqs`,
    silently attends as if it belonged to the last sequence, and comes back as
    a plausible-looking wrong tensor with no exception anywhere.

    Host-side is itself a contract, not an accident: it reads metadata VALUES,
    so calling it from inside a trace is a programming error and it says so
    rather than letting a ConcretizationTypeError surface from an int() three
    frames down.
    """
    try:
        ragged_tokens = int(attn_metadata.query_start_loc[-1])
        longest = (int(attn_metadata.seq_lens.max())
                   if attn_metadata.seq_lens.size else 0)
    except jax.errors.ConcretizationTypeError as exc:
        raise RuntimeError(
            "validate_metadata reads metadata VALUES and so must be called on "
            "the host, where the metadata is constructed -- not from inside a "
            "trace, where every field is a tracer and nothing can be checked."
        ) from exc

    # A token array SHORTER than the batch it describes (attention/paged.py). A
    # LONGER one is the compiled bucket, which `ragged` supports and masks.
    if num_query_tokens < ragged_tokens:
        raise ValueError(
            f"validate_metadata: {num_query_tokens} query tokens but "
            f"query_start_loc[-1] is {ragged_tokens}: the token array is SHORTER "
            f"than the ragged batch it describes, so rows the metadata claims "
            f"exist would never be computed."
        )

    # Under-allocated pages are a wrong answer, not a slow one.
    max_kv = attn_metadata.pages_per_seq * page_size
    if longest > max_kv:
        raise ValueError(
            f"validate_metadata: seq_lens max is {longest} but the block table "
            f"only reaches pages_per_seq * page_size = "
            f"{attn_metadata.pages_per_seq} * {page_size} = {max_kv}; this "
            f"request is under-allocated."
        )


class Engine:
    #: The method a model defines when its `input_positions` are NOT `arange`.
    #: On `core` and `ragged` this names a hook the engine calls, and HERE it
    #: does too -- :meth:`prefill` asks the named builder for the whole run and
    #: windows this chunk out of it. Held at the same string as the other two
    #: modes and as the golden freezer, which the harness checks across all
    #: instances, because a lookup that keys on a stale name finds nothing,
    #: falls back to `arange` and rotates every image token wrong.
    POSITIONS_HOOK = "get_input_positions"

    #: The OPTIONAL method a model may define to turn its own forward keywords
    #: into TRACEABLE ones, on the host. JAX-only -- see the module docstring's
    #: difference 4 and :meth:`_model_inputs`. Held at `core`'s and `ragged`'s
    #: spelling for the reason POSITIONS_HOOK is: the grade resolves it by name
    #: off the candidate's own `Engine` class, so a second spelling here would
    #: be silently un-findable rather than loudly wrong.
    INPUTS_HOOK = "prepare_model_inputs"

    def __init__(self, model, run_config: RunConfig) -> None:
        self.model = model
        self.cfg = run_config
        cfg = model.config
        # ONE GROUP. This family's layers all agree on their KV geometry, so the
        # container's single array IS the rectangular cache this engine used to
        # hold -- which is what keeps the frozen `kv_cache_state.npz` and
        # `cache_spec.json` where they are.
        #
        # THIS BLOCK OWNS THE CONTAINER'S GEOMETRY AND NOTHING ELSE. The layer
        # -> (group, slot) map that INDEXES it is applied in modeling.py, on the
        # attention module that does the indexing, and both are spelled from the
        # one definition in cache/kv_cache.py (`identity_layer_slot` /
        # `identity_layer_writes`). gemma4-E2B changes BOTH: two groups here,
        # because its sliding layers have head_dim 256 and its full ones 512,
        # and a non-identity map there, because it allocates 15 slots for 35
        # layers. Neither file is the whole extension point and neither should
        # claim to be; `cache/kv_cache._require_slot` is what refuses the
        # half-done version where only one of them moved.
        self.kv_cache = make_kv_cache(
            ((cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim),),
            run_config.num_pages, run_config.page_size,
            getattr(jnp, run_config.dtype),
        )
        self._pool = BlockPool(run_config.num_pages, run_config.page_size)
        # The single owner of shared pages ABOVE the pool (cache/prefix_cache.py's
        # own docstring). One instance per engine lifetime, rebuilt in reset()
        # alongside the pool it wraps -- see the comment there for why a stale
        # PrefixCache tied to a replaced pool would be worse than no cache at
        # all. Public (not `_prefix_cache`): the engine test suite and the
        # graded invariance test (test_behavioral.py) both read
        # `engine.prefix_cache.stats()["hits"]` as an honest count of REAL
        # cross-request sharing.
        self.prefix_cache = PrefixCache(self._pool, run_config.page_size)
        self._pages: dict[int, list[int]] = {}
        self._lens: dict[int, int] = {}
        # The NEXT position column each live request will use, at that model's
        # own rope rank. See the torch twin for why this is not derived from
        # `_lens`: with M-RoPE a request's next POSITION and its next token
        # INDEX are different numbers.
        self._next_pos: dict[int, jax.Array] = {}

        # How many times the forward has been LOWERED, not how many times it has
        # been called: the body of a jitted function runs once per trace. This
        # is the only handle on "does the decode loop re-compile per step" --
        # `nnx.jit` exposes no cache-size accessor in flax 0.12, so the
        # alternative is no observation at all. The harness keys off it.
        self._traces = 0

        def _forward(model, input_ids, kv_cache, attn_metadata, slot_mapping,
                     output_hidden_states, model_kwargs):
            self._traces += 1
            return model(input_ids, kv_cache, attn_metadata, slot_mapping,
                         output_hidden_states=output_hidden_states,
                         **model_kwargs)

        # output_hidden_states is static: it changes the number of returned
        # values, so it cannot be a traced argument. Everything else -- the
        # model included -- is graph data or a pytree, and `model_kwargs` is a
        # pytree of arrays by the time it gets here (:meth:`_model_inputs`
        # refuses anything else by name). `{}` for every text-only family, so
        # the lowered function is the one it always was -- and that is every
        # `chunked` instance shipped today, which is why the frozen decode
        # geometry does not move. `decode_step` passes the literal `{}` on
        # every family: the tower runs once, at prefill.
        self._jit_fwd = nnx.jit(_forward, static_argnums=(5,))

    def reset(self) -> None:
        self._pool = BlockPool(self.cfg.num_pages, self.cfg.page_size)
        # CONTENT tracking cannot survive: the old PrefixCache's `_by_hash`/
        # `_refcount`/`_owners` still claim specific physical page ids as
        # CACHED after `kv_cache` is zeroed below and the new `self._pool`
        # above has forgotten they were ever allocated. Keeping it would let
        # a future `acquire()` hand back a page id that is simultaneously
        # FREE (in the new pool's free list) and OWNED (per the stale hit
        # record) -- the exact double-owner state
        # `allocator_reuses_live_page` exists to refuse, just reached through
        # the cache instead of through `_grow`.
        #
        # The hit/miss COUNTERS are a different kind of state: pure lifetime
        # diagnostics with no physical resource behind them, so nothing about
        # rebuilding the pool invalidates them. Carried over rather than
        # wiped with the rest -- the same choice this file already makes for
        # `_traces`, just below, which `reset()` also leaves untouched.
        stats = self.prefix_cache.stats()
        self.prefix_cache = PrefixCache(self._pool, self.cfg.page_size)
        self.prefix_cache._hits = stats["hits"]
        self.prefix_cache._misses = stats["misses"]
        self._pages.clear()
        self._lens.clear()
        self._next_pos.clear()
        # EVERY group, not just the first. One group today, so a `reset` that
        # reached only `kv_cache[0]` would be green on all thirteen instances
        # and silently wrong for the first family that allocates two.
        self.kv_cache = tuple(jnp.zeros_like(group) for group in self.kv_cache)

    @staticmethod
    def _advance(positions: jax.Array) -> jax.Array:
        """The position column ONE STEP PAST `positions`, at the same rank.

        The torch twin's rule and its argument: one past the furthest position
        ANY rope section reached, which is vLLM's `mrope_position_delta`. At
        rank 1 over the run `[0, seq_len)` it is `seq_len` -- exactly the
        `jnp.array([seq_len], dtype=jnp.int32)` :meth:`prefill` used to write,
        same shape, same dtype, same value, on every family whose positions are
        `arange`.

        THE EMPTY CLAUSE IS THIS MODE'S, and `core`/`ragged` do not carry it.
        What is handed in here is THE RUN SO FAR, `[0, seq_len)`, not the
        chunk's window -- :meth:`prefill` says why -- so the clause is
        unreachable for a request that has any tokens at all. It is kept for
        the one call that has none: a `prefill` of a zero-token `input_ids`,
        which this engine accepts today and answers `[0]`, and which the rule
        above cannot answer because an empty run reached no position.

        `jnp.max` on a zero-size array RAISES
        `ValueError: zero-size array to reduction operation max which has no
        identity` -- eagerly AND under `jax.jit`, because the check is on the
        shape, which is static. So the failure this clause replaces is a crash on
        both sides, not a silent wrong answer; the clause exists to keep the
        zero-token call's ANSWER, not to dodge a silent one.
        """
        return jnp.full((*positions.shape[:-1], 1),
                        int(positions.max()) + 1 if positions.size else 0,
                        dtype=jnp.int32)

    def _model_inputs(self, input_ids, **model_kwargs) -> dict:
        """The model's extra forward keywords, PREPARED ON THE HOST if it asks.

        THE FOURTH JAX-FORCED DIFFERENCE (module docstring), and the reason it
        is forced rather than chosen is worth stating where the code is, because
        it is not what a reader of the torch twin would predict.

        `jnp.arange(4)` INSIDE a `jit` trace is a tracer.
        Not "if it depends on an argument" -- always. So `int(x)`, `x.tolist()`
        and `if x > 0` are unavailable on any `jnp` value under the trace,
        whatever was passed in and however it was marked static.

        AND THE HONEST STATEMENT OF THE CONSEQUENCE, because the obvious one is
        too strong: `np.arange(4)` in the SAME trace is concrete and `int()` on
        it works. So a subsystem is stuck outside the trace
        only if it spells its own geometry in `jnp`. A vision tower whose window
        permutation, per-chunk attention bounds and rotary range check are
        Python control flow over `jnp` intermediates is; the same tower with
        that arithmetic in `numpy` over a static grid would not be, because
        every one of those reads is a function of `image_grid_thw` alone and
        never of `pixel_values`. Compiling it is a rewrite of the tower, not a
        change to this engine -- and it is deliberately not attempted here,
        because the tower is a de-vendored port whose shape is graded against
        upstream's.

        WHAT IT COSTS, so a later reader does not have to guess: PREFILL ONLY.
        The tower runs once per request and never in the decode loop (the torch
        twin's `prefill` docstring says why), while the perf gate times
        steady-state DECODE. So an uncompiled tower does not reach the graded
        number.

        THE IDS ARE THE CHUNK'S, which is this mode's one adaptation of `core`'s
        method and is the whole of it. :meth:`prefill` calls this with
        `forward_ids` -- the tokens THIS call actually embeds -- because that is
        what the keywords have to line up with; the request's COMPLETE ids go to
        the POSITIONS builder instead, and the two are different arrays the
        moment a chunk is not the whole prompt.

        The model gets one chance, on the host, to turn keywords it must read
        concretely into keywords the trace can carry -- typically by running the
        tower here and returning `inputs_embeds` instead of `pixel_values`. The
        default (:func:`default_model_inputs`) returns them unchanged, which is
        every text-only family.

        WHAT COMES BACK MUST BE ARRAYS, and a non-array is refused BY NAME
        rather than left to jax. jax would flatten a Python int into a traced
        scalar without complaint, and the model would then fail somewhere deep
        with a ConcretizationTypeError naming a line that is not the mistake.
        """
        prepare = getattr(self.model, self.INPUTS_HOOK, default_model_inputs)
        prepared = prepare(input_ids, **model_kwargs)
        stragglers = sorted(name for name, value in prepared.items()
                            if not isinstance(value, jax.Array))
        if stragglers:
            raise TypeError(
                f"{self.INPUTS_HOOK} returned non-array keyword(s) "
                f"{stragglers}, which the jitted forward would trace into "
                f"tracers. Anything the model has to read as a CONCRETE value "
                f"must be consumed on the host, inside {self.INPUTS_HOOK}, and "
                f"only its array result handed back.")
        return prepared

    def _metadata(self, positions, page_ids, seq_len, q_len, distribution):
        # `positions` IS PASSED THROUGH AT WHATEVER RANK IT ARRIVES: (T,) here,
        # and (R, T) for an M-RoPE family, where R is the number of rope
        # sections. Normalising the rank HERE would change input_positions'
        # frozen shape for every text-only instance, and that shape is graded;
        # layers/rotary.py is where the two ranks are made uniform, at the one
        # point that actually indexes the cos/sin table.
        #
        # Every array field is int32, which is the AttentionMetadata contract
        # the harness pins -- and which this side gets for free under
        # jax_enable_x64=False, where the torch side has to ask for it. The
        # engine is the PRODUCTION source of the metadata, so this is what the
        # golden freezer reads.
        #
        # BUCKETED TO max_num_seqs, WHICH IS WHAT `ragged` MAKES OBSERVABLE.
        # AttentionMetadata's own docstring declares block_tables to be
        # `(max_num_seqs * pages_per_seq,)` and seq_lens `(max_num_seqs,)`, and
        # derives `pages_per_seq` as `block_tables.shape[0] // num_seqs`. On the
        # `core` mode max_num_seqs is 1, so the bucket and the occupancy coincide
        # and an unbucketed builder satisfies the contract by accident; here they
        # do not, and a one-row table against a three-request seq_lens makes that
        # derivation read a third of the table as one request's pages.
        rows = [self._pool.block_table_row(page_ids, self.cfg.pages_per_seq)]
        pad = self.cfg.max_num_seqs - 1
        rows += [jnp.zeros(self.cfg.pages_per_seq, dtype=jnp.int32)] * pad
        meta = AttentionMetadata(
            input_positions=positions,
            block_tables=jnp.concatenate(rows) if pad else rows[0],
            # A request slot nothing scheduled has no KV and contributes no query
            # token, so it pads with a zero length and with the running offset.
            seq_lens=jnp.array([seq_len] + [0] * pad, dtype=jnp.int32),
            query_start_loc=jnp.array([0] + [q_len] * (pad + 1), dtype=jnp.int32),
            request_distribution=jnp.array(distribution, dtype=jnp.int32),
            padded_num_reqs=self.cfg.max_num_seqs,
        )
        # Here, and not inside the model: this is the last point at which the
        # fields are concrete values rather than tracers.
        validate_metadata(meta, q_len, self.cfg.page_size)
        return meta

    def _grow(self, request_id: int, new_len: int) -> list[int]:
        """Ensure the request owns enough pages for `new_len` tokens."""
        # Named, not incidental. Without this the overrun dies further down as a
        # shape error out of block_table_row, which says nothing about the
        # cause. paged_attention and BlockPool.free both refuse their own bad
        # inputs by name; the engine is the one layer that used to opt out.
        if new_len > self.cfg.max_model_len:
            raise ValueError(
                f"request {request_id} would reach {new_len} tokens but "
                f"max_model_len is {self.cfg.max_model_len}"
            )
        pages = self._pages.setdefault(request_id, [])
        need = (new_len + self.cfg.page_size - 1) // self.cfg.page_size
        if need > len(pages):
            pages.extend(self._pool.allocate((need - len(pages)) * self.cfg.page_size))
        return pages

    def prefill(self, input_ids: jax.Array, request_id: int = 0,
                output_hidden_states: bool = False,
                continue_request: bool = False,
                full_input_ids: jax.Array | None = None,
                **model_kwargs) -> PrefillResult:
        """THE POSITIONS ARE A WINDOW ONTO THE RUN THE WHOLE PROMPT DEFINES.

        A chunk's positions are not `arange(cached, seq_len)` for every family.
        With M-RoPE an image collapses several tokens onto fewer positions, so
        rebuilding from the chunk's own bounds restarts the run and loses the
        offset. `full_input_ids` is the request's COMPLETE token sequence; the
        model's own builder is asked for the whole run and this chunk's slice is
        taken out of it.

        `full_input_ids` defaults to `input_ids`, which is exactly right for a
        first chunk and for every family with no positions hook -- a window of an
        `arange` run IS `arange`. That default is what keeps this identical to
        the previous behaviour on all four shipped chunked instances, none of
        which defines the hook. The one shape that default cannot cover -- a
        hook family's chunk with `continue_request=True`, whose run really is
        longer than the ids in hand -- is refused by name below rather than
        computed short.

        `**model_kwargs` REACHES BOTH THE POSITIONS BUILDER AND THE FORWARD, and
        the two get different ids: the builder is asked for the whole run
        (`full_ids`), while the forward's keywords are prepared from
        `forward_ids`, the tokens this call actually embeds. A family whose
        FORWARD needs a keyword -- a vision tower's `pixel_values` -- is
        therefore servable here, through :meth:`_model_inputs`.
        """
        n = int(jnp.asarray(input_ids).shape[0])
        full_ids = input_ids if full_input_ids is None else full_input_ids
        build = getattr(self.model, self.POSITIONS_HOOK, None)
        live = request_id in self._lens
        # THE TWO ERRORS ARE DIFFERENT REPAIRS, so they are different messages.
        # Re-prefilling a live request would silently reuse its pages and
        # rewind its length, interleaving two sequences in one page set --
        # benign in `core` (one sequence, reset() between runs) but this same
        # file serves the `ragged` and `chunked` modes, where concurrent
        # requests are the point and silent state clobbering there is
        # unfindable. Continuing a request nobody prefilled is the opposite
        # bug: it attends to a cache nobody wrote.
        if live and not continue_request:
            raise ValueError(
                f"request {request_id} is already active with "
                f"{self._lens[request_id]} tokens; call reset() before "
                "prefilling it again, or pass continue_request=True to append "
                "a chunk to it"
            )
        if continue_request and not live:
            raise ValueError(
                f"request {request_id} is not active, so there is no cached "
                f"prefix for this chunk to continue; prefill its first chunk "
                "without continue_request"
            )
        # A HOOK FAMILY'S CONTINUATION NEEDS THE WHOLE RUN, and this is the
        # last point at which saying so is still free. `full_ids` fell back to
        # this chunk while `continue_request` says the chunk does not start the
        # run, so the builder would be handed a sequence that restarts at 0 and
        # the window taken out of it at absolute `[cached, seq_len)` would come
        # back short or empty. With a collapsing builder attached, the second of
        # three chunks produces ONE position where seven were required and the
        # prefill returns seven rows of logits anyway -- a wrong answer, not a
        # crash. The refusal here is diagnosable, and it keeps that property for
        # the one caller shape that cannot work while leaving every other one
        # computing.
        #
        # `continue_request`, NOT `cached > 0`. `cached` is a CACHE OUTCOME, not
        # a place in the run: a request's FIRST chunk that hits the prefix cache
        # has `cached > 0` while its `input_ids` ARE the whole run -- the case
        # this method's docstring calls "exactly right for a first chunk".
        # Keying on `cached` refuses that legitimate call and breaks the very
        # path `full_input_ids` exists to enable. `continue_request` is the
        # caller's own statement that earlier chunks of this request exist,
        # which is the property the window actually depends on.
        #
        # `full_input_ids is None`, NOT `full_ids is input_ids`: the `None`
        # default is the only thing separating NOT SUPPLIED from
        # supplied-and-equal-to-the-chunk, and only the first is a caller bug.
        #
        # ABOVE `acquire`/`_grow`, DELIBERATELY: raising after them leaves the
        # refcount incremented and `_owners[request_id]` overwritten, so a
        # caller that catches this and retries with the full ids inherits a
        # cache that has already counted a hit it never served. BELOW the two
        # lifecycle errors, equally deliberately: those describe a request that
        # should not be here at all, neither of them mutates anything, and their
        # diagnosis is the more specific one when both apply.
        if build is not None and full_input_ids is None and continue_request:
            raise NotImplementedError(
                f"{type(self.model).__name__} defines {self.POSITIONS_HOOK}, "
                f"so its positions are a run this chunk is only a WINDOW onto "
                f"-- but this call passed continue_request=True and no "
                f"`full_input_ids`, so the only ids available are this chunk's "
                f"and the run they describe restarts at 0. Pass the request's "
                f"COMPLETE token sequence as `full_input_ids` on every chunk.")
        # TWO INDEPENDENT SOURCES OF "ALREADY CACHED", never consulted together
        # in the same call. `PrefixCache.acquire` hashes its argument as a
        # prompt starting at absolute position 0 -- true only of a request's
        # FIRST chunk -- so it is called exactly once per request lifetime,
        # right here, and never again on a later chunk of the SAME request.
        # `_owners` (prefix_cache.py) tracks one page list per request_id; a
        # second `acquire` call for the same id would overwrite that entry
        # rather than extend it. A continuation chunk's own "already cached"
        # is `self._lens[request_id]`: tokens THIS request wrote in
        # an earlier call, invisible to `acquire` and to every other request.
        # Both resolve to the identical `cached` role below -- the branch is
        # which call site produces the number, not two formulas to reconcile.
        if continue_request:
            cached = self._lens[request_id]
            seq_len = cached + n
            pages = self._grow(request_id, seq_len)
            forward_ids = input_ids
        else:
            # Named here too, not only inside `_grow`: this branch does not
            # call `_grow`, so without this check the overrun would instead
            # surface from `PrefixCache._allocate` as an opaque "out of KV
            # pages" -- or worse, succeed by evicting pages a live request
            # still needs.
            seq_len = n
            if seq_len > self.cfg.max_model_len:
                raise ValueError(
                    f"request {request_id} would reach {seq_len} tokens but "
                    f"max_model_len is {self.cfg.max_model_len}"
                )
            # `acquire` returns the request's COMPLETE page list -- any shared
            # (cache-hit) pages first, then freshly allocated ones covering
            # the unmatched suffix -- so it is used AS the block table
            # directly. Calling `_grow` too would allocate the hit's pages a
            # second time, from a pool that no longer owns them to give.
            pages, cached = self.prefix_cache.acquire(
                jnp.asarray(input_ids).tolist(), request_id)
            # `list(pages)`, NOT `pages` itself: `acquire` also stores this
            # exact list object in `PrefixCache._owners[request_id]`. Sharing
            # the object would let `_grow`'s later `pages.extend(...)` (on a
            # continuation chunk) silently mutate `_owners`' copy too, so the
            # cache would believe it still owns pages that were actually
            # allocated straight from the pool and never registered in its
            # own `_refcount` -- a `release()` on that request then raises
            # KeyError on the ungoverned page instead of freeing it cleanly.
            self._pages[request_id] = list(pages)
            # A prefix hit's tokens are already resident at the right
            # absolute positions -- valid because the ids are identical and
            # both requests started at position 0 -- so only the UNCACHED
            # tail is fed through the forward; the hit portion is read off
            # the shared page, never recomputed.
            forward_ids = input_ids[cached:]
        # `seq_len` is the whole sequence; `q_len` is only the tokens THIS
        # call feeds through the forward. They are equal on a cold, unshared,
        # unchunked prefill -- the ONLY reason `core` never needed this
        # distinction -- and unequal whenever either cached source is
        # nonzero. `seq_len` still describes the whole sequence either way,
        # which is what lets paged_attention derive q_pos from the difference
        # (attention/paged.py:162) instead of from a positions array that
        # would otherwise have to restart at 0.
        q_len = seq_len - cached
        slot_mapping = self._pool.slot_mapping(pages, cached, q_len)
        distribution = [0, 1, 0] if cached == 0 else [0, 0, 1]
        # THE WINDOW, taken from the full run rather than rebuilt from the
        # chunk. `cached` is this chunk's first absolute token index and
        # `seq_len` is one past its last, so `[cached, seq_len)` is exactly the
        # `q_len` tokens this call feeds through the forward. NOT
        # `[cached, cached + n)`: on a FIRST chunk that hit the prefix cache `n`
        # is the whole prompt while only its uncached tail is forwarded, and the
        # two bounds differ by the hit length.
        #
        # `full` IS BOUND ON BOTH PATHS so the cursor below can read the run
        # instead of calling the builder a second time -- a hook is a model
        # method this engine does not own and must not invoke twice per chunk.
        # `None` there is the same fact `build is None` states; a builder that
        # returned `None` would already have died on the slice.
        full = None
        if build is None:
            positions = jnp.arange(cached, seq_len, dtype=jnp.int32)
        else:
            full = build(full_ids, **model_kwargs)
            positions = full[..., cached:seq_len]
        meta = self._metadata(positions, pages, seq_len, q_len, distribution)
        out = self._jit_fwd(self.model, forward_ids, self.kv_cache, meta,
                            slot_mapping, output_hidden_states,
                            self._model_inputs(forward_ids, **model_kwargs))
        if output_hidden_states:
            hidden, hs, self.kv_cache = out
        else:
            hs = None
            hidden, self.kv_cache = out
        self._lens[request_id] = seq_len
        # THE CURSOR IS A PROPERTY OF THE WHOLE SEQUENCE, NOT OF THIS CHUNK'S
        # WINDOW -- the torch twin's comment, and now the torch twin's code.
        # It is derived at THE MODEL'S OWN ROPE RANK by `_advance`, so an M-RoPE
        # family gets the `(R, 1)` column its rotary indexes with and a
        # text-only one gets the same rank-1 `[seq_len]` this line wrote before.
        #
        # THE ARGUMENT IS THE RUN SO FAR, `[0, seq_len)`, AND THAT IS THE WHOLE
        # RESOLUTION OF THE PROBLEM THAT KEPT THIS AT RANK 1. `max() + 1` over
        # THIS CHUNK'S WINDOW has no answer when the chunk is entirely a prefix
        # hit -- `q_len == 0`, `positions` empty, and both frameworks refuse an
        # empty max (jax raises `ValueError: zero-size array to reduction
        # operation max which has no identity`, eagerly and under jit; torch
        # raises `RuntimeError: max(): Expected reduction dim ...`).
        # The RUN's prefix is empty only when the request has no tokens at all,
        # so the entirely-a-hit chunk is answered from the positions its own
        # earlier tokens already occupy rather than from the nothing it
        # forwarded. It is also the more faithful rule: vLLM's
        # `mrope_position_delta` is `max_position + 1` over the WHOLE prompt,
        # not over the last chunk of it.
        #
        # `full` is the whole run the request's ids define and can reach past
        # `seq_len` on any chunk but the last, so the slice is not decoration:
        # the cursor must be one past the furthest position reached SO FAR, not
        # one past the furthest the prompt will ever reach.
        self._next_pos[request_id] = self._advance(
            jnp.arange(seq_len, dtype=jnp.int32) if full is None
            else full[..., :seq_len])
        return PrefillResult(hidden, self.model.compute_logits(hidden),
                             slot_mapping, meta, list(pages), hs)

    def decode_step(self, request_id: int, token_id: int) -> DecodeResult:
        pos = self._lens[request_id]
        new_len = pos + 1
        pages = self._grow(request_id, new_len)
        slot_mapping = self._pool.slot_mapping(pages, pos, 1)
        # The SLOT comes from the token index and the POSITION does not -- see
        # the torch twin, where the two coincide for a text-only family and
        # diverge for one with M-RoPE.
        #
        # `positions + 1` BELOW IS RANK-AGNOSTIC AND WAS VERIFIED RATHER THAN
        # ASSUMED: `_advance` hands back `(*rope_sections, 1)`, and adding a
        # Python int broadcasts over it, so a `(3, 1)` M-RoPE column advances
        # all three sections by one exactly as a `(1,)` text column advances
        # its single one. Upstream does the same thing -- one delta added to
        # every section -- so this needed no change when the cursor gained a
        # rank.
        positions = self._next_pos[request_id]
        meta = self._metadata(positions, pages, new_len, 1, [1, 0, 0])
        hidden, self.kv_cache = self._jit_fwd(
            self.model, jnp.array([token_id], dtype=jnp.int32), self.kv_cache,
            meta, slot_mapping, False, {})
        self._lens[request_id] = new_len
        self._next_pos[request_id] = positions + 1
        return DecodeResult(hidden, self.model.compute_logits(hidden), slot_mapping, meta)

    def generate(self, input_ids: jax.Array, num_steps: int,
                 params: SamplingParams | None = None,
                 request_id: int = 0, **model_kwargs) -> GenerateResult:
        """Greedy by default; returns per-step chosen-token logprobs.

        The FINAL decode_step's logits are discarded on purpose -- do not
        "optimise" the loop to skip it. It is what leaves the cache holding
        prompt_len + num_steps tokens, which is the end state the frozen
        geometry assumes: instance.yaml's num_decode_steps=8 takes a 12-token
        prompt to 20, and the last token therefore lands on a page the PROMPT
        never reached -- which is only true if the last step actually writes.
        Which page that is follows from page_size and is deliberately not named
        here: at the current page_size of 8 the prompt already spans pages 0-1
        and token 16 opens page 2. It also leaves the request ready to keep
        generating.

        `tokens` is int32, not torch's int64: jax_enable_x64 is off for the
        whole class, so int64 would be silently downcast with a warning. The
        VALUES are what the parity gate compares.

        `**model_kwargs` REACHES THE PREFILL AND NOTHING ELSE -- see
        :meth:`prefill`. What stays out is `continue_request` and
        `full_input_ids`: taking those here would advertise a chunk-by-chunk
        serving path this method does not have, which is true of those two
        parameters and says nothing whatever about the keywords. This
        method drives ONE un-chunked prompt through ONE prefill call -- exactly
        as `core`'s and `ragged`'s do, and both of those take the keywords -- and
        that call's `full_input_ids` already defaults to the whole prompt. So the
        chunking parameters stay out and the model keywords come in. A caller
        that really has a chunk-by-chunk path still drives :meth:`prefill`
        directly.

        WHY THAT IS NOT COSMETIC: the golden freezer reaches the engine ONLY
        through this method -- for both `generation.npz` and
        `fp32_generation.npz` -- and calls it `**self.mm_inputs`
        UNCONDITIONALLY -- `{pixel_values, image_grid_thw}` for any instance
        whose `instance.yaml` says `multimodal`, from one freezer class with no
        branch on mode. Without this parameter the first freeze of a chunked
        multimodal cell dies with `TypeError: Engine.generate() got an
        unexpected keyword argument 'pixel_values'` before either artifact is
        written, and `INPUTS_HOOK` above would be a hook no freezer call could
        reach.
        """
        params = params or SamplingParams()
        logits = self.prefill(input_ids, request_id, **model_kwargs).logits[-1]
        tokens, logprobs, per_step = [], [], []
        for _ in range(num_steps):
            # No `.clone()`: jax arrays are immutable, so the reference is the
            # snapshot torch has to copy for.
            per_step.append(logits)
            # The TOKEN comes from the processed distribution; the LOGPROB comes
            # from the raw one. That asymmetry is deliberate and it is vLLM v1's
            # default, logprobs_mode="raw_logprobs" (vllm/config/model.py) --
            # upstream exposes the alternative as a flag, so it is a convention
            # to match, not a detail to pick. It only shows up once the sampler
            # grid moves off the identity: at instance.yaml's
            # {temperature: 0.7, top_k: 50} the two differ by ~0.4 nats, and a
            # mismatch there reads as broken sampler math rather than as a
            # convention difference -- expensive to diagnose at the I1-9 gate.
            token = int(greedy(process_logits(jnp.expand_dims(logits, 0), params))[0])
            logprobs.append(
                float(jax.nn.log_softmax(logits.astype(jnp.float32), -1)[token]))
            tokens.append(token)
            logits = self.decode_step(request_id, token).logits[0]
        return GenerateResult(
            jnp.array(tokens, dtype=jnp.int32),
            jnp.array(logprobs, dtype=jnp.float32),
            per_step,
        )
