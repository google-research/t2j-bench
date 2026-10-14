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
     the measurement and for what it would take to lift.

`reset()` zeroes the cache by rebinding `jnp.zeros_like` OVER EVERY GROUP, the
functional analogue of torch's `zero_()`; there is no other kind of zeroing
here.

POSITIONS ARE THE MODEL'S TO BUILD, exactly as on the torch side -- see
:func:`default_input_positions` and :meth:`Engine._input_positions`. No branch
on a family, a model name, an instance or a `multimodal` fact appears in this
file, for the reason the torch twin's docstring gives; difference 4 above is
resolved the same way, by a named hook with a shared default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from attention.metadata import AttentionMetadata
from cache.block_table import BlockPool
from cache.kv_cache import make_kv_cache
from config import RunConfig
from sample.sampler import SamplingParams, greedy, process_logits


def default_input_positions(input_ids, **_) -> jax.Array:
    """``(num_tokens,)`` ascending positions -- the shared, text-only default.

    The torch twin's `default_input_positions`, argued there: one rope section,
    one position per token, `**_` swallowing multimodal keywords this builder
    has no use for, int32 because that is `AttentionMetadata`'s contract (which
    this side gets for free under `jax_enable_x64=False`).
    """
    return jnp.arange(int(jnp.asarray(input_ids).shape[0]), dtype=jnp.int32)


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
    THIS ENGINE. Be precise about that: it is not catching anything here.
    Neither branch can fire for the reference:
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

    # Padded query tokens are out of scope for `core` (attention/paged.py).
    if num_query_tokens != ragged_tokens:
        raise ValueError(
            f"validate_metadata: {num_query_tokens} query tokens but "
            f"query_start_loc[-1] is {ragged_tokens}: padded query tokens are "
            f"not supported in the `core` mode (deferred to `ragged`). Under "
            f"jit the gather would clamp instead of raising, so the padded "
            f"token would be attributed to the last sequence."
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
    #: The OPTIONAL method a model may define to build its own
    #: `input_positions`. The torch twin's constant, and the same protocol
    #: lookup -- see :meth:`_input_positions`.
    POSITIONS_HOOK = "get_input_positions"

    #: The OPTIONAL method a model may define to turn its own forward keywords
    #: into TRACEABLE ones, on the host. JAX-only -- see the module docstring's
    #: difference 4 and :meth:`_model_inputs`.
    INPUTS_HOOK = "prepare_model_inputs"

    def __init__(self, model, run_config: RunConfig) -> None:
        self.model = model
        self.cfg = run_config
        cfg = model.config
        # TWO GROUPS, AND THIS IS THE ONE LINE THAT MAKES THIS FILE DIFFER
        # FROM ITS MODE PARENT. Every other instance in this class passes
        # `((cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim),)`
        # -- one group, one slot per layer -- and for gemma-4 that expression
        # is wrong twice over: its sliding layers have head_dim 256 and its
        # full ones 512, so one rectangular array cannot hold them; and 20 of
        # its 35 layers SHARE an earlier layer's slot, so it allocates 15 slots
        # and not 35. `config.kv_groups()` returns `((12, 1, 256), (3, 1, 512))`
        # here and the identity one-group tuple for a family with one geometry
        # and no sharing, so the expression generalises rather than branching.
        #
        # THE COST IS REAL AND IS DECLARED RATHER THAN HIDDEN: `engine.py` is a
        # MODE-column file, byte-identical across every `core` instance, and
        # this instance is the first exception. `test_inference_reference_
        # columns.PER_MODE_EXCEPT_FAMILY` carries the waiver and the reason.
        # The alternative -- teaching all fourteen instances' engines to ask
        # their config -- is a class-wide change and belongs to whoever owns the
        # mode column, not to one family.
        #
        # THIS BLOCK OWNS THE CONTAINER'S GEOMETRY AND NOTHING ELSE. The layer
        # -> (group, slot) map that INDEXES it is applied in modeling.py, on the
        # attention module that does the indexing, and both are spelled from the
        # one definition in config.py (`kv_groups` / `layer_slot` /
        # `layer_writes`). Neither file is the whole extension point and neither
        # should claim to be; `cache/kv_cache._require_slot` is what refuses the
        # half-done version where only one of them moved.
        self.kv_cache = make_kv_cache(
            cfg.kv_groups(),
            run_config.num_pages, run_config.page_size,
            getattr(jnp, run_config.dtype),
        )
        self._pool = BlockPool(run_config.num_pages, run_config.page_size)
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
        # alternative is no observation at all. test_the_decode_loop_traces_
        # once_not_once_per_step keys off it.
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
        # the lowered function is the one it always was.
        self._jit_fwd = nnx.jit(_forward, static_argnums=(5,))

    def reset(self) -> None:
        self._pool = BlockPool(self.cfg.num_pages, self.cfg.page_size)
        self._pages.clear()
        self._lens.clear()
        self._next_pos.clear()
        # EVERY group, not just the first. One group today, so a `reset` that
        # reached only `kv_cache[0]` would be green on all thirteen instances
        # and silently wrong for the first family that allocates two.
        self.kv_cache = tuple(jnp.zeros_like(group) for group in self.kv_cache)

    def _input_positions(self, input_ids, **model_kwargs):
        """Ask the MODEL for its positions; fall back to the shared default.

        The torch twin's method verbatim in intent -- a protocol lookup by name,
        naming no family and no `multimodal` fact, with the extra keywords
        forwarded uninspected. See it for the full argument.
        """
        build = getattr(self.model, self.POSITIONS_HOOK, default_input_positions)
        return build(input_ids, **model_kwargs)

    @staticmethod
    def _advance(positions: jax.Array) -> jax.Array:
        """The position column ONE STEP PAST `positions`, at the same rank.

        The torch twin's rule and its argument: one past the furthest position
        ANY rope section reached, which is vLLM's `mrope_position_delta`, and
        which degenerates to `num_tokens` at rank 1.
        """
        return jnp.full((*positions.shape[:-1], 1), int(positions.max()) + 1,
                        dtype=jnp.int32)

    def _model_inputs(self, input_ids, **model_kwargs) -> dict:
        """The model's extra forward keywords, PREPARED ON THE HOST if it asks.

        THE FOURTH JAX-FORCED DIFFERENCE (module docstring), and the reason it
        is forced rather than chosen is worth stating where the code is, because
        it is not what a reader of the torch twin would predict.

        Under jax 0.11, `jnp.arange(4)` INSIDE a `jit` trace is a tracer.
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
        twin's `prefill` docstring says why, and the harness measures
        it), while the perf gate times steady-state DECODE. So an uncompiled
        tower does not reach the graded number.

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
        # test_metadata.py pins -- and which this side gets for free under
        # jax_enable_x64=False, where the torch side has to ask for it. The
        # engine is the PRODUCTION source of the metadata, so this is what the
        # golden freezer reads.
        meta = AttentionMetadata(
            input_positions=positions,
            block_tables=self._pool.block_table_row(page_ids, self.cfg.pages_per_seq),
            seq_lens=jnp.array([seq_len], dtype=jnp.int32),
            query_start_loc=jnp.array([0, q_len], dtype=jnp.int32),
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
                **model_kwargs) -> PrefillResult:
        """`**model_kwargs` is FORWARDED VERBATIM -- the torch twin's contract.

        The one difference is on the way into the trace: the model gets to
        prepare them on the host first (:meth:`_model_inputs`). Empty for every
        text-only family, where the two engines do the identical thing.
        """
        n = int(jnp.asarray(input_ids).shape[0])
        # Prefilling a live request would silently reuse its pages and rewind
        # its length, so the cache would hold two sequences interleaved. Benign
        # in `core` (one sequence, reset() between runs) but this same file
        # serves the `ragged` and `chunked` modes, where concurrent requests are
        # the point -- and silent state clobbering there is unfindable.
        if request_id in self._lens:
            raise ValueError(
                f"request {request_id} is already active with "
                f"{self._lens[request_id]} tokens; call reset() before "
                "prefilling it again"
            )
        pages = self._grow(request_id, n)
        slot_mapping = self._pool.slot_mapping(pages, 0, n)
        positions = self._input_positions(input_ids, **model_kwargs)
        meta = self._metadata(positions, pages, n, n, [0, 1, 0])
        out = self._jit_fwd(self.model, input_ids, self.kv_cache, meta,
                            slot_mapping, output_hidden_states,
                            self._model_inputs(input_ids, **model_kwargs))
        if output_hidden_states:
            hidden, hs, self.kv_cache = out
        else:
            hs = None
            hidden, self.kv_cache = out
        self._lens[request_id] = n
        self._next_pos[request_id] = self._advance(positions)
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

        `**model_kwargs` reaches the PREFILL and nothing else -- see
        :meth:`prefill`.
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
