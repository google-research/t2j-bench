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

"""The prefill -> decode driver for the `core` mode.

This is the de-vendored stand-in for vllm/v1/worker/gpu_model_runner.py (7751
LOC) and the scheduler (3055 LOC), both far out of scope. What survives is the
part that defines the CONTRACT: build attention metadata, allocate pages,
compute the slot mapping, run the model, sample, append.

`core` is one sequence at a time -- request_distribution is pure-prefill then
pure-decode, never mixed. The `ragged` and `chunked` modes reuse this file with
a metadata builder that mixes stages; that is the only thing that changes.

POSITIONS ARE A WINDOW ONTO THE RUN THE WHOLE PROMPT DEFINES.
`core` and `ragged` delegate: they ask the model for its own `input_positions`
through `Engine.POSITIONS_HOOK`, so a family whose position is a TUPLE per token
(M-RoPE: t, h, w) supplies its own in per-family code and the shared engine
never learns a family name. THIS FILE DELEGATES TOO, and it needs one thing more
to do it: a chunk's positions are a WINDOW `[cached, seq_len)` onto a run the
whole prompt defines, and a continuation chunk is handed only the chunk, so
there is no sequence to give a positions builder. :meth:`Engine.prefill`
therefore takes the request's COMPLETE ids as `full_input_ids`, asks the builder
for the whole run, and slices this chunk out of it.

A model that defines NO builder gets `arange(cached, seq_len)`, which is the
identical array that slice would have returned -- so every text-only family,
which is every `chunked` instance shipped today, is bit-for-bit unaffected.

`default_input_positions` and `_input_positions` are the two pieces of the other
modes' delegation machinery that are STILL deliberately absent, and that is a
consequence rather than a leftover: this fork has to know whether the model has
a builder AT ALL (`getattr(..., None)`), because the un-plumbed continuation it
refuses by name is a shape only a hook family can be in, and a `getattr` with a
real fallback cannot answer that. `_advance` IS here now -- the decode cursor is
derived at the model's own rope rank rather than written at rank 1 -- and on the
JAX side so are `INPUTS_HOOK` / `_model_inputs`. This file hands its extra
keywords straight into the forward the way `core`'s does, and the JAX twin
prepares them on the host first because a tower written in `jnp` cannot be
traced; `generate` carries them too, because that is the only entry point
the offline freezing step uses.

WHAT THAT DOES AND DOES NOT MEAN, stated narrowly on purpose. THE ENGINE no
longer refuses a multimodal family: a keyword can reach the traced forward, a
chunk's positions are a window onto the run its full ids define, and the decode
cursor is a column at the model's own rope rank. Whether a full multimodal
instance runs end to end is a different question and this file cannot answer it
-- nothing here has been exercised against a real tower, no `chunked` instance
defines either model hook, and the harness has its own blockers. Do not read
this paragraph as a claim that a multimodal instance is complete.

There is still no `if multimodal` here, and no branch on a family, a model name
or an instance -- the delegation keys on whether the MODEL implements a protocol
method, which is the same capability lookup the other two modes make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from attention.metadata import AttentionMetadata
from cache.block_table import BlockPool
from cache.kv_cache import make_kv_cache
from cache.prefix_cache import PrefixCache
from config import RunConfig
from sample.sampler import SamplingParams, greedy, process_logits


@dataclass
class PrefillResult:
    hidden: torch.Tensor
    logits: torch.Tensor
    slot_mapping: torch.Tensor
    attn_metadata: AttentionMetadata
    page_ids: list[int]
    hidden_states: list[torch.Tensor] | None = None


@dataclass
class DecodeResult:
    hidden: torch.Tensor
    logits: torch.Tensor
    slot_mapping: torch.Tensor
    attn_metadata: AttentionMetadata


@dataclass
class GenerateResult:
    tokens: torch.Tensor
    step_logprobs: torch.Tensor
    logits_per_step: list[torch.Tensor] = field(default_factory=list)


class Engine:
    #: The method a model defines when its `input_positions` are NOT `arange`.
    #: On `core` and `ragged` this names a hook the engine calls, and HERE it
    #: does too -- :meth:`prefill` asks the named builder for the whole run and
    #: windows this chunk out of it. Held at the same string as the other two
    #: modes and as the offline freezing step's own POSITIONS_HOOK (which the
    #: harness checks across all seven instances), because a lookup that keys on
    #: a stale name finds nothing, falls back to `arange` and rotates every
    #: image token wrong.
    POSITIONS_HOOK = "get_input_positions"

    def __init__(self, model, run_config: RunConfig) -> None:
        self.model = model
        self.cfg = run_config
        cfg = model.config
        # ONE GROUP. This family's layers all agree on their KV geometry, so the
        # container's single array IS the rectangular cache this engine holds,
        # which is what keeps the frozen `kv_cache_state.npz` and
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
            getattr(torch, run_config.dtype),
        )
        self._pool = BlockPool(run_config.num_pages, run_config.page_size)
        # The single owner of shared pages ABOVE the pool (cache/prefix_cache.py's
        # own docstring). One instance per engine lifetime, rebuilt in reset()
        # alongside the pool it wraps -- see the comment there for why a stale
        # PrefixCache tied to a replaced pool would be worse than no cache at
        # all. Public (not `_prefix_cache`): test_engine's own suite and the
        # graded invariance test (test_behavioral.py) both read
        # `engine.prefix_cache.stats()["hits"]` as an honest count of REAL
        # cross-request sharing.
        self.prefix_cache = PrefixCache(self._pool, run_config.page_size)
        self._pages: dict[int, list[int]] = {}
        self._lens: dict[int, int] = {}
        # The NEXT position column each live request will use, at that model's
        # own rope rank -- `(1,)` for a text-only family, `(R, 1)` for one with
        # R rope sections. Kept beside `_lens` and not derived from it because
        # the two are different quantities the moment a family has M-RoPE: an
        # image occupies `max(h, w)` positions and one KV slot per token, so a
        # VL request's next POSITION is smaller than its next INDEX.
        self._next_pos: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self._pool = BlockPool(self.cfg.num_pages, self.cfg.page_size)
        # CONTENT tracking cannot survive: the old PrefixCache's `_by_hash`/
        # `_refcount`/`_owners` still claim specific physical page ids as
        # CACHED after the `kv_cache` zeroing below has erased their content and
        # the new `self._pool` above has forgotten they were ever allocated.
        # Keeping it would let a future `acquire()` hand back a page id that
        # is simultaneously FREE (in the new pool's free list) and OWNED (per
        # the stale hit record) -- the exact double-owner state
        # `allocator_reuses_live_page` exists to refuse, just reached through
        # the cache instead of through `_grow`.
        #
        # The hit/miss COUNTERS are a different kind of state: pure lifetime
        # diagnostics with no physical resource behind them, so nothing about
        # rebuilding the pool invalidates them. Carried over rather than
        # wiped with the rest -- the same choice this file already makes for
        # `_traces` on the JAX side, which `reset()` also leaves untouched.
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
        for group in self.kv_cache:
            group.zero_()

    @staticmethod
    def _advance(positions: torch.Tensor) -> torch.Tensor:
        """The position column ONE STEP PAST `positions`, at the same rank.

        "One past the furthest position any rope section has reached", which is
        rank-agnostic and is upstream's own rule: vLLM stores exactly this as
        `mrope_position_delta` (`MRotaryEmbedding.get_next_input_positions`,
        `max_position + 1 - len(input_tokens)`) and adds it to the token index
        for every decode step. The GLOBAL max and not a per-row one: a prompt
        that ENDS with an image leaves its three rows at different values, and
        upstream advances all of them from the same place.

        At rank 1 over the run `[0, seq_len)` it is `seq_len` -- exactly a
        rank-1 `torch.tensor([seq_len], dtype=torch.int32)`. Same shape, same
        dtype, same value, on every family whose
        positions are `arange`, which is every `chunked` instance shipped
        today.

        THE EMPTY CLAUSE IS THIS MODE'S, and `core`/`ragged` do not carry it.
        What is handed in here is THE RUN SO FAR, `[0, seq_len)`, not the
        chunk's window -- :meth:`prefill` says why -- so the clause is
        unreachable for a request that has any tokens at all. It is kept for
        the one call that has none: a `prefill` of a zero-token `input_ids`,
        which this engine accepts today and answers `[0]`, and which the rule
        above cannot answer because an empty run reached no position.

        `torch.max()` on a zero-element tensor
        raises `RuntimeError: max(): Expected reduction dim to be specified for
        input.numel() == 0`, and `jnp.max` on a zero-size array raises
        `ValueError: zero-size array to reduction operation max which has no
        identity` -- eagerly AND under `jax.jit`, because the check is on the
        shape, which is static. Neither side returns `-inf`. So the failure this
        clause replaces is a crash on both sides, not a silent wrong answer.
        """
        return torch.full((*positions.shape[:-1], 1),
                          int(positions.max()) + 1 if positions.numel() else 0,
                          dtype=torch.int32)

    def _metadata(self, positions, page_ids, seq_len, q_len, distribution):
        # `positions` IS PASSED THROUGH AT WHATEVER RANK IT ARRIVES: (T,) here,
        # and (R, T) for an M-RoPE family, where R is the number of rope
        # sections. Normalising the rank HERE would change input_positions'
        # frozen shape for every text-only instance, and that shape is graded;
        # layers/rotary.py is where the two ranks are made uniform, at the one
        # point that actually indexes the cos/sin table.
        #
        # Every tensor field is int32, which is the AttentionMetadata contract
        # test_metadata.py pins. This is the PRODUCTION source of the metadata --
        # what the offline freezing step reads -- so an int64 input_positions here would
        # be baked into the frozen scenario, and the JAX twin emits int32 under
        # jax_enable_x64=False. The graded test_attention_metadata_schema would
        # then fail on a dtype difference that means nothing.
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
        rows += [torch.zeros(self.cfg.pages_per_seq, dtype=torch.int32)] * pad
        return AttentionMetadata(
            input_positions=positions,
            block_tables=torch.cat(rows) if pad else rows[0],
            # A request slot nothing scheduled has no KV and contributes no query
            # token, so it pads with a zero length and with the running offset.
            seq_lens=torch.tensor([seq_len] + [0] * pad, dtype=torch.int32),
            query_start_loc=torch.tensor([0] + [q_len] * (pad + 1), dtype=torch.int32),
            request_distribution=torch.tensor(distribution, dtype=torch.int32),
            padded_num_reqs=self.cfg.max_num_seqs,
        )

    def _grow(self, request_id: int, new_len: int) -> list[int]:
        """Ensure the request owns enough pages for `new_len` tokens."""
        # Named, not incidental. Without this the overrun dies two frames down in
        # block_table_row as "RuntimeError: The expanded size of the tensor (4)
        # must match the existing size (5)", which says nothing about the cause.
        # paged_attention and BlockPool.free both refuse their own bad inputs by
        # name; the engine is the one layer that would otherwise opt out.
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

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, request_id: int = 0,
                output_hidden_states: bool = False,
                continue_request: bool = False,
                full_input_ids: torch.Tensor | None = None,
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

        `**model_kwargs` IS FORWARDED VERBATIM, to the positions builder and to
        the model, and is empty for every text-only family. The two get
        different ids: the builder is asked for the whole run (`full_ids`),
        while the model is handed this call's own `forward_ids` and the keywords
        beside them. A family whose FORWARD needs one -- a vision tower's
        `pixel_values` -- is therefore servable here. `decode_step` deliberately
        takes none: the tower runs once, at prefill, and its rows are already in
        the cache by the time a decode step reads them.
        """
        n = int(input_ids.shape[0])
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
        # three chunks produces ONE position where seven are required and the
        # prefill returns seven rows of logits anyway -- a wrong answer, not a
        # crash. Refusing this call by name is diagnosable, and this keeps that
        # property for the one caller shape that cannot work while leaving every
        # other one computing.
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
                input_ids.tolist(), request_id)
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
            positions = torch.arange(cached, seq_len, dtype=torch.int32)
        else:
            full = build(full_ids, **model_kwargs)
            positions = full[..., cached:seq_len]
        meta = self._metadata(positions, pages, seq_len, q_len, distribution)
        out = self.model(forward_ids, self.kv_cache, meta, slot_mapping,
                         output_hidden_states=output_hidden_states,
                         **model_kwargs)
        hidden, hs = out if output_hidden_states else (out, None)
        self._lens[request_id] = seq_len
        # THE CURSOR IS A PROPERTY OF THE WHOLE SEQUENCE, NOT OF THIS CHUNK'S
        # WINDOW, and it is now DERIVED rather than written at rank 1: the same
        # `_advance` helper `core` and `ragged` use, so an M-RoPE family gets
        # the `(R, 1)` column its rotary indexes with and a text-only one gets
        # the same rank-1 `[seq_len]` this line wrote before.
        #
        # THE ARGUMENT IS THE RUN SO FAR, `[0, seq_len)`, AND THAT IS THE WHOLE
        # RESOLUTION OF THE PROBLEM THAT KEPT THIS AT RANK 1. `max() + 1` over
        # THIS CHUNK'S WINDOW has no answer when the chunk is entirely a prefix
        # hit -- `q_len == 0`, nothing computed, `positions` empty, and
        # `torch.max()` on a zero-element tensor raises
        # (`test_a_first_chunk_that_is_entirely_a_prefix_hit` is exactly that
        # case; the JAX twin's `jnp.max` raises too). The RUN's prefix is empty only
        # when the request has no tokens at all, so the entirely-a-hit chunk is
        # answered from the positions its own earlier tokens already occupy
        # rather than from the nothing it forwarded. It is also the more
        # faithful rule: vLLM's `mrope_position_delta` is `max_position + 1`
        # over the WHOLE prompt, not over the last chunk of it.
        #
        # `full` is the whole run the request's ids define and can reach past
        # `seq_len` on any chunk but the last, so the slice is not decoration:
        # the cursor must be one past the furthest position reached SO FAR, not
        # one past the furthest the prompt will ever reach.
        self._next_pos[request_id] = self._advance(
            torch.arange(seq_len, dtype=torch.int32) if full is None
            else full[..., :seq_len])
        return PrefillResult(hidden, self.model.compute_logits(hidden),
                             slot_mapping, meta, list(pages), hs)

    @torch.no_grad()
    def decode_step(self, request_id: int, token_id: int) -> DecodeResult:
        pos = self._lens[request_id]
        new_len = pos + 1
        pages = self._grow(request_id, new_len)
        slot_mapping = self._pool.slot_mapping(pages, pos, 1)
        # THE SLOT COMES FROM THE TOKEN INDEX AND THE POSITION DOES NOT, and on a
        # text-only family that distinction is invisible because the two numbers
        # are equal. `slot_mapping` addresses the KV cache, which has one entry
        # per TOKEN; `input_positions` indexes the rope table, which a family
        # with M-RoPE advances by its own rule -- see `_advance` and the cursor
        # :meth:`prefill` derives with it, at the model's own rope rank.
        #
        # `positions + 1` BELOW IS RANK-AGNOSTIC: `_advance` hands back
        # `(*rope_sections, 1)`, and adding a
        # Python int broadcasts over it, so a `(3, 1)` M-RoPE column advances
        # all three sections by one exactly as a `(1,)` text column advances
        # its single one. Upstream does the same thing -- one delta added to
        # every section -- so this needed no change when the cursor gained a
        # rank.
        positions = self._next_pos[request_id]
        meta = self._metadata(positions, pages, new_len, 1, [1, 0, 0])
        hidden = self.model(torch.tensor([token_id]), self.kv_cache, meta, slot_mapping)
        self._lens[request_id] = new_len
        self._next_pos[request_id] = positions + 1
        return DecodeResult(hidden, self.model.compute_logits(hidden), slot_mapping, meta)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, num_steps: int,
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

        `**model_kwargs` REACHES THE PREFILL AND NOTHING ELSE -- see
        :meth:`prefill`. What stays out is `continue_request` and
        `full_input_ids`: taking those here would advertise a chunk-by-chunk
        serving path this method does not have. That reasoning applies to those
        two parameters and says nothing whatever about the model keywords. This
        method drives ONE un-chunked prompt through ONE prefill call -- exactly
        as `core`'s and `ragged`'s do, and both of those take the keywords -- and
        that call's `full_input_ids` already defaults to the whole prompt. So the
        chunking parameters stay out and the model keywords come in. A caller
        that really has a chunk-by-chunk path still drives :meth:`prefill`
        directly.

        WHY THAT IS NOT COSMETIC: the offline freezing step reaches the engine
        ONLY through this method and calls it with `**self.mm_inputs`
        UNCONDITIONALLY -- `{pixel_values, image_grid_thw}` for any instance
        whose `instance.yaml` says `multimodal`, from one freezing routine with
        no branch on mode. Without this parameter the first freeze of a chunked
        multimodal instance dies with `TypeError: Engine.generate() got an
        unexpected keyword argument 'pixel_values'` before either artifact
        (`generation.npz` / `fp32_generation.npz`) is written, and the keywords
        this file now splats into its forward would be keywords no call from the
        freezing step could deliver.
        """
        params = params or SamplingParams()
        logits = self.prefill(input_ids, request_id, **model_kwargs).logits[-1]
        tokens, logprobs, per_step = [], [], []
        for _ in range(num_steps):
            per_step.append(logits.clone())
            # The TOKEN comes from the processed distribution; the LOGPROB comes
            # from the raw one. That asymmetry is deliberate and it is vLLM v1's
            # default, logprobs_mode="raw_logprobs" (vllm/config/model.py) --
            # upstream exposes the alternative as a flag, so it is a convention
            # to match, not a detail to pick. It only shows up once the sampler
            # grid moves off the identity: at instance.yaml's
            # {temperature: 0.7, top_k: 50} the two differ by ~0.4 nats, and a
            # mismatch there reads as broken sampler math rather than as a
            # convention difference -- expensive to diagnose at the I1-9 gate.
            token = int(greedy(process_logits(logits.unsqueeze(0), params))[0])
            logprobs.append(float(torch.log_softmax(logits.float(), -1)[token]))
            tokens.append(token)
            logits = self.decode_step(request_id, token).logits[0]
        return GenerateResult(
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor(logprobs, dtype=torch.float32),
            per_step,
        )
