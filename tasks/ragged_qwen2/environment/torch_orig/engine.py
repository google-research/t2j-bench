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

POSITIONS ARE THE MODEL'S TO BUILD, NOT THIS FILE'S -- see
:func:`default_input_positions` and :meth:`Engine._input_positions`. There is
deliberately NO `if multimodal` here, and no branch on a family, a model name or
an instance: a branch would make this shared file per-family, which is the cost
model the whole class rests on. What replaces it is delegation with a shared
default, so a family whose position is a scalar per token gets `arange` for free
and a family whose position is a TUPLE per token (M-RoPE: t, h, w) supplies its
own in per-family code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from attention.metadata import AttentionMetadata
from cache.block_table import BlockPool
from cache.kv_cache import make_kv_cache
from config import RunConfig
from sample.sampler import SamplingParams, greedy, process_logits


def default_input_positions(input_ids: torch.Tensor, **_) -> torch.Tensor:
    """``(num_tokens,)`` ascending positions -- the shared, text-only default.

    ONE ROPE SECTION, one position per token, starting at 0. That is every
    family whose position is a scalar; `layers/rotary.py` normalises this to
    ``(1, num_tokens)`` at the one point that indexes the cos/sin table, so the
    rank-1 result here is bit-identical to what a single-section rope computed
    before `positions` grew a section axis at all.

    ``**_`` SWALLOWS THE MULTIMODAL KEYWORDS ON PURPOSE. :meth:`Engine.prefill`
    forwards whatever extra keywords it was handed to BOTH the positions builder
    and the model, without inspecting them -- that opacity is what keeps the
    engine family-agnostic -- so a text-only default has to accept and ignore
    the ones it has no use for rather than make the engine decide which builder
    wants which keyword.

    int32, which is `AttentionMetadata`'s contract (`Engine._metadata`).
    """
    return torch.arange(int(input_ids.shape[0]), dtype=torch.int32)


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
    #: The OPTIONAL method a model may define to build its own
    #: `input_positions`. Resolved by NAME, with :func:`default_input_positions`
    #: as the fallback, so this file never mentions a family, a model class or a
    #: `multimodal` fact -- see :meth:`_input_positions`. Read by
    #: the offline golden-authoring process, which has to build the same
    #: positions the engine would.
    POSITIONS_HOOK = "get_input_positions"

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
            getattr(torch, run_config.dtype),
        )
        self._pool = BlockPool(run_config.num_pages, run_config.page_size)
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
        self._pages.clear()
        self._lens.clear()
        self._next_pos.clear()
        # EVERY group, not just the first. One group today, so a `reset` that
        # reached only `kv_cache[0]` would be green on all thirteen instances
        # and silently wrong for the first family that allocates two.
        for group in self.kv_cache:
            group.zero_()

    def _input_positions(self, input_ids: torch.Tensor, **model_kwargs):
        """Ask the MODEL for its positions; fall back to the shared default.

        DELEGATION, NOT A BRANCH. `getattr(obj, name, default)` is a protocol
        lookup: it asks whether this model implements the hook, and it names no
        family, no model class, no instance and no `multimodal` flag. A model
        that does not implement it gets :func:`default_input_positions`, which
        is why the four text-only families needed no per-family code and why
        their frozen `input_positions` did not move.

        The multimodal keywords go to the builder UNINSPECTED, the same set that
        goes to the model itself: the engine cannot know that `image_grid_thw`
        is the one that shapes positions while `pixel_values` is not, and a
        table here saying so would be the family knowledge this design exists to
        keep out of the shared column.
        """
        build = getattr(self.model, self.POSITIONS_HOOK, default_input_positions)
        return build(input_ids, **model_kwargs)

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

        At rank 1 it is `positions.max() + 1 == num_tokens`, i.e. exactly the
        `torch.tensor([self._lens[request_id]])` this replaced -- same shape,
        same dtype, same value.
        """
        return torch.full((*positions.shape[:-1], 1), int(positions.max()) + 1,
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
        # the harness pins. This is the PRODUCTION source of the metadata --
        # what the golden-authoring process reads -- so an int64 input_positions here would
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
        # name; the engine is the one layer that used to opt out.
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
                **model_kwargs) -> PrefillResult:
        """`**model_kwargs` is FORWARDED VERBATIM, to the positions builder and
        to the model, and is empty for every text-only family.

        It is how a multimodal family gets its `pixel_values` / `image_grid_thw`
        to a model that declares them, without this file learning either name.
        `decode_step` deliberately takes none: the tower runs once, at prefill,
        and its rows are already in the cache by the time a decode step reads
        them -- re-running it per step would be both wrong and expensive.
        """
        n = int(input_ids.shape[0])
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
        out = self.model(input_ids, self.kv_cache, meta, slot_mapping,
                         output_hidden_states=output_hidden_states, **model_kwargs)
        hidden, hs = out if output_hidden_states else (out, None)
        self._lens[request_id] = n
        self._next_pos[request_id] = self._advance(positions)
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
        # with M-RoPE advances by its own rule (`_advance`).
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

        `**model_kwargs` reaches the PREFILL and nothing else -- see
        :meth:`prefill`.
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
            # convention difference -- expensive to diagnose.
            token = int(greedy(process_logits(logits.unsqueeze(0), params))[0])
            logprobs.append(float(torch.log_softmax(logits.float(), -1)[token]))
            tokens.append(token)
            logits = self.decode_step(request_id, token).logits[0]
        return GenerateResult(
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor(logprobs, dtype=torch.float32),
            per_step,
        )
