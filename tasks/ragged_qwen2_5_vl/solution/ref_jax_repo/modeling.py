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

"""Inference-only Qwen2.5-VL in flax nnx, mirroring
``environment/torch_orig/modeling.py``.

Structurally the same file: same classes, same attribute names, same order of
operations. Read the torch twin for the upstream provenance (TWO vLLM files,
because Qwen2.5-VL's decoder literally IS Qwen2.5's) and for the two naming
decisions this file inherits. They are restated here because they bind the
PORT, not the reference:

  1. THE TEXT STACK HANGS AT ``model.`` AND ``lm_head``, NOT AT
     ``language_model.``. vLLM's wrapper owns a ``language_model`` attribute
     only because ``init_vllm_registered_model`` returns a whole registered
     ``Qwen2ForCausalLM``, and vLLM's own loader has to RENAME the checkpoint to
     invent the level. The shipped checkpoint spells it ``model.layers.N.*`` /
     ``visual.*`` -- 824 keys -- and ``canonical_param_key``
     normalises no intermediate path component, so keeping the checkpoint's own
     spelling is what makes ``test_param_correspondence``'s stated contract true
     for this family. Consequence: this family's reference-unit paths are
     ``core_llama3``'s unchanged, and ``model/loader.py`` needs no rename table.
  2. THE TOWER HANGS AT ``visual.``, with vLLM's and the checkpoint's own
     internals. There the two upstreams agree with each other and with the
     checkpoint, so there was nothing to decide. It lives in ``vision.py`` -- a
     declared partition violation, argued there.

Three differences from the torch twin, all of them forced by JAX and all of them
the point of the exercise:

  1. ``nn.Module`` -> ``nnx.Module``, ``forward`` -> ``__call__``,
     ``nn.ModuleList`` -> ``nnx.data([...])``.
  2. EVERY METHOD THAT TOUCHES THE CACHE RETURNS IT. jax arrays are immutable,
     so ``write_kv`` produces a new array and the caller has to thread it: the
     attention returns ``(out, kv_cache)``, the decoder layer returns
     ``(hidden, residual, kv_cache)``, and both models return the cache last.
     torch_orig hides this inside an in-place scatter; a port that keeps the
     torch signature silently drops every cache write.
  3. ``inputs_embeds[mask] = image_embeds`` is an in-place masked write on the
     torch side and a ``.at[rows].set(...)`` here -- see
     :meth:`Qwen2_5_VLForConditionalGeneration.get_input_embeddings`.

Unchanged, because they are the graded contract:
  * K is written to the cache AFTER ``rotary_emb``; V is written un-rotated.
  * ``lm_head.weight`` IS ``model.embed_tokens.weight`` -- the same nnx.Param
    object, not a copy -- when ``config.tie_word_embeddings``.
  * The forward is token-flat: [T, H], never [B, L, H].
  * ``mrope_section`` reaches ``RotaryEmbedding`` from the config, so
    ``attn_metadata.input_positions`` must arrive as ``(3, num_tokens)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from attention.metadata import AttentionMetadata
from attention.paged import paged_attention
from cache.kv_cache import (
    _require_slot,
    identity_layer_slot,
    identity_layer_writes,
    write_kv,
)
from config import ModelConfig
from layers.activation import SiluAndMul
from layers.layernorm import RMSNorm
from layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding
from vision import Qwen2_5_VisionTransformer


class Qwen2MLP(nnx.Module):
    def __init__(self, config: ModelConfig, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False,
            dtype=dtype, rngs=rngs,
        )
        self.act_fn = SiluAndMul()

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2Attention(nnx.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.layer_idx = layer_idx
        # WHERE THIS LAYER'S KV LIVES IN THE CACHE CONTAINER, resolved once
        # here rather than re-derived at every forward. The identity for every
        # family shipped today: group 0, slot == layer_idx, and this layer owns
        # the slot it reads.
        #
        # THIS IS WHERE THE MAP IS APPLIED; engine.py declares the container's
        # GEOMETRY, and cache/kv_cache.py holds the one definition both spell
        # it from. gemma4-E2B moves both: its sliding and full layers land in
        # different groups because their head_dims differ, and its KV-SHARED
        # layers get `kv_writes = False` so they attend another layer's slot
        # without overwriting it. Moving only one of the two is what
        # `_require_slot` refuses at the read site below.
        self.kv_group, self.kv_slot = identity_layer_slot(
            config.num_hidden_layers)[layer_idx]
        self.kv_writes = identity_layer_writes(
            config.num_hidden_layers)[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.num_heads,
            self.num_kv_heads,
            # Qwen2's QKV bias is ARCHITECTURAL, not configurable: Qwen2.5's
            # config.json carries no `attention_bias` key at all, so reading
            # config.attention_bias here would silently build a bias-free QKV
            # and leave three tensors of the checkpoint unloaded. vLLM's
            # Qwen2Attention hardcodes bias=True the same way
            # (vllm/model_executor/models/qwen2.py, at the pinned SHA), and
            # environment/torch_orig/modeling.py mirrors it identically --
            # test_param_correspondence grades the two key sets against each
            # other, so they must agree by construction.
            bias=True,
            dtype=dtype, rngs=rngs,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False,
            dtype=dtype, rngs=rngs,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            rope_scaling=config.rope_scaling,
            # THE ONE LINE THAT MAKES THIS DECODER M-RoPE. Everything else in
            # this class is core_qwen2's verbatim. `mrope_section` is
            # `[16, 24, 24]` for this checkpoint -- three sections partitioning
            # the 64-wide (head_dim // 2) frequency band between the t, h and w
            # position rows -- and `attn_metadata.input_positions` must
            # therefore arrive as `(3, num_tokens)`. `layers/rotary.py` REFUSES a
            # row count that disagrees with the section count rather than
            # broadcasting, so a rank-1 `positions` fails loudly here instead of
            # producing a plausible, wrongly-rotated model. That refusal is a
            # SHAPE check, so it fires under jit too.
            #
            # `ModelConfig.mrope_section` returns None for a checkpoint with no
            # such key, which is the text-only default.
            mrope_section=config.mrope_section,
        )

    def __call__(self, hidden: jax.Array, kv_cache: tuple[jax.Array, ...],
                 attn_metadata: AttentionMetadata, slot_mapping: jax.Array
                 ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        t = hidden.shape[0]
        q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        q = q.reshape(t, self.num_heads, self.head_dim)
        k = k.reshape(t, self.num_kv_heads, self.head_dim)
        v = v.reshape(t, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(attn_metadata.input_positions, q, k)

        # BEFORE the index, because JAX would not complain about a bad one: it
        # CLAMPS the read and DROPS the write where torch raises, so a
        # half-generalised family is a wrong number here and a crash there. See
        # cache/kv_cache._require_slot for the measurement.
        _require_slot(kv_cache, self.kv_group, self.kv_slot)

        # Post-RoPE K and pre-RoPE V go into the cache -- the order matters and
        # is the classic port bug: caching pre-RoPE K silently breaks decode.
        group = kv_cache[self.kv_group]
        if self.kv_writes:
            group = group.at[self.kv_slot].set(
                write_kv(group[self.kv_slot], k, v, slot_mapping)
            )
            kv_cache = (kv_cache[:self.kv_group] + (group,)
                        + kv_cache[self.kv_group + 1:])

        attn = paged_attention(
            q, group[self.kv_slot], attn_metadata,
            num_kv_heads=self.num_kv_heads, sm_scale=self.scaling,
        )
        return self.o_proj(attn.reshape(t, self.num_heads * self.head_dim)), kv_cache


class Qwen2DecoderLayer(nnx.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.self_attn = Qwen2Attention(config, layer_idx, dtype=dtype, rngs=rngs)
        self.mlp = Qwen2MLP(config, dtype=dtype, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                       dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps, dtype=dtype)

    def __call__(self, hidden, residual, kv_cache, attn_metadata, slot_mapping):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, kv_cache = self.self_attn(hidden, kv_cache, attn_metadata, slot_mapping)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual, kv_cache


class Qwen2Model(nnx.Module):
    def __init__(self, config: ModelConfig, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size,
                                                   dtype=dtype, rngs=rngs)
        # nnx.data, not a bare list: flax 0.12 classifies every attribute as
        # static structure or graph data, and a bare list of modules is refused
        # as "data on a static attribute".
        self.layers = nnx.data([
            Qwen2DecoderLayer(config, i, dtype=dtype, rngs=rngs)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def __call__(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                 output_hidden_states=False, inputs_embeds=None):
        # `inputs_embeds` IS THE MULTIMODAL ENTRY POINT and is upstream's own
        # (qwen2.py:397-410): the wrapper embeds the text ids, scatters the
        # image embeddings over the placeholder rows, and hands the result down.
        # When it is None this is byte-for-byte core_qwen2's forward.
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        collected = [hidden] if output_hidden_states else None
        residual = None
        for layer in self.layers:
            hidden, residual, kv_cache = layer(hidden, residual, kv_cache,
                                               attn_metadata, slot_mapping)
            if output_hidden_states:
                collected.append(hidden + residual)
        hidden, _ = self.norm(hidden, residual)
        if output_hidden_states:
            collected[-1] = hidden
            return hidden, collected, kv_cache
        return hidden, kv_cache


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    """The wrapper: a Qwen2.5 decoder, a ViT tower, and the scatter between them.

    ``...ForConditionalGeneration``, not ``...ForCausalLM``: the checkpoint's own
    ``architectures`` says so, vLLM's class says so, and it is the first model in
    this class whose head is a wrapper around a text decoder PLUS an encoder.
    The harness reads that name, so it is
    the one string here that other files depend on literally.
    """

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs | None = None, *,
                 dtype=jnp.float32) -> None:
        self.config = config
        # `norm_eps` comes from the TEXT config, exactly as upstream does it
        # (`getattr(config, "rms_norm_eps", 1e-6)`): the vision_config carries no
        # eps of its own and HF hardcodes 1e-6 in Qwen2_5_VLRMSNorm. Both are
        # 1e-6 for this checkpoint, so the choice is invisible today and would
        # not be for a family that retunes it.
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config, norm_eps=config.rms_norm_eps,
            dtype=dtype, rngs=rngs)
        self.model = Qwen2Model(config, dtype=dtype, rngs=rngs)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                      dtype=dtype, rngs=rngs)
        if config.tie_word_embeddings:
            # The SAME nnx.Param object, not a copy. flax's graph dedups shared
            # references, so `lm_head.weight` disappears from the parameter
            # pytree exactly as it disappears from torch's named_parameters(),
            # and a write through either name is visible through both.
            self.lm_head.weight = self.model.embed_tokens.weight

    # -- multimodal ---------------------------------------------------------#

    def get_image_embeds(self, pixel_values: jax.Array,
                         image_grid_thw) -> jax.Array:
        """``[num_merged_tokens, hidden_size]``, one row per ``<|image_pad|>``.

        Upstream's ``_process_image_input`` additionally splits the result per
        image with ``grid_thw.prod(-1) // merge_size ** 2``; the split exists so
        the model runner can attribute rows to requests, and the very next thing
        it does is flatten them again for the scatter. Nothing here attributes
        rows to requests, so the split is dropped and the concatenated form is
        returned -- which is also what makes :meth:`get_input_embeddings` a
        single scatter.
        """
        return self.visual(pixel_values, image_grid_thw)

    def get_input_embeddings(self, input_ids: jax.Array,
                             image_embeds: jax.Array | None = None
                             ) -> jax.Array:
        """Text embeddings with the tower's rows written over the placeholders.

        De-vendored from ``vllm/model_executor/models/utils.py``
        ``_merge_multimodal_embeddings``: a boolean mask over
        ``input_ids == image_token_id`` and one masked assignment.

        THE ASSIGNMENT IS THE ONE LINE THE TWO SIDES CANNOT SHARE. torch writes
        ``inputs_embeds[mask] = image_embeds`` in place; a boolean mask has a
        data-dependent output shape, which JAX refuses, so the placeholder rows
        are resolved to explicit indices first. ``jnp.nonzero`` is given a static
        ``size`` -- the tower's own row count, which the check above has just
        proved equals the placeholder count -- so the gather stays a fixed shape
        and the two sides write the same rows in the same order.

        The count check is upstream's too, promoted from a ``RuntimeError``
        rescue to a precondition: a prompt whose placeholder count disagrees with
        the grid is a scenario-construction bug, and neither framework's own
        message for it names the image or the prompt. It reads ``int(mask.sum())``
        on the host, exactly as the torch side does -- this whole method is
        called once per request, before the decode loop, and never from inside
        ``Engine``'s jitted forward (which passes neither ``pixel_values`` nor
        ``inputs_embeds``).
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        if image_embeds is None:
            return inputs_embeds
        mask = input_ids == self.config.image_token_id
        placeholders = int(mask.sum())
        if placeholders != int(image_embeds.shape[0]):
            raise ValueError(
                f"the prompt carries {placeholders} <|image_pad|> "
                f"(id {self.config.image_token_id}) placeholders but the tower "
                f"produced {int(image_embeds.shape[0])} embeddings. Each merged "
                "vision token needs exactly one placeholder: check that the "
                "prompt was built from the SAME image_grid_thw the processor "
                "returned.")
        rows = jnp.nonzero(mask, size=placeholders)[0]
        return inputs_embeds.at[rows].set(
            image_embeds.astype(inputs_embeds.dtype))

    def prepare_model_inputs(self, input_ids: jax.Array, *, pixel_values=None,
                             image_grid_thw=None, **rest) -> dict:
        """THE ENGINE'S HOST-SIDE HOOK (``engine.Engine.INPUTS_HOOK``).

        RUNS THE TOWER HERE, EAGERLY, and hands the jitted forward an
        already-merged ``inputs_embeds`` in place of the two image keywords.
        This method has no torch counterpart and is not a JAX-side liberty --
        see ``engine.Engine._model_inputs`` for the measurement that forces it:
        on jax 0.11 every array created inside a ``jit`` trace is a tracer, so
        ``vision.py``'s window permutation (:meth:`window_index_thw`), its
        per-chunk attention bounds (:class:`Qwen2_5_VisionAttention`) and its
        rotary range check (:meth:`Qwen2_5_VisionRotaryEmbedding.__call__`) --
        all of them Python control flow over ``jnp`` intermediates -- cannot
        execute under the trace. Marking ``image_grid_thw`` static does not
        rescue them: the arrays derived from it are still born inside the trace.

        THE TOWER IS NOT COMPILED; IT IS NOT UNCOMPILABLE. Every one of those
        reads is a function of ``image_grid_thw`` ALONE and never of
        ``pixel_values``, and ``np.arange`` inside a trace is concrete where
        ``jnp.arange`` is not -- so a tower that did its geometry in
        ``numpy`` over the static grid would trace. That is a rewrite of
        ``vision.py``, whose shape is graded against upstream's, and it is
        deliberately not attempted for a cost that lands on PREFILL only: the
        tower runs once per request, never in the decode loop, and the perf gate
        times steady-state decode.

        The torch twin needs none of this because nothing there is traced; its
        ``__call__`` runs the tower inline, which is upstream's own shape, and
        this method reaches the SAME model code by the other entry
        (``__call__``'s ``inputs_embeds`` branch) one call earlier. So the two
        sides compute the same thing in the same order; only the boundary
        between host and device moves.

        A caller that passes no image gets its keywords back unchanged, which is
        the shared default's behaviour and what makes a text-only prompt through
        this family cost nothing extra.
        """
        if pixel_values is None:
            return dict(rest)
        if image_grid_thw is None:
            raise ValueError(
                "pixel_values was given without image_grid_thw. The tower "
                "cannot infer a grid from a flat patch list -- the two come "
                "out of the processor together.")
        return {**rest, "inputs_embeds": self.get_input_embeddings(
            input_ids, self.get_image_embeds(pixel_values, image_grid_thw))}

    def get_input_positions(self, input_ids: jax.Array, *,
                            image_grid_thw=None, **_) -> jax.Array:
        """THE ENGINE'S HOOK (``engine.Engine.POSITIONS_HOOK``). M-RoPE here.

        The torch twin's method, argued there: a family-neutral name the shared
        engine resolves by string so that it never learns this family's
        position is a (t, h, w) triple, delegating to upstream's own
        ``get_mrope_input_positions``; ``**_`` absorbs the ``pixel_values`` the
        engine forwards to every builder without inspecting it.
        """
        return self.get_mrope_input_positions(input_ids, image_grid_thw)

    def get_mrope_input_positions(self, input_ids: jax.Array,
                                  image_grid_thw=None) -> jax.Array:
        """``(3, num_tokens)`` M-RoPE positions: the t, h and w row per token.

        De-vendored from ``get_mrope_input_positions``, with one deliberate
        change of INPUT: upstream reads the image spans off ``mm_features``, a
        scheduler object that does not exist here, so the spans are recovered
        from the ids instead -- each maximal run of ``image_token_id`` is one
        image, in prompt order, which is precisely what the placeholder
        expansion guarantees.

        THE RULE, in one sentence: a text token advances all three rows by one;
        an image contributes its (t, h, w) grid indices, so the whole image
        occupies ``max(h, w)`` positions rather than one per token, and the text
        after it resumes from ``max + 1``. That is why a VL prompt's last
        position is generally larger than its length, and why this cannot be
        replaced by ``broadcast_to(arange(n), (3, n))`` for anything but a
        text-only prompt (where it reduces to exactly that -- and where it is
        also bit-identical to the rank-1 rope, because three equal rows select
        the same frequency band the single-section default does).

        Returned as int32, matching ``AttentionMetadata``'s contract that every
        tensor field is int32 -- see ``engine.py::_metadata``.
        """
        ids = [int(i) for i in input_ids.reshape(-1).tolist()]
        merge = self.config.vision_config.spatial_merge_size
        grids = [] if image_grid_thw is None else [
            tuple(int(v) for v in entry) for entry in image_grid_thw]

        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(ids):
            if ids[index] == self.config.image_token_id:
                start = index
                while index < len(ids) and ids[index] == self.config.image_token_id:
                    index += 1
                spans.append((start, index - start))
            else:
                index += 1
        if len(spans) != len(grids):
            raise ValueError(
                f"the prompt has {len(spans)} run(s) of <|image_pad|> but "
                f"{len(grids)} grid(s) were supplied. Positions cannot be built "
                "for an image whose geometry is unknown.")

        rows: list[jax.Array] = []
        seen = 0
        for (offset, length), (grid_t, grid_h, grid_w) in zip(spans, grids):
            llm_t, llm_h, llm_w = grid_t, grid_h // merge, grid_w // merge
            if llm_t * llm_h * llm_w != length:
                raise ValueError(
                    f"grid {(grid_t, grid_h, grid_w)} merges to "
                    f"{llm_t * llm_h * llm_w} tokens but the prompt reserves "
                    f"{length} placeholders for it.")
            text_len = offset - seen
            start = int(rows[-1].max()) + 1 if rows else 0
            rows.append(jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32), (3, text_len)) + start)
            t_idx = jnp.broadcast_to(
                jnp.arange(llm_t).reshape(llm_t, 1, 1), (llm_t, llm_h, llm_w))
            h_idx = jnp.broadcast_to(
                jnp.arange(llm_h).reshape(1, llm_h, 1), (llm_t, llm_h, llm_w))
            w_idx = jnp.broadcast_to(
                jnp.arange(llm_w).reshape(1, 1, llm_w), (llm_t, llm_h, llm_w))
            grid = jnp.stack([t_idx.reshape(-1), h_idx.reshape(-1),
                              w_idx.reshape(-1)]).astype(jnp.int32)
            rows.append(grid + text_len + start)
            seen = offset + length

        if seen < len(ids):
            text_len = len(ids) - seen
            start = int(rows[-1].max()) + 1 if rows else 0
            rows.append(jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32), (3, text_len)) + start)
        return jnp.concatenate(rows, axis=1)

    # -- forward ------------------------------------------------------------#

    def __call__(self, input_ids, kv_cache, attn_metadata, slot_mapping=None,
                 output_hidden_states=False, pixel_values=None,
                 image_grid_thw=None, inputs_embeds=None):
        """The engine's entry point, widened by three optional keywords.

        The first four positional parameters and ``output_hidden_states`` are the
        contract ``engine.py`` calls and every other family in this class
        implements. ``pixel_values`` / ``image_grid_thw`` run the tower and merge
        its output; ``inputs_embeds`` skips both and takes an already-merged
        table, which is how a caller that ran the tower once can decode without
        re-running it.
        """
        if slot_mapping is None:
            slot_mapping = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
        if inputs_embeds is None and pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError(
                    "pixel_values was given without image_grid_thw. The tower "
                    "cannot infer a grid from a flat patch list -- the two come "
                    "out of the processor together.")
            inputs_embeds = self.get_input_embeddings(
                input_ids, self.get_image_embeds(pixel_values, image_grid_thw))
        return self.model(input_ids, kv_cache, attn_metadata, slot_mapping,
                          output_hidden_states=output_hidden_states,
                          inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden: jax.Array) -> jax.Array:
        return self.lm_head.project(hidden)
