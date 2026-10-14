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

"""Inference-only Qwen2.5-VL, de-vendored from vLLM.

TWO upstream files, at upstream sha ``06ecec7a...``, because upstream
itself is two files:

  * ``vllm/model_executor/models/qwen2_5_vl.py`` :1217-2060 --
    ``Qwen2_5_VLForConditionalGeneration``. Of it, what survives is ``__init__``
    :1337-1370, ``get_mrope_input_positions`` :1289-1326,
    ``_process_image_input`` :1423-1446, ``forward`` :1993-2020 and
    ``compute_logits`` :2023-2027.
  * ``vllm/model_executor/models/qwen2.py`` :80-500 -- the TEXT stack. That is
    not an analogy: ``qwen2_5_vl.py`` :1361-1366 literally builds its language
    model with ``init_vllm_registered_model(..., architectures=
    ["Qwen2ForCausalLM"])``, so Qwen2.5-VL's decoder IS Qwen2.5's, and the four
    classes below are ``core_qwen2``'s with one addition (the M-RoPE section
    widths, read from the config).

The vision tower is in ``vision.py`` -- a DECLARED partition violation, argued
in that file's docstring.

Structural differences from upstream, all of them removals:
  * TP=1, so every *ParallelLinear collapses to a plain linear (layers/linear.py)
    while KEEPING the fused qkv_proj / gate_up_proj parameter layout, which is
    what test_param_correspondence grades.
  * No @support_torch_compile, no VllmConfig, no quantization, no LoRA,
    no pipeline parallel (PPMissingLayer / make_layers), no EVS video pruning,
    no encoder CUDA graphs, no multimodal registry / processor plumbing.
  * vllm's Attention layer is replaced by an explicit write_kv + paged_attention
    pair, so the cache write is visible in the model rather than hidden inside
    a backend -- the candidate has to port it, and the grader can gate it.
  * Video is out of scope. ``video_token_id`` is read into ``ModelConfig``
    because the checkpoint declares it, and nothing here consumes it.

TWO NAMING DECISIONS, BOTH BINDING ON THE JAX PORT, both stated here because
``test_reference_units`` freezes one dotted attribute path per unit and walks
that same string against the candidate:

  1. THE TEXT STACK HANGS AT ``model.`` AND ``lm_head``, NOT AT
     ``language_model.``. vLLM's wrapper owns a ``language_model`` attribute
     (:1362) purely because ``init_vllm_registered_model`` returns a whole
     registered ``Qwen2ForCausalLM`` -- it is model-registry infrastructure, of
     exactly the kind this class de-vendors away along with ``VllmConfig`` and
     ``*ParallelLinear``. There is no registry here to compose from, and vLLM's
     own weight loader has to RENAME the checkpoint (``"model."`` ->
     ``"language_model.model."``, :1236-1245) to invent the level. The shipped
     checkpoint spells it ``model.layers.N.*`` / ``visual.*`` -- measured, from
     ``model.safetensors.index.json`` -- and ``canonical_param_key`` normalises
     no part of an intermediate path, so keeping the checkpoint's own spelling
     is what makes ``test_param_correspondence``'s stated contract ("every path
     component must match the checkpoint's spelling") true for this family as
     well as the other four. Consequence: this family's reference-unit paths are
     ``core_llama3``'s unchanged.
  2. THE TOWER HANGS AT ``visual.``, with vLLM's and the checkpoint's own
     internals (``blocks.N.attn.qkv``, ``blocks.N.mlp.gate_up_proj``,
     ``merger.mlp.0``). There the two upstreams agree with each other and with
     the checkpoint, so there was nothing to decide.
"""

from __future__ import annotations

import torch
from torch import nn

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


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
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
            # (vllm/model_executor/models/qwen2.py:161, at the pinned SHA), and
            # Qwen2.5-VL-3B-Instruct's checkpoint does ship
            # `model.layers.N.self_attn.{q,k,v}_proj.bias`.
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
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
            # producing a plausible, wrongly-rotated model.
            #
            # Upstream reaches the same place through the config rather than
            # through a keyword: `get_rope` dispatches on
            # `rope_scaling["type"] == "mrope"` and hands the section list to
            # `MRotaryEmbedding` (qwen2.py:180-185, mrope.py:286).
            # `ModelConfig.mrope_section` returns None for a checkpoint with no
            # such key, which is the text-only default.
            mrope_section=config.mrope_section,
        )

    def forward(self, hidden: torch.Tensor,
                kv_cache: tuple[torch.Tensor, ...],
                attn_metadata: AttentionMetadata,
                slot_mapping: torch.Tensor) -> torch.Tensor:
        t = hidden.shape[0]
        q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(attn_metadata.input_positions, q, k)

        # BEFORE the index, so that BOTH sides refuse a bad (group, slot) the
        # same way: torch would raise IndexError here anyway, but JAX clamps the
        # read and drops the write, and the two references have to agree about
        # what a broken map does. See cache/kv_cache._require_slot.
        _require_slot(kv_cache, self.kv_group, self.kv_slot)

        # Post-RoPE K and pre-RoPE V go into the cache -- the order matters and
        # is the classic port bug: caching pre-RoPE K silently breaks decode.
        group = kv_cache[self.kv_group]
        if self.kv_writes:
            write_kv(group[self.kv_slot], k, v, slot_mapping)

        attn = paged_attention(
            q, group[self.kv_slot], attn_metadata,
            num_kv_heads=self.num_kv_heads, sm_scale=self.scaling,
        )
        return self.o_proj(attn.reshape(t, self.num_heads * self.head_dim))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden, residual, kv_cache, attn_metadata, slot_mapping):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(hidden, kv_cache, attn_metadata, slot_mapping)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                output_hidden_states=False, inputs_embeds=None):
        # `inputs_embeds` IS THE MULTIMODAL ENTRY POINT and is upstream's own
        # (qwen2.py:397-410): the wrapper embeds the text ids, scatters the
        # image embeddings over the placeholder rows, and hands the result down.
        # When it is None this is byte-for-byte core_qwen2's forward.
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        collected = [hidden] if output_hidden_states else None
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, kv_cache, attn_metadata, slot_mapping)
            if output_hidden_states:
                collected.append(hidden + residual)
        hidden, _ = self.norm(hidden, residual)
        if output_hidden_states:
            collected[-1] = hidden
            return hidden, collected
        return hidden


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    """The wrapper: a Qwen2.5 decoder, a ViT tower, and the scatter between them.

    ``...ForConditionalGeneration``, not ``...ForCausalLM``: the checkpoint's own
    ``architectures`` says so, vLLM's class says so, and it is the first model in
    this class whose head is a wrapper around a text decoder PLUS an encoder.
    The offline task-generation and golden-authoring tooling reads that name, so
    it is the one string here that other files depend on literally.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # `norm_eps` comes from the TEXT config, exactly as upstream does it
        # (qwen2_5_vl.py:1355, `getattr(config, "rms_norm_eps", 1e-6)`): the
        # vision_config carries no eps of its own and HF hardcodes 1e-6 in
        # Qwen2_5_VLRMSNorm. Both are 1e-6 for this checkpoint, so the choice is
        # invisible today and would not be for a family that retunes it.
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config, norm_eps=config.rms_norm_eps)
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # -- multimodal ---------------------------------------------------------#

    def get_image_embeds(self, pixel_values: torch.Tensor,
                         image_grid_thw) -> torch.Tensor:
        """``[num_merged_tokens, hidden_size]``, one row per ``<|image_pad|>``.

        Upstream's ``_process_image_input`` (:1423-1446) additionally splits the
        result per image with ``grid_thw.prod(-1) // merge_size ** 2``; the split
        exists so the model runner can attribute rows to requests, and the very
        next thing it does is flatten them again for the scatter. Nothing here
        attributes rows to requests, so the split is dropped and the concatenated
        form is returned -- which is also what makes
        :meth:`get_input_embeddings` a single index-put.
        """
        return self.visual(pixel_values, image_grid_thw)

    def get_input_embeddings(self, input_ids: torch.Tensor,
                             image_embeds: torch.Tensor | None = None
                             ) -> torch.Tensor:
        """Text embeddings with the tower's rows written over the placeholders.

        De-vendored from ``vllm/model_executor/models/utils.py``
        ``_merge_multimodal_embeddings`` :682-716: a boolean mask over
        ``input_ids == image_token_id`` and one masked assignment. Upstream
        writes IN PLACE into the freshly-embedded tensor and so does this --
        ``embed_tokens`` returns a new tensor on every call, so there is nothing
        aliased to corrupt.

        The count check is upstream's too (:704-712), promoted from a
        ``RuntimeError`` rescue to a precondition: a prompt whose placeholder
        count disagrees with the grid is a scenario-construction bug, and torch's
        own message for it ("shape mismatch: value tensor of shape [4, 2048]
        cannot be broadcast...") names neither the image nor the prompt.
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
        inputs_embeds[mask] = image_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def get_input_positions(self, input_ids: torch.Tensor, *,
                            image_grid_thw=None, **_) -> torch.Tensor:
        """THE ENGINE'S HOOK (``engine.Engine.POSITIONS_HOOK``). M-RoPE here.

        Two names for one computation, and the split is the whole point of the
        design. ``get_input_positions`` is the FAMILY-NEUTRAL name the shared
        engine looks up -- it appears in ``engine.py`` as a string constant and
        nowhere else, so the engine can ask every model for its positions
        without knowing that this one's are a (t, h, w) triple.
        ``get_mrope_input_positions`` is upstream's own name for the arithmetic
        (``qwen2_5_vl.py`` :1289-1326) and stays, because the de-vendoring
        citation is to that function and the harness grades it by it.

        A model that does NOT define this method gets
        ``engine.default_input_positions`` -- ``arange``, one section -- which
        is why the four text-only families needed no counterpart.

        ``**_`` ABSORBS ``pixel_values``. The engine forwards its whole
        multimodal keyword set to the positions builder and to the forward
        alike, without inspecting it (see ``Engine._input_positions``): deciding
        which builder wants which keyword is exactly the per-family knowledge
        that must not live in the shared column. So the cost of that opacity is
        paid here, in one ``**_``, by the family that has the knowledge.
        """
        return self.get_mrope_input_positions(input_ids, image_grid_thw)

    def get_mrope_input_positions(self, input_ids: torch.Tensor,
                                  image_grid_thw=None) -> torch.Tensor:
        """``(3, num_tokens)`` M-RoPE positions: the t, h and w row per token.

        De-vendored from ``get_mrope_input_positions`` :1289-1326, with one
        deliberate change of INPUT: upstream reads the image spans off
        ``mm_features``, a scheduler object that does not exist here, so the
        spans are recovered from the ids instead -- each maximal run of
        ``image_token_id`` is one image, in prompt order, which is precisely
        what the placeholder expansion guarantees.

        THE RULE, in one sentence: a text token advances all three rows by one;
        an image contributes its (t, h, w) grid indices, so the whole image
        occupies ``max(h, w)`` positions rather than one per token, and the text
        after it resumes from ``max + 1``. That is why a VL prompt's last
        position is generally larger than its length, and why this cannot be
        replaced by ``arange(n).expand(3, n)`` for anything but a text-only
        prompt (where it reduces to exactly that -- and where it is also
        bit-identical to the rank-1 rope, because three equal rows select the
        same frequency band the single-section default does).

        Returned as int32, matching ``AttentionMetadata``'s contract that every
        tensor field is int32 -- see ``engine.py::_metadata``.
        """
        ids = input_ids.reshape(-1).tolist()
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

        rows: list[torch.Tensor] = []
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
            rows.append(
                torch.arange(text_len, dtype=torch.int32).expand(3, text_len) + start)
            t_idx = torch.arange(llm_t).view(llm_t, 1, 1).expand(llm_t, llm_h, llm_w)
            h_idx = torch.arange(llm_h).view(1, llm_h, 1).expand(llm_t, llm_h, llm_w)
            w_idx = torch.arange(llm_w).view(1, 1, llm_w).expand(llm_t, llm_h, llm_w)
            grid = torch.stack([t_idx.reshape(-1), h_idx.reshape(-1),
                                w_idx.reshape(-1)]).to(torch.int32)
            rows.append(grid + text_len + start)
            seen = offset + length

        if seen < len(ids):
            text_len = len(ids) - seen
            start = int(rows[-1].max()) + 1 if rows else 0
            rows.append(
                torch.arange(text_len, dtype=torch.int32).expand(3, text_len) + start)
        return torch.cat(rows, dim=1)

    # -- forward ------------------------------------------------------------#

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping=None,
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
            slot_mapping = torch.arange(input_ids.shape[0], dtype=torch.long)
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

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.project(hidden)
