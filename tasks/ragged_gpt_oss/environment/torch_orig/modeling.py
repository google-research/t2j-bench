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

"""Inference-only gpt-oss, de-vendored from vllm/model_executor/models/gpt_oss.py.

THE CLASS NAMES ARE VLLM'S, NOT `core_llama3`'S, AND THAT IS DELIBERATE.
``OAIAttention`` / ``MLPBlock`` / ``TransformerBlock`` / ``GptOssModel`` /
``GptOssForCausalLM``, and ``GptOssRoutedExperts`` in `layers/moe.py`, are the
names upstream uses (`gpt_oss.py` :82, :318, :384, :432, :1367, :192). They break
`core_llama3`'s ``{Family}MLP`` / ``{Family}Attention`` / ``{Family}DecoderLayer``
convention on purpose: ``test_reference_units`` grades a port on mirroring THIS
file's submodule attribute names, so the convention has to be "mirror vLLM"
rather than "mirror core_llama3", or every family whose upstream disagrees would
have to be renamed away from its own source.

ONE NAME IS NOT VLLM'S, and it is the only one: the embedding is
``model.embed_tokens``, where vLLM's ``GptOssModel`` calls it ``self.embedding``.
Three reasons, all of them about the harness rather than about taste --
``embed_tokens`` is the checkpoint's own key, so `model/loader.py` needs no
rename for it; it is what every other instance in this class uses, so the shared
`_t2j` driver suite's ``model.embed_tokens.weight`` assertion keeps holding; and
vLLM itself is the odd one out here (its other 200-odd models use
``embed_tokens``). Recorded rather than left to be noticed.

Structural differences from upstream, all of them removals, exactly as
`core_llama3` states them: TP=1 so every *ParallelLinear collapses to a plain
linear while KEEPING the fused qkv_proj layout; no @support_torch_compile, no
VllmConfig, no LoRA, no pipeline parallel; and vLLM's `Attention` layer is
replaced by an explicit write_kv + paged_attention pair so the cache write is
visible in the model rather than hidden inside a backend.

WHAT IS NEW HERE relative to the two dense families, each with its own home:

  * a routed MoE (`layers/moe.py`) whose 19.1B parameters stay MXFP4-packed;
  * attention sinks -- a learned per-head logit in the softmax denominator
    (`attention/paged.py`, the `sinks` keyword);
  * an ALTERNATING sliding window: `config.layer_types` is 24 entries of
    sliding/full starting sliding, with `sliding_window: 128`. Read off the
    config, not off `layer_idx % 2` -- see `config.sliding_window_for_layer`;
  * YaRN RoPE with a 1.34657 attention gain (`layers/rotary.py`);
  * an UNTIED lm_head. `tie_word_embeddings: false`, unlike llama3 and qwen2,
    which are both tied. A port that ties anyway loads the embedding over the
    unembedding and is wrong at the output layer ONLY -- invisible to every
    hidden-state check.

The forward is token-flat: [T, H], never [B, L, H].
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
from layers.layernorm import RMSNorm
from layers.linear import QKVParallelLinear, RowParallelLinear
from layers.moe import MLPBlock
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding


class OAIAttention(nn.Module):
    """vLLM's `OAIAttention` (:82). Sinks at :121, the window policy at :167."""

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
        # NOT hidden_size // num_attention_heads: gpt-oss is 2880 / 64 with
        # head_dim 64, so the derived form gives 45. config.py makes it a field.
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        # Read from layer_types. vLLM writes it as `layer_idx % 2 == 0`, which
        # agrees on this checkpoint and is a consequence rather than the source.
        self.sliding_window = config.sliding_window_for_layer(layer_idx)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.num_heads,
            self.num_kv_heads, bias=config.attention_bias,
        )
        # WITH a bias, unlike llama3's o_proj. attention_bias is true for
        # gpt-oss and the checkpoint really ships `self_attn.o_proj.bias`.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias,
        )
        # One learned logit per QUERY head, joined to the softmax denominator
        # with no matching value. A plain Parameter, not a buffer: it is a
        # checkpoint tensor (`model.layers.*.self_attn.sinks`) and has to appear
        # in named_parameters() for the loader's coverage rule to reach it.
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            rope_scaling=config.rope_scaling,
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
            sliding_window=self.sliding_window, sinks=self.sinks,
        )
        return self.o_proj(attn.reshape(t, self.num_heads * self.head_dim))


class TransformerBlock(nn.Module):
    """vLLM's `TransformerBlock` (:384): ``attn`` and ``mlp``, not ``self_attn``."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OAIAttention(config, layer_idx)
        self.mlp = MLPBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(self, hidden, residual, kv_cache, attn_metadata, slot_mapping):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.attn(hidden, kv_cache, attn_metadata, slot_mapping)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class GptOssModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                output_hidden_states=False):
        hidden = self.embed_tokens(input_ids)
        collected = [hidden] if output_hidden_states else None
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, kv_cache, attn_metadata,
                                     slot_mapping)
            if output_hidden_states:
                collected.append(hidden + residual)
        hidden, _ = self.norm(hidden, residual)
        if output_hidden_states:
            collected[-1] = hidden
            return hidden, collected
        return hidden


class GptOssForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # NO TIE ARM. gpt-oss sets tie_word_embeddings: false and the checkpoint
        # ships its own 201088 x 2880 lm_head.weight. The conditional is kept --
        # rather than deleted as dead code -- because it is the one line that
        # states the difference from llama3 and qwen2, and because a config that
        # did ask for a tie must not silently get an independent head.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping=None,
                output_hidden_states=False):
        if slot_mapping is None:
            slot_mapping = torch.arange(input_ids.shape[0], dtype=torch.long)
        return self.model(input_ids, kv_cache, attn_metadata, slot_mapping,
                          output_hidden_states=output_hidden_states)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.project(hidden)
