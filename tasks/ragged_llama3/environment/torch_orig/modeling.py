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

"""Inference-only Llama, de-vendored from vllm/model_executor/models/llama.py.

Structural differences from upstream, all of them removals:
  * TP=1, so every *ParallelLinear collapses to a plain linear (layers/linear.py)
    while KEEPING the fused qkv_proj / gate_up_proj parameter layout, which is
    what test_param_correspondence grades.
  * No @support_torch_compile, no VllmConfig, no quantization, no LoRA,
    no pipeline parallel (PPMissingLayer / make_layers).
  * vllm's Attention layer is replaced by an explicit write_kv + paged_attention
    pair, so the cache write is visible in the model rather than hidden inside
    a backend -- the candidate has to port it, and the grader can gate it.

The forward is token-flat: [T, H], never [B, L, H]. Batching is expressed by
query_start_loc, not by a batch dimension. That is the whole point of ragged
inference and it is the shape the candidate must reproduce.
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


class LlamaMLP(nn.Module):
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


class LlamaAttention(nn.Module):
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
            self.num_kv_heads, bias=config.attention_bias,
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


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
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


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                output_hidden_states=False):
        hidden = self.embed_tokens(input_ids)
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


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
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
