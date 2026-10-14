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

"""Inference-only Qwen3 in flax nnx, mirroring environment/torch_orig/modeling.py.

Structurally the same file: same classes, same attribute names, same order of
operations. Three differences, all of them forced by JAX and all of them the
point of the exercise:

  1. `nn.Module` -> `nnx.Module`, `forward` -> `__call__`, `nn.ModuleList` -> a
     plain list (`nnx.data`, so flax treats it as graph data rather than static
     structure).
  2. EVERY METHOD THAT TOUCHES THE CACHE RETURNS IT. jax arrays are immutable,
     so `write_kv` produces a new array and the caller has to thread it: the
     attention returns `(out, kv_cache)`, the decoder layer returns
     `(hidden, residual, kv_cache)`, and both models return the cache last.
     torch_orig hides this inside an in-place scatter; a port that keeps the
     torch signature silently drops every cache write.
  3. The per-layer write becomes
     `group = group.at[slot].set(write_kv(group[slot], k, v, slot_mapping))`
     on the one container group this layer owns (`self.kv_group` /
     `self.kv_slot`),
     and the attention then reads the UPDATED layer -- exactly as torch_orig
     reads the layer it just mutated in place.

Unchanged, because they are the graded contract:
  * QK-NORM IS ARCHITECTURAL, NOT CONFIGURABLE, and runs BEFORE `rotary_emb`.
    RoPE is a rotation, and a non-uniform per-channel scale does not commute
    with it, so normalising after the rotation gives a different and wrong
    answer. The weight is [head_dim], SHARED ACROSS HEADS; both q and k are
    already (t, heads, head_dim) at that point, so RMSNorm over the last axis
    -- with no reshape -- IS the per-head norm.
  * K is written to the cache AFTER `rotary_emb` (and after `k_norm`, which now
    runs before it); V is written un-rotated. This is invisible to a single
    prefill (the cache is written and read in the same step) and only diverges
    once decode reads K that an earlier step wrote.
  * `lm_head.weight` IS `model.embed_tokens.weight` -- the same nnx.Param
    object, not a copy -- when config.tie_word_embeddings.
  * The forward is token-flat: [T, H], never [B, L, H]. Batching is expressed by
    query_start_loc, not by a batch dimension.
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


class Qwen3MLP(nnx.Module):
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


class Qwen3Attention(nnx.Module):
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
            self.num_kv_heads, bias=config.attention_bias,
            dtype=dtype, rngs=rngs,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False,
            dtype=dtype, rngs=rngs,
        )

        # QK-NORM IS ARCHITECTURAL, NOT CONFIGURABLE. Qwen3's config.json carries
        # no key for it; the only evidence is the checkpoint, which ships
        # `self_attn.{q,k}_norm.weight` at [head_dim]. A port that reads only the
        # config builds an attention with no QK-norm, loads every other tensor
        # cleanly, and is silently wrong -- the same shape of defect as Qwen2.5's
        # hardcoded QKV bias. vLLM's Qwen3Attention constructs both the same way.
        #
        # The weight is [head_dim], SHARED ACROSS HEADS, and the norm is applied
        # per head: RMSNorm over the last axis of (t, num_heads, head_dim).
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            rope_scaling=config.rope_scaling,
        )

    def __call__(self, hidden: jax.Array, kv_cache: tuple[jax.Array, ...],
                 attn_metadata: AttentionMetadata, slot_mapping: jax.Array
                 ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        t = hidden.shape[0]
        q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        q = q.reshape(t, self.num_heads, self.head_dim)
        k = k.reshape(t, self.num_kv_heads, self.head_dim)
        v = v.reshape(t, self.num_kv_heads, self.head_dim)

        # BEFORE RoPE, and the order is load-bearing: normalising after the
        # rotation would rescale the rotated vectors and change the attention
        # logits. Both q and k are already (t, heads, head_dim) here, so RMSNorm
        # over the last axis is the per-head norm with no reshape.
        q = self.q_norm(q)
        k = self.k_norm(k)

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


class Qwen3DecoderLayer(nnx.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.self_attn = Qwen3Attention(config, layer_idx, dtype=dtype, rngs=rngs)
        self.mlp = Qwen3MLP(config, dtype=dtype, rngs=rngs)
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


class Qwen3Model(nnx.Module):
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
            Qwen3DecoderLayer(config, i, dtype=dtype, rngs=rngs)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def __call__(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                 output_hidden_states=False):
        hidden = self.embed_tokens(input_ids)
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


class Qwen3ForCausalLM(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs | None = None, *,
                 dtype=jnp.float32) -> None:
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                      dtype=dtype, rngs=rngs)
        if config.tie_word_embeddings:
            # The SAME nnx.Param object, not a copy. flax's graph dedups shared
            # references, so `lm_head.weight` disappears from the parameter
            # pytree exactly as it disappears from torch's named_parameters(),
            # and a write through either name is visible through both.
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids, kv_cache, attn_metadata, slot_mapping=None,
                 output_hidden_states=False):
        if slot_mapping is None:
            slot_mapping = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
        return self.model(input_ids, kv_cache, attn_metadata, slot_mapping,
                          output_hidden_states=output_hidden_states)

    def compute_logits(self, hidden: jax.Array) -> jax.Array:
        return self.lm_head.project(hidden)
