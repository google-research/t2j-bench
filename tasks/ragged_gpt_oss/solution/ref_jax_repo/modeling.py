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

"""Inference-only gpt-oss in flax nnx, mirroring torch_orig/modeling.py.

Structurally the same file: same classes, same attribute names, same order of
operations. Three differences, all forced by JAX and all of them the point of
the exercise:

  1. `nn.Module` -> `nnx.Module`, `forward` -> `__call__`, `nn.ModuleList` -> a
     plain list wrapped in `nnx.data` (flax 0.12 refuses data on a static
     attribute).
  2. EVERY METHOD THAT TOUCHES THE CACHE RETURNS IT. jax arrays are immutable,
     so `write_kv` produces a new array and the caller has to thread it: the
     attention returns `(out, kv_cache)`, the transformer block returns
     `(hidden, residual, kv_cache)`, and both models return the cache last.
     torch_orig hides this inside an in-place scatter; a port that keeps the
     torch signature silently drops every cache write.
  3. The per-layer write becomes
     `group = group.at[slot].set(write_kv(group[slot], k, v, slot_mapping))`
     on the one container group this layer owns (`self.kv_group` /
     `self.kv_slot`),
     and the attention then reads the UPDATED layer.

THE CLASS NAMES ARE VLLM'S. `OAIAttention` / `MLPBlock` / `TransformerBlock` /
`GptOssModel` / `GptOssForCausalLM`, and `GptOssRoutedExperts` in
`layers/moe.py`. See the torch twin's docstring for why this family breaks
`core_llama3`'s `{Family}MLP` convention, and for the one name that is NOT
vLLM's (`model.embed_tokens`, where vLLM says `self.embedding`).

THIS IS A RESTRUCTURING OF UPSTREAM, NOT A DE-VENDORING OF IT.
`tpu_inference/models/jax/gpt_oss.py` is a single 555-line `nnx.Module` with an
inline `__call__` and no sub-modules at all, unlike its own `qwen2.py`. There
was no JAX decomposition to follow; there is a torch one, and this mirrors it.

Unchanged from core_llama3, because they are the graded contract:
  * K is written to the cache AFTER `rotary_emb`; V is written un-rotated.
  * The forward is token-flat: [T, H], never [B, L, H].

NOT tied: `tie_word_embeddings` is false for gpt-oss, so `lm_head.weight` is its
own nnx.Param and the tie arm below never fires.
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
from layers.layernorm import RMSNorm
from layers.linear import QKVParallelLinear, RowParallelLinear
from layers.moe import MLPBlock
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding


class OAIAttention(nnx.Module):
    """vLLM's `OAIAttention` (:82). Sinks at :121, the window policy at :167."""

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
            dtype=dtype, rngs=rngs,
        )
        # WITH a bias, unlike llama3's o_proj. attention_bias is true for
        # gpt-oss and the checkpoint really ships `self_attn.o_proj.bias`.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias, dtype=dtype, rngs=rngs,
        )
        # One learned logit per QUERY head, joined to the softmax denominator
        # with no matching value. An nnx.Param, not a plain attribute: it is a
        # checkpoint tensor (`model.layers.*.self_attn.sinks`) and has to be in
        # the parameter pytree for the loader's coverage rule to reach it.
        self.sinks = nnx.Param(jnp.zeros((self.num_heads,), dtype))
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
            sliding_window=self.sliding_window, sinks=self.sinks[...],
        )
        return (self.o_proj(attn.reshape(t, self.num_heads * self.head_dim)),
                kv_cache)


class TransformerBlock(nnx.Module):
    """vLLM's `TransformerBlock` (:384): ``attn`` and ``mlp``, not ``self_attn``."""

    def __init__(self, config: ModelConfig, layer_idx: int, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.attn = OAIAttention(config, layer_idx, dtype=dtype, rngs=rngs)
        self.mlp = MLPBlock(config, dtype=dtype, rngs=rngs)
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
        hidden, kv_cache = self.attn(hidden, kv_cache, attn_metadata, slot_mapping)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual, kv_cache


class GptOssModel(nnx.Module):
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
            TransformerBlock(config, i, dtype=dtype, rngs=rngs)
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


class GptOssForCausalLM(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs | None = None, *,
                 dtype=jnp.float32) -> None:
        self.config = config
        self.model = GptOssModel(config, dtype=dtype, rngs=rngs)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                      dtype=dtype, rngs=rngs)
        # NO TIE ON THIS CHECKPOINT. gpt-oss sets tie_word_embeddings: false and
        # ships its own 201088 x 2880 lm_head.weight. The conditional is kept --
        # rather than deleted as dead code -- because it is the one line that
        # states the difference from llama3 and qwen2, and because a config that
        # did ask for a tie must not silently get an independent head.
        if config.tie_word_embeddings:
            # The SAME nnx.Param object, not a copy.
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids, kv_cache, attn_metadata, slot_mapping=None,
                 output_hidden_states=False):
        if slot_mapping is None:
            slot_mapping = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
        return self.model(input_ids, kv_cache, attn_metadata, slot_mapping,
                          output_hidden_states=output_hidden_states)

    def compute_logits(self, hidden: jax.Array) -> jax.Array:
        return self.lm_head.project(hidden)
