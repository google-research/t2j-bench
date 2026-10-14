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

"""Inference-only gemma-4 in flax nnx, mirroring torch_orig/modeling.py.

Structurally the same file: same classes, same attribute names, same order of
operations. Three differences, all forced by JAX and all of them the point of
the exercise:

  1. `nn.Module` -> `nnx.Module`, `forward` -> `__call__`, `nn.ModuleList` -> a
     plain list wrapped in `nnx.data` (flax 0.12 refuses data on a static
     attribute).
  2. EVERY METHOD THAT TOUCHES THE CACHE RETURNS IT. jax arrays are immutable,
     so `write_kv` produces a new array and the caller has to thread it: the
     attention returns `(out, kv_cache)`, the decoder layer returns
     `(hidden, kv_cache)`, and the model returns the cache last. torch_orig
     hides this inside an in-place scatter; a port that keeps the torch
     signature silently drops every cache write.
  3. The per-layer write becomes
     `group = group.at[slot].set(write_kv(group[slot], k, v, slot_mapping))`
     on the one container group this layer owns (`self.kv_group` /
     `self.kv_slot`), and the attention then reads the UPDATED layer.

THE CLASS NAMES ARE VLLM'S: `Gemma4MLP` / `Gemma4Attention` /
`Gemma4DecoderLayer` / `Gemma4Model` / `Gemma4ForCausalLM`. Unlike gpt_oss they
also happen to match `core_llama3`'s `{Family}MLP` convention -- that is a
coincidence, and the names were READ OFF `vllm/model_executor/models/gemma4.py`
(:225, :375, :554, :949, :1494) rather than derived from the family string. See
the torch twin's docstring for the submodule names and where each came from.

See the torch twin for everything that is about the MODEL rather than about
JAX -- the four heterogeneity axes, the three traps, the residual pattern and
the PLE algorithm are all argued there and not repeated here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from attention.metadata import AttentionMetadata
from attention.paged import paged_attention
from cache.kv_cache import _require_slot, write_kv
from config import ModelConfig
from layers.activation import GeluAndMul
from layers.layernorm import RMSNorm
from layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding


class RMSNormNoWeight(nnx.Module):
    """RMS normalisation with NO learned scale. gemma-4's `v_norm`, and only that.

    See the torch twin for the whole argument -- in short: upstream spells this
    `RMSNorm(..., has_weight=False)`, `layers/layernorm.py` in this class is
    byte-identical across fourteen instances and gaining a flag for one family
    would be a class-wide edit, and the checkpoint ships no `v_norm.weight`, so
    a parameter-bearing RMSNorm here would fail the loader's coverage rule.

    The arithmetic is `layers/layernorm.RMSNorm`'s with the final multiply
    removed, cast order included: accumulate the reciprocal-sqrt in fp32, cast
    back, and stop.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.variance_epsilon = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.variance_epsilon)
        return x.astype(orig_dtype)


class Gemma4MLP(nnx.Module):
    """vLLM's `Gemma4MLP` (:225). `intermediate_size` is passed IN, not read."""

    def __init__(self, config: ModelConfig, intermediate_size: int, *,
                 dtype=jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [intermediate_size] * 2, bias=False,
            dtype=dtype, rngs=rngs,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, config.hidden_size, bias=False,
            dtype=dtype, rngs=rngs,
        )
        self.act_fn = GeluAndMul(config.hidden_activation)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Gemma4Attention(nnx.Module):
    """vLLM's `Gemma4Attention` (:375). The KV-share branch is at :534-546."""

    def __init__(self, config: ModelConfig, layer_idx: int,
                 rotary_emb: RotaryEmbedding, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.layer_idx = layer_idx
        # WHERE THIS LAYER'S KV LIVES IN THE CACHE CONTAINER, resolved once here
        # rather than re-derived at every forward -- and for this family it is
        # NOT the identity in either component. `config.layer_slot()` puts the
        # 15 allocating layers into two groups (sliding, head_dim 256; full,
        # head_dim 512) and points the 20 shared ones back at slot 11 of group 0
        # or slot 2 of group 1; `config.layer_writes()` is False on exactly
        # those 20.
        #
        # THIS IS WHERE THE MAP IS APPLIED; engine.py declares the container's
        # GEOMETRY (`config.kv_groups()`), and config.py holds the one
        # definition both spell it from. Moving only one of the two is what
        # `_require_slot` refuses at the read site below.
        self.kv_group, self.kv_slot = config.layer_slot()[layer_idx]
        self.kv_writes = config.layer_writes()[layer_idx]
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads_for_layer(layer_idx)
        self.head_dim = config.head_dim_for_layer(layer_idx)
        # 1.0, NOT head_dim ** -0.5. See the torch twin.
        self.scaling = 1.0
        self.sliding_window = config.sliding_window_for_layer(layer_idx)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.num_heads,
            self.num_kv_heads, bias=config.attention_bias,
            dtype=dtype, rngs=rngs,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias, dtype=dtype, rngs=rngs,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.v_norm = RMSNormNoWeight(eps=config.rms_norm_eps)
        # SHARED BETWEEN EVERY LAYER OF THIS TYPE, and handed in rather than
        # constructed. See `Gemma4Model` and `layers/rotary.py` for the
        # arithmetic that forces it.
        self.rotary_emb = rotary_emb

    def __call__(self, hidden: jax.Array, kv_cache: tuple[jax.Array, ...],
                 attn_metadata: AttentionMetadata, slot_mapping: jax.Array
                 ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        t = hidden.shape[0]
        q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        q = q.reshape(t, self.num_heads, self.head_dim)
        k = k.reshape(t, self.num_kv_heads, self.head_dim)
        v = v.reshape(t, self.num_kv_heads, self.head_dim)

        # Q IS ALWAYS NORMED AND ALWAYS ROTATED, on every layer. K and V are
        # neither, on the 20 shared ones. See the torch twin.
        q = self.q_norm(q)
        if not self.is_kv_shared_layer:
            k = self.k_norm(k)
            q, k = self.rotary_emb(attn_metadata.input_positions, q, k)
            v = self.v_norm(v)
        else:
            q, _ = self.rotary_emb(attn_metadata.input_positions, q, k)

        # BEFORE the index, because JAX would not complain about a bad one: it
        # CLAMPS the read and DROPS the write where torch raises, so a
        # half-generalised family is a wrong number here and a crash there. This
        # is THE family the guard was written for -- 15 slots for 35 layers.
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
            sliding_window=self.sliding_window,
        )
        return (self.o_proj(attn.reshape(t, self.num_heads * self.head_dim)),
                kv_cache)


class Gemma4DecoderLayer(nnx.Module):
    """vLLM's `Gemma4DecoderLayer` (:554). Four norms, PLE, and `layer_scalar`."""

    def __init__(self, config: ModelConfig, layer_idx: int,
                 rotary_emb: RotaryEmbedding, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.layer_idx = layer_idx
        h, eps = config.hidden_size, config.rms_norm_eps
        p = config.hidden_size_per_layer_input

        self.self_attn = Gemma4Attention(config, layer_idx, rotary_emb,
                                         dtype=dtype, rngs=rngs)
        # DOUBLE WIDE ON EXACTLY THE KV-SHARED LAYERS -- 12288 rather than the
        # declared 6144 on 20 of the 35. Derived from `config`, never from a
        # second copy of the `>= 15` test.
        self.mlp = Gemma4MLP(config, config.intermediate_size_for_layer(layer_idx),
                             dtype=dtype, rngs=rngs)

        self.input_layernorm = RMSNorm(h, eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(h, eps=eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps, dtype=dtype)

        self.per_layer_input_gate = ReplicatedLinear(h, p, bias=False,
                                                     dtype=dtype, rngs=rngs)
        self.per_layer_projection = ReplicatedLinear(p, h, bias=False,
                                                     dtype=dtype, rngs=rngs)
        self.post_per_layer_input_norm = RMSNorm(h, eps=eps, dtype=dtype)

        # AN nnx.Param, NOT A BUFFER, and that is a deliberate divergence from
        # vLLM -- see the torch twin for the whole argument.
        self.layer_scalar = nnx.Param(jnp.ones((1,), dtype))

    def __call__(self, hidden: jax.Array, kv_cache: tuple[jax.Array, ...],
                 attn_metadata: AttentionMetadata, slot_mapping: jax.Array,
                 per_layer_input: jax.Array):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, kv_cache = self.self_attn(hidden, kv_cache, attn_metadata,
                                          slot_mapping)
        hidden = self.post_attention_layernorm(hidden)
        hidden = hidden + residual

        residual = hidden
        hidden = self.pre_feedforward_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)
        hidden = hidden + residual

        gate = jax.nn.gelu(self.per_layer_input_gate(hidden), approximate=True)
        contribution = self.post_per_layer_input_norm(
            self.per_layer_projection(gate * per_layer_input))
        hidden = hidden + contribution

        # LAST, after PLE and after both residual adds.
        return hidden * self.layer_scalar[...], kv_cache


class Gemma4Model(nnx.Module):
    def __init__(self, config: ModelConfig, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.config = config
        h = config.hidden_size
        p = config.hidden_size_per_layer_input
        ell = config.num_hidden_layers
        if p <= 0:
            raise ValueError(
                "hidden_size_per_layer_input is 0, so this checkpoint has no "
                "per-layer embeddings. This reference implements the PLE "
                "variants (E2B/E4B) and refuses the others by name rather than "
                "running a model that is missing 46.6% of its parameters.")

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, h,
                                                   dtype=dtype, rngs=rngs)
        self.embed_tokens_per_layer = VocabParallelEmbedding(
            config.vocab_size_per_layer_input, ell * p, dtype=dtype, rngs=rngs)
        self.per_layer_model_projection = ColumnParallelLinear(
            h, ell * p, bias=False, dtype=dtype, rngs=rngs)
        self.per_layer_projection_norm = RMSNorm(p, eps=config.rms_norm_eps,
                                                 dtype=dtype)

        # TWO ROTARY TABLES FOR THIRTY-FIVE LAYERS. See `layers/rotary.py` for
        # the arithmetic (35 private tables would be 5.6 GB per model copy and
        # the grader holds three; two shared ones are 402 MB). A table is a pure
        # function of (base, head_size, max_position) and every layer of a type
        # asks for the same one, so nothing about the values changes.
        rotary: dict[str, RotaryEmbedding] = {}
        for i in range(ell):
            kind = config.layer_type(i)
            if kind in rotary:
                continue
            rope = config.rope_for_layer(i)
            rotary[kind] = RotaryEmbedding(
                head_size=config.head_dim_for_layer(i),
                # rotary_dim == head_size. NOT a truncated rotary_dim -- see
                # `layers/rotary.py`, which is where this family's sharpest
                # trap lives.
                rotary_dim=config.head_dim_for_layer(i),
                max_position=config.max_position_embeddings,
                base=float(rope.get("rope_theta", 10000.0)),
                is_neox_style=True,
                rope_scaling=rope,
                partial_rotary_factor=float(
                    rope.get("partial_rotary_factor", 1.0)),
            )
        # nnx.data, not a bare list: flax 0.12 classifies every attribute as
        # static structure or graph data, and a bare list of modules is refused
        # as "data on a static attribute".
        self.layers = nnx.data([
            Gemma4DecoderLayer(config, i, rotary[config.layer_type(i)],
                               dtype=dtype, rngs=rngs)
            for i in range(ell)
        ])
        self.norm = RMSNorm(h, eps=config.rms_norm_eps, dtype=dtype)

        # THE THREE PLE SCALES AND THE EMBEDDING SCALE, as Python floats.
        # Plain floats and not arrays: they are derived from `config`, so an
        # nnx.Param would put four tensors into the parameter pytree with no
        # counterpart in the torch state dict.
        self.embedding_scale = float(h) ** 0.5
        self.embed_scale_per_layer = float(p) ** 0.5
        self.per_layer_input_scale = 1.0 / (2.0 ** 0.5)
        self.per_layer_projection_scale = float(h) ** -0.5

    def get_per_layer_inputs(self, input_ids: jax.Array) -> jax.Array:
        """Track A: the PLE lookup, `[T, L, P]`. See the torch twin.

        The out-of-range mask is a no-op on this checkpoint and is kept anyway,
        for the reason the twin gives.
        """
        mask = ((input_ids >= 0)
                & (input_ids < self.config.vocab_size_per_layer_input))
        ple_ids = jnp.where(mask, input_ids, jnp.zeros_like(input_ids))
        embeds = self.embed_tokens_per_layer(ple_ids) * self.embed_scale_per_layer
        return embeds.reshape(*input_ids.shape,
                              self.config.num_hidden_layers,
                              self.config.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: jax.Array,
                                 per_layer_inputs: jax.Array) -> jax.Array:
        """Track B, and the combination. See the torch twin."""
        projection = self.per_layer_model_projection(inputs_embeds)
        projection = projection * self.per_layer_projection_scale
        projection = projection.reshape(*inputs_embeds.shape[:-1],
                                        self.config.num_hidden_layers,
                                        self.config.hidden_size_per_layer_input)
        projection = self.per_layer_projection_norm(projection)
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def compute_per_layer_inputs(self, input_ids: jax.Array,
                                 inputs_embeds: jax.Array) -> jax.Array:
        """`[T, L, P]`, computed ONCE per forward and sliced per layer."""
        return self.project_per_layer_inputs(inputs_embeds,
                                             self.get_per_layer_inputs(input_ids))

    def __call__(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                 output_hidden_states=False):
        hidden = self.embed_tokens(input_ids) * self.embedding_scale
        per_layer_inputs = self.compute_per_layer_inputs(input_ids, hidden)
        collected = [hidden] if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            hidden, kv_cache = layer(hidden, kv_cache, attn_metadata,
                                     slot_mapping, per_layer_inputs[:, i, :])
            if output_hidden_states:
                collected.append(hidden)
        hidden = self.norm(hidden)
        if output_hidden_states:
            collected[-1] = hidden
            return hidden, collected, kv_cache
        return hidden, kv_cache


class Gemma4ForCausalLM(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs | None = None, *,
                 dtype=jnp.float32) -> None:
        self.config = config
        self.model = Gemma4Model(config, dtype=dtype, rngs=rngs)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                      dtype=dtype, rngs=rngs)
        # TIED, and the checkpoint really ships no `lm_head.weight` (measured:
        # 600 text tensors, none of them an unembedding).
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
        logits = self.lm_head.project(hidden)
        # FINAL LOGIT SOFT-CAPPING, in the head's own dtype -- see the torch twin.
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = jnp.tanh(logits / cap) * cap
        return logits
