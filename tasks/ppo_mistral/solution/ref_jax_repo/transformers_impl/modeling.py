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

"""Native-JAX (flax.linen) Mistral — mirror of torch_orig/transformers_impl/modeling.py.

Same class names / hierarchy as the torch impl (``MistralRMSNorm``, ``MistralMLP``,
``MistralAttention``, ``MistralDecoderLayer``, ``MistralModel``, ``MistralForCausalLM``); flax
adjustments are the usual ones: ``forward`` -> ``__call__`` + external params via
``module.apply({"params": p}, ...)``, and the parameter pytree uses
``model/...`` paths (top submodule named ``model``; tied output embeddings via
``embed_tokens.attend`` — no separate ``lm_head`` param when tied). The math is
byte-faithful to torch: RMSNorm in fp32, NeoX ``rotate_half`` RoPE (theta=rope_theta;
plain rotary per §12.6), GQA ``repeat_kv``, scaled-dot-product with a causal+padding
mask, softmax in fp32, SwiGLU MLP. Unlike Qwen2, Mistral attention has **no q/k/v bias**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration import MistralConfig, config_from_pretrained

_NEG = -1e30  # additive-mask "-inf" (finite so softmax stays well-defined)


class MistralRMSNorm(nn.Module):
    """RMSNorm (mirror of torch ``MistralRMSNorm``): fp32 ``x * rsqrt(mean(x^2)+eps)``,
    then multiply by the ``scale`` weight (cast back to the input dtype first)."""

    dim: int
    eps: float = 1e-5
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        in_dtype = x.dtype
        cdt = jnp.promote_types(in_dtype, jnp.float32)  # f32 for bf16/f16/f32; f64 for the x64 grad probe
        x32 = x.astype(cdt)
        var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
        x32 = x32 * jax.lax.rsqrt(var + self.eps)
        return (scale.astype(cdt) * x32).astype(in_dtype)


def rotate_half(x):
    """``cat(-x2, x1)`` where ``x1, x2`` are the two halves of the last dim (NeoX)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to q, k. ``cos``/``sin`` are ``(b, s, head_dim)``; broadcast over heads."""
    cos = cos[:, None, :, :]  # (b, 1, s, hd)
    sin = sin[:, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rope_cos_sin(positions, head_dim: int, theta: float, dtype):
    """cos/sin for NeoX RoPE. ``positions`` is ``(b, s)``; returns ``(b, s, head_dim)``."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]  # (b, s, hd/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (b, s, hd)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def repeat_kv(hidden, n_rep: int):
    """(b, n_kv, s, hd) -> (b, n_kv*n_rep, s, hd) (GQA), mirror of torch ``repeat_kv``."""
    if n_rep == 1:
        return hidden
    b, n_kv, s, hd = hidden.shape
    hidden = jnp.broadcast_to(hidden[:, :, None, :, :], (b, n_kv, n_rep, s, hd))
    return hidden.reshape(b, n_kv * n_rep, s, hd)


def lora_a_init(in_d: int):
    """PEFT's ``lora_A`` initialiser, exactly.

    ``peft.LoraLayer.reset_lora_parameters`` does ``kaiming_uniform_(lora_A.weight, a=√5)``
    on a ``(rank, in_features)`` weight, so ``fan_in = in_features``, the gain is
    ``√(2/(1+5)) = √(1/3)`` and the bound is ``gain * √(3/fan_in) = 1/√in_features``. That
    is a *centred* uniform; ``nn.initializers.uniform(s)`` is ``U[0, s)``, which is why this
    is spelled out rather than reused. ``lora_B`` is zeros in PEFT and here, which is what
    makes the adapter a no-op at step 0 and the step-0 forward RNG-independent.
    """
    bound = float(in_d) ** -0.5

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def _proj(module, x, features, name, cfg, pd, use_bias: bool = False):
    """Linear projection ``Dense(x)``, plus an **in-forward LoRA** term when
    ``cfg.lora_rank > 0`` — the JAX mirror of torch_orig's PEFT LoRA on this linear layer.

    Keeps the base kernel at the original path ``<name>/kernel`` (so the converted
    weights load unchanged) and adds sibling adapters ``<name>_lora_A`` / ``<name>_lora_B``. The
    low-rank term is formed inside the matmul — ``scale * (x @ A) @ B`` with
    ``scale = alpha/rank`` — so the merged ``W + A@B`` weight is never materialized and a
    frozen 7B base fits one GPU. With ``lora_rank == 0`` (full finetuning / grader default)
    this is exactly ``nn.Dense`` — no extra params, byte-identical to the mirror.
    """
    y = nn.Dense(features, use_bias=use_bias, name=name, param_dtype=pd)(x)
    r = int(getattr(cfg, "lora_rank", 0) or 0)
    if r > 0:
        in_d = x.shape[-1]
        scale = float(getattr(cfg, "lora_alpha", 0) or (r * 2)) / r
        a = module.param(f"{name}_lora_A", lora_a_init(in_d), (in_d, r), pd)
        b = module.param(f"{name}_lora_B", nn.initializers.zeros, (r, features), pd)
        y = y + scale * ((x @ a) @ b)
    return y


class MistralMLP(nn.Module):
    """SwiGLU MLP (mirror of torch ``MistralMLP``): ``down(silu(gate(x)) * up(x))``."""

    config: MistralConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        gate = _proj(self, x, cfg.intermediate_size, "gate_proj", cfg, pd)
        up = _proj(self, x, cfg.intermediate_size, "up_proj", cfg, pd)
        return _proj(self, nn.silu(gate) * up, cfg.hidden_size, "down_proj", cfg, pd)


class MistralAttention(nn.Module):
    """GQA attention with RoPE (mirror of torch ``MistralAttention``). Mistral has **no**
    q/k/v/o bias; scaling = ``head_dim ** -0.5``."""

    config: MistralConfig

    @nn.compact
    def __call__(self, hidden_states, cos, sin, attention_mask):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = hidden_states.shape
        hd = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        q = _proj(self, hidden_states, n_heads * hd, "q_proj", cfg, pd)
        k = _proj(self, hidden_states, n_kv * hd, "k_proj", cfg, pd)
        v = _proj(self, hidden_states, n_kv * hd, "v_proj", cfg, pd)

        q = q.reshape(b, s, n_heads, hd).transpose(0, 2, 1, 3)  # (b, nh, s, hd)
        k = k.reshape(b, s, n_kv, hd).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, n_kv, hd).transpose(0, 2, 1, 3)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, n_heads // n_kv)
        v = repeat_kv(v, n_heads // n_kv)

        scaling = hd ** -0.5
        cdt = jnp.promote_types(q.dtype, jnp.float32)  # f32 for bf16; f64 for the x64 grad probe
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) * scaling
        attn = attn + attention_mask.astype(cdt)  # (b, 1, q, k) additive
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (b, nh, q, hd)
        context = context.transpose(0, 2, 1, 3).reshape(b, s, n_heads * hd)
        return _proj(self, context, cfg.hidden_size, "o_proj", cfg, pd)


class MistralDecoderLayer(nn.Module):
    """Decoder layer (mirror of torch ``MistralDecoderLayer``): pre-norm attention +
    pre-norm MLP with residuals."""

    config: MistralConfig

    @nn.compact
    def __call__(self, hidden, cos, sin, attention_mask):
        cfg = self.config
        residual = hidden
        h = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="input_layernorm")(hidden)
        h = MistralAttention(cfg, name="self_attn")(h, cos, sin, attention_mask)
        hidden = residual + h

        residual = hidden
        h = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="post_attention_layernorm")(hidden)
        h = MistralMLP(cfg, name="mlp")(h)
        return residual + h


@dataclass
class BaseModelOutput:
    """Lightweight stand-in for HF ``BaseModelOutputWithPast``."""

    last_hidden_state: Any
    hidden_states: Any = None
    word_embeddings: Any = None  # the Embed module (for tied lm_head via attend)


class MistralModel(nn.Module):
    """The Mistral transformer stack (mirror of torch ``MistralModel``; named ``model`` so
    param paths match what ``convert_checkpoint`` emits)."""

    config: MistralConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        cfg = self.config
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((b, s), dtype=jnp.int32)

        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)

        # Default (no-cache) position ids = arange, matching torch's cache_position.
        positions = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))
        cos, sin = _rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta, hidden.dtype)

        # causal + padding additive mask, shape (b, 1, q, k).
        causal = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        mask = causal + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)

        all_hidden = [hidden] if output_hidden_states else None
        for i in range(cfg.num_hidden_layers):
            hidden = MistralDecoderLayer(cfg, name=f"layers_{i}")(hidden, cos, sin, mask)
            if output_hidden_states:
                all_hidden.append(hidden)
        hidden = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="norm")(hidden)
        if output_hidden_states:
            all_hidden[-1] = hidden
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden, word_embeddings=embed)


@flax.struct.dataclass
class CausalLMOutput:
    """Stand-in for HF ``CausalLMOutputWithPast`` (exposes ``.logits``); registered as a
    flax pytree so it is a valid JAX type, and dict-subscriptable for ``["logits"]``."""

    logits: Any
    hidden_states: Any = None
    values: Any = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class MistralForCausalLM(nn.Module):
    """Mistral causal LM (mirror of torch ``MistralForCausalLM``). Tied output embeddings
    (Mistral-3.2): ``logits = model.embed_tokens.attend(hidden)`` (no separate ``lm_head``
    param); falls back to an explicit ``lm_head`` Dense when the checkpoint is untied."""

    config: MistralConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        outputs = MistralModel(cfg, name="model")(input_ids, attention_mask, output_hidden_states)
        hidden = outputs.last_hidden_state
        if cfg.tie_word_embeddings:
            logits = outputs.word_embeddings.attend(hidden)  # tied embeddings
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head", param_dtype=cfg.dtype)(hidden)
        values = None
        if cfg.with_value_head:
            values = nn.Dense(1, use_bias=False, name="v_head", param_dtype=cfg.dtype)(hidden)[..., 0]
        return CausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, values=values)

    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 8, **kwargs):
        """Greedy decode (JAX adjustment of HF ``generate``); params stashed by the loader."""
        params = {"params": object.__getattribute__(self, "_gen_params")}
        ids = jnp.asarray(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        for _ in range(int(max_new_tokens)):
            mask = jnp.ones(ids.shape, dtype=jnp.int32)
            logits = self.apply(params, input_ids=ids, attention_mask=mask).logits
            nxt = jnp.argmax(logits[:, -1, :].astype(jnp.float32), axis=-1)
            ids = jnp.concatenate([ids, nxt[:, None]], axis=1)
        return ids


__all__ = [
    "MistralRMSNorm",
    "MistralMLP",
    "MistralAttention",
    "MistralDecoderLayer",
    "MistralModel",
    "MistralForCausalLM",
    "MistralConfig",
    "config_from_pretrained",
    "lora_a_init",
]
