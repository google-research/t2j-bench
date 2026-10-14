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

"""Native-JAX (flax.linen) Phi-3/Phi-4 — mirror of torch_orig/transformers_impl/modeling.py.

Same class names / hierarchy as the torch impl (``Phi3RMSNorm``, ``Phi3MLP``,
``Phi3Attention``, ``Phi3DecoderLayer``, ``Phi3Model``, ``Phi3ForCausalLM``). Phi3
uses **fused** projections — ``qkv_proj`` (split into q/k/v) and ``gate_up_proj`` (split
into gate/up) — and a **partial rotary**: only the first ``rotary_dim = round(head_dim *
partial_rotary_factor)`` dims of q/k are rotated, the rest pass through. No attention
bias; tied embeddings; RMSNorm in fp32; GQA via ``repeat_kv``; SwiGLU MLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration import Phi3Config, config_from_pretrained

_NEG = -1e30


class Phi3RMSNorm(nn.Module):
    """RMSNorm (mirror of torch ``Phi3RMSNorm``): fp32 ``x * rsqrt(mean(x^2)+eps)``."""

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
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _rope_cos_sin(positions, rotary_dim: int, theta: float, dtype, short_factor=(), attention_factor: float = 1.0):
    """cos/sin for the ROTARY sub-dimension (partial rotary). Returns ``(b, s, rotary_dim)``.

    Longrope (phi3/phi4): ``short_factor`` scales ``inv_freq`` per-dim (~1.0 at short
    sequences) and ``attention_factor`` (mscale, ~1.19) scales cos/sin globally.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    if short_factor:
        inv_freq = inv_freq / jnp.asarray(short_factor, dtype=jnp.float32)
    freqs = positions.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return (jnp.cos(emb) * attention_factor).astype(dtype), (jnp.sin(emb) * attention_factor).astype(dtype)


def apply_partial_rotary(q, k, cos, sin, rotary_dim: int):
    """Rotate only the first ``rotary_dim`` dims of q/k (Phi3 partial rotary); pass the rest."""
    cos = cos[:, None, :, :]  # (b, 1, s, rotary_dim)
    sin = sin[:, None, :, :]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return jnp.concatenate([q_rot, q_pass], axis=-1), jnp.concatenate([k_rot, k_pass], axis=-1)


def repeat_kv(hidden, n_rep: int):
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

    Keeps the base kernel at the original path ``<name>/kernel`` (so the converted weights
    load unchanged) and adds sibling adapters ``<name>_lora_A`` / ``<name>_lora_B``. The
    low-rank term is formed inside the matmul — ``scale * (x @ A) @ B`` with
    ``scale = alpha/rank`` — so the merged ``W + A@B`` weight is never materialized and a
    frozen base fits one GPU. Phi3's projections are **fused** (``qkv_proj`` covers q/k/v,
    ``gate_up_proj`` covers gate/up), and PEFT adapts them exactly as fused too: one
    ``lora_A``/``lora_B`` pair per ``nn.Linear``, so the fusion needs no special handling.
    With ``lora_rank == 0`` (full finetuning, which is what every instance but the four 7B
    ones asks for) this is exactly ``nn.Dense``: no extra params, byte-identical to the
    pre-LoRA mirror.
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


class Phi3MLP(nn.Module):
    """SwiGLU MLP with a **fused** ``gate_up_proj`` (mirror of torch ``Phi3MLP``)."""

    config: Phi3Config

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        gate_up = _proj(self, x, 2 * cfg.intermediate_size, "gate_up_proj", cfg, pd)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return _proj(self, nn.silu(gate) * up, cfg.hidden_size, "down_proj", cfg, pd)


class Phi3Attention(nn.Module):
    """GQA attention with a **fused** ``qkv_proj`` and **partial** RoPE (mirror of torch
    ``Phi3Attention``). No bias; scaling = ``head_dim ** -0.5``."""

    config: Phi3Config

    @nn.compact
    def __call__(self, hidden_states, cos, sin, attention_mask):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = hidden_states.shape
        hd = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        qkv = _proj(self, hidden_states, (n_heads + 2 * n_kv) * hd, "qkv_proj", cfg, pd)
        q, k, v = jnp.split(qkv, [n_heads * hd, (n_heads + n_kv) * hd], axis=-1)

        q = q.reshape(b, s, n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, n_kv, hd).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, n_kv, hd).transpose(0, 2, 1, 3)

        q, k = apply_partial_rotary(q, k, cos, sin, cfg.rotary_dim)
        k = repeat_kv(k, n_heads // n_kv)
        v = repeat_kv(v, n_heads // n_kv)

        cdt = jnp.promote_types(q.dtype, jnp.float32)  # f32 for bf16; f64 for the x64 grad probe
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) * (hd ** -0.5)
        attn = attn + attention_mask.astype(cdt)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        context = context.transpose(0, 2, 1, 3).reshape(b, s, n_heads * hd)
        return _proj(self, context, cfg.hidden_size, "o_proj", cfg, pd)


class Phi3DecoderLayer(nn.Module):
    """Decoder layer (mirror of torch ``Phi3DecoderLayer``): pre-norm attention + pre-norm MLP."""

    config: Phi3Config

    @nn.compact
    def __call__(self, hidden, cos, sin, attention_mask):
        cfg = self.config
        residual = hidden
        h = Phi3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="input_layernorm")(hidden)
        h = Phi3Attention(cfg, name="self_attn")(h, cos, sin, attention_mask)
        hidden = residual + h

        residual = hidden
        h = Phi3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="post_attention_layernorm")(hidden)
        h = Phi3MLP(cfg, name="mlp")(h)
        return residual + h


@dataclass
class BaseModelOutput:
    last_hidden_state: Any
    hidden_states: Any = None
    word_embeddings: Any = None


class Phi3Model(nn.Module):
    """The Phi3 transformer stack (mirror of torch ``Phi3Model``; named ``model``)."""

    config: Phi3Config

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        cfg = self.config
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((b, s), dtype=jnp.int32)

        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)

        positions = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))
        cos, sin = _rope_cos_sin(
            positions, cfg.rotary_dim, cfg.rope_theta, hidden.dtype,
            cfg.rope_short_factor, cfg.rope_attention_factor,
        )

        causal = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        mask = causal + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)

        all_hidden = [hidden] if output_hidden_states else None
        for i in range(cfg.num_hidden_layers):
            hidden = Phi3DecoderLayer(cfg, name=f"layers_{i}")(hidden, cos, sin, mask)
            if output_hidden_states:
                all_hidden.append(hidden)
        hidden = Phi3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="norm")(hidden)
        if output_hidden_states:
            all_hidden[-1] = hidden
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden, word_embeddings=embed)


@flax.struct.dataclass
class CausalLMOutput:
    logits: Any
    hidden_states: Any = None
    values: Any = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Phi3ForCausalLM(nn.Module):
    """Phi3 causal LM (mirror of torch ``Phi3ForCausalLM``). Tied output embeddings."""

    config: Phi3Config

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        outputs = Phi3Model(cfg, name="model")(input_ids, attention_mask, output_hidden_states)
        hidden = outputs.last_hidden_state
        if cfg.tie_word_embeddings:
            logits = outputs.word_embeddings.attend(hidden)
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head", param_dtype=cfg.dtype)(hidden)
        values = None
        if cfg.with_value_head:
            values = nn.Dense(1, use_bias=False, name="v_head", param_dtype=cfg.dtype)(hidden)[..., 0]
        return CausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, values=values)

    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 8, **kwargs):
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
    "Phi3RMSNorm",
    "Phi3MLP",
    "Phi3Attention",
    "Phi3DecoderLayer",
    "Phi3Model",
    "Phi3ForCausalLM",
    "Phi3Config",
    "config_from_pretrained",
    "lora_a_init",
]
