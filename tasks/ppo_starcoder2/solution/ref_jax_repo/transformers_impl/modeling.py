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

"""Native-JAX (flax.linen) Starcoder2 — mirror of torch_orig/transformers_impl/modeling.py.

Same class names / hierarchy as the torch impl (``Starcoder2LayerNorm``, ``Starcoder2MLP``,
``Starcoder2Attention``, ``Starcoder2DecoderLayer``, ``Starcoder2Model``,
``Starcoder2ForCausalLM``). Unlike the Llama family, Starcoder2 uses **LayerNorm**
(scale+bias, fp32) instead of RMSNorm, a **GELU dense** MLP (``c_fc``/``c_proj`` with
bias) instead of SwiGLU, and **bias on q/k/v/o**. RoPE is plain NeoX (theta=rope_theta),
GQA via ``repeat_kv``, softmax in fp32, tied output embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration import Starcoder2Config, config_from_pretrained

_NEG = -1e30


class Starcoder2LayerNorm(nn.Module):
    """LayerNorm (mirror of torch ``nn.LayerNorm``): fp32 mean/var normalize, then
    ``scale * normed + bias`` (cast back to the input dtype)."""

    dim: int
    eps: float = 1e-5
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
        in_dtype = x.dtype
        cdt = jnp.promote_types(in_dtype, jnp.float32)  # f32 for bf16/f16/f32; f64 for the x64 grad probe
        xf = x.astype(cdt)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        normed = (xf - mean) * jax.lax.rsqrt(var + self.eps)
        return (scale.astype(cdt) * normed + bias.astype(cdt)).astype(in_dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _rope_cos_sin(positions, head_dim: int, theta: float, dtype):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


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
    frozen base fits one GPU. Starcoder2's linears carry a bias (``cfg.use_bias``); PEFT
    adapts the weight only, so the bias rides along on the base ``nn.Dense`` untouched.
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


class Starcoder2MLP(nn.Module):
    """GELU dense MLP (mirror of torch ``Starcoder2MLP``): ``c_proj(gelu(c_fc(x)))`` with
    bias on both projections."""

    config: Starcoder2Config

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        cfg = self.config
        pd = cfg.dtype
        h = _proj(self, x, cfg.intermediate_size, "c_fc", cfg, pd, use_bias=cfg.use_bias)
        h = jax.nn.gelu(h, approximate=cfg.gelu_approximate)
        h = _proj(self, h, cfg.hidden_size, "c_proj", cfg, pd, use_bias=cfg.use_bias)
        # torch applies residual_dropout to the MLP output (functional, gated on training).
        return nn.Dropout(rate=cfg.residual_dropout, deterministic=deterministic)(h)


class Starcoder2Attention(nn.Module):
    """GQA attention with RoPE (mirror of torch ``Starcoder2Attention``). q/k/v/o all
    carry bias (``use_bias``); scaling = ``head_dim ** -0.5``."""

    config: Starcoder2Config

    @nn.compact
    def __call__(self, hidden_states, cos, sin, attention_mask, deterministic: bool = True):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = hidden_states.shape
        hd = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        ub = cfg.use_bias

        q = _proj(self, hidden_states, n_heads * hd, "q_proj", cfg, pd, use_bias=ub)
        k = _proj(self, hidden_states, n_kv * hd, "k_proj", cfg, pd, use_bias=ub)
        v = _proj(self, hidden_states, n_kv * hd, "v_proj", cfg, pd, use_bias=ub)

        q = q.reshape(b, s, n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, n_kv, hd).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, n_kv, hd).transpose(0, 2, 1, 3)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, n_heads // n_kv)
        v = repeat_kv(v, n_heads // n_kv)

        cdt = jnp.promote_types(q.dtype, jnp.float32)  # f32 for bf16; f64 for the x64 grad probe
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) * (hd ** -0.5)
        attn = attn + attention_mask.astype(cdt)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        # torch applies attention_dropout to the softmaxed weights (after the f32->dtype cast).
        attn = nn.Dropout(rate=cfg.attention_dropout, deterministic=deterministic)(attn)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        context = context.transpose(0, 2, 1, 3).reshape(b, s, n_heads * hd)
        out = _proj(self, context, cfg.hidden_size, "o_proj", cfg, pd, use_bias=ub)
        # torch applies residual_dropout to the attention output projection.
        return nn.Dropout(rate=cfg.residual_dropout, deterministic=deterministic)(out)


class Starcoder2DecoderLayer(nn.Module):
    """Decoder layer (mirror of torch ``Starcoder2DecoderLayer``): pre-LayerNorm attention
    + pre-LayerNorm MLP with residuals."""

    config: Starcoder2Config

    @nn.compact
    def __call__(self, hidden, cos, sin, attention_mask, deterministic: bool = True):
        cfg = self.config
        residual = hidden
        h = Starcoder2LayerNorm(cfg.hidden_size, cfg.norm_epsilon, cfg.dtype, name="input_layernorm")(hidden)
        h = Starcoder2Attention(cfg, name="self_attn")(h, cos, sin, attention_mask, deterministic=deterministic)
        hidden = residual + h

        residual = hidden
        h = Starcoder2LayerNorm(cfg.hidden_size, cfg.norm_epsilon, cfg.dtype, name="post_attention_layernorm")(hidden)
        h = Starcoder2MLP(cfg, name="mlp")(h, deterministic=deterministic)
        return residual + h


@dataclass
class BaseModelOutput:
    last_hidden_state: Any
    hidden_states: Any = None
    word_embeddings: Any = None


class Starcoder2Model(nn.Module):
    """The Starcoder2 transformer stack (mirror of torch ``Starcoder2Model``; named
    ``model`` so param paths match what ``convert_checkpoint`` emits)."""

    config: Starcoder2Config

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, deterministic: bool = True):
        cfg = self.config
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((b, s), dtype=jnp.int32)

        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)
        # torch applies embedding_dropout to inputs_embeds before the decoder stack.
        hidden = nn.Dropout(rate=cfg.embedding_dropout, deterministic=deterministic)(hidden)

        positions = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))
        cos, sin = _rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta, hidden.dtype)

        causal = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        mask = causal + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)

        all_hidden = [hidden] if output_hidden_states else None
        for i in range(cfg.num_hidden_layers):
            hidden = Starcoder2DecoderLayer(cfg, name=f"layers_{i}")(hidden, cos, sin, mask, deterministic=deterministic)
            if output_hidden_states:
                all_hidden.append(hidden)
        hidden = Starcoder2LayerNorm(cfg.hidden_size, cfg.norm_epsilon, cfg.dtype, name="norm")(hidden)
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


class Starcoder2ForCausalLM(nn.Module):
    """Starcoder2 causal LM (mirror of torch ``Starcoder2ForCausalLM``). Tied output
    embeddings: ``logits = model.embed_tokens.attend(hidden)``."""

    config: Starcoder2Config

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, deterministic: bool = True, **kwargs):
        cfg = self.config
        outputs = Starcoder2Model(cfg, name="model")(input_ids, attention_mask, output_hidden_states, deterministic=deterministic)
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
    "Starcoder2LayerNorm",
    "Starcoder2MLP",
    "Starcoder2Attention",
    "Starcoder2DecoderLayer",
    "Starcoder2Model",
    "Starcoder2ForCausalLM",
    "Starcoder2Config",
    "config_from_pretrained",
    "lora_a_init",
]
