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

"""Native-flax BLOOM implementation — mirror of
``torch_orig/transformers_impl/modeling.py``.

Same public surface as the torch module: ``build_alibi_tensor``, ``BloomGelu``,
``BloomAttention``, ``BloomMLP``, ``BloomBlock``, ``BloomModel``,
``BloomForCausalLM``.

JAX adjustments (documented once here):
- ``flax.linen`` modules with ``__call__`` (not ``forward``); parameters are external
  and supplied via ``module.apply({"params": p}, ...)``.
- Parameter paths mirror torch's module tree with two deliberate exceptions: the
  transformer submodule is named ``model`` (torch names it ``transformer``) and output
  embeddings are **tied** via ``embed.attend`` (no separate ``lm_head`` parameter,
  exactly as the checkpoint stores them). Everything else keeps torch's names:
  ``word_embeddings``, ``word_embeddings_layernorm``, ``h_{i}``, ``self_attention``
  (``query_key_value``/``dense``), ``mlp`` (``dense_h_to_4h``/``dense_4h_to_h``),
  ``input_layernorm``, ``post_attention_layernorm``, ``ln_f``.
- ALiBi + causal/padding bias are added in fp32 for numerical stability; dropout is
  omitted (DPO/eval disables it).
- Returns are plain dicts / dataclasses (``.logits``) instead of HF output objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp

from .configuration import BloomConfig, config_from_pretrained  # noqa: F401 (re-export)

_NEG = jnp.float32(-1e30)


# --------------------------------------------------------------------------- #
# Standalone ops (mirror the torch module-level functions).
# --------------------------------------------------------------------------- #
def _alibi_slopes(num_heads: int) -> jnp.ndarray:
    """Per-head ALiBi slopes (mirror of the slope construction in build_alibi_tensor)."""
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = list(range(1, 1 + closest_power_of_2))
    slopes = [base ** p for p in powers]
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = list(range(1, 1 + 2 * num_remaining, 2))
        slopes += [extra_base ** p for p in extra_powers]
    return jnp.asarray(slopes, dtype=jnp.float32)


def build_alibi_tensor(attention_mask, num_heads: int, dtype=jnp.bfloat16) -> jnp.ndarray:
    """ALiBi bias, shape ``(batch_size * num_heads, 1, seq_len)`` — mirror of torch's
    ``build_alibi_tensor``. ``arange = (cumsum(mask)-1)*mask`` gives per-key positions;
    ``alibi = slope_h * arange``. Added to the query-key scores (broadcast over query),
    relying on softmax translation-invariance."""
    attention_mask = jnp.asarray(attention_mask)
    batch_size, seq_length = attention_mask.shape
    slopes = _alibi_slopes(num_heads)  # (num_heads,)
    arange_tensor = ((jnp.cumsum(attention_mask, axis=-1) - 1) * attention_mask)[:, None, :]  # (b,1,s)
    alibi = slopes[None, :, None] * arange_tensor.astype(jnp.float32)  # (b, num_heads, s)
    return alibi.reshape(batch_size * num_heads, 1, seq_length).astype(dtype)


def bloom_gelu(x):
    """BLOOM's tanh-approximation GELU (mirror of bloom_gelu_forward)."""
    return jax.nn.gelu(x, approximate=True)


class BloomGelu(nn.Module):
    """GELU activation (mirror of torch ``BloomGelu``; no parameters)."""

    @nn.compact
    def __call__(self, x):
        return bloom_gelu(x)


class LayerNorm(nn.Module):
    """LayerNorm with ``scale``+``bias`` params (flax equivalent of torch ``nn.LayerNorm``;
    the converter emits ``scale``/``bias``). Computed in fp32."""

    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), jnp.bfloat16)
        bias = self.param("bias", nn.initializers.zeros, (self.dim,), jnp.bfloat16)
        xf = x.astype(jnp.float32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        normed = (xf - mean) * jax.lax.rsqrt(var + self.eps)
        return (scale.astype(jnp.float32) * normed + bias.astype(jnp.float32)).astype(x.dtype)


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
    frozen base fits one GPU. With ``lora_rank == 0`` (full finetuning, which is what every
    instance but the four 7B ones asks for) this is exactly ``nn.Dense``: no extra params,
    byte-identical to the pre-LoRA mirror.

    BLOOM pins ``pd`` to ``jnp.bfloat16`` at every call site (the checkpoint's storage
    dtype), so the adapter term is cast back to the base projection's dtype before the
    add — a caller that hands in fp32 adapters must not silently widen the whole trunk.
    """
    y = nn.Dense(features, use_bias=use_bias, name=name, param_dtype=pd)(x)
    r = int(getattr(cfg, "lora_rank", 0) or 0)
    if r > 0:
        in_d = x.shape[-1]
        scale = float(getattr(cfg, "lora_alpha", 0) or (r * 2)) / r
        a = module.param(f"{name}_lora_A", lora_a_init(in_d), (in_d, r), pd)
        b = module.param(f"{name}_lora_B", nn.initializers.zeros, (r, features), pd)
        y = y + (scale * ((x @ a) @ b)).astype(y.dtype)
    return y


# --------------------------------------------------------------------------- #
# Transformer submodules (mirror torch class hierarchy).
# --------------------------------------------------------------------------- #
class BloomAttention(nn.Module):
    """Multi-head attention with fused interleaved QKV + ALiBi (mirror of
    torch ``BloomAttention``). ``__call__(hidden_states, residual, alibi, attention_mask)``
    returns ``dense(attn) + residual`` (the torch ``dropout_add`` residual, dropout off)."""

    config: BloomConfig

    @nn.compact
    def __call__(self, hidden_states, residual, alibi, attention_mask):
        cfg = self.config
        b, s, _ = hidden_states.shape
        h, hd = cfg.num_attention_heads, cfg.head_dim
        fused_qkv = _proj(self, hidden_states, 3 * h * hd, "query_key_value", cfg, jnp.bfloat16, use_bias=True)
        # BLOOM packs qkv interleaved per head: [b, s, num_heads, 3, head_dim].
        fused_qkv = fused_qkv.reshape(b, s, h, 3, hd)
        q = fused_qkv[..., 0, :].transpose(0, 2, 1, 3)  # [b, h, s, hd]
        k = fused_qkv[..., 1, :].transpose(0, 2, 1, 3)
        v = fused_qkv[..., 2, :].transpose(0, 2, 1, 3)

        inv_norm_factor = hd ** -0.5
        scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * inv_norm_factor
        # alibi: (b*h, 1, k) -> (b, h, 1, k); broadcast over query. mask: (b, 1, q, k).
        scores = scores + alibi.reshape(b, h, 1, s).astype(jnp.float32) + attention_mask
        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        context = jnp.einsum("bhqk,bhkd->bhqd", weights, v).transpose(0, 2, 1, 3).reshape(b, s, h * hd)
        attn_output = _proj(self, context, cfg.hidden_size, "dense", cfg, jnp.bfloat16, use_bias=True)
        return attn_output + residual


class BloomMLP(nn.Module):
    """GELU MLP (mirror of torch ``BloomMLP``). ``__call__(hidden_states, residual)``
    returns ``dense_4h_to_h(gelu(dense_h_to_4h(x))) + residual``."""

    config: BloomConfig

    @nn.compact
    def __call__(self, hidden_states, residual):
        cfg = self.config
        h = _proj(self, hidden_states, 4 * cfg.hidden_size, "dense_h_to_4h", cfg, jnp.bfloat16, use_bias=True)
        h = bloom_gelu(h)
        h = _proj(self, h, cfg.hidden_size, "dense_4h_to_h", cfg, jnp.bfloat16, use_bias=True)
        return h + residual


class BloomBlock(nn.Module):
    """One transformer block (mirror of torch ``BloomBlock``;
    ``apply_residual_connection_post_layernorm=False``)."""

    config: BloomConfig

    @nn.compact
    def __call__(self, hidden_states, alibi, attention_mask):
        cfg = self.config
        layernorm_output = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps, name="input_layernorm")(hidden_states)
        residual = (
            layernorm_output if cfg.apply_residual_connection_post_layernorm else hidden_states
        )
        attention_output = BloomAttention(cfg, name="self_attention")(
            layernorm_output, residual, alibi, attention_mask
        )
        layernorm_output = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps, name="post_attention_layernorm")(
            attention_output
        )
        residual = (
            layernorm_output if cfg.apply_residual_connection_post_layernorm else attention_output
        )
        return BloomMLP(cfg, name="mlp")(layernorm_output, residual)


@dataclass
class BaseModelOutput:
    """Lightweight stand-in for HF ``BaseModelOutputWithPastAndCrossAttentions``."""

    last_hidden_state: Any
    hidden_states: Any = None
    word_embeddings: Any = None  # the Embed module (for tied lm_head via attend)


class BloomModel(nn.Module):
    """The BLOOM transformer stack (mirror of torch ``BloomModel``; named ``model`` so
    param paths match what ``convert_checkpoint`` emits)."""

    config: BloomConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        cfg = self.config
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((b, s), dtype=jnp.int32)

        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="word_embeddings", param_dtype=jnp.bfloat16)
        hidden = embed(input_ids)
        hidden = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps, name="word_embeddings_layernorm")(hidden)

        alibi = build_alibi_tensor(attention_mask, cfg.num_attention_heads, dtype=jnp.float32)
        # causal + padding additive mask, shape (b, 1, q, k).
        causal = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        mask = causal + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)

        all_hidden = [hidden] if output_hidden_states else None
        for i in range(cfg.num_hidden_layers):
            hidden = BloomBlock(cfg, name=f"h_{i}")(hidden, alibi, mask)
            if output_hidden_states:
                all_hidden.append(hidden)
        hidden = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps, name="ln_f")(hidden)
        if output_hidden_states:
            all_hidden[-1] = hidden
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden, word_embeddings=embed)


@flax.struct.dataclass
class CausalLMOutput:
    """Stand-in for HF ``CausalLMOutputWithCrossAttentions`` (exposes ``.logits``).

    Registered as a flax pytree so it is a valid JAX type (traceable by
    ``jax.make_jaxpr``/``jit``); also dict-subscriptable for callers that read
    ``["logits"]``/``.get(...)``."""

    logits: Any
    hidden_states: Any = None
    values: Any = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class BloomForCausalLM(nn.Module):
    """BLOOM causal LM (mirror of torch ``BloomForCausalLM``). Tied output embeddings:
    ``logits = model.word_embeddings.attend(hidden)`` (no separate ``lm_head`` param,
    matching the checkpoint)."""

    config: BloomConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        outputs = BloomModel(cfg, name="model")(input_ids, attention_mask, output_hidden_states)
        hidden = outputs.last_hidden_state
        logits = outputs.word_embeddings.attend(hidden)  # tied embeddings
        values = None
        if cfg.with_value_head:
            values = nn.Dense(1, use_bias=False, name="v_head", param_dtype=jnp.bfloat16)(hidden)[..., 0]
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
    "build_alibi_tensor",
    "bloom_gelu",
    "BloomGelu",
    "BloomAttention",
    "BloomMLP",
    "BloomBlock",
    "BloomModel",
    "BloomForCausalLM",
    "BloomConfig",
    "config_from_pretrained",
    "lora_a_init",
]
