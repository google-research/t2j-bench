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

"""Native-JAX (flax.linen) Falcon (RW) — mirror of torch_orig/transformers_impl/modeling.py.

falcon-rw: **ALiBi** position bias (no rotary), sequential blocks with **two**
LayerNorms (input / post_attention), a biased **interleaved** fused ``query_key_value``
(``[num_heads, 3, head_dim]``, MHA), a biased output ``dense``, a biased GELU MLP.
Param names mirror torch's: ``transformer/word_embeddings/embedding``,
``transformer/h/{i}/self_attention/{query_key_value,dense}``,
``transformer/h/{i}/mlp/{dense_h_to_4h,dense_4h_to_h}``,
``transformer/h/{i}/{input_layernorm,post_attention_layernorm}``, ``transformer/ln_f``.
Falcon scales the ALiBi bias by ``1/sqrt(head_dim)`` too (baddbmm beta=alpha), unlike
bloom (beta=1): ``scores = (q@k + alibi) / sqrt(head_dim)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration import FalconConfig, config_from_pretrained

_NEG = -1e30


class FalconLayerNorm(nn.Module):
    """LayerNorm (scale+bias), fp32 compute (dtype-following for the x64 grad probe)."""

    dim: int
    eps: float = 1e-5
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
        in_dtype = x.dtype
        cdt = jnp.promote_types(in_dtype, jnp.float32)
        xf = x.astype(cdt)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        normed = (xf - mean) * jax.lax.rsqrt(var + self.eps)
        return (scale.astype(cdt) * normed + bias.astype(cdt)).astype(in_dtype)


def _alibi_slopes(n_heads: int) -> jnp.ndarray:
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = pow2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = pow2(closest) + pow2(2 * closest)[0::2][: n_heads - closest]
    return jnp.asarray(slopes, dtype=jnp.float32)


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
    byte-identical to the pre-LoRA mirror. Falcon's targeted projections
    (``query_key_value``, ``dense``, ``dense_h_to_4h``, ``dense_4h_to_h``) are all biased;
    PEFT adapts the weight only, so the bias stays on the frozen base.
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


class FalconAttention(nn.Module):
    """MHA with ALiBi + interleaved fused QKV (mirror of torch ``FalconAttention``)."""

    config: FalconConfig

    @nn.compact
    def __call__(self, x, alibi, attn_mask):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h, hd = cfg.num_attention_heads, cfg.head_dim
        qkv = _proj(self, x, 3 * h * hd, "query_key_value", cfg, pd, use_bias=True)
        qkv = qkv.reshape(b, s, h, 3, hd)  # interleaved per head [num_heads, 3, head_dim]
        q = qkv[..., 0, :].transpose(0, 2, 1, 3)
        k = qkv[..., 1, :].transpose(0, 2, 1, 3)
        v = qkv[..., 2, :].transpose(0, 2, 1, 3)
        cdt = jnp.promote_types(q.dtype, jnp.float32)  # f32 for bf16; f64 for the x64 grad probe
        scores = (jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) + alibi.astype(cdt)) * (hd ** -0.5)
        scores = scores + attn_mask.astype(cdt)
        w = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(b, s, h * hd)
        return _proj(self, out, cfg.hidden_size, "dense", cfg, pd, use_bias=True)


class FalconMLP(nn.Module):
    """Biased GELU MLP (mirror of torch ``FalconMLP``)."""

    config: FalconConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        h = _proj(self, x, 4 * cfg.hidden_size, "dense_h_to_4h", cfg, pd, use_bias=True)
        h = jax.nn.gelu(h, approximate=False)  # exact erf GELU (torch GELUActivation default)
        return _proj(self, h, cfg.hidden_size, "dense_4h_to_h", cfg, pd, use_bias=True)


class FalconDecoderLayer(nn.Module):
    """Decoder block (mirror of torch ``FalconDecoderLayer``): two LayerNorms, sequential
    attention + MLP with residuals."""

    config: FalconConfig

    @nn.compact
    def __call__(self, x, alibi, attn_mask):
        cfg = self.config
        h = x + FalconAttention(cfg, name="self_attention")(
            FalconLayerNorm(cfg.hidden_size, cfg.layer_norm_epsilon, cfg.dtype, name="input_layernorm")(x),
            alibi, attn_mask)
        return h + FalconMLP(cfg, name="mlp")(
            FalconLayerNorm(cfg.hidden_size, cfg.layer_norm_epsilon, cfg.dtype, name="post_attention_layernorm")(h))


class _FalconLayers(nn.Module):
    """Container named ``h`` so params land at ``transformer/h/{i}`` (mirror of the torch
    ``ModuleList``)."""

    config: FalconConfig

    @nn.compact
    def __call__(self, x, alibi, attn_mask, output_hidden_states=False):
        # Per-layer activation capture (TEST_PLAN §14.1): record each layer input;
        # the caller appends the post-ln_f state -> length num_hidden_layers + 1.
        all_hidden = [] if output_hidden_states else None
        for i in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden.append(x)
            x = FalconDecoderLayer(self.config, name=str(i))(x, alibi, attn_mask)
        return x, all_hidden


@dataclass
class BaseModelOutput:
    last_hidden_state: Any
    hidden_states: Any = None
    word_embeddings: Any = None


class FalconModel(nn.Module):
    """The Falcon transformer stack (mirror of torch ``FalconModel``; named ``transformer``)."""

    config: FalconConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        cfg = self.config
        b, s = input_ids.shape
        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="word_embeddings", param_dtype=cfg.dtype)
        hidden = embed(input_ids)
        # ALiBi bias, mirroring torch ``build_alibi_tensor``: the per-head slopes are cast to
        # bfloat16 and multiplied by the (absolute) key position in bf16 *before* the fp32
        # upcast, so the bias carries torch's bf16 quantization even on the fp32 graded path.
        # It depends only on the key position (a query-only offset is softmax-invariant), so it
        # broadcasts across queries as [1, h, 1, k] — matching torch's ``alibi.view(b, h, 1, -1)``.
        pos = jnp.arange(s)
        slopes = _alibi_slopes(cfg.num_attention_heads).astype(jnp.bfloat16)  # [h]
        alibi = (slopes[:, None] * pos[None, :].astype(jnp.bfloat16)).astype(jnp.float32)  # [h, k]
        alibi = alibi[None, :, None, :]  # [1, h, 1, k]
        mask = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        if attention_mask is not None:
            mask = mask + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)
        hidden, all_hidden = _FalconLayers(cfg, name="h")(hidden, alibi, mask, output_hidden_states)
        hidden = FalconLayerNorm(cfg.hidden_size, cfg.layer_norm_epsilon, cfg.dtype, name="ln_f")(hidden)
        if output_hidden_states:
            all_hidden.append(hidden)
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


class FalconForCausalLM(nn.Module):
    """Falcon causal LM (mirror of torch ``FalconForCausalLM``; tied for dpo/ppo)."""

    config: FalconConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        outputs = FalconModel(cfg, name="transformer")(input_ids, attention_mask, output_hidden_states)
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
            logits = self.apply(params, input_ids=ids, attention_mask=jnp.ones(ids.shape, jnp.int32)).logits
            ids = jnp.concatenate([ids, jnp.argmax(logits[:, -1].astype(jnp.float32), -1)[:, None]], axis=1)
        return ids


__all__ = [
    "FalconLayerNorm",
    "FalconAttention",
    "FalconMLP",
    "FalconDecoderLayer",
    "FalconModel",
    "FalconForCausalLM",
    "FalconConfig",
    "config_from_pretrained",
    "lora_a_init",
]
