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

"""Native-JAX (flax.linen) Qwen3VLForConditionalGeneration language-model path — mirror of
torch_orig/transformers_impl/modeling.py.

The graded forward is text-only: a causal decoder nested under the LM prefix
(``model/language_model/layers/{i}``), covering the qwen2/qwen3/llama/gemma LM cores
up to config switches (attention bias, qk-norm, gemma (1+w) RMSNorm / gelu MLP / embed
scale). ``convert_checkpoint`` converts the FULL checkpoint (vision/audio towers
included) so the spec test matches; flax leaves those leaves untouched at apply time.
Norms/softmax are dtype-following (f32 for bf16; f64 for the x64 gradient probe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration import AutoConfig, Qwen3VLConfig

_NEG = -1e30


def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rotary(position_ids, head_dim, theta, dtype):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _apply_rotary(q, k, cos, sin):
    cdt = jnp.promote_types(q.dtype, jnp.float32)
    cos, sin = cos[:, None, :, :].astype(cdt), sin[:, None, :, :].astype(cdt)
    qf, kf = q.astype(cdt), k.astype(cdt)
    return (qf * cos + _rotate_half(qf) * sin).astype(q.dtype), (kf * cos + _rotate_half(kf) * sin).astype(k.dtype)


def _mrope_cos_sin(position_ids, head_dim, theta, mrope_section, interleaved, dtype):
    """3D M-RoPE cos/sin (mirror of torch ``Qwen3VLTextRotaryEmbedding``).

    ``position_ids`` is (3, batch, seq): the (temporal, height, width) index of each token
    from ``get_rope_index``. Each of the head_dim/2 frequencies is taken from ONE of the
    three axes. Qwen3-VL *interleaves* the sections as (t, h, w, t, h, w, ...) within the
    first ``3 * section`` slots (``apply_interleaved_mrope``) rather than three contiguous
    blocks. Either way a text-only sequence -- where all three axes equal ``arange`` --
    reduces exactly to 1-D RoPE, so the text goldens stay bit-identical. Returns cos/sin of
    shape (batch, seq, head_dim), the layout ``_apply_rotary`` expects.
    """
    import numpy as np

    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = position_ids.astype(jnp.float32)                     # (3, b, s)
    freqs = pos[..., None] * inv_freq[None, None, None, :]     # (3, b, s, half)
    # sel[i] = which of the three (t, h, w) axes frequency i is drawn from.
    sel = np.zeros(half, dtype=np.int64)
    if interleaved:
        # torch apply_interleaved_mrope: freqs_t = freqs[0]; for dim in (1, 2) overwrite
        # freqs_t[..., dim : mrope_section[dim] * 3 : 3] = freqs[dim, ...].
        for dim in (1, 2):
            length = int(mrope_section[dim]) * 3
            sel[dim:length:3] = dim
    else:
        start = 0
        for dim, s in enumerate(mrope_section):
            sel[start:start + int(s)] = dim
            start += int(s)
    onehot = jax.nn.one_hot(jnp.asarray(sel), 3, axis=0, dtype=freqs.dtype)  # (3, half)
    freqs_t = jnp.einsum("absi,ai->bsi", freqs, onehot)       # (b, s, half)
    emb = jnp.concatenate([freqs_t, freqs_t], axis=-1)        # (b, s, head_dim)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _scatter_image_features(hidden, input_ids, image_features, image_token_id):
    """The flax mirror of torch's ``inputs_embeds.masked_scatter(image_mask, feats)``.

    Replace the embeddings at ``input_ids == image_token_id`` with the packed visual
    features, in row-major order (row 0's placeholders first). ``image_features`` has one
    row per placeholder token across the whole batch. Runs only in the eager
    training/eval forward, where the sequence length is concrete.
    """
    b, s, d = hidden.shape
    mask_flat = (input_ids == image_token_id).reshape(-1)
    idx = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1
    n = image_features.shape[0]
    gathered = image_features[jnp.clip(idx, 0, n - 1)].astype(hidden.dtype)
    merged = jnp.where(mask_flat[:, None], gathered, hidden.reshape(-1, d))
    return merged.reshape(b, s, d)


def _scatter_add_deepstack(hidden, mask_flat, embeds):
    """Add per-visual-token ``embeds`` back at the image-token positions (mirror of torch
    ``Qwen3VLTextModel._deepstack_process``: ``hidden[visual_pos_masks] += visual_embeds``),
    row-major over the batch. Runs only in the eager multimodal forward."""
    b, s, d = hidden.shape
    idx = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1
    n = embeds.shape[0]
    gathered = embeds[jnp.clip(idx, 0, n - 1)].astype(hidden.dtype)
    add = jnp.where(mask_flat[:, None], gathered, jnp.zeros((), hidden.dtype))
    return (hidden.reshape(-1, d) + add).reshape(b, s, d)


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
    """
    y = nn.Dense(features, use_bias=use_bias, name=name, param_dtype=pd)(x)
    r = int(getattr(cfg, "lora_rank", 0) or 0)
    if r > 0:
        in_d = x.shape[-1]
        scale = float(getattr(cfg, "lora_alpha", 0) or (r * 2)) / r
        a = module.param(f"{name}_lora_A", lora_a_init(in_d), (in_d, r), pd)
        b = module.param(f"{name}_lora_B", nn.initializers.zeros, (r, features), pd)
        y = (y + scale * ((x @ a) @ b)).astype(y.dtype)
    return y


class VLRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    add_one: bool = False
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.zeros if self.add_one else nn.initializers.ones,
                           (self.dim,), self.param_dtype)
        cdt = jnp.promote_types(x.dtype, jnp.float32)
        w = scale.astype(cdt) + (1.0 if self.add_one else 0.0)
        xf = x.astype(cdt)
        normed = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
        return (w * normed).astype(x.dtype)


class VLAttention(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h, kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        # Every q/k/v/o projection goes through ``_proj`` (base Dense + optional in-forward
        # LoRA); the factory keeps the call sites below unchanged.
        dense = lambda f, n, bias: (lambda z: _proj(self, z, f, n, cfg, pd, use_bias=bias))
        q = dense(h * hd, "q_proj", cfg.attn_bias)(x).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
        k = dense(kv * hd, "k_proj", cfg.attn_bias)(x).reshape(b, s, kv, hd).transpose(0, 2, 1, 3)
        v = dense(kv * hd, "v_proj", cfg.attn_bias)(x).reshape(b, s, kv, hd).transpose(0, 2, 1, 3)
        if cfg.qk_norm:
            q = VLRMSNorm(hd, cfg.rms_norm_eps, False, pd, name="q_norm")(q)
            k = VLRMSNorm(hd, cfg.rms_norm_eps, False, pd, name="k_norm")(k)
        q, k = _apply_rotary(q, k, cos, sin)
        n_rep = h // kv
        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)
        cdt = jnp.promote_types(q.dtype, jnp.float32)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) * (hd ** -0.5)
        scores = scores + attn_bias.astype(cdt)
        w = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(b, s, h * hd)
        return dense(cfg.hidden_size, "o_proj", cfg.o_bias)(out)


class VLMLP(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        # gate/up/down all go through ``_proj`` — PEFT's "all-linear" adapts the MLP too.
        dense = lambda f, n: (lambda z: _proj(self, z, f, n, cfg, pd))
        gate = dense(cfg.intermediate_size, "gate_proj")(x)
        up = dense(cfg.intermediate_size, "up_proj")(x)
        act = jax.nn.gelu(gate, approximate=True) if cfg.mlp_act == "gelu" else nn.silu(gate)
        return dense(cfg.hidden_size, "down_proj")(act * up)


class VLDecoderLayer(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        norm = lambda n: VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_add_one, cfg.dtype, name=n)
        h = x + VLAttention(cfg, name="self_attn")(norm("input_layernorm")(x), cos, sin, attn_bias)
        return h + VLMLP(cfg, name="mlp")(norm("post_attention_layernorm")(h))


class _VLDecoders(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias, deepstack=None, vmask=None):
        for i in range(self.config.num_hidden_layers):
            x = VLDecoderLayer(self.config, name=str(i))(x, cos, sin, attn_bias)
            # Qwen3-VL deepstack: after decoder layers 0/1/2, add the merged features from
            # vision blocks [5,11,17] back at the image-token positions (torch
            # ``Qwen3VLTextModel._deepstack_process``). None on the text-only golden path.
            if deepstack is not None and i < len(deepstack):
                x = _scatter_add_deepstack(x, vmask, deepstack[i])
        return x


class VLLanguageModel(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, image_features=None,
                 position_ids=None, deepstack=None):
        cfg = self.config
        b, s = input_ids.shape
        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)
        if image_features is not None:
            hidden = _scatter_image_features(hidden, input_ids, image_features, cfg.image_token_id)
        if cfg.embed_scale:
            hidden = (hidden.astype(jnp.float32) * math.sqrt(cfg.hidden_size)).astype(hidden.dtype)
        if position_ids is not None:
            # 3D interleaved M-RoPE from the batch's (t, h, w) position_ids (the multimodal
            # training path). Text-only goldens pass no position_ids -> the 1-D arange path
            # below, which interleaved M-RoPE equals for pure-text sequences, so goldens stay
            # bit-identical.
            cos, sin = _mrope_cos_sin(position_ids, cfg.head_dim, cfg.rope_theta,
                                      cfg.mrope_section, cfg.mrope_interleaved, hidden.dtype)
        else:
            cos, sin = _rotary(jnp.arange(s)[None, :].repeat(b, axis=0), cfg.head_dim, cfg.rope_theta, hidden.dtype)
        attn_bias = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        if attention_mask is not None:
            attn_bias = attn_bias + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)
        vmask = (input_ids == cfg.image_token_id).reshape(-1) if deepstack is not None else None
        hidden = _VLDecoders(cfg, name="layers")(hidden, cos, sin, attn_bias, deepstack, vmask)
        hidden = VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_add_one, cfg.dtype, name="norm")(hidden)
        return hidden, embed


# --------------------------------------------------------------------------- #
# Qwen3-VL vision tower (mirror of torch ``Qwen3VLVisionModel`` + ``Qwen3VLVisionPatchMerger``).
# Frozen at train time (``freeze_vision_tower`` default, no LoRA adapter), so no gradient
# flows here; it runs in the forward to turn ``<|image_pad|>`` placeholders into embeddings
# and to emit the three deepstack features. The graded golden batch parks every image token
# at attention_mask=0, so this whole path is inert on the goldens and only moves the
# training loss.
# --------------------------------------------------------------------------- #


class _VisionLayerNorm(nn.Module):
    """torch ``nn.LayerNorm`` (weight+bias, float32 reduction). Names its gain ``scale`` /
    shift ``bias`` so the converted ViT weights load unchanged."""

    dim: int
    eps: float
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
        cdt = jnp.promote_types(x.dtype, jnp.float32)
        xf = x.astype(cdt)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        normed = (xf - mean) * jax.lax.rsqrt(var + self.eps)
        return (scale.astype(cdt) * normed + bias.astype(cdt)).astype(x.dtype)


def _vision_rope_blockmask(grid_list, head_dim, theta, merge):
    """Host-side (concrete grid) 2D vision RoPE tables + per-frame block-diagonal mask.

    ``grid_list`` is a list of concrete (t, h, w) patch-grid triples. Mirrors torch
    ``Qwen3VLVisionModel.rot_pos_emb`` + the ``cu_seqlens`` full-attention-per-frame mask
    (Qwen3-VL attends every patch within a frame, no windowing). Returns cos/sin
    (total_patches, head_dim) and an additive (total, total) mask (0 within a frame, -inf
    across frames).
    """
    import numpy as np

    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    pos_ids = []
    seqlens = []
    for (t, h, w) in grid_list:
        hpos = np.broadcast_to(np.arange(h)[:, None], (h, w))
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        wpos = np.broadcast_to(np.arange(w)[None, :], (h, w))
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        stacked = np.stack([hpos, wpos], axis=-1)  # (h*w, 2), 2x2-window order
        pos_ids.append(np.tile(stacked, (t, 1)))   # repeat over temporal frames
        seqlens.extend([h * w] * t)
    pos_ids = np.concatenate(pos_ids, axis=0)  # (total, 2)
    max_grid = max(max(h, w) for (_, h, w) in grid_list)
    table = np.outer(np.arange(max_grid), inv_freq)  # (max_grid, rotary_dim/2)
    rpe = table[pos_ids].reshape(pos_ids.shape[0], -1)  # (total, rotary_dim)
    emb = np.concatenate([rpe, rpe], axis=-1)           # (total, head_dim)
    cos = np.cos(emb)
    sin = np.sin(emb)
    block_id = np.concatenate([np.full(sl, i, dtype=np.int64) for i, sl in enumerate(seqlens)])
    same = block_id[:, None] == block_id[None, :]
    mask = np.where(same, 0.0, _NEG)
    return cos, sin, mask


def _fast_pos_embed_indices(grid_list, num_grid_per_side, merge):
    """Host-side bilinear indices/weights for the learned position embedding.

    Mirror of torch ``Qwen3VLVisionModel.fast_pos_embed_interpolate``: the learned
    (num_grid_per_side x num_grid_per_side) grid is sampled bilinearly at each of the h x w
    patch centres, tiled over the ``t`` frames, and reordered into the 2x2-merge-window
    patch order (so it lines up with the merger's fold and with ``_vision_rope_blockmask``).
    Returns ``idx`` (total_patches, 4) into the pos-embed table and ``weight``
    (total_patches, 4).
    """
    import numpy as np

    ng = num_grid_per_side
    out_idx, out_w = [], []
    for (t, h, w) in grid_list:
        h_idxs = np.linspace(0, ng - 1, h)
        w_idxs = np.linspace(0, ng - 1, w)
        h_floor = h_idxs.astype(np.int64)
        w_floor = w_idxs.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, None, ng - 1)
        w_ceil = np.clip(w_floor + 1, None, ng - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        base = h_floor * ng
        base_c = h_ceil * ng
        idx = np.stack([
            (base[:, None] + w_floor[None, :]).reshape(-1),
            (base[:, None] + w_ceil[None, :]).reshape(-1),
            (base_c[:, None] + w_floor[None, :]).reshape(-1),
            (base_c[:, None] + w_ceil[None, :]).reshape(-1),
        ], axis=-1)                                            # (h*w, 4)
        wt = np.stack([
            ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
            ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
            (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
            (dh[:, None] * dw[None, :]).reshape(-1),
        ], axis=-1)                                            # (h*w, 4)
        # window permute: output order [hb][wb][hi][wi] -> input flat (hb*m+hi)*w+(wb*m+wi)
        hb, wb = h // merge, w // merge
        a = np.arange(hb)[:, None, None, None]
        b_ = np.arange(wb)[None, :, None, None]
        hi = np.arange(merge)[None, None, :, None]
        wi = np.arange(merge)[None, None, None, :]
        perm = ((a * merge + hi) * w + (b_ * merge + wi)).reshape(-1)
        idx, wt = idx[perm], wt[perm]
        for _ in range(int(t)):
            out_idx.append(idx)
            out_w.append(wt)
    return (np.concatenate(out_idx, 0).astype(np.int64),
            np.concatenate(out_w, 0).astype(np.float64))


def _apply_vision_rotary(q, k, cos, sin):
    """Rotate (seq, heads, head_dim) q/k by per-token vision cos/sin (seq, head_dim)."""
    cdt = jnp.promote_types(q.dtype, jnp.float32)
    cos = cos[:, None, :].astype(cdt)
    sin = sin[:, None, :].astype(cdt)
    qf, kf = q.astype(cdt), k.astype(cdt)
    return ((qf * cos + _rotate_half(qf) * sin).astype(q.dtype),
            (kf * cos + _rotate_half(kf) * sin).astype(k.dtype))


class _Conv3dProj(nn.Module):
    """Conv3d patch projection kept as its native 5-D kernel [T, P, P, C, out] -- the layout
    the checkpoint loader's ``moveaxis`` produces for ``patch_embed.proj.weight`` -- applied
    by an einsum (kernel == stride == the whole patch). torch ``patch_embed.proj`` carries a
    bias, unlike Qwen2-VL."""

    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        C, T, P = cfg.vision_in_chans, cfg.vision_temporal_patch_size, cfg.vision_patch_size
        out = cfg.vision_embed_dim
        kernel = self.param("kernel", nn.initializers.zeros, (T, P, P, C, out), cfg.dtype)
        bias = self.param("bias", nn.initializers.zeros, (out,), cfg.dtype)
        xr = x.reshape(x.shape[0], C, T, P, P)
        y = jnp.einsum("nctpq,tpqco->no", xr, kernel.astype(x.dtype))
        return y + bias.astype(y.dtype)


class Qwen3VisionPatchEmbed(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x):
        return _Conv3dProj(self.config, name="proj")(x)


class Qwen3VisionAttention(nn.Module):
    """Fused qkv Linear(+bias) + proj Linear(+bias); MHA with per-frame full attention and
    2D vision RoPE (torch ``Qwen3VLVisionAttention``)."""

    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x, cos, sin, block_bias):
        cfg = self.config
        pd = cfg.dtype
        seq, _ = x.shape
        h = cfg.vision_num_heads
        hd = cfg.vision_embed_dim // h
        qkv = nn.Dense(3 * cfg.vision_embed_dim, use_bias=True, name="qkv", param_dtype=pd)(x)
        qkv = qkv.reshape(seq, 3, h, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (seq, heads, hd)
        q, k = _apply_vision_rotary(q, k, cos, sin)
        q = q.transpose(1, 0, 2)  # (heads, seq, hd)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        cdt = jnp.promote_types(q.dtype, jnp.float32)
        scores = jnp.einsum("hqd,hkd->hqk", q, k).astype(cdt) * (hd ** -0.5)
        scores = scores + block_bias.astype(cdt)[None]
        w = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,hkd->hqd", w, v).transpose(1, 0, 2).reshape(seq, h * hd)
        return nn.Dense(cfg.vision_embed_dim, use_bias=True, name="proj", param_dtype=pd)(out)


class Qwen3VisionMLP(nn.Module):
    """linear_fc1 -> gelu_pytorch_tanh -> linear_fc2 (torch ``Qwen3VLVisionMLP``, both
    linears biased)."""

    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        x = nn.Dense(cfg.vision_intermediate_size, use_bias=True, name="linear_fc1", param_dtype=pd)(x)
        x = jax.nn.gelu(x, approximate=True)  # gelu_pytorch_tanh
        return nn.Dense(cfg.vision_embed_dim, use_bias=True, name="linear_fc2", param_dtype=pd)(x)


class Qwen3VisionBlock(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x, cos, sin, block_bias):
        cfg = self.config
        ln = lambda n: _VisionLayerNorm(cfg.vision_embed_dim, cfg.vision_layer_norm_eps, cfg.dtype, name=n)
        x = x + Qwen3VisionAttention(cfg, name="attn")(ln("norm1")(x), cos, sin, block_bias)
        x = x + Qwen3VisionMLP(cfg, name="mlp")(ln("norm2")(x))
        return x


class _Qwen3VisionBlocks(nn.Module):
    """The ``blocks.{i}`` scope; also collects the intermediate hidden states at the
    deepstack indexes for the deepstack mergers."""

    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, x, cos, sin, block_bias):
        cfg = self.config
        wanted = set(cfg.vision_deepstack_indexes)
        collected = {}
        for i in range(cfg.vision_depth):
            x = Qwen3VisionBlock(cfg, name=str(i))(x, cos, sin, block_bias)
            if i in wanted:
                collected[i] = x
        return x, collected


class Qwen3VisionMerger(nn.Module):
    """PatchMerger: LayerNorm + fold the 2x2 window into channels -> linear_fc1 -> gelu ->
    linear_fc2 (torch ``Qwen3VLVisionPatchMerger``). ``postshuffle`` selects whether the
    norm runs on the folded merged width (the deepstack mergers, use_postshuffle_norm=True)
    or the per-patch width (the main merger)."""

    config: Qwen3VLConfig
    postshuffle: bool = False

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        merged_in = cfg.vision_embed_dim * (cfg.vision_spatial_merge_size ** 2)
        eps = cfg.vision_layer_norm_eps
        if self.postshuffle:
            x = x.reshape(-1, merged_in)
            x = _VisionLayerNorm(merged_in, eps, pd, name="norm")(x)
        else:
            x = _VisionLayerNorm(cfg.vision_embed_dim, eps, pd, name="norm")(x)
            x = x.reshape(-1, merged_in)  # 2x2-window ordered -> 4 consecutive = one block
        x = nn.Dense(merged_in, use_bias=True, name="linear_fc1", param_dtype=pd)(x)
        x = jax.nn.gelu(x, approximate=False)  # nn.GELU() default is exact erf
        return nn.Dense(cfg.vision_out_hidden_size, use_bias=True, name="linear_fc2", param_dtype=pd)(x)


class _DeepstackMergerList(nn.Module):
    """The ``deepstack_merger_list.{j}`` scope (one PatchMerger per deepstack index)."""

    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, collected_list):
        outs = []
        for j, feat in enumerate(collected_list):
            outs.append(Qwen3VisionMerger(self.config, postshuffle=True, name=str(j))(feat))
        return outs


class Qwen3VisionModel(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, pixel_values, pe_idx, pe_w, cos, sin, block_bias):
        cfg = self.config
        pd = cfg.dtype
        x = Qwen3VisionPatchEmbed(cfg, name="patch_embed")(pixel_values)   # (total, hidden)
        # learned position embedding, bilinearly interpolated to the (h, w) grid
        emb = nn.Embed(cfg.vision_num_position_embeddings, cfg.vision_embed_dim,
                       name="pos_embed", param_dtype=pd)
        pe = emb(pe_idx)                                        # (total, 4, hidden)
        pos = jnp.sum(pe * pe_w[:, :, None].astype(pe.dtype), axis=1)  # (total, hidden)
        x = x + pos.astype(x.dtype)
        x, collected = _Qwen3VisionBlocks(cfg, name="blocks")(x, cos, sin, block_bias)
        deep = _DeepstackMergerList(cfg, name="deepstack_merger_list")(
            [collected[i] for i in cfg.vision_deepstack_indexes])
        merged = Qwen3VisionMerger(cfg, postshuffle=False, name="merger")(x)
        return merged, deep


class _Wrap(nn.Module):
    child: nn.Module

    @nn.compact
    def __call__(self, *a, **k):
        return self.child(*a, **k)


class _NestedLM(nn.Module):
    """Emits the ``<prefix>`` wrapper scopes then the LanguageModel (prefix[0] is this
    module's own name; nest the rest so paths match ``<prefix>/layers/{i}``). This is the
    ``model`` scope, so the Qwen3-VL ViT (``model/visual``) -- a sibling of
    ``language_model`` in the checkpoint -- is built here."""

    config: Qwen3VLConfig

    def _image_features(self, pixel_values, image_grid_thw):
        """Qwen3-VL ViT -> (merged features, [deepstack_0, deepstack_1, deepstack_2]).

        ``merged`` is the flat (total_merged_tokens, hidden) block torch scatters over the
        ``<|image_pad|>`` placeholders; each deepstack entry is added back at the same
        positions after decoder layers 0/1/2. ``image_grid_thw`` must be concrete (eager
        training/eval), which it is when this branch runs."""
        import numpy as np

        cfg = self.config
        grid = np.asarray(image_grid_thw).reshape(-1, 3)
        grid_list = [(int(t), int(h), int(w)) for (t, h, w) in grid]
        hd = cfg.vision_embed_dim // cfg.vision_num_heads
        cos, sin, mask = _vision_rope_blockmask(
            grid_list, hd, cfg.vision_rope_theta, cfg.vision_spatial_merge_size)
        ng = int(round(cfg.vision_num_position_embeddings ** 0.5))
        pe_idx, pe_w = _fast_pos_embed_indices(grid_list, ng, cfg.vision_spatial_merge_size)
        pv = jnp.asarray(pixel_values).astype(cfg.dtype)
        merged, deep = Qwen3VisionModel(cfg, name="visual")(
            pv, jnp.asarray(pe_idx), jnp.asarray(pe_w, cfg.dtype),
            jnp.asarray(cos, cfg.dtype), jnp.asarray(sin, cfg.dtype),
            jnp.asarray(mask, jnp.float32))
        merged = merged.reshape(-1, merged.shape[-1])
        deep = [d.reshape(-1, d.shape[-1]) for d in deep]
        return merged, deep

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, pixel_values=None,
                 image_grid_thw=None, position_ids=None, run_vision=None,
                 image_features=None, deepstack=None, return_patch_features=False,
                 pe_idx=None, pe_w=None, vcos=None, vsin=None, block_bias=None):
        cfg = self.config
        # Vision-precompute fast half (dpo/trainer.py ``_precompute_image_features``): run the
        # frozen Qwen3-VL ViT ONCE over a set-wide-max padded patch batch. The grid-derived
        # vision-RoPE / learned-pos-embed / block-diagonal-attention tables are computed
        # host-side (concrete grid) and passed straight in, so this path carries NO
        # image_grid_thw host control flow and jits ONCE. Returns (merged features,
        # [deepstack_0, deepstack_1, deepstack_2]); the trainer slices off the inert pad rows
        # and feeds them back in below as ``image_features``/``deepstack`` (tower then skipped).
        if return_patch_features:
            return Qwen3VisionModel(cfg, name="visual")(
                jnp.asarray(pixel_values).astype(cfg.dtype), pe_idx, pe_w, vcos, vsin, block_bias)
        # When the trainer supplies precomputed ``image_features``/``deepstack`` the frozen tower
        # is skipped entirely (its output is CONSTANT across steps); otherwise compute them from
        # the tower exactly as before. The golden / text-only path passes neither -> both stay
        # None -> byte-identical to the pre-precompute forward.
        if image_features is None and pixel_values is not None and image_grid_thw is not None:
            # Run the ViT only when an <|image_pad|> placeholder sits at an attended
            # position. The graded forward and the finite-difference probe mask every image
            # token (attention_mask=0), so features reach no compared position and the graph
            # stays the text-only LM -- baseline-identical, probe leaf-span unchanged.
            # Real training leaves image tokens unmasked, so the tower runs. The
            # try/except mirrors the internvl pilot: jax.make_jaxpr lifts the closed-over
            # batch to tracers, so bool(jnp.any(...)) raises ConcretizationTypeError.
            if run_vision is not None:
                # In-band real-training path: ``run_vision`` is a STATIC python bool baked into
                # the jitted step, so this branch resolves at trace time (never a tracer -> no
                # ConcretizationTypeError) and the Qwen3-VL ViT runs UNDER jit -- fused, one
                # compile per pixel_values shape, ~torch speed instead of the eager op-by-op
                # dispatch (11-15x slower). Numerically identical to the eager guard below: real
                # training leaves image tokens unmasked, so that guard was always True here.
                if run_vision:
                    image_features, deepstack = self._image_features(pixel_values, image_grid_thw)
            else:
                try:
                    if attention_mask is None or bool(
                        jnp.any((input_ids == cfg.image_token_id) & (attention_mask > 0))
                    ):
                        image_features, deepstack = self._image_features(pixel_values, image_grid_thw)
                except jax.errors.ConcretizationTypeError:
                    image_features, deepstack = None, None
        # 3D M-RoPE + deepstack only when the tower actually ran; the golden text-only path
        # keeps the 1-D arange RoPE (which interleaved M-RoPE equals for pure-text ids).
        pos = position_ids if image_features is not None else None
        inner = cfg.prefix[1:]
        if len(inner) <= 1:
            return VLLanguageModel(cfg, name=inner[0] if inner else "language_model")(
                input_ids, attention_mask, image_features, pos, deepstack)
        node = VLLanguageModel(cfg, name=inner[-1])
        for nm in reversed(inner[:-1]):
            node = _Wrap(node, name=nm)
        return node(input_ids, attention_mask)


@flax.struct.dataclass
class CausalLMOutput:
    logits: Any
    hidden_states: Any = None
    values: Any = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3VLForConditionalGeneration LM-path causal LM (mirror of the torch class; text-only forward).
    The LM is nested under ``config.prefix`` (``model/language_model``)."""

    config: Qwen3VLConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        # Vision precompute (jittable half): run the frozen ViT on a fixed-width padded patch
        # batch (grid tables passed in) and hand back (merged features, [deepstack_0..2]). The
        # trainer caches this ONCE (frozen tower -> constant across steps) and feeds it back as
        # ``image_features``/``deepstack``; text-only / golden callers never take this path.
        if kwargs.get("return_patch_features"):
            return _NestedLM(cfg, name=cfg.prefix[0])(
                input_ids, attention_mask,
                pixel_values=kwargs.get("pixel_values"),
                return_patch_features=True,
                pe_idx=kwargs.get("pe_idx"), pe_w=kwargs.get("pe_w"),
                vcos=kwargs.get("vcos"), vsin=kwargs.get("vsin"),
                block_bias=kwargs.get("block_bias"),
            )
        # pixel_values / image_grid_thw / position_ids arrive via kwargs (the collator's
        # mm_inputs). Text-only callers (generate(), the test battery) pass none, so the
        # forward is byte-identical to before; on the graded golden the image tokens sit at
        # attention_mask=0 so the vision path stays inert. ``image_features``/``deepstack`` (when
        # supplied) are the trainer's precomputed frozen blocks -- the tower is then skipped.
        hidden, embed = _NestedLM(cfg, name=cfg.prefix[0])(
            input_ids, attention_mask,
            pixel_values=kwargs.get("pixel_values"),
            image_grid_thw=kwargs.get("image_grid_thw"),
            position_ids=kwargs.get("position_ids"),
            run_vision=kwargs.get("run_vision"),
            image_features=kwargs.get("image_features"),
            deepstack=kwargs.get("deepstack"),
        )
        if cfg.tie_word_embeddings:
            logits = embed.attend(hidden)
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head", param_dtype=cfg.dtype)(hidden)
        values = None
        if cfg.with_value_head:
            values = nn.Dense(1, use_bias=False, name="v_head", param_dtype=cfg.dtype)(hidden)[..., 0]
        return CausalLMOutput(logits=logits, hidden_states=None, values=values)

    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 8, **kwargs):
        params = {"params": object.__getattribute__(self, "_gen_params")}
        ids = jnp.asarray(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        for _ in range(int(max_new_tokens)):
            logits = self.apply(params, input_ids=ids, attention_mask=jnp.ones(ids.shape, jnp.int32)).logits
            ids = jnp.concatenate([ids, jnp.argmax(logits[:, -1].astype(jnp.float32), -1)[:, None]], axis=1)
        return ids

    def get_rope_index(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        """Numpy port of torch ``Qwen3VLForConditionalGeneration.get_rope_index`` (3D M-RoPE).

        The multimodal collator calls this to add ``position_ids`` (3, batch, seq) and
        ``rope_deltas`` (batch, 1) to the batch — mirroring what torch's model method
        produces, so the candidate's batch schema matches the reference's."""
        import numpy as np

        cfg = self.config
        sms = cfg.spatial_merge_size
        image_token_id, video_token_id = cfg.image_token_id, cfg.video_token_id
        vision_start_token_id = cfg.vision_start_token_id
        input_ids = np.asarray(input_ids)
        image_grid_thw = np.asarray(image_grid_thw) if image_grid_thw is not None else None
        video_grid_thw = np.asarray(video_grid_thw) if video_grid_thw is not None else None
        if video_grid_thw is not None:
            video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
            video_grid_thw[:, 0] = 1
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total = input_ids
            if attention_mask is None:
                attention_mask = np.ones_like(total)
            attention_mask = np.asarray(attention_mask)
            position_ids = np.ones((3, total.shape[0], total.shape[1]), dtype=total.dtype)
            image_index = video_index = 0
            for i in range(total.shape[0]):
                ids_i = total[i][attention_mask[i] == 1]
                vstart = np.argwhere(ids_i == vision_start_token_id).reshape(-1)
                vtok = ids_i[vstart + 1] if vstart.size else ids_i[:0]
                image_nums = int((vtok == image_token_id).sum())
                video_nums = int((vtok == video_token_id).sum())
                toks = ids_i.tolist()
                pos_list, st = [], 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    ed_image = toks.index(image_token_id, st) if (image_token_id in toks and remain_images > 0) else len(toks) + 1
                    ed_video = toks.index(video_token_id, st) if (video_token_id in toks and remain_videos > 0) else len(toks) + 1
                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]; image_index += 1; remain_images -= 1; ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]; video_index += 1; remain_videos -= 1; ed = ed_video
                    gt, gh, gw = int(t), int(h) // sms, int(w) // sms
                    text_len = ed - st
                    st_idx = int(pos_list[-1].max()) + 1 if pos_list else 0
                    pos_list.append(np.broadcast_to(np.arange(text_len)[None, :], (3, text_len)) + st_idx)
                    t_index = np.broadcast_to(np.arange(gt).reshape(-1, 1), (gt, gh * gw)).flatten()
                    h_index = np.broadcast_to(np.arange(gh).reshape(1, -1, 1), (gt, gh, gw)).flatten()
                    w_index = np.broadcast_to(np.arange(gw).reshape(1, 1, -1), (gt, gh, gw)).flatten()
                    pos_list.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + gt * gh * gw
                if st < len(toks):
                    st_idx = int(pos_list[-1].max()) + 1 if pos_list else 0
                    text_len = len(toks) - st
                    pos_list.append(np.broadcast_to(np.arange(text_len)[None, :], (3, text_len)) + st_idx)
                llm_positions = np.concatenate(pos_list, axis=1).reshape(3, -1)
                position_ids[:, i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(int(llm_positions.max()) + 1 - int(total[i].shape[0]))
            return position_ids, np.array(mrope_position_deltas).reshape(-1, 1)
        # text-only fallback (no vision): let the collator skip position_ids/rope_deltas
        return None


__all__ = ["Qwen3VLForConditionalGeneration", "Qwen3VLConfig", "AutoConfig", "lora_a_init"]
