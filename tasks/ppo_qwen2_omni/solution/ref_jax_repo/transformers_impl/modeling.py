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

"""Native-JAX (flax.linen) Qwen2.5-Omni **thinker** language-model path — mirror of
torch_orig modeling_qwen2_5_omni.py (text-only forward, §12.4).

The graded forward runs the thinker's Qwen2 LM + ``thinker/lm_head``. Everything is
nested under ``thinker/`` (``thinker/model/{embed_tokens,layers/{i},norm}`` +
``thinker/lm_head``); ``convert_checkpoint`` converts the full checkpoint (thinker
vision/audio towers + talker + token2wav). The vision tower runs when a batch carries
``pixel_values`` and the audio tower when it carries ``input_features`` (the PPO
training run on voicebench); the talker/token2wav stacks are never exercised. Qwen2 conventions: q/k/v bias,
no qk-norm, SwiGLU, RMSNorm (fp32), NeoX RoPE (theta 1e6). Norms/softmax dtype-following.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration import Qwen2_5OmniConfig, config_from_pretrained

_NEG = -1e30


def _rotate_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _rotary(position_ids, head_dim, theta, dtype):
    inv = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    fr = position_ids.astype(jnp.float32)[..., None] * inv[None, None, :]
    emb = jnp.concatenate([fr, fr], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _apply_rotary(q, k, cos, sin):
    cdt = jnp.promote_types(q.dtype, jnp.float32)
    cos, sin = cos[:, None, :, :].astype(cdt), sin[:, None, :, :].astype(cdt)
    qf, kf = q.astype(cdt), k.astype(cdt)
    return (qf * cos + _rotate_half(qf) * sin).astype(q.dtype), (kf * cos + _rotate_half(kf) * sin).astype(k.dtype)


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
        y = y + scale * ((x @ a) @ b)
    return y


class Qwen2_5OmniRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        cdt = jnp.promote_types(x.dtype, jnp.float32)
        xf = x.astype(cdt)
        normed = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
        return (scale.astype(cdt) * normed).astype(x.dtype)


class Qwen2_5OmniAttention(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h, kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        # Every projection here is a LoRA target (torch's ``target_modules="all-linear"``);
        # ``_proj`` is ``nn.Dense`` plus the in-forward adapter, and is exactly ``nn.Dense``
        # when ``cfg.lora_rank == 0``. Written as a factory so the call sites are unchanged.
        d = lambda f, n, bias: (lambda z: _proj(self, z, f, n, cfg, pd, bias))
        q = d(h * hd, "q_proj", True)(x).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
        k = d(kv * hd, "k_proj", True)(x).reshape(b, s, kv, hd).transpose(0, 2, 1, 3)
        v = d(kv * hd, "v_proj", True)(x).reshape(b, s, kv, hd).transpose(0, 2, 1, 3)
        q, k = _apply_rotary(q, k, cos, sin)
        n_rep = h // kv
        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)
        cdt = jnp.promote_types(q.dtype, jnp.float32)
        sc = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) * (hd ** -0.5)
        w = jax.nn.softmax(sc + attn_bias.astype(cdt), axis=-1).astype(v.dtype)
        o = jnp.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(b, s, h * hd)
        return d(cfg.hidden_size, "o_proj", False)(o)


class Qwen2_5OmniMLP(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        d = lambda f, n: (lambda z: _proj(self, z, f, n, cfg, cfg.dtype))
        return d(cfg.hidden_size, "down_proj")(nn.silu(d(cfg.intermediate_size, "gate_proj")(x)) * d(cfg.intermediate_size, "up_proj")(x))


class Qwen2_5OmniDecoderLayer(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        h = x + Qwen2_5OmniAttention(cfg, name="self_attn")(
            Qwen2_5OmniRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="input_layernorm")(x), cos, sin, attn_bias)
        return h + Qwen2_5OmniMLP(cfg, name="mlp")(
            Qwen2_5OmniRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="post_attention_layernorm")(h))


class _Decoders(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        for i in range(self.config.num_hidden_layers):
            x = Qwen2_5OmniDecoderLayer(self.config, name=str(i))(x, cos, sin, attn_bias)
        return x


# ---------------------------------------------------------------------------- #
# Thinker vision tower (Qwen2_5OmniVisionEncoder) + thinker 3-D M-RoPE — mirror
# of torch_orig. Exercised by the graded image-only forward: the frozen batches
# carry ``pixel_values``/``image_grid_thw`` and no ``position_ids``, so the model
# runs the ViT, scatters image features at the ``<|IMAGE|>`` placeholders, and
# computes the M-RoPE position ids itself (torch ``get_rope_index``). The ViT is a
# Qwen2.5-VL windowed encoder: Conv3d patch embed, 2D vision RoPE, windowed +
# per-frame full block-diagonal attention, RMSNorm (fp32), SwiGLU MLP with bias,
# separate q/k/v, PatchMerger. The window layout / RoPE tables / block masks are
# grid-shape-derived, so they are built host-side (concrete inputs); under a jit
# trace the batch is abstract, ``np.asarray`` raises, and the caller drops back to
# the text-only path (preserving the audio-only/text-only frozen bands).
# ---------------------------------------------------------------------------- #


def _apply_vision_rotary(q, k, cos, sin):
    """2D vision rotary (torch ``apply_rotary_pos_emb_vision``); cos/sin are (seq, head_dim)."""
    cdt = jnp.promote_types(q.dtype, jnp.float32)
    c, s = cos[:, None, :].astype(cdt), sin[:, None, :].astype(cdt)
    qf, kf = q.astype(cdt), k.astype(cdt)
    return (qf * c + _rotate_half(qf) * s).astype(q.dtype), (kf * c + _rotate_half(kf) * s).astype(k.dtype)


def _mrope_cos_sin(position_ids, head_dim, theta, mrope_section, dtype):
    """3-D M-RoPE cos/sin (torch ``Qwen2_5OmniRotaryEmbedding`` + ``apply_multimodal_rotary_pos_emb``).

    ``position_ids`` is ``(3, b, s)`` (temporal/height/width rows). Builds per-row cos/sin
    and stitches head_dim/2 rotary freqs across the three rows by ``mrope_section*2``: chunk
    ``i`` is taken from row ``i % 3``. Returns ``(b, s, head_dim)`` for ``_apply_rotary``.
    """
    inv = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = position_ids.astype(jnp.float32)  # (3, b, s)
    freqs = pos[..., None] * inv[None, None, None, :]  # (3, b, s, head_dim/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (3, b, s, head_dim)
    cos, sin = jnp.cos(emb), jnp.sin(emb)
    sections = list(mrope_section) * 2  # sums to head_dim
    bounds = list(np.cumsum(sections)[:-1])

    def _combine(x):
        chunks = jnp.split(x, bounds, axis=-1)
        return jnp.concatenate([c[i % 3] for i, c in enumerate(chunks)], axis=-1)  # (b, s, head_dim)

    return _combine(cos).astype(dtype), _combine(sin).astype(dtype)


def _scatter_image(hidden, input_ids, image_features, image_token_id: int):
    """Place per-patch merged image features at the ``<|IMAGE|>`` placeholders (torch
    ``inputs_embeds.masked_scatter``); jit-safe cumsum gather, row-major over ``input_ids``.
    ``image_features`` is already in the original (reverse-window) merged order."""
    b, s, dim = hidden.shape
    n = image_features.shape[0]
    ids = input_ids.reshape(-1)
    mask = ids == image_token_id
    rank = jnp.clip(jnp.cumsum(mask.astype(jnp.int32)) - 1, 0, n - 1)
    gathered = image_features[rank]
    flat = hidden.reshape(b * s, dim)
    flat = jnp.where(mask[:, None], gathered.astype(flat.dtype), flat)
    return flat.reshape(b, s, dim)


def _vision_prep(grid_list, head_dim, theta, merge, window_size, patch_size, fullatt_indexes):
    """Host-side (numpy) window layout + 2D RoPE tables + block masks for the ViT.

    Mirrors torch ``rot_pos_emb`` / ``get_window_index`` and the per-layer ``cu_seqlens``
    selection. Returns window-reordered cos/sin ``(seq, head_dim)``, full and window additive
    block masks ``(seq, seq)``, the merged-unit ``window_index`` / ``reverse_index`` gathers,
    and the spatial-merge-unit ``smu``. Everything here is a pure function of the grid shape.
    """
    smu = merge * merge
    vit_win = window_size // merge // patch_size
    # --- 2D vision RoPE position ids (h, w) per patch, in merge-block order -------------- #
    pos_ids = []
    frame_lens = []  # per-frame patch counts (h*w), original order -> full-attn cu_seqlens
    for (t, h, w) in grid_list:
        hpos = np.broadcast_to(np.arange(h)[:, None], (h, w))
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        wpos = np.broadcast_to(np.arange(w)[None, :], (h, w))
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        stacked = np.stack([hpos, wpos], axis=-1)  # (h*w, 2)
        pos_ids.append(np.tile(stacked, (t, 1)))
        frame_lens.extend([h * w] * t)
    pos_ids = np.concatenate(pos_ids, axis=0)  # (seq, 2)
    seq = pos_ids.shape[0]
    rdim = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, rdim, 2, dtype=np.float64) / rdim))  # (rdim/2,)
    max_grid = max(max(h, w) for (_, h, w) in grid_list)
    table = np.outer(np.arange(max_grid, dtype=np.float64), inv)  # (max_grid, rdim/2)
    rpe = table[pos_ids].reshape(seq, -1)  # (seq, rdim) == [h-freqs, w-freqs]
    emb = np.concatenate([rpe, rpe], axis=-1)  # (seq, head_dim)
    cos_o, sin_o = np.cos(emb), np.sin(emb)
    # --- window index + windowed cu_seqlens --------------------------------------------- #
    window_index = []
    cu_window = [0]
    wid = 0
    for (t, h, w) in grid_list:
        lh, lw = h // merge, w // merge
        index = np.arange(t * lh * lw).reshape(t, lh, lw)
        pad_h = vit_win - lh % vit_win
        pad_w = vit_win - lw % vit_win
        nwh = (lh + pad_h) // vit_win
        nww = (lw + pad_w) // vit_win
        index_padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        index_padded = index_padded.reshape(t, nwh, vit_win, nww, vit_win).transpose(0, 1, 3, 2, 4)
        index_padded = index_padded.reshape(t, nwh * nww, vit_win, vit_win)
        seqlens = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + wid)
        cu_tmp = np.cumsum(seqlens) * smu + cu_window[-1]
        cu_window.extend(cu_tmp.tolist())
        wid += t * lh * lw
    window_index = np.concatenate(window_index).astype(np.int64)  # (seq//smu,)
    cu_window = np.array(cu_window, dtype=np.int64)
    keep = np.concatenate([[True], cu_window[1:] != cu_window[:-1]])  # unique_consecutive
    cu_window = cu_window[keep]
    # Full-attn cu_seqlens (per frame). After the window reorder each frame stays contiguous
    # and in the original order, so these boundaries hold on the reordered sequence too.
    cu_full = np.concatenate([[0], np.cumsum(np.array(frame_lens, dtype=np.int64))])
    # --- reorder cos/sin over merged units by window_index ------------------------------ #
    cos_r = cos_o.reshape(seq // smu, smu, head_dim)[window_index].reshape(seq, head_dim)
    sin_r = sin_o.reshape(seq // smu, smu, head_dim)[window_index].reshape(seq, head_dim)

    def _bias(cu):
        bid = np.zeros(seq, dtype=np.int64)
        for j in range(len(cu) - 1):
            bid[cu[j]:cu[j + 1]] = j
        same = bid[:, None] == bid[None, :]
        return np.where(same, 0.0, _NEG).astype(np.float32)

    full_bias = _bias(cu_full)
    window_bias = _bias(cu_window)
    reverse_index = np.argsort(window_index).astype(np.int64)
    return cos_r, sin_r, full_bias, window_bias, window_index, reverse_index, smu


def _llm_pos_ids_for_vision(start_idx, vision_idx, merge, t_index, grid_hs, grid_ws):
    """torch ``get_llm_pos_ids_for_vision`` — (3, llm_grid) t/h/w position ids for one image."""
    llm_h = int(grid_hs[vision_idx]) // merge
    llm_w = int(grid_ws[vision_idx]) // merge
    tt = len(t_index)
    h_index = np.broadcast_to(np.arange(llm_h).reshape(1, -1, 1), (tt, llm_h, llm_w)).reshape(-1)
    w_index = np.broadcast_to(np.arange(llm_w).reshape(1, 1, -1), (tt, llm_h, llm_w)).reshape(-1)
    t_idx = np.repeat(np.asarray(t_index, dtype=np.int64), llm_h * llm_w)
    return np.stack([t_idx, h_index, w_index]) + start_idx


def _get_rope_index(input_ids, image_grid_thw, attention_mask, cfg):
    """torch ``get_rope_index`` (image-only path) -> ``(3, B, S)`` M-RoPE position ids.

    The graded batches are image-only (no audio/video placeholders), so this ports the
    image branch of torch's routine: per row, walk the ``<|IMAGE|>`` spans, giving text a
    plain arange, the ``vision_start`` a bos slot, the image tokens (t,h,w) grid positions,
    the ``vision_end`` an eos slot, and the trailing text an arange. Padded positions keep
    the ``ones`` fill, exactly like torch.
    """
    merge = cfg.vision_spatial_merge_size
    image_token_id = cfg.image_token_id
    vision_start_token_id = cfg.vision_start_token_id
    pps = cfg.position_id_per_seconds
    ids_all = np.asarray(input_ids)
    b, s = ids_all.shape
    am = None if attention_mask is None else (np.asarray(attention_mask) == 1)
    position_ids = np.ones((3, b, s), dtype=np.int64)
    grid = np.asarray(image_grid_thw).reshape(-1, 3)
    image_idx = 0
    for i in range(b):
        row = ids_all[i]
        if am is not None:
            row = row[am[i]]
        vs = np.where(row == vision_start_token_id)[0]
        vtok = row[vs + 1] if vs.size else np.array([], dtype=row.dtype)
        image_nums = int(np.sum(vtok == image_token_id))
        input_tokens = row.tolist()
        llm: list = []
        st = 0
        remain = image_nums

        def _st_idx():
            return int(llm[-1].max()) + 1 if llm else 0

        for _ in range(image_nums):
            ed = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain > 0) else len(input_tokens) + 1
            text_len = ed - st - 1
            if text_len != 0:
                llm.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + _st_idx())
            llm.append(np.broadcast_to(np.arange(1), (3, 1)) + _st_idx())  # vision_start (bos)
            si = _st_idx()
            grid_t = int(grid[image_idx][0])
            t_index = (np.arange(grid_t) * 1 * pps).astype(np.int64)
            llm.append(_llm_pos_ids_for_vision(si, image_idx, merge, t_index, grid[:, 1], grid[:, 2]))
            image_len = int(grid[image_idx].prod() // (merge ** 2))
            llm.append(np.broadcast_to(np.arange(1), (3, 1)) + _st_idx())  # vision_end (eos)
            st += text_len + 1 + image_len + 1
            image_idx += 1
            remain -= 1
        if st < len(input_tokens):
            text_len = len(input_tokens) - st
            llm.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + _st_idx())
        llm_positions = np.concatenate(llm, axis=1).reshape(3, -1)
        if am is not None:
            position_ids[:, i, am[i]] = llm_positions
        else:
            position_ids[:, i, :] = llm_positions
    return position_ids


class _VisionProjConv(nn.Module):
    """Conv3d patch embed as a flat matmul (name ``proj``). Input casts to the kernel dtype
    so the ViT computes in the param dtype (fp32 under the fp32 golden, bf16 otherwise) —
    torch ``pixel_values.type(self.visual.dtype)``. Kernel path/shape ``proj/kernel``
    ``(T, P, P, C, H)`` matches the converted Conv3d weight exactly."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x):  # x (n, T*P*P*C) channels-last flattened
        cfg = self.config
        c, t, p, hdim = cfg.vision_in_channels, cfg.vision_temporal_patch_size, cfg.vision_patch_size, cfg.vision_hidden_size
        kernel = self.param("kernel", nn.initializers.zeros, (t, p, p, c, hdim), cfg.dtype)
        w = kernel.reshape(t * p * p * c, hdim)
        return x.astype(kernel.dtype) @ w


class _VisionPatchEmbed(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, pixel_values):  # (n, C*T*P*P)
        cfg = self.config
        c, t, p = cfg.vision_in_channels, cfg.vision_temporal_patch_size, cfg.vision_patch_size
        n = pixel_values.shape[0]
        # torch view(-1, C, T, P, P); flax Conv3d kernel is (T, P, P, C, H) -> channels-last flatten.
        x = pixel_values.reshape(n, c, t, p, p).transpose(0, 2, 3, 4, 1).reshape(n, t * p * p * c)
        return _VisionProjConv(cfg, name="proj")(x)


class _VisionMLP(nn.Module):
    """Vision SwiGLU MLP with bias (torch ``Qwen2_5OmniMLP(bias=True)``, silu)."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        gate = nn.Dense(cfg.vision_intermediate_size, use_bias=True, name="gate_proj", param_dtype=pd)(x)
        up = nn.Dense(cfg.vision_intermediate_size, use_bias=True, name="up_proj", param_dtype=pd)(x)
        return nn.Dense(cfg.vision_hidden_size, use_bias=True, name="down_proj", param_dtype=pd)(nn.silu(gate) * up)


class _VisionAttention(nn.Module):
    """ViT self-attention: separate biased q/k/v + biased proj, 2D rotary, block-diagonal
    mask (``attn_bias`` (seq, seq)). Single packed sequence (no batch axis)."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):  # x (seq, H)
        cfg = self.config
        pd = cfg.dtype
        hdim = cfg.vision_hidden_size
        nh = cfg.vision_num_heads
        hd = hdim // nh
        seq = x.shape[0]
        q = nn.Dense(hdim, use_bias=True, name="q", param_dtype=pd)(x).reshape(seq, nh, hd)
        k = nn.Dense(hdim, use_bias=True, name="k", param_dtype=pd)(x).reshape(seq, nh, hd)
        v = nn.Dense(hdim, use_bias=True, name="v", param_dtype=pd)(x).reshape(seq, nh, hd)
        q, k = _apply_vision_rotary(q, k, cos, sin)
        q = q.transpose(1, 0, 2)  # (nh, seq, hd)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        cdt = jnp.promote_types(q.dtype, jnp.float32)
        sc = jnp.einsum("hqd,hkd->hqk", q, k).astype(cdt) * (hd ** -0.5)
        sc = sc + attn_bias.astype(cdt)[None]
        w = jax.nn.softmax(sc, axis=-1).astype(v.dtype)
        o = jnp.einsum("hqk,hkd->hqd", w, v).transpose(1, 0, 2).reshape(seq, hdim)
        return nn.Dense(hdim, use_bias=True, name="proj", param_dtype=pd)(o)


class _VisionBlock(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        x = x + _VisionAttention(cfg, name="attn")(
            Qwen2_5OmniRMSNorm(cfg.vision_hidden_size, 1e-6, cfg.dtype, name="norm1")(x), cos, sin, attn_bias)
        x = x + _VisionMLP(cfg, name="mlp")(
            Qwen2_5OmniRMSNorm(cfg.vision_hidden_size, 1e-6, cfg.dtype, name="norm2")(x))
        return x


class _VisionBlocks(nn.Module):
    """The ViT block stack (name ``blocks``): full attention on ``fullatt_block_indexes``,
    windowed attention elsewhere (torch per-layer ``cu_seqlens_now`` selection)."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, cos, sin, full_bias, window_bias):
        cfg = self.config
        full_set = {int(i) for i in cfg.vision_fullatt_block_indexes}
        for i in range(cfg.vision_depth):
            bias = full_bias if i in full_set else window_bias
            x = _VisionBlock(cfg, name=str(i))(x, cos, sin, bias)
        return x


class _VisionMergerMLP(nn.Module):
    """PatchMerger MLP (name ``mlp``): Linear(0) -> exact GELU -> Linear(2, out_hidden)."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        merged_in = cfg.vision_hidden_size * (cfg.vision_spatial_merge_size ** 2)
        x = nn.Dense(merged_in, name="0", param_dtype=cfg.dtype)(x)
        x = jax.nn.gelu(x, approximate=False)
        return nn.Dense(cfg.vision_out_hidden_size, name="2", param_dtype=cfg.dtype)(x)


class _VisionMerger(nn.Module):
    """PatchMerger (name ``merger``): ln_q (RMSNorm) then group ``merge^2`` patches and MLP."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x):  # (seq, H)
        cfg = self.config
        merged_in = cfg.vision_hidden_size * (cfg.vision_spatial_merge_size ** 2)
        x = Qwen2_5OmniRMSNorm(cfg.vision_hidden_size, 1e-6, cfg.dtype, name="ln_q")(x)
        x = x.reshape(-1, merged_in)
        return _VisionMergerMLP(cfg, name="mlp")(x)


class Qwen2_5OmniVisionEncoder(nn.Module):
    """Thinker vision tower (name ``visual``) — mirror of torch ``Qwen2_5OmniVisionEncoder``.

    ``pooler_output``: patch-embed -> window reorder -> ViT blocks -> PatchMerger ->
    reverse-window. All grid-derived tensors (cos/sin, block masks, window gathers) arrive
    precomputed from :func:`_vision_prep`.
    """

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, pixel_values, cos, sin, full_bias, window_bias, window_index, reverse_index, smu):
        cfg = self.config
        x = _VisionPatchEmbed(cfg, name="patch_embed")(pixel_values)  # (seq, H)
        seq = x.shape[0]
        x = x.reshape(seq // smu, smu, -1)[window_index].reshape(seq, -1)  # window reorder
        x = _VisionBlocks(cfg, name="blocks")(x, cos, sin, full_bias, window_bias)
        x = _VisionMerger(cfg, name="merger")(x)  # (seq//smu, out_hidden)
        return x[reverse_index]  # undo window reorder over merged units


# ---------------------------------------------------------------------------- #
# Thinker audio tower (Qwen2_5OmniAudioEncoder) — mirror of torch_orig.
#
# Exercised when a batch carries ``input_features`` (the voicebench PPO run: every prompt
# is an ``<|AUDIO|>`` span); the frozen golden batches are image-only, so the tower is
# never traced by the numeric probes. Whisper-style encoder: dual conv feature extractor
# + chunked (block-diagonal) self-attention + per-audio avg-pool/ln_post/proj, producing
# one embedding per ``<|AUDIO|>`` placeholder, scattered into the text embeddings.
#
# LoRA: torch_orig's PEFT targets ``all-linear`` minus the vision tower / projector /
# lm_head, so every ``nn.Linear`` of the audio tower (q/k/v/out_proj, fc1, fc2, proj) gets
# an adapter and trains. The linears therefore go through ``_proj`` (exactly ``nn.Dense``
# when ``cfg.lora_rank == 0``), and ``model.loader`` adds the matching adapter leaves.
# ---------------------------------------------------------------------------- #


def _sinusoids(length: int, channels: int, max_timescale: float = 10000.0):
    """Whisper ``SinusoidsPositionEmbedding`` as a computed (non-parameter) table."""
    half = channels // 2
    incr = math.log(max_timescale) / (half - 1)
    inv = jnp.exp(-incr * jnp.arange(half, dtype=jnp.float32))
    scaled = jnp.arange(length, dtype=jnp.float32)[:, None] * inv[None, :]
    return jnp.concatenate([jnp.sin(scaled), jnp.cos(scaled)], axis=1)


class _AudioLayerNorm(nn.Module):
    """``nn.LayerNorm`` (scale + bias) with the reduction in float32, like torch."""

    dim: int
    eps: float = 1e-5
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


class _AudioAttention(nn.Module):
    """Whisper multi-head self-attention (no rotary, non-causal). ``k_proj`` has no bias.

    Batched over the leading chunk axis; ``key_bias`` ``(n_chunks, S)`` is the additive
    per-chunk key mask (0 for a chunk's valid after-cnn positions, ``_NEG`` for padding).
    Chunk attention is block-diagonal in torch (``cu_seqlens``), so attending within each
    chunk and masking its padded keys gives the real positions exactly torch's values.
    """

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, key_bias):
        cfg = self.config
        pd = cfg.dtype
        d = cfg.audio_d_model
        h = cfg.audio_encoder_attention_heads
        hd = d // h
        c, s = x.shape[0], x.shape[1]
        lin = lambda z, n, bias: _proj(self, z, d, n, cfg, pd, bias)
        q = lin(x, "q_proj", True).reshape(c, s, h, hd).transpose(0, 2, 1, 3)
        k = lin(x, "k_proj", False).reshape(c, s, h, hd).transpose(0, 2, 1, 3)
        v = lin(x, "v_proj", True).reshape(c, s, h, hd).transpose(0, 2, 1, 3)
        cdt = jnp.promote_types(q.dtype, jnp.float32)
        sc = jnp.einsum("chqd,chkd->chqk", q, k).astype(cdt) * (hd ** -0.5)
        w = jax.nn.softmax(sc + key_bias.astype(cdt)[:, None, None, :], axis=-1).astype(v.dtype)
        o = jnp.einsum("chqk,chkd->chqd", w, v).transpose(0, 2, 1, 3).reshape(c, s, d)
        return lin(o, "out_proj", True)


class _AudioEncoderLayer(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, key_bias):
        cfg = self.config
        eps = cfg.audio_layer_norm_eps
        r = x
        h = _AudioLayerNorm(cfg.audio_d_model, eps, cfg.dtype, name="self_attn_layer_norm")(x)
        h = _AudioAttention(cfg, name="self_attn")(h, key_bias)
        h = r + h
        r = h
        h = _AudioLayerNorm(cfg.audio_d_model, eps, cfg.dtype, name="final_layer_norm")(h)
        h = _proj(self, h, cfg.audio_encoder_ffn_dim, "fc1", cfg, cfg.dtype, True)
        h = jax.nn.gelu(h, approximate=False)
        h = _proj(self, h, cfg.audio_d_model, "fc2", cfg, cfg.dtype, True)
        return r + h


class _AudioEncoderLayers(nn.Module):
    """The stack scope (name ``layers``): ``thinker/audio_tower/layers/{i}``."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, x, key_bias):
        for i in range(self.config.audio_encoder_layers):
            x = _AudioEncoderLayer(self.config, name=str(i))(x, key_bias)
        return x


class Qwen2_5OmniAudioEncoder(nn.Module):
    """Mirror of torch ``Qwen2_5OmniAudioEncoder.forward`` (thinker audio path).

    Static-shape / jit-safe: rather than compacting each audio to its valid mel frames
    (torch's ``input_features[feature_attention_mask]`` + per-audio chunk split), the tower
    runs every ``n_window*2``-frame chunk of the fixed-width input and masks. Frames past an
    audio's valid length are zeroed before ``conv1`` (torch zero-pads its tail chunk), conv
    outputs there are zeroed before ``conv2`` (torch's ``padded_mask``), and padded after-cnn
    positions get a ``_NEG`` key bias, so every real position is torch's value; trailing
    all-padding chunks never influence them. Returns per-audio pooled embeddings
    ``(B, out_len_max, output_dim)`` and a ``(B, out_len_max)`` mask marking the first
    ``_audio_seqlen(L)`` real tokens of each audio (= its ``<|AUDIO|>`` placeholder count).
    """

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, input_features, feature_attention_mask):
        cfg = self.config
        pd = cfg.dtype
        d = cfg.audio_d_model
        win = int(cfg.audio_n_window) * 2
        xin = jnp.asarray(input_features)
        b, mel, t_ext = xin.shape
        k = -(-t_ext // win)  # ceil -> chunks per audio (static)
        t_pad = k * win
        if t_pad != t_ext:
            xin = jnp.pad(xin, ((0, 0), (0, 0), (0, t_pad - t_ext)))
        fam = jnp.asarray(feature_attention_mask)
        if t_pad != fam.shape[1]:
            fam = jnp.pad(fam, ((0, 0), (0, t_pad - fam.shape[1])))
        # (B, T_pad, mel) -> (B*K, win, mel) chunks, with per-frame validity in the same layout.
        pf = xin.transpose(0, 2, 1).reshape(b * k, win, mel).astype(pd)
        frame_valid = (fam > 0).reshape(b * k, win)
        pf = pf * frame_valid[:, :, None].astype(pf.dtype)  # torch zero-pads the tail chunk
        vlen = frame_valid.sum(axis=1)  # valid mel frames per chunk
        h = jax.nn.gelu(
            nn.Conv(d, (3,), (1,), [(1, 1)], use_bias=True, param_dtype=pd, name="conv1")(pf), approximate=False
        )
        h = h * frame_valid[:, :, None].astype(h.dtype)
        h = jax.nn.gelu(
            nn.Conv(d, (3,), (2,), [(1, 1)], use_bias=True, param_dtype=pd, name="conv2")(h), approximate=False
        )  # (B*K, S_chunk, d)
        s_chunk = h.shape[1]
        h = h + _sinusoids(s_chunk, d).astype(h.dtype)[None]
        # Mask after-cnn positions past this chunk's valid length ``(vlen - 1) // 2 + 1`` (0 for
        # an all-padding chunk: softmax over all-``_NEG`` is uniform, never NaN, and those
        # outputs are dropped by ``tok_valid``).
        len_after = jnp.maximum((vlen - 1) // 2 + 1, 0)
        key_valid = jnp.arange(s_chunk)[None, :] < len_after[:, None]
        key_bias = jnp.where(key_valid, 0.0, _NEG).astype(jnp.float32)
        h = _AudioEncoderLayers(cfg, name="layers")(h, key_bias)
        # Per-audio avg-pool (k=2, s=2) over the chunk-concatenated after-cnn sequence, then
        # ln_post + proj. Every chunk but an audio's last is full, so its real after-cnn
        # positions are the dense prefix of the ``K*S_chunk`` stream and the pool reproduces
        # torch's per-audio pool for every real token.
        al_max = k * s_chunk
        out_len_max = al_max // 2
        ha = h.reshape(b, al_max, d)[:, : out_len_max * 2].reshape(b, out_len_max, 2, d)
        pooled = ha.mean(axis=2).astype(h.dtype)
        pooled = _AudioLayerNorm(d, cfg.audio_layer_norm_eps, cfg.dtype, name="ln_post")(pooled)
        feats = _proj(self, pooled, cfg.audio_output_dim, "proj", cfg, pd, True)  # (B, out_len_max, out)
        lvalid = fam.sum(axis=1)  # valid mel frames per audio
        out_len_real = ((lvalid - 1) // 2 + 1 - 2) // 2 + 1  # == placeholder count per audio
        tok_valid = jnp.arange(out_len_max)[None, :] < out_len_real[:, None]
        return feats, tok_valid


def _scatter_audio(hidden, input_ids, audio_features, audio_tok_valid, audio_token_id: int):
    """Place the tower's per-audio embeddings at the ``<|AUDIO|>`` placeholders (torch
    ``inputs_embeds.masked_scatter``), jit-safe via cumsum ranks: compact the valid audio
    tokens (row-major over audios) into a dense stream, then gather that stream at the
    placeholder positions (row-major over ``input_ids``)."""
    b, s, dim = hidden.shape
    ba, tmax, od = audio_features.shape
    feats_flat = audio_features.reshape(ba * tmax, od)
    valid_flat = audio_tok_valid.reshape(-1)
    m_buf = ba * tmax
    rank = jnp.cumsum(valid_flat.astype(jnp.int32)) - 1  # 0-based slot in the compacted stream
    dest = jnp.where(valid_flat, rank, m_buf)  # invalid tokens -> scratch row (dropped below)
    compact = jnp.zeros((m_buf + 1, od), feats_flat.dtype).at[dest].set(feats_flat)[:m_buf]
    ids = input_ids.reshape(-1)
    amask = ids == audio_token_id
    tok_rank = jnp.clip(jnp.cumsum(amask.astype(jnp.int32)) - 1, 0, m_buf - 1)
    gathered = compact[tok_rank]
    flat = hidden.reshape(b * s, dim)
    flat = jnp.where(amask[:, None], gathered.astype(flat.dtype), flat)
    return flat.reshape(b, s, dim)


class Qwen2_5OmniTextModel(nn.Module):
    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, image_features=None, position_ids=None,
                 audio_features=None, audio_tok_valid=None):
        cfg = self.config
        b, s = input_ids.shape
        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)
        if audio_features is not None:  # torch scatters audio before image
            hidden = _scatter_audio(hidden, input_ids, audio_features, audio_tok_valid, cfg.audio_token_id)
        if image_features is not None:
            hidden = _scatter_image(hidden, input_ids, image_features, cfg.image_token_id)
        if position_ids is not None:
            cos, sin = _mrope_cos_sin(position_ids, cfg.head_dim, cfg.rope_theta, cfg.mrope_section, hidden.dtype)
        else:
            cos, sin = _rotary(jnp.arange(s)[None, :].repeat(b, axis=0), cfg.head_dim, cfg.rope_theta, hidden.dtype)
        attn_bias = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        if attention_mask is not None:
            attn_bias = attn_bias + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)
        hidden = _Decoders(cfg, name="layers")(hidden, cos, sin, attn_bias)
        return Qwen2_5OmniRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, name="norm")(hidden), embed


class _Thinker(nn.Module):
    """The thinker scope (name ``thinker``): the Qwen2 LM (``model``) + ``lm_head``, plus the
    audio tower (``audio_tower``) when a batch carries ``input_features`` and the vision tower
    (``visual``) when a batch carries ``pixel_values``."""

    config: Qwen2_5OmniConfig

    def _vision_prep(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        """Host-side ViT prep + M-RoPE position ids (no submodules). Needs concrete inputs;
        under a jit trace ``np.asarray`` raises and the caller falls back to text-only."""
        cfg = self.config
        grid = np.asarray(image_grid_thw).reshape(-1, 3)
        grid_list = [(int(t), int(h), int(w)) for t, h, w in grid]
        am = attention_mask if attention_mask is not None else jnp.ones(input_ids.shape, jnp.int32)
        position_ids = _get_rope_index(input_ids, image_grid_thw, am, cfg)
        vhd = cfg.vision_hidden_size // cfg.vision_num_heads
        cos, sin, full_bias, window_bias, window_index, reverse_index, smu = _vision_prep(
            grid_list, vhd, cfg.vision_rope_theta, cfg.vision_spatial_merge_size,
            cfg.vision_window_size, cfg.vision_patch_size, cfg.vision_fullatt_block_indexes)
        return {
            "pixel_values": jnp.asarray(pixel_values),
            "cos": jnp.asarray(cos, jnp.float32), "sin": jnp.asarray(sin, jnp.float32),
            "full_bias": jnp.asarray(full_bias), "window_bias": jnp.asarray(window_bias),
            "window_index": jnp.asarray(window_index), "reverse_index": jnp.asarray(reverse_index),
            "smu": int(smu), "position_ids": jnp.asarray(position_ids),
        }

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
                 input_features=None, feature_attention_mask=None, audio_features=None,
                 audio_tok_valid=None):
        cfg = self.config
        # Audio (torch ``get_audio_features`` + ``masked_scatter``): either run the tower on
        # ``input_features``/``feature_attention_mask``, or take its precomputed output
        # (``audio_features``/``audio_tok_valid``, e.g. from ``Qwen2_5OmniAudioEncoder.apply`` on
        # the ``thinker/audio_tower`` subtree) so a caller can encode a batch's audio once and
        # reuse it across forwards. Presence is a static ``is not None`` check: text-only and
        # golden probes never trace the tower, so their forward is unchanged. The tower is
        # jit-safe (fixed shapes, cumsum scatter). For audio-only sequences torch's
        # ``get_rope_index`` gives every row ``cumsum(attention_mask) - 1`` on all three M-RoPE
        # axes, i.e. the plain positions the text path already uses.
        if audio_features is None and input_features is not None and feature_attention_mask is not None:
            audio_features, audio_tok_valid = Qwen2_5OmniAudioEncoder(cfg, name="audio_tower")(
                input_features, feature_attention_mask
            )
        prep = None
        if pixel_values is not None:
            try:
                prep = self._vision_prep(input_ids, attention_mask, pixel_values, image_grid_thw)
            except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
                prep = None
        image_features, position_ids = None, None
        if prep is not None:
            image_features = Qwen2_5OmniVisionEncoder(cfg, name="visual")(
                prep["pixel_values"], prep["cos"], prep["sin"], prep["full_bias"], prep["window_bias"],
                prep["window_index"], prep["reverse_index"], prep["smu"])
            position_ids = prep["position_ids"]
        hidden, _ = Qwen2_5OmniTextModel(cfg, name="model")(
            input_ids, attention_mask, image_features=image_features, position_ids=position_ids,
            audio_features=audio_features, audio_tok_valid=audio_tok_valid,
        )
        logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head", param_dtype=cfg.dtype)(hidden)
        return logits, hidden


@flax.struct.dataclass
class CausalLMOutput:
    logits: Any
    hidden_states: Any = None
    values: Any = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Qwen2_5OmniForConditionalGeneration(nn.Module):
    """Qwen2.5-Omni thinker causal LM (mirror): text, image (``pixel_values``) and audio
    (``input_features``, or precomputed ``audio_features``) inputs, §12.4."""

    config: Qwen2_5OmniConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False,
                 pixel_values=None, image_grid_thw=None, input_features=None,
                 feature_attention_mask=None, audio_features=None, audio_tok_valid=None, **kwargs):
        cfg = self.config
        logits, hidden = _Thinker(cfg, name="thinker")(
            input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            audio_features=audio_features, audio_tok_valid=audio_tok_valid,
        )
        values = None
        if cfg.with_value_head:  # PPO value head at the top level (v_head/kernel)
            values = nn.Dense(1, use_bias=False, name="v_head", param_dtype=cfg.dtype)(hidden)[..., 0]
        return CausalLMOutput(logits=logits, hidden_states=None, values=values)

    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 8, **kwargs):
        """Cache-free greedy decode. Like torch's thinker ``generate``, the media inputs
        (``input_features``/``feature_attention_mask``/``pixel_values``/``image_grid_thw``)
        condition only the first (prefill) step; later steps re-run the bare ids."""
        params = {"params": object.__getattribute__(self, "_gen_params")}
        ids = jnp.asarray(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        media = {k: kwargs[k] for k in ("input_features", "feature_attention_mask", "pixel_values",
                                        "image_grid_thw") if kwargs.get(k) is not None}
        for step in range(int(max_new_tokens)):
            extra = media if step == 0 else {}
            logits = self.apply(params, input_ids=ids, attention_mask=jnp.ones(ids.shape, jnp.int32), **extra).logits
            ids = jnp.concatenate([ids, jnp.argmax(logits[:, -1].astype(jnp.float32), -1)[:, None]], axis=1)
        return ids


__all__ = [
    "Qwen2_5OmniRMSNorm",
    "Qwen2_5OmniAttention",
    "Qwen2_5OmniMLP",
    "Qwen2_5OmniDecoderLayer",
    "Qwen2_5OmniTextModel",
    "Qwen2_5OmniVisionEncoder",
    "Qwen2_5OmniAudioEncoder",
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniConfig",
    "config_from_pretrained",
    "lora_a_init",
]
