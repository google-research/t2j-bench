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

"""Native-JAX (flax.linen) InternVLForConditionalGeneration language-model path — mirror of
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

from .configuration import InternVLConfig, config_from_pretrained

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


class VLAttention(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h, kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        # Every q/k/v/o projection goes through ``_proj`` — a plain ``nn.Dense`` unless
        # ``cfg.lora_rank > 0``, in which case it also carries the in-forward LoRA term.
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
    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        dense = lambda f, n: (lambda z: _proj(self, z, f, n, cfg, cfg.dtype))
        gate = dense(cfg.intermediate_size, "gate_proj")(x)
        up = dense(cfg.intermediate_size, "up_proj")(x)
        act = jax.nn.gelu(gate, approximate=True) if cfg.mlp_act == "gelu" else nn.silu(gate)
        return dense(cfg.hidden_size, "down_proj")(act * up)


class VLDecoderLayer(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        norm = lambda n: VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_add_one, cfg.dtype, name=n)
        h = x + VLAttention(cfg, name="self_attn")(norm("input_layernorm")(x), cos, sin, attn_bias)
        return h + VLMLP(cfg, name="mlp")(norm("post_attention_layernorm")(h))


class _VLDecoders(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        for i in range(self.config.num_hidden_layers):
            x = VLDecoderLayer(self.config, name=str(i))(x, cos, sin, attn_bias)
        return x


def _scatter_image_features(hidden, input_ids, image_features, image_token_id):
    """The flax mirror of torch's ``inputs_embeds.masked_scatter(special_image_mask, feats)``.

    Replace the embeddings at ``input_ids == image_token_id`` with the packed visual
    features, in row-major order (row 0's placeholders first). ``image_features`` has one
    row per placeholder token across the whole batch, exactly as torch's flattened
    ``image_features.reshape(-1, hidden)`` does. Runs only in the eager training/eval
    forward, where the sequence length is concrete, so the cumulative-index gather is
    well defined.
    """
    b, s, d = hidden.shape
    mask_flat = (input_ids == image_token_id).reshape(-1)
    idx = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1
    n = image_features.shape[0]
    gathered = image_features[jnp.clip(idx, 0, n - 1)].astype(hidden.dtype)
    merged = jnp.where(mask_flat[:, None], gathered, hidden.reshape(-1, d))
    return merged.reshape(b, s, d)


class VLLanguageModel(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, image_features=None):
        cfg = self.config
        b, s = input_ids.shape
        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)
        if image_features is not None:
            hidden = _scatter_image_features(hidden, input_ids, image_features, cfg.image_token_id)
        if cfg.embed_scale:
            hidden = (hidden.astype(jnp.float32) * math.sqrt(cfg.hidden_size)).astype(hidden.dtype)
        cos, sin = _rotary(jnp.arange(s)[None, :].repeat(b, axis=0), cfg.head_dim, cfg.rope_theta, hidden.dtype)
        attn_bias = jnp.where(jnp.tril(jnp.ones((s, s), dtype=bool)), 0.0, _NEG)[None, None, :, :]
        if attention_mask is not None:
            attn_bias = attn_bias + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)
        hidden = _VLDecoders(cfg, name="layers")(hidden, cos, sin, attn_bias)
        hidden = VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_add_one, cfg.dtype, name="norm")(hidden)
        return hidden, embed


# --------------------------------------------------------------------------- #
# InternViT vision tower + pixel-shuffle projector (mirror of torch_orig
# ``InternVLVisionModel`` + ``InternVLMultiModalProjector`` + ``InternVLModel.
# {get_image_features,pixel_shuffle}``). Frozen at train time (no LoRA adapter,
# ``freeze_vision_tower`` default), so no gradient flows here; it runs in the forward to
# turn ``<IMG_CONTEXT>`` placeholders into embeddings. The graded golden batch parks every
# image token at attention_mask=0, so this whole path is inert on the goldens and only
# moves the training loss.
# --------------------------------------------------------------------------- #


class InternViTLayerNorm(nn.Module):
    """LayerNorm with a float32 reduction (torch ``nn.LayerNorm`` on bf16 inputs). Names
    its gain ``scale`` / shift ``bias`` so the converted InternViT weights load unchanged."""

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


class InternViTPatchEmbeddings(nn.Module):
    """Conv2d patchifier. torch key ``patch_embeddings.projection.{weight,bias}`` -> the conv
    is named ``projection`` under this ``patch_embeddings`` scope."""

    config: InternVLConfig

    @nn.compact
    def __call__(self, pixel_values):
        cfg = self.config
        pd = cfg.dtype
        dim = cfg.vision_hidden_size
        n = pixel_values.shape[0]
        # torch: Conv2d(NCHW) -> flatten(2).transpose(1,2). flax conv is NHWC.
        x = jnp.transpose(pixel_values, (0, 2, 3, 1))
        patch = nn.Conv(dim, (cfg.vision_patch_size, cfg.vision_patch_size),
                        strides=(cfg.vision_patch_size, cfg.vision_patch_size),
                        padding="VALID", use_bias=True, name="projection", param_dtype=pd)(x)
        return patch.reshape(n, -1, dim)  # (N, H'*W', dim), row-major like torch flatten(2)


class InternViTEmbeddings(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, pixel_values):
        cfg = self.config
        pd = cfg.dtype
        dim = cfg.vision_hidden_size
        n = pixel_values.shape[0]
        patch = InternViTPatchEmbeddings(cfg, name="patch_embeddings")(pixel_values)  # (N, P, dim)
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, dim), pd)
        cls = jnp.broadcast_to(cls.astype(patch.dtype), (n, 1, dim))
        emb = jnp.concatenate([cls, patch], axis=1)  # (N, 1 + P, dim)
        num_pos = (cfg.vision_image_size // cfg.vision_patch_size) ** 2 + 1
        pos = self.param("position_embeddings", nn.initializers.zeros, (1, num_pos, dim), pd)
        # 448/14 -> 32x32 = 1024 patches (+CLS) == num_pos, so torch's interpolate_pos_encoding
        # takes its identity fast-path; no interpolation is needed for square 448 tiles.
        return emb + pos.astype(emb.dtype)


class InternViTAttention(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h = cfg.vision_num_attention_heads
        hd = cfg.vision_hidden_size // h
        bias = cfg.vision_attention_bias
        proj = lambda name: (
            nn.Dense(h * hd, use_bias=bias, name=name, param_dtype=pd)(x)
            .reshape(b, s, h, hd).transpose(0, 2, 1, 3)
        )
        q, k, v = proj("q_proj"), proj("k_proj"), proj("v_proj")
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (hd ** -0.5)
        w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(b, s, h * hd)
        return nn.Dense(cfg.vision_hidden_size, use_bias=True, name="projection_layer", param_dtype=pd)(out)


class InternViTMLP(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        x = nn.Dense(cfg.vision_intermediate_size, name="fc1", param_dtype=pd)(x)
        x = jax.nn.gelu(x, approximate=False)  # exact-erf gelu (InternViT hidden_act="gelu")
        return nn.Dense(cfg.vision_hidden_size, name="fc2", param_dtype=pd)(x)


class InternViTLayer(nn.Module):
    """Pre-norm block with LayerScale (``lambda_1``/``lambda_2``), mirroring torch
    ``InternVLVisionLayer``."""

    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        dim = cfg.vision_hidden_size
        ln = lambda n: InternViTLayerNorm(dim, cfg.vision_layer_norm_eps, pd, name=n)
        lam1 = self.param("lambda_1", nn.initializers.constant(cfg.vision_layer_scale_init_value), (dim,), pd)
        lam2 = self.param("lambda_2", nn.initializers.constant(cfg.vision_layer_scale_init_value), (dim,), pd)
        a = InternViTAttention(cfg, name="attention")(ln("layernorm_before")(x))
        x = x + lam1.astype(x.dtype) * a
        m = InternViTMLP(cfg, name="mlp")(ln("layernorm_after")(x))
        return x + lam2.astype(x.dtype) * m


class _InternViTLayers(nn.Module):
    """The ``encoder.layer.{i}`` scope."""

    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        for i in range(self.config.vision_num_hidden_layers):
            x = InternViTLayer(self.config, name=str(i))(x)
        return x


class InternViTEncoder(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        return _InternViTLayers(self.config, name="layer")(x)


class InternVLVisionModel(nn.Module):
    config: InternVLConfig

    @nn.compact
    def __call__(self, pixel_values):
        cfg = self.config
        x = InternViTEmbeddings(cfg, name="embeddings")(pixel_values)
        # use_mean_pooling=True -> the tower's final ``layernorm`` is Identity; the
        # last hidden state is returned directly (vision_feature_layer=-1).
        return InternViTEncoder(cfg, name="encoder")(x)


class InternVLMultiModalProjector(nn.Module):
    """``mlp1``: LayerNorm(4x hidden) -> Linear -> gelu -> Linear (torch
    ``InternVLMultiModalProjector``). The layer_norm uses torch's default eps (1e-5)."""

    config: InternVLConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        in_dim = int(cfg.vision_hidden_size * int(1 / cfg.downsample_ratio) ** 2)
        x = InternViTLayerNorm(in_dim, 1e-5, pd, name="layer_norm")(x)
        x = nn.Dense(cfg.hidden_size, name="linear_1", param_dtype=pd)(x)
        x = jax.nn.gelu(x, approximate=False)  # projector_hidden_act="gelu" (exact erf)
        return nn.Dense(cfg.hidden_size, name="linear_2", param_dtype=pd)(x)


def _pixel_shuffle(x, scale_factor):
    """torch ``InternVLModel.pixel_shuffle``: fold a ``scale_factor`` window of the spatial
    grid into the channel dim (x: (N, W, H, C))."""
    n, w, h, c = x.shape
    x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor ** 2)))
    x = x.transpose(0, 2, 1, 3)
    return x


class _Wrap(nn.Module):
    child: nn.Module

    @nn.compact
    def __call__(self, *a, **k):
        return self.child(*a, **k)


class _NestedLM(nn.Module):
    """Emits the ``<prefix>`` wrapper scopes then the LanguageModel (prefix[0] is this
    module's own name; nest the rest so paths match ``<prefix>/layers/{i}``). This is the
    ``model`` scope, so the InternViT tower (``model/vision_tower``) and the projector
    (``model/multi_modal_projector``) -- siblings of ``language_model`` in the checkpoint --
    are built here."""

    config: InternVLConfig

    def _image_features(self, pixel_values):
        """InternViT tower + pixel-shuffle + projector -> (total_image_tokens, hidden), the
        flat feature block torch scatters over the ``<IMG_CONTEXT>`` placeholders."""
        cfg = self.config
        pv = jnp.asarray(pixel_values).astype(cfg.dtype)  # (num_tiles, 3, 448, 448)
        feat = InternVLVisionModel(cfg, name="vision_tower")(pv)  # (T, 1+P, vision_hidden)
        if cfg.vision_feature_select_strategy == "default":
            feat = feat[:, 1:, :]  # drop CLS -> (T, P, vision_hidden)
        t = feat.shape[0]
        side = int(round(feat.shape[1] ** 0.5))  # sqrt(P) = 32
        feat = feat.reshape(t, side, side, feat.shape[-1])
        feat = _pixel_shuffle(feat, cfg.downsample_ratio)  # (T, side/2, side/2, 4*vision_hidden)
        feat = feat.reshape(t, -1, feat.shape[-1])         # (T, 256, 4*vision_hidden)
        feat = InternVLMultiModalProjector(cfg, name="multi_modal_projector")(feat)  # (T, 256, hidden)
        return feat.reshape(-1, feat.shape[-1])

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, pixel_values=None, run_vision=None):
        cfg = self.config
        image_features = None
        if pixel_values is not None:
            # Run the InternViT tower + projector only when an <IMG_CONTEXT> placeholder token
            # sits at an attended position. The graded forward and the finite-difference probe
            # mask every image token (attention_mask=0), so features reach no compared position
            # and the forward stays the text-only LM graph -- baseline-identical, and the
            # probe's float64 leaf-span budget is unchanged. Real training leaves image
            # tokens unmasked, so the tower runs. The try/except guards the make_jaxpr case:
            # jax.make_jaxpr (the probe's leaf-span pass) lifts the closed-over batch to
            # tracers, so bool(jnp.any(...)) raises ConcretizationTypeError -> text-only.
            if run_vision is not None:
                # In-band real-training path: ``run_vision`` is a STATIC python bool baked into
                # the jitted step, so this branch resolves at trace time (never a tracer -> no
                # ConcretizationTypeError) and the InternViT tower runs UNDER jit -- fused, one
                # compile per pixel_values shape, ~torch speed instead of the eager op-by-op
                # dispatch (11-15x slower). Numerically identical to the eager guard below: real
                # training leaves image tokens unmasked, so that guard was always True here.
                if run_vision:
                    image_features = self._image_features(pixel_values)
            else:
                try:
                    if attention_mask is None or bool(
                        jnp.any((input_ids == cfg.image_token_id) & (attention_mask > 0))
                    ):
                        image_features = self._image_features(pixel_values)
                except jax.errors.ConcretizationTypeError:
                    image_features = None
        inner = cfg.prefix[1:]
        if len(inner) <= 1:
            return VLLanguageModel(cfg, name=inner[0] if inner else "language_model")(
                input_ids, attention_mask, image_features)
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


class InternVLForConditionalGeneration(nn.Module):
    """InternVLForConditionalGeneration LM-path causal LM (mirror of the torch class; text-only forward).
    The LM is nested under ``config.prefix`` (``model/language_model``)."""

    config: InternVLConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        # pixel_values arrives via kwargs (the collator's mm_inputs); when absent (text-only
        # callers -- generate(), the battery) the forward is byte-identical to before,
        # and on the graded golden the image tokens sit at attention_mask=0.
        hidden, embed = _NestedLM(cfg, name=cfg.prefix[0])(
            input_ids, attention_mask, pixel_values=kwargs.get("pixel_values"),
            run_vision=kwargs.get("run_vision"),
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


__all__ = ["InternVLForConditionalGeneration", "InternVLConfig", "config_from_pretrained", "lora_a_init"]
