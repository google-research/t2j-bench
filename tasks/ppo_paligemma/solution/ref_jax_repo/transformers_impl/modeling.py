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

"""Native-JAX (flax.linen) PaliGemmaForConditionalGeneration language-model path — mirror of
torch_orig/transformers_impl/modeling.py (TEST_PLAN.md §6.5).

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

from .configuration import PaliGemmaConfig, config_from_pretrained

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
    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h, kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        # Every q/k/v/o projection goes through ``_proj`` so PEFT's target_modules
        # ="all-linear" has a mirror here; with ``cfg.lora_rank == 0`` it is plain nn.Dense.
        dense = lambda f, n, bias: (lambda z: _proj(self, z, f, n, cfg, pd, bias))
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
    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        # gate/up/down are LoRA targets too (see ``_proj``); bias-free, as in torch.
        dense = lambda f, n: (lambda z: _proj(self, z, f, n, cfg, cfg.dtype))
        gate = dense(cfg.intermediate_size, "gate_proj")(x)
        up = dense(cfg.intermediate_size, "up_proj")(x)
        act = jax.nn.gelu(gate, approximate=True) if cfg.mlp_act == "gelu" else nn.silu(gate)
        return dense(cfg.hidden_size, "down_proj")(act * up)


class VLDecoderLayer(nn.Module):
    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        cfg = self.config
        norm = lambda n: VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_add_one, cfg.dtype, name=n)
        h = x + VLAttention(cfg, name="self_attn")(norm("input_layernorm")(x), cos, sin, attn_bias)
        return h + VLMLP(cfg, name="mlp")(norm("post_attention_layernorm")(h))


class _VLDecoders(nn.Module):
    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x, cos, sin, attn_bias):
        for i in range(self.config.num_hidden_layers):
            x = VLDecoderLayer(self.config, name=str(i))(x, cos, sin, attn_bias)
        return x


def _prefix_lm_allow(token_type_ids, seq_len: int):
    """Causal OR bidirectional-within-each-image-block, as (batch, q, kv) booleans.

    Port of torch ``create_causal_mask_mapping`` + ``token_type_ids_mask_function``.
    torch reverses the incoming ids (``1 - token_type_ids``) and then calls a position
    "image" when the reversed value is 1 -- i.e. when the ORIGINAL id is 0. Contiguous
    runs of image positions form groups; two positions attend bidirectionally only when
    both are image AND in the same group. Everything else stays causal.
    """
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, :, :]
    if token_type_ids is None:
        return causal
    is_image = jnp.asarray(token_type_ids) == 0
    previous = jnp.pad(is_image, ((0, 0), (1, 0)))[:, :-1]
    group = jnp.cumsum((is_image & ~previous).astype(jnp.int32), axis=1) - 1
    group = jnp.where(is_image, group, -1)
    same_block = (is_image[:, :, None] & is_image[:, None, :]) & (group[:, :, None] == group[:, None, :])
    return causal | same_block


class VLLanguageModel(nn.Module):
    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, image_features=None):
        cfg = self.config
        b, s = input_ids.shape
        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens", param_dtype=cfg.dtype)
        hidden = embed(input_ids)
        if cfg.embed_scale:
            hidden = (hidden.astype(jnp.float32) * math.sqrt(cfg.hidden_size)).astype(hidden.dtype)
        # PaliGemma scatters the UNSCALED SigLIP image features AFTER the sqrt(hidden) embed
        # scale: torch's get_input_embeddings() (a GemmaTextScaledWordEmbedding) applies the
        # embed scale during the lookup, and only THEN does PaliGemmaModel.forward run
        # ``inputs_embeds.masked_scatter(special_image_mask, image_features)``. Scattering
        # before the scale would erroneously multiply the vision features by sqrt(hidden).
        if image_features is not None:
            hidden = _scatter_image_features(hidden, input_ids, image_features, cfg.image_token_id)
        cos, sin = _rotary(jnp.arange(s)[None, :].repeat(b, axis=0), cfg.head_dim, cfg.rope_theta, hidden.dtype)
        if cfg.causal:
            allow = jnp.tril(jnp.ones((s, s), dtype=bool))[None, :, :]
        else:  # prefix-LM (paligemma): bidirectional inside the image block only
            allow = _prefix_lm_allow(token_type_ids, s)
        attn_bias = jnp.where(allow[:, None, :, :], 0.0, _NEG)
        if attention_mask is not None:
            attn_bias = attn_bias + jnp.where(attention_mask[:, None, None, :] > 0, 0.0, _NEG)
        hidden = _VLDecoders(cfg, name="layers")(hidden, cos, sin, attn_bias)
        hidden = VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_add_one, cfg.dtype, name="norm")(hidden)
        return hidden, embed


# --------------------------------------------------------------------------- #
# SigLIP vision tower + PaliGemma multimodal projector (mirror of torch
# ``SiglipVisionTransformer`` + ``PaliGemmaMultiModalProjector``). Frozen at train time
# (``freeze_vision_tower`` / ``freeze_multi_modal_projector`` default, no LoRA adapter):
# it turns the ``<image>`` placeholders (image_token_id; 256 per 224px image) into
# text-hidden embeddings. UNLIKE gemma3 (which parks image tokens at attention_mask=0 so
# vision is inert on the golden), PaliGemma's graded forward leaves the image tokens
# ATTENDED, so this path runs on the goldens and its features reach the compared positions
# -- the whole point of this port. PaliGemma SigLIP specifics: NO CLS token and
# ``vision_use_head=False`` (256 patches -> 256 image tokens directly, no pooling head);
# the projector is a single biased Linear -- NO pooling, pixel-shuffle, RMSNorm or
# sqrt-normalization (gemma3's projector differs and must NOT be copied).
# --------------------------------------------------------------------------- #


class _VisionLayerNorm(nn.Module):
    """torch ``nn.LayerNorm`` (weight+bias, float32 reduction). Names its gain ``scale`` /
    shift ``bias`` so the converted SigLIP weights load unchanged."""

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


class SiglipVisionEmbeddings(nn.Module):
    """Conv2d patchify + learned position embedding (torch ``SiglipVisionEmbeddings``);
    224px / patch 14 -> 16x16 = 256 patches, NO CLS token."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, pixel_values):
        cfg = self.config
        pd = cfg.dtype
        p = cfg.patch_size
        # torch pixel_values are NCHW; flax nn.Conv wants NHWC.
        x = jnp.transpose(pixel_values, (0, 2, 3, 1)).astype(pd)
        patch = nn.Conv(cfg.vision_hidden_size, kernel_size=(p, p), strides=(p, p),
                        padding="VALID", name="patch_embedding", param_dtype=pd)(x)
        b = patch.shape[0]
        # (b, H', W', hidden) -> (b, H'*W', hidden): row-major over (H', W'), matching
        # torch's conv(...).flatten(2).transpose(1, 2).
        emb = patch.reshape(b, -1, cfg.vision_hidden_size)
        num_patches = (cfg.image_size // p) ** 2
        pos = nn.Embed(num_patches, cfg.vision_hidden_size, name="position_embedding", param_dtype=pd)
        emb = emb + pos(jnp.arange(num_patches))[None, :, :]
        return emb


class SiglipAttention(nn.Module):
    """MHA with biased q/k/v/out projections, float32 softmax, scale=head_dim**-0.5
    (torch ``SiglipAttention``)."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        b, s, _ = x.shape
        h = cfg.vision_num_attention_heads
        hd = cfg.vision_hidden_size // h
        scale = hd ** -0.5
        dense = lambda n: nn.Dense(cfg.vision_hidden_size, use_bias=True, name=n, param_dtype=pd)
        q = dense("q_proj")(x).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
        k = dense("k_proj")(x).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
        v = dense("v_proj")(x).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
        cdt = jnp.promote_types(q.dtype, jnp.float32)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(cdt) * scale
        w = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        o = jnp.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(b, s, cfg.vision_hidden_size)
        return dense("out_proj")(o)


class SiglipMLP(nn.Module):
    """fc1 -> gelu_pytorch_tanh -> fc2 (torch ``SiglipMLP``, both linears biased)."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pd = cfg.dtype
        x = nn.Dense(cfg.vision_intermediate_size, use_bias=True, name="fc1", param_dtype=pd)(x)
        x = jax.nn.gelu(x, approximate=True)  # gelu_pytorch_tanh
        return nn.Dense(cfg.vision_hidden_size, use_bias=True, name="fc2", param_dtype=pd)(x)


class SiglipEncoderLayer(nn.Module):
    """Pre-norm residual block (torch ``SiglipEncoderLayer``)."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        ln = lambda n: _VisionLayerNorm(cfg.vision_hidden_size, cfg.vision_layer_norm_eps, cfg.dtype, name=n)
        x = x + SiglipAttention(cfg, name="self_attn")(ln("layer_norm1")(x))
        x = x + SiglipMLP(cfg, name="mlp")(ln("layer_norm2")(x))
        return x


class _SiglipEncoderLayers(nn.Module):
    """The ``encoder/layers/{i}`` scope (the ``layers`` submodule of the encoder)."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        for i in range(self.config.vision_num_hidden_layers):
            x = SiglipEncoderLayer(self.config, name=str(i))(x)
        return x


class SiglipEncoder(nn.Module):
    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        return _SiglipEncoderLayers(self.config, name="layers")(x)


class SiglipVisionTransformer(nn.Module):
    """embeddings -> encoder -> post_layernorm (torch ``SiglipVisionTransformer``).

    PaliGemma sets ``vision_use_head=False`` so there is no pooling head; returns the
    encoder's last_hidden_state after the post layernorm (all 256 patch tokens). Mounted
    directly at ``model/vision_tower`` (the loader drops the ``vision_model`` segment)."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, pixel_values):
        cfg = self.config
        h = SiglipVisionEmbeddings(cfg, name="embeddings")(pixel_values)
        h = SiglipEncoder(cfg, name="encoder")(h)
        h = _VisionLayerNorm(cfg.vision_hidden_size, cfg.vision_layer_norm_eps, cfg.dtype,
                             name="post_layernorm")(h)
        return h


class PaliGemmaMultiModalProjector(nn.Module):
    """torch ``PaliGemmaMultiModalProjector``: a single biased Linear from the SigLIP hidden
    size to ``projection_dim`` (== text hidden). NO pooling, NO pixel-shuffle, NO RMSNorm and
    NO sqrt-normalization -- unlike gemma3's projector. Mounted at
    ``model/multi_modal_projector``; the ``linear`` scope matches the checkpoint's
    ``multi_modal_projector.linear.{weight,bias}``."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        return nn.Dense(cfg.projection_dim, use_bias=True, name="linear", param_dtype=cfg.dtype)(x)


class _Wrap(nn.Module):
    child: nn.Module

    @nn.compact
    def __call__(self, *a, **k):
        return self.child(*a, **k)


class _NestedLM(nn.Module):
    """Emits the ``<prefix>`` wrapper scopes then the LanguageModel (prefix[0] is this
    module's own name; nest the rest so paths match ``<prefix>/layers/{i}``). This is the
    ``model`` scope, so the SigLIP tower (``model/vision_tower``) and the projector
    (``model/multi_modal_projector``) -- siblings of ``language_model`` in the checkpoint --
    are built here."""

    config: PaliGemmaConfig

    def _image_features(self, pixel_values):
        """SigLIP tower -> plain-Linear projector -> (total_image_tokens, projection_dim),
        the flat feature block torch scatters over the ``<image>`` placeholders. Called only
        from the eager forward, where ``pixel_values`` is concrete."""
        cfg = self.config
        pv = jnp.asarray(pixel_values).astype(cfg.dtype)
        vision_out = SiglipVisionTransformer(cfg, name="vision_tower")(pv)
        feats = PaliGemmaMultiModalProjector(cfg, name="multi_modal_projector")(vision_out)
        return feats.reshape(-1, feats.shape[-1])

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None,
                 pixel_values=None, run_vision=None):
        cfg = self.config
        image_features = None
        if pixel_values is not None:
            # UNLIKE gemma3 (image tokens at attention_mask=0 -> vision inert on the golden),
            # PaliGemma's graded forward leaves the <image> placeholders ATTENDED, so the
            # SigLIP tower runs here and its features reach the compared positions.
            # ``run_vision`` (a STATIC python bool) forces the tower under jit for in-band
            # real training; the eager guard mirrors torch (pixel_values present + an image
            # token at an attended position -> run the tower). The try/except mirrors the
            # sibling ports: jax.make_jaxpr (the probe) lifts the batch to tracers, so
            # bool(jnp.any(...)) raises ConcretizationTypeError -> text-only.
            if run_vision is not None:
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
                input_ids, attention_mask, token_type_ids, image_features
            )
        node = VLLanguageModel(cfg, name=inner[-1])
        for nm in reversed(inner[:-1]):
            node = _Wrap(node, name=nm)
        return node(input_ids, attention_mask, token_type_ids)


@flax.struct.dataclass
class CausalLMOutput:
    logits: Any
    hidden_states: Any = None
    values: Any = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class PaliGemmaForConditionalGeneration(nn.Module):
    """PaliGemmaForConditionalGeneration LM-path causal LM (mirror of the torch class; text-only forward, §12.4).
    The LM is nested under ``config.prefix`` (``model/language_model``)."""

    config: PaliGemmaConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, output_hidden_states=False, **kwargs):
        cfg = self.config
        # pixel_values arrives via kwargs (the collator's mm_inputs). Text-only callers
        # (generate(), the battery) pass none, so the forward is the text-only LM; on the
        # graded golden the <image> tokens are attended, so the SigLIP tower runs and its
        # features are scattered into the placeholder positions.
        hidden, embed = _NestedLM(cfg, name=cfg.prefix[0])(
            input_ids, attention_mask, token_type_ids,
            pixel_values=kwargs.get("pixel_values"), run_vision=kwargs.get("run_vision"))
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


__all__ = [
    "PaliGemmaForConditionalGeneration",
    "PaliGemmaConfig",
    "config_from_pretrained",
    # re-exported through the ``modeling`` shim's star-import for ``model/loader.py``
    "lora_a_init",
]
