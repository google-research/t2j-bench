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

"""The Qwen2.5-VL vision tower in flax nnx, mirroring
``environment/torch_orig/vision.py``.

Same classes, same attribute names, same order of operations. Read that file for
the UPSTREAM provenance (vLLM line ranges), for why this module exists at all
(a declared partition violation -- the tower is a second, structurally unrelated
transformer) and for the full list of structural differences from vLLM. This
header records only what is different HERE, i.e. what JAX forces.

THE PARAMETER PATHS ARE THE CONTRACT, and they are identical to the torch side's
because ``test_param_correspondence`` compares the two key sets through
``canonical_param_key``, which normalises no intermediate path component::

    visual.patch_embed.proj.weight                 # rank-5, [hidden, C, tps, ps, ps]
    visual.blocks.N.norm1.weight / norm2.weight
    visual.blocks.N.attn.qkv.{weight,bias}         # ALREADY packed in the checkpoint
    visual.blocks.N.attn.proj.{weight,bias}
    visual.blocks.N.mlp.gate_up_proj.{weight,bias} # FUSED from gate_proj/up_proj
    visual.blocks.N.mlp.down_proj.{weight,bias}
    visual.merger.ln_q.weight
    visual.merger.mlp.0.{weight,bias}              # a LIST; the GELU sits at index 1
    visual.merger.mlp.2.{weight,bias}

``merger.mlp`` is an ``nnx.data`` LIST rather than three named attributes,
precisely so that flax emits ``merger.mlp.0`` / ``merger.mlp.2`` -- the
checkpoint's own spelling, with the activation occupying index 1 and
contributing no leaf. Naming the two Dense layers ``mlp_0``/``mlp_2`` would also
canonicalize (``_N`` folds to ``.N``), but renumbering them to 0 and 1 would NOT.

DIFFERENCES FROM THE TORCH TWIN, all of them forced by JAX:

  1. ``nn.Module`` -> ``nnx.Module``, ``forward`` -> ``__call__``,
     ``nn.ModuleList`` -> ``nnx.data([...])``.
  2. THE CUMULATIVE-LENGTH VECTORS ARE PYTHON ``tuple[int, ...]``, NOT ARRAYS.
     ``Qwen2_5_VisionAttention`` slices its input per chunk, and a slice bound
     has to be a static Python int for the loop to be traceable at all; the
     torch side reaches the same place with a ``cu_seqlens.tolist()`` inside
     every block's forward, which its own docstring flags as 32 host syncs per
     image and tells this port to hoist. It is hoisted: ``encoder_metadata``
     derives them ONCE, from a grid that is static Python geometry on both
     sides. ``window_index`` / ``reverse_indices`` stay arrays -- they are
     gather indices, not control flow.
  3. ``torch.empty_like(q)`` + per-chunk in-place writes become a list of chunk
     outputs and one ``concatenate``. The coverage check that guards the torch
     buffer is KEPT, and its justification changes rather than disappearing:
     there is no uninitialised memory to leak here, but a ``cu_seqlens`` that
     does not start at 0 and end at ``seq_len`` would silently return a
     DIFFERENTLY-SHAPED tensor whose rows are not the rows it was asked for, and
     the two sides must refuse the same inputs for the same reason.
  4. ``nn.Conv3d`` becomes an explicit ``[hidden, C*tps*ps*ps]`` matmul against
     the SAME rank-5 parameter -- see :class:`Qwen2_5_VisionPatchEmbed`.

NUMERIC CONTRACTS COPIED EXACTLY, because they are the ones a plausible port
gets wrong silently:

  * :func:`apply_rotary_emb_vision` runs in fp32 REGARDLESS of the surrounding
    dtype and casts back. That follows HF, not vLLM (vLLM's
    ``ApplyRotaryEmb(enforce_enable=True)`` selects the custom-op path and says
    nothing about precision); the torch twin argues the choice in full and pins
    it with a bf16 dtype-contract test.
  * Vision attention accumulates in fp32 -- both matmuls AND the softmax --
    per ``cu_seqlens`` chunk, with NO CAUSAL MASK. The tower is bidirectional; a
    port that reuses the decoder's masked attention here still produces
    plausible embeddings.
  * The merger's GELU is the EXACT erf form (``approximate=False``), not the
    tanh approximation several Qwen ports reach for. Measured deviation of the
    tanh mutant on the torch side's frozen recording: 1.284e-03.
  * :data:`ROPE_THETA` is 10000.0 -- the TOWER's base, NOT ``config.rope_theta``
    (1e6, the text stack's). Reading the text theta here keeps every shape right
    and moves only the image attention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from config import VisionConfig
from layers.activation import SiluAndMul
from layers.layernorm import RMSNorm
from layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

#: How many grid positions the 2-D rotary table covers. vLLM builds the same
#: table with ``max_position=8192``; HF instead rebuilds it per call at
#: ``max(h, w)``. 8192 is taken because it is upstream's number and because a
#: cached table makes the forward a pure gather, exactly as ``layers/rotary.py``
#: does for the text rope. A grid larger than this raises rather than wrapping.
MAX_GRID_SIZE = 8192

#: The tower's rope base. 10000 in both upstreams, and NOT ``config.rope_theta``
#: (1e6), which belongs to the text stack. Named so the two cannot be confused.
ROPE_THETA = 10000.0


def apply_rotary_emb_vision(
    x: jax.Array, cos: jax.Array, sin: jax.Array
) -> jax.Array:
    """Rotate-half over the FULL head, with a half-width cos/sin.

    ``x`` is ``[S, H, D]``; ``cos``/``sin`` are ``[S, D // 2]`` and broadcast
    over the head axis. Upstream duplicates the band instead and multiplies the
    full width; the two are the same arithmetic written twice, and this is the
    form ``layers/rotary.py`` already uses so the two ropes read alike.

    IN FLOAT32, ALWAYS, and cast back afterwards. See the module docstring and
    the torch twin: this follows HF rather than vLLM, deliberately, and the
    torch side pins it with a bf16 test that a narrow rotation fails.
    """
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    cos = jnp.expand_dims(cos.astype(jnp.float32), 1)
    sin = jnp.expand_dims(sin.astype(jnp.float32), 1)
    x1, x2 = jnp.split(x, 2, axis=-1)
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(orig_dtype)


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):
    """The 2-D grid rope's frequency table: ``[MAX_GRID_SIZE, head_dim // 4]``.

    ``dim`` is ``head_dim // 2`` -- upstream's ``partial_rotary_factor: 0.5`` --
    so the table is ``dim // 2 == head_dim // 4`` wide. One row is looked up per
    AXIS, and the h-row and w-row are concatenated into the ``head_dim // 2``
    band :func:`apply_rotary_emb_vision` consumes. That concatenation is where
    "2-D" actually lives.

    THE TABLE IS NOT AN ``nnx.Param``, for the reason ``layers/rotary.py`` gives
    at length: it is derived from ``(theta, dim, max_grid_size)``, so it is a
    BUFFER. torch_orig says the same with ``register_buffer(persistent=False)``.
    Promoting it would put two tensors into the parameter pytree that no
    checkpoint key answers for, and ``test_param_correspondence`` grades that
    correspondence 1:1.
    """

    def __init__(self, dim: int, theta: float = ROPE_THETA,
                 max_grid_size: int = MAX_GRID_SIZE) -> None:
        self.dim = dim
        self.max_grid_size = max_grid_size
        inv_freq = 1.0 / (
            theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        freqs = jnp.outer(jnp.arange(max_grid_size, dtype=jnp.float32), inv_freq)
        self.cos_cached = jnp.cos(freqs)
        self.sin_cached = jnp.sin(freqs)

    def __call__(self, pos_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``pos_ids`` ``[S, 2]`` (h, w) -> two ``[S, dim]`` arrays.

        ``cos_cached[pos_ids]`` is ``[S, 2, dim // 2]``; flattening the last two
        axes concatenates the h band and the w band IN THAT ORDER, which is the
        order the checkpoint's heads were trained with. Swapping them keeps every
        shape and produces an image encoder that is wrong in a way no shape check
        can see.

        The bounds check reads ``int(pos_ids.max())`` to drive a Python
        ``raise``, exactly as the torch twin does. That is safe here and is not
        an accident: ``pos_ids`` comes from :meth:`Qwen2_5_VisionTransformer.
        pos_ids_thw`, which builds it out of ``jnp.arange`` over a grid that is
        static Python geometry on both sides, so it is a concrete array even
        inside a trace. Nothing in this module reads a traced value on the host.
        """
        largest = int(pos_ids.max()) if pos_ids.size else 0
        if largest >= self.max_grid_size:
            raise ValueError(
                f"grid position {largest} is outside the rotary table's "
                f"{self.max_grid_size} rows. Rebuild the table larger rather "
                "than letting the gather wrap, which would silently reuse "
                "another patch's rotation.")
        rows = pos_ids.shape[0]
        return (self.cos_cached[pos_ids].reshape(rows, -1),
                self.sin_cached[pos_ids].reshape(rows, -1))


class _PatchProjection(nnx.Module):
    """The Conv3d's parameter, and the matmul that is equivalent to it.

    NOT a public name, mirroring the torch side where ``nn.Conv3d`` is torch's
    and not this module's. It exists so the parameter path is
    ``patch_embed.proj.weight`` on both sides.
    """

    def __init__(self, in_channels: int, hidden_size: int,
                 kernel_size: tuple[int, int, int], *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        key = rngs.params() if rngs is not None else jax.random.key(0)
        shape = (hidden_size, in_channels) + tuple(kernel_size)
        self.weight = nnx.Param(nnx.initializers.zeros(key, shape, dtype))
        self.hidden_size = hidden_size

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [N, C * tps * ps * ps], already in the kernel's own C-order. The
        # convolution's stride EQUALS its kernel, so each output row is the full
        # contraction of one patch against one filter -- i.e. a plain matmul
        # against the flattened kernel. Zeros as the init, mirroring
        # layers/linear.py: a projection that has not been through the loader
        # holds no meaningful values on either side.
        w = self.weight[...].reshape(self.hidden_size, -1)
        return x @ w.T


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    """Conv3d over ``(temporal_patch_size, patch_size, patch_size)``.

    The processor hands ``[num_patches, C * tps * ps * ps]`` -- already
    patchified, so the convolution's stride equals its kernel and it is a
    per-patch linear map wearing a convolution's parameter layout.

    THE PARAMETER KEEPS THE RANK-5 SHAPE ``[hidden, C, tps, ps, ps]``, because
    that is what the CHECKPOINT stores and what ``model/loader.py`` copies over
    untouched. Only the ARITHMETIC is rewritten (see :class:`_PatchProjection`):
    flattening the kernel's trailing four axes in C-order and contracting is the
    same sum of products the convolution computes, term for term, because the
    input arrives flattened in that same order.
    """

    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, hidden_size: int, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = _PatchProjection(in_channels, hidden_size, kernel_size,
                                     dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        num_patches = x.shape[0]
        return self.proj(x.reshape(num_patches, -1)).reshape(
            num_patches, self.hidden_size)


class Qwen2_5_VisionMLP(nnx.Module):
    """SwiGLU, with BIASES -- unlike the text stack's, which has none.

    The checkpoint is the evidence: ``visual.blocks.N.mlp.{gate,up,down}_proj``
    each ship a ``.bias`` and the text ``model.layers.N.mlp.*`` do not. A port
    that copies the text MLP here loads cleanly, drops three tensors per block,
    and is quietly wrong.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 bias: bool = True, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.gate_up_proj = MergedColumnParallelLinear(
            in_features, [hidden_features, hidden_features], bias=bias,
            dtype=dtype, rngs=rngs)
        self.down_proj = RowParallelLinear(hidden_features, in_features,
                                           bias=bias, dtype=dtype, rngs=rngs)
        self.act_fn = SiluAndMul()

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2_5_VisionAttention(nnx.Module):
    """Bidirectional attention, chunked by ``cu_seqlens``, no KV cache.

    ``num_kv_heads == num_heads``: the tower is not GQA, and the checkpoint's
    ``attn.qkv.weight`` is ``[3 * dim, dim]``, which only ``QKVParallelLinear``
    with equal head counts produces.

    WHY ``cu_seqlens`` AND NOT A MASK. The same module serves both the windowed
    blocks and the full-attention blocks; the ONLY difference between them is
    which cumulative-length vector arrives. Attending per chunk makes that
    difference structural instead of a mask the caller has to build correctly
    twice.

    ``cu_seqlens`` ARRIVES AS A ``tuple[int, ...]``, not an array, and that is
    the one signature difference from the torch twin. The chunk bounds index
    Python slices, so they have to be static; the torch side calls
    ``cu_seqlens.tolist()`` here on every block and its own docstring says the
    JAX mirror should hoist that to geometry time instead. It is hoisted, into
    :meth:`Qwen2_5_VisionTransformer.encoder_metadata`.
    """

    def __init__(self, embed_dim: int, num_heads: int, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = QKVParallelLinear(embed_dim, self.head_dim, num_heads,
                                     num_heads, bias=True, dtype=dtype, rngs=rngs)
        self.proj = RowParallelLinear(embed_dim, embed_dim, bias=True,
                                      dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, cu_seqlens, cos: jax.Array,
                 sin: jax.Array) -> jax.Array:
        seq_len = x.shape[0]
        q, k, v = self.qkv.split(self.qkv(x))
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)

        q = apply_rotary_emb_vision(q, cos, sin)
        k = apply_rotary_emb_vision(k, cos, sin)

        bounds = [int(b) for b in cu_seqlens]
        # THE CHUNKS MUST COVER EVERY ROW, and that is checked rather than
        # absorbed. The torch twin writes into an uninitialised `empty_like`
        # buffer, so a partial cover would return heap contents there; here the
        # chunks are concatenated instead, so a partial cover returns a tensor
        # of the WRONG LENGTH whose rows do not line up with the residual stream
        # it is about to be added to. Different symptom, same defect, so both
        # sides refuse the same inputs with the same message. `forward` already
        # refuses a `cu_seqlens` that disagrees with the patch count, but only
        # for the FULL vector, and the windowed blocks are handed a different one.
        if not bounds or bounds[0] != 0 or bounds[-1] != seq_len:
            raise ValueError(
                f"cu_seqlens {bounds[:4]}..{bounds[-1:]} does not cover the "
                f"{seq_len} rows it is attending over: it must start at 0 and "
                "end at the sequence length. A partial cover would leave "
                "uninitialised rows in the output.")
        chunks = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                # Upstream's padded ``cu_seqlens`` can repeat its last offset to
                # keep a CUDA-graph buffer a fixed size; an empty chunk is the
                # residue of that and contributes nothing. Skipped rather than
                # fed to softmax, which would produce NaN over zero keys.
                continue
            qc = jnp.swapaxes(q[start:end], 0, 1).astype(jnp.float32)  # [H, L, D]
            kc = jnp.swapaxes(k[start:end], 0, 1).astype(jnp.float32)
            vc = jnp.swapaxes(v[start:end], 0, 1).astype(jnp.float32)
            # FP32 THROUGHOUT -- both matmuls AND the softmax, matching
            # attention/paged.py, which upcasts q and k into its einsum for the
            # same reason. This is WIDER THAN EITHER UPSTREAM. NO CAUSAL MASK --
            # the tower is bidirectional, and a port that reuses the decoder's
            # masked attention here still produces plausible embeddings.
            scores = jnp.matmul(qc, jnp.swapaxes(kc, -1, -2)) * self.scaling
            probs = jax.nn.softmax(scores, axis=-1)
            chunks.append(
                jnp.swapaxes(jnp.matmul(probs, vc), 0, 1).astype(q.dtype))

        out = jnp.concatenate(chunks, axis=0)
        return self.proj(out.reshape(seq_len, self.num_heads * self.head_dim))


class Qwen2_5_VisionBlock(nnx.Module):
    """Pre-norm attention + MLP, with RMSNorm -- not LayerNorm.

    Qwen2-VL used LayerNorm here and Qwen2.5-VL replaced it, which is why
    ``visual.blocks.N.norm1`` ships a ``.weight`` and no ``.bias``. That is the
    single easiest tensor set to mis-port from the older family's code, and it
    fails loudly (a missing ``.bias`` in the checkpoint) rather than silently.
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int,
                 eps: float = 1e-6, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.norm1 = RMSNorm(dim, eps=eps, dtype=dtype)
        self.norm2 = RMSNorm(dim, eps=eps, dtype=dtype)
        self.attn = Qwen2_5_VisionAttention(dim, num_heads, dtype=dtype, rngs=rngs)
        self.mlp = Qwen2_5_VisionMLP(dim, mlp_hidden_dim, bias=True,
                                     dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, cu_seqlens, cos: jax.Array,
                 sin: jax.Array) -> jax.Array:
        x_attn = self.attn(self.norm1(x), cu_seqlens, cos, sin)
        # The FUSED two-argument RMSNorm, as upstream writes it: `residual` is
        # `x + x_attn` and the block's output is that residual plus the MLP of
        # its norm. Written this way and not as two separate adds so the
        # residual stream is threaded exactly as the text decoder threads it.
        normed, residual = self.norm2(x, residual=x_attn)
        return residual + self.mlp(normed)


class _ExactGELU(nnx.Module):
    """``nn.GELU()`` -- the ERF form, and index 1 of the merger's MLP list.

    A module rather than a bare callable so it can sit in an ``nnx.data`` list
    between two Dense layers, which is what makes the surrounding parameters
    ``mlp.0.*`` and ``mlp.2.*``. It holds no state, so it contributes no leaf --
    exactly like the ``nn.GELU()`` it mirrors.

    ``approximate=False`` IS THE CONTRACT. The tanh approximation is what several
    Qwen ports reach for because other Qwen models use it; it differs by up to
    ~1e-3 here (measured on the torch side's frozen recording: 1.284e-03, 64x
    that suite's tolerance) and is the single likeliest silent numeric
    divergence in this module.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x, approximate=False)


class Qwen2_5_VisionPatchMerger(nnx.Module):
    """RMSNorm, regroup ``merge**2`` patches into one row, then a 2-layer MLP.

    ``mlp`` is a LIST and the GELU sits at index 1, which is why the
    checkpoint's keys are ``visual.merger.mlp.0.*`` and ``visual.merger.mlp.2.*``
    with nothing at 1. The indices are the contract: a port may name its two
    Dense layers ``mlp_0``/``mlp_2`` (``canonical_param_key`` folds ``_N`` to
    ``.N``) but it may NOT renumber them to 0 and 1.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2, eps: float = 1e-6, *,
                 dtype=jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=eps, dtype=dtype)
        # nnx.data, not a bare list: flax 0.12 classifies every attribute as
        # static structure or graph data, and a bare list of modules is refused
        # as "data on a static attribute". This is also what emits the `.0` /
        # `.2` path components -- see the class docstring.
        self.mlp = nnx.data([
            ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True,
                                 dtype=dtype, rngs=rngs),
            _ExactGELU(),
            RowParallelLinear(self.hidden_size, d_model, bias=True,
                              dtype=dtype, rngs=rngs),
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            out = layer(out)
        return out


class Qwen2_5_VisionTransformer(nnx.Module):
    """The tower: patch embed -> 32 windowed/full blocks -> patch merger.

    ``__call__(pixel_values, grid_thw)`` returns ``[num_merged_tokens,
    out_hidden_size]`` -- one row per ``<|image_pad|>`` placeholder in the
    prompt, in PROMPT order, which is what makes the merge in ``modeling.py`` a
    plain scatter.

    THE WINDOW PERMUTATION IS THE HARD PART, and it is worth stating outright
    because it has no analogue anywhere else in this class. Patches are
    REORDERED before the blocks run, so that each local window is contiguous and
    window attention is expressible as ``cu_seqlens`` over the reordered
    sequence; the merger's output is then permuted BACK. Four things ride on
    that and each is independently wrong-able:

      1. the reorder is applied to groups of ``spatial_merge_unit`` patches, not
         to single patches;
      2. the rotary cos/sin are reordered with the patches, not left in grid
         order;
      3. the inverse permutation is applied AFTER the merger, at merged-token
         granularity, not before it;
      4. blocks in ``fullatt_block_indexes`` see the whole image's
         ``cu_seqlens``; every other block sees the window one.

    Get 1 and 3 both wrong in the same way and the output is still a permutation
    of the right rows; get only one wrong and the image is scrambled with no
    shape change anywhere.
    """

    def __init__(self, vision_config: VisionConfig, norm_eps: float = 1e-6, *,
                 dtype=jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        self.config = vision_config
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.out_hidden_size = vision_config.out_hidden_size
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = vision_config.spatial_merge_unit
        self.fullatt_block_indexes = tuple(vision_config.fullatt_block_indexes)

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=rngs,
        )
        # `head_dim // 2`, per upstream's partial_rotary_factor of 0.5. See
        # Qwen2_5_VisionRotaryEmbedding's docstring.
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(
            vision_config.head_dim // 2)
        self.blocks = nnx.data([
            Qwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                eps=norm_eps,
                dtype=dtype,
                rngs=rngs,
            )
            for _ in range(vision_config.depth)
        ])
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            eps=norm_eps,
            dtype=dtype,
            rngs=rngs,
        )

    # -- geometry ---------------------------------------------------------- #

    def pos_ids_thw(self, grid_t: int, grid_h: int, grid_w: int) -> jax.Array:
        """``[grid_t * grid_h * grid_w, 2]`` of (h, w) per patch, MERGE-ORDERED.

        The reshape/transpose pair is not a flatten: it puts the
        ``spatial_merge_size ** 2`` patches that will become ONE llm token
        adjacent to each other, which is the order ``pixel_values`` itself
        arrives in from the processor. Reading it as a plain row-major
        ``arange`` gives the right shape and rotates every patch by the wrong
        position.
        """
        merge = self.spatial_merge_size
        hpos = jnp.broadcast_to(
            jnp.arange(grid_h)[:, None], (grid_h, grid_w))
        wpos = jnp.broadcast_to(
            jnp.arange(grid_w)[None, :], (grid_h, grid_w))
        blocks = (grid_h // merge, merge, grid_w // merge, merge)
        hpos = jnp.transpose(hpos.reshape(blocks), (0, 2, 1, 3)).reshape(-1)
        wpos = jnp.transpose(wpos.reshape(blocks), (0, 2, 1, 3)).reshape(-1)
        return jnp.tile(jnp.stack([hpos, wpos], axis=-1), (grid_t, 1))

    def window_index_thw(self, grid_t: int, grid_h: int,
                         grid_w: int) -> tuple[jax.Array, tuple[int, ...]]:
        """``(window_index, cu_window_seqlens)`` for one image.

        ``window_index`` is a permutation of the MERGED token indices; the
        ``-100`` padding is how a window that runs off the edge of the grid is
        made rectangular and then dropped, so windows at the right and bottom
        edges are legitimately shorter than the rest. ``cu_window_seqlens``
        counts RAW patches (hence the ``* spatial_merge_unit``), because that is
        the granularity attention runs at, and it is a Python tuple because it
        drives slice bounds -- see :class:`Qwen2_5_VisionAttention`.
        """
        merge_window = self.window_size // self.spatial_merge_size // self.patch_size
        llm_h = grid_h // self.spatial_merge_size
        llm_w = grid_w // self.spatial_merge_size
        index = jnp.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
        pad_h = merge_window - llm_h % merge_window
        pad_w = merge_window - llm_w % merge_window
        num_windows_h = (llm_h + pad_h) // merge_window
        num_windows_w = (llm_w + pad_w) // merge_window
        padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)),
                         constant_values=-100)
        padded = padded.reshape(grid_t, num_windows_h, merge_window,
                                num_windows_w, merge_window)
        padded = jnp.transpose(padded, (0, 1, 3, 2, 4)).reshape(
            grid_t, num_windows_h * num_windows_w, merge_window, merge_window)
        seqlens = jnp.sum(padded != -100, axis=(2, 3)).reshape(-1)
        flat = padded.reshape(-1)
        # `flat[flat != -100]` is the torch spelling and has a DATA-DEPENDENT
        # output shape, which JAX refuses under a trace. The count is known
        # statically -- exactly the merged-token count -- so `nonzero` is given
        # it and the gather stays a fixed shape.
        kept = jnp.nonzero(flat != -100, size=grid_t * llm_h * llm_w)[0]
        window_index = flat[kept]
        # The cumulative vector, in Python ints. `torch.unique_consecutive`
        # drops the zero-length windows a fully padded window would contribute;
        # so does the `!=` guard below, and the attention loop's empty-chunk
        # guard is the belt to these braces.
        cu: list[int] = []
        running = 0
        for length in seqlens.tolist():
            running += int(length) * self.spatial_merge_unit
            if not cu or cu[-1] != running:
                cu.append(running)
        return window_index, tuple(cu)

    @staticmethod
    def invert_permutation(perm: jax.Array) -> jax.Array:
        """The inverse of a permutation, in O(n).

        ``argsort(perm)`` returns the same array for a genuine permutation. The
        scatter is upstream's and is kept because it is also a cheap assertion:
        a non-permutation leaves holes, which the caller's shape checks catch.
        """
        return jnp.zeros_like(perm).at[perm].set(
            jnp.arange(perm.size, dtype=perm.dtype))

    def encoder_metadata(self, grid_thw) -> dict:
        """Everything ``__call__`` needs that depends only on the GRID.

        Pure geometry: it never touches a parameter, and every value it returns
        is a function of ``grid_thw`` alone. The two cumulative-length vectors
        come back as Python ``tuple[int, ...]`` rather than arrays, which is the
        hoist the torch twin's docstring asks this port for -- the alternative
        is a ``.tolist()`` inside all 32 blocks.
        """
        cos_parts: list[jax.Array] = []
        sin_parts: list[jax.Array] = []
        window_parts: list[jax.Array] = []
        cu_window: list[int] = [0]
        frame_lens: list[int] = []

        window_offset = 0
        cu_window_last = 0
        for entry in grid_thw:
            grid_t, grid_h, grid_w = (int(v) for v in entry)
            cos, sin = self.rotary_pos_emb(self.pos_ids_thw(grid_t, grid_h, grid_w))
            window_index, cu = self.window_index_thw(grid_t, grid_h, grid_w)

            # The rope travels WITH the patches through the window permutation:
            # regroup into merge-units, gather, flatten back.
            unit = self.spatial_merge_unit
            cos = cos.reshape(cos.shape[0] // unit, unit, -1)[window_index]
            sin = sin.reshape(sin.shape[0] // unit, unit, -1)[window_index]
            cos_parts.append(cos.reshape(-1, cos.shape[-1]))
            sin_parts.append(sin.reshape(-1, sin.shape[-1]))

            window_parts.append(window_index + window_offset)
            window_offset += grid_t * (grid_h // self.spatial_merge_size) * (
                grid_w // self.spatial_merge_size)

            for value in cu:
                shifted = value + cu_window_last
                if shifted != cu_window[-1]:
                    cu_window.append(shifted)
            cu_window_last = cu_window[-1]

            # Full attention is per FRAME, not per image: a t-frame video
            # contributes t sequences of h*w patches each.
            frame_lens.extend([grid_h * grid_w] * grid_t)

        window_index = jnp.concatenate(window_parts)
        cu_seqlens: list[int] = [0]
        for length in frame_lens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        return {
            "cos": jnp.concatenate(cos_parts),
            "sin": jnp.concatenate(sin_parts),
            "window_index": window_index,
            "reverse_indices": self.invert_permutation(window_index),
            "cu_seqlens": tuple(cu_seqlens),
            "cu_window_seqlens": tuple(cu_window),
        }

    # -- forward ----------------------------------------------------------- #

    def __call__(self, pixel_values: jax.Array, grid_thw) -> jax.Array:
        """``[num_patches, C*tps*ps*ps]`` + grid -> ``[num_merged, out_hidden]``.

        ``grid_thw`` is any iterable of ``(t, h, w)`` triples -- a
        ``[num_images, 3]`` array or a list of lists both work, because upstream
        takes a list here and the processor produces an array.
        """
        hidden = self.patch_embed(pixel_values.astype(self.dtype))
        seq_len = hidden.shape[0]
        meta = self.encoder_metadata(grid_thw)
        expected = int(meta["cu_seqlens"][-1])
        if seq_len != expected:
            raise ValueError(
                f"the tower got {seq_len} patches but grid_thw accounts for "
                f"{expected}. pixel_values and image_grid_thw come out of the "
                "processor together and must stay together; a mismatch here "
                "would otherwise surface as a scrambled image rather than an "
                "error.")

        unit = self.spatial_merge_unit
        hidden = hidden.reshape(seq_len // unit, unit, -1)
        hidden = hidden[meta["window_index"]].reshape(seq_len, -1)

        for index, block in enumerate(self.blocks):
            cu = (meta["cu_seqlens"] if index in self.fullatt_block_indexes
                  else meta["cu_window_seqlens"])
            hidden = block(hidden, cu, meta["cos"], meta["sin"])

        # The merger consumes `spatial_merge_unit` rows per output row, so the
        # inverse permutation that follows is at MERGED granularity -- which is
        # exactly the granularity `window_index` was built at.
        return self.merger(hidden)[meta["reverse_indices"]]

    @property
    def dtype(self):
        return self.patch_embed.proj.weight[...].dtype
