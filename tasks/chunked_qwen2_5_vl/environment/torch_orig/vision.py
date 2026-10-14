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

"""The Qwen2.5-VL vision tower, de-vendored from vLLM.

Source: ``vllm/model_executor/models/qwen2_5_vl.py`` at upstream sha
``06ecec7a...``. Line ranges, upstream class by upstream class:

  * ``Qwen2_5_VisionMLP``           :309-344
  * ``Qwen2_5_VisionAttention``     :347-458
  * ``Qwen2_5_VisionBlock``         :461-524
  * ``Qwen2_5_VisionPatchEmbed``    :527-560
  * ``Qwen2_5_VisionPatchMerger``   :563-613
  * ``Qwen2_5_VisionTransformer``   :616-1131, of which the parts that survive
    are ``rotary_pos_emb_thw`` :717-761, ``get_window_index_thw`` :763-798,
    ``get_rope_by_thw`` :800-820, ``invert_permutation`` :835-840, the
    metadata loop inside ``prepare_encoder_metadata`` :915-958, and ``forward``
    :1067-1127.

THIS FILE IS A DECLARED PARTITION VIOLATION: a new module per side (the vision
tower) is one of two deliberate departures from the per-family column
(``modeling.py`` / ``config.py`` / ``model/loader.py`` / ``serve.py``). It is
here, and not in ``modeling.py``, because the tower is a second, structurally
unrelated transformer -- bidirectional, windowed, 2-D rotary, no KV cache -- and
folding six more classes into ``modeling.py`` would have made the one file every
other family uses to state its decoder unreadable for this one. A reader who
finds an extra module in this instance is looking at a planned violation, not at
drift, and does not have to reconstruct which.

STRUCTURAL DIFFERENCES FROM UPSTREAM. All but the last are a removal or a
consequence of TP=1; the last one is an ADDITION and is called out as such,
because a divergence hidden inside a list titled "removals" is the shape of
defect this header exists to prevent:

  * TP=1, so every ``*ParallelLinear`` is the plain linear in ``layers/linear.py``
    while KEEPING the fused ``qkv`` / ``gate_up_proj`` layout -- which is what
    ``test_param_correspondence`` grades, and which is also what the CHECKPOINT
    ships (``visual.blocks.N.attn.qkv.weight`` is already packed there;
    ``mlp.gate_proj`` / ``mlp.up_proj`` are not, and ``model/loader.py`` fuses
    them).
  * No ``@support_torch_compile``, no ``VllmConfig``, no quantization, no FP8
    padding, no data-parallel ViT sharding, no CUDA-graph encoder buffers, and
    no FlashAttention/FlashInfer backend selection. Upstream's
    ``prepare_encoder_metadata`` is 190 lines of which ~150 exist to pad buffers
    for CUDA-graph replay and to re-derive ``cu_seqlens`` per backend; none of
    that survives at batch 1 with no graph capture.
  * ``MMEncoderAttention`` is replaced by an explicit per-chunk softmax over
    ``cu_seqlens`` -- the same shape HF's non-FlashAttention path takes
    (``transformers.models.qwen2_5_vl.modeling_qwen2_5_vl``:
    ``Qwen2_5_VLVisionAttention.forward``, the ``else`` branch) -- so the
    windowing is visible in the model rather than hidden inside a kernel's
    ``cu_seqlens`` argument. The candidate has to port it and the grader can
    gate it.
  * TOKEN-FLAT, ``[S, C]``, where upstream carries ``[S, 1, C]``
    (``forward`` :1098 ``hidden_states.unsqueeze(1)``). The batch axis is
    vestigial upstream -- it exists so ``einops.rearrange`` can name it -- and
    this class's whole convention is that batching is expressed by
    ``cu_seqlens``, never by a leading axis.
  * ``float16`` overflow clamping (``cast_overflow_tensors`` :1121-1122) is
    dropped: this class serves bf16 and fp32 only, and a clamp that never fires
    is a branch a port has to reproduce for nothing.
  * **WIDER ARITHMETIC THAN vLLM, in two places, and this is the one item on
    this list that ADDS something.** The rotary and the attention accumulation
    both run at fp32 here regardless of the serving dtype
    (:func:`apply_rotary_emb_vision`, :meth:`Qwen2_5_VisionAttention.forward`).
    vLLM does NEITHER: its ``ApplyRotaryEmb(enforce_enable=True)``
    (qwen2_5_vl.py:398) leaves ``enable_fp32_compute`` at its ``False`` default
    (rotary_embedding/common.py:132) -- ``enforce_enable`` selects the custom-op
    path over the native one (custom_op.py:130, :185) and says nothing about
    precision -- and its attention is a kernel whose accumulation width is the
    backend's business. HF upcasts the rotary
    (``apply_rotary_pos_emb_vision`` opens ``q, k = q.float(), k.float()``) and
    upcasts the SOFTMAX only, not the two matmuls.
    So: the rotary follows HF over vLLM, and the attention is wider than either.
    Both are deliberate -- they match ``attention/paged.py``'s house rule that
    the accumulation is what has to be wide, and they match the oracle the
    frozen recording came from -- but they are DIVERGENCES FROM THE DECLARED
    SOURCE, they are why the fp64 agreement in the vision fixture bottoms out at
    4.6e-07, and the JAX mirror must reproduce them. Both are pinned by the
    harness's two bf16 dtype-contract tests.

THE 2-D ROTARY IS NOT ``layers/rotary.py``'s. That module implements the TEXT
rope: one position per token, optionally partitioned into M-RoPE sections. The
tower's is a different thing wearing a similar name -- each patch carries a
(h, w) PAIR, the two halves of the frequency band are indexed by DIFFERENT
positions, and there is no cache of absolute positions to gather from because
the grid size is an input. Sharing the module would have meant a ``if 2d:``
branch in shared code, which is exactly what the rank-agnostic ``positions``
work exists to avoid. The rotation arithmetic itself is the same rotate-half
expression, and is written once here in :func:`apply_rotary_emb_vision`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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
#: table with ``max_position=8192`` (qwen2_5_vl.py:670-675); HF instead rebuilds
#: it per call at ``max(h, w)``. 8192 is taken because it is upstream's number
#: and because a cached table makes the forward a pure gather, exactly as
#: ``layers/rotary.py`` does for the text rope. A grid larger than this raises
#: rather than wrapping: at ``patch_size`` 14 it is a 114,688-pixel side, which
#: no processor output can reach under ``max_pixels``.
MAX_GRID_SIZE = 8192

#: The tower's rope base. vLLM reaches it through ``get_rope``'s default and HF
#: writes it as ``Qwen2_5_VisionRotaryEmbedding(dim, theta=10000.0)``; it is
#: 10000 in both and is NOT ``config.rope_theta`` (1e6), which belongs to the
#: text stack. Reading the text theta here is a silent, plausible-looking defect
#: -- every shape stays right and only the image attention moves.
ROPE_THETA = 10000.0


def apply_rotary_emb_vision(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate-half over the FULL head, with a half-width cos/sin.

    ``x`` is ``[S, H, D]``; ``cos``/``sin`` are ``[S, D // 2]`` and broadcast
    over the head axis. Upstream duplicates the band instead
    (``torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)`` in HF, an
    ``ApplyRotaryEmb`` op in vLLM) and multiplies the full width; the two are
    the same arithmetic written twice, and this is the form ``layers/rotary.py``
    already uses so the two ropes read alike.

    IN FLOAT32, ALWAYS, and cast back afterwards -- WHICH FOLLOWS HF AND NOT
    vLLM. ``apply_rotary_pos_emb_vision`` in transformers 5.4.0 opens with
    ``q, k = q.float(), k.float()`` and closes by restoring the original dtype.
    vLLM does not: ``qwen2_5_vl.py:398`` builds
    ``ApplyRotaryEmb(enforce_enable=True)``, and ``enforce_enable`` is
    ``CustomOp``'s "take the custom-op path rather than the native one"
    argument (custom_op.py:130, :185) -- the precision flag is the SEPARATE
    ``enable_fp32_compute``, which defaults ``False``
    (rotary_embedding/common.py:132) and is not passed there. So vLLM's tower
    rotates at the serving dtype.

    This file takes HF's side deliberately, for two reasons worth writing down
    rather than leaving to be re-derived: a bf16 rotation is a measurable,
    silent accuracy loss in every block, and HF is the oracle the frozen
    recording was taken from, so matching vLLM here would have put a known bias
    between this module and its own de-vendoring proof. It is listed as a
    divergence in the module docstring and pinned by the harness.
    """
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(orig_dtype)


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    """The 2-D grid rope's frequency table: ``[MAX_GRID_SIZE, head_dim // 4]``.

    ``dim`` is ``head_dim // 2`` -- upstream's ``partial_rotary_factor: 0.5``
    (vLLM qwen2_5_vl.py:674) and HF's ``Qwen2_5_VisionRotaryEmbedding(head_dim
    // 2)`` say the same thing -- so the table is ``dim // 2 == head_dim // 4``
    wide. One row of it is looked up per AXIS, and the h-row and w-row are
    concatenated into the ``head_dim // 2`` band :func:`apply_rotary_emb_vision`
    consumes. That concatenation is where "2-D" actually lives.

    The cache is a non-persistent BUFFER, like ``layers/rotary.py``'s, so it
    never reaches ``state_dict()`` and never appears in the parameter
    correspondence.
    """

    def __init__(self, dim: int, theta: float = ROPE_THETA,
                 max_grid_size: int = MAX_GRID_SIZE) -> None:
        super().__init__()
        self.dim = dim
        self.max_grid_size = max_grid_size
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        freqs = torch.outer(
            torch.arange(max_grid_size, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``pos_ids`` ``[S, 2]`` (h, w) -> two ``[S, dim]`` tensors.

        ``cos_cached[pos_ids]`` is ``[S, 2, dim // 2]``; flattening the last two
        axes concatenates the h band and the w band in that order, which is the
        order the checkpoint's heads were trained with. Swapping them keeps
        every shape and produces an image encoder that is wrong in a way no
        shape check can see.

        FOR THE JAX MIRROR: the bounds check below reads ``int(pos_ids.max())``
        to drive a Python ``raise``, which is not expressible under ``jit`` --
        the same trap as ``Qwen2_5_VisionAttention``'s ``cu_seqlens.tolist()``.
        Both belong at geometry time, where the grid is a static Python tuple,
        not inside the traced forward.
        """
        largest = int(pos_ids.max()) if pos_ids.numel() else 0
        if largest >= self.max_grid_size:
            raise ValueError(
                f"grid position {largest} is outside the rotary table's "
                f"{self.max_grid_size} rows. Rebuild the table larger rather "
                "than letting the gather wrap, which would silently reuse "
                "another patch's rotation.")
        return (self.cos_cached[pos_ids].flatten(1),
                self.sin_cached[pos_ids].flatten(1))


class Qwen2_5_VisionPatchEmbed(nn.Module):
    """Conv3d over ``(temporal_patch_size, patch_size, patch_size)``.

    The processor hands ``[num_patches, C * tps * ps * ps]`` -- already
    patchified, so the convolution's stride equals its kernel and it is a
    per-patch linear map wearing a convolution's parameter layout. It stays a
    Conv3d because the CHECKPOINT stores it as one
    (``visual.patch_embed.proj.weight`` is ``[hidden, C, tps, ps, ps]``); a port
    that reshapes it into a Linear must transpose nothing but must still produce
    that 5-D shape, which ``test_param_correspondence`` checks up to a
    permutation of the dims.
    """

    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel_size,
                              stride=kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_patches = x.shape[0]
        x = x.view(num_patches, self.in_channels, self.temporal_patch_size,
                   self.patch_size, self.patch_size)
        return self.proj(x).view(num_patches, self.hidden_size)


class Qwen2_5_VisionMLP(nn.Module):
    """SwiGLU, with BIASES -- unlike the text stack's, which has none.

    The checkpoint is the evidence: ``visual.blocks.N.mlp.{gate,up,down}_proj``
    each ship a ``.bias`` and the text ``model.layers.N.mlp.*`` do not. vLLM
    constructs this block with ``bias=True`` (qwen2_5_vl.py:499) for the same
    reason. A port that copies the text MLP here loads cleanly, drops three
    tensors per block, and is quietly wrong.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            in_features, [hidden_features, hidden_features], bias=bias)
        self.down_proj = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2_5_VisionAttention(nn.Module):
    """Bidirectional attention, chunked by ``cu_seqlens``, no KV cache.

    ``num_kv_heads == num_heads``: the tower is not GQA, and the checkpoint's
    ``attn.qkv.weight`` is ``[3 * dim, dim]``, which only ``QKVParallelLinear``
    with equal head counts produces.

    WHY ``cu_seqlens`` AND NOT A MASK. The same module serves both the windowed
    blocks and the full-attention blocks; the ONLY difference between them is
    which cumulative-length vector arrives (see
    :meth:`Qwen2_5_VisionTransformer.forward`). Attending per chunk makes that
    difference structural instead of a mask the caller has to build correctly
    twice.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True)
        self.proj = RowParallelLinear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0]
        q, k, v = self.qkv.split(self.qkv(x))
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        q = apply_rotary_emb_vision(q, cos, sin)
        k = apply_rotary_emb_vision(k, cos, sin)

        bounds = cu_seqlens.tolist()
        # THE CHUNKS MUST COVER EVERY ROW, and that is checked rather than
        # absorbed. `out` is uninitialised memory, so a `cu_seqlens` that
        # started late or stopped short would return whatever was on the heap
        # for the uncovered rows -- finite, plausible, and wrong. Zeroing the
        # buffer instead would trade that for a different plausible wrong
        # answer; `forward` already refuses a `cu_seqlens` that disagrees with
        # the patch count, but only for the FULL vector, and the windowed
        # blocks are handed a different one.
        if not bounds or bounds[0] != 0 or bounds[-1] != seq_len:
            raise ValueError(
                f"cu_seqlens {bounds[:4]}..{bounds[-1:]} does not cover the "
                f"{seq_len} rows it is attending over: it must start at 0 and "
                "end at the sequence length. A partial cover would leave "
                "uninitialised rows in the output.")
        out = torch.empty_like(q)
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                # Upstream's padded ``cu_seqlens`` can repeat its last offset to
                # keep a CUDA-graph buffer a fixed size; an empty chunk is the
                # residue of that and contributes nothing. Skipped rather than
                # fed to softmax, which would produce NaN over zero keys.
                continue
            qc = q[start:end].transpose(0, 1).float()      # [H, L, D]
            kc = k[start:end].transpose(0, 1).float()
            vc = v[start:end].transpose(0, 1).float()
            # FP32 THROUGHOUT -- both matmuls AND the softmax, matching
            # attention/paged.py, which upcasts q and k into its einsum for the
            # same reason. This is WIDER THAN EITHER UPSTREAM: HF's eager path
            # matmuls at the input dtype and upcasts only the softmax, and
            # vLLM's accumulation width is whatever its kernel chooses. Listed
            # as a divergence in the module docstring; pinned by the harness.
            # NO CAUSAL MASK -- the tower
            # is bidirectional, and a port that reuses the decoder's masked
            # attention here still produces plausible embeddings.
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * self.scaling
            probs = torch.softmax(scores, dim=-1)
            out[start:end] = torch.matmul(probs, vc).transpose(0, 1).to(q.dtype)

        return self.proj(out.reshape(seq_len, self.num_heads * self.head_dim))


class Qwen2_5_VisionBlock(nn.Module):
    """Pre-norm attention + MLP, with RMSNorm -- not LayerNorm.

    Qwen2-VL used LayerNorm here and Qwen2.5-VL replaced it, which is why
    ``visual.blocks.N.norm1`` ships a ``.weight`` and no ``.bias``. That is the
    single easiest tensor set to mis-port from the older family's code, and it
    fails loudly (a missing ``.bias`` in the checkpoint) rather than silently,
    which is the one merciful thing about it.
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.attn = Qwen2_5_VisionAttention(dim, num_heads)
        self.mlp = Qwen2_5_VisionMLP(dim, mlp_hidden_dim, bias=True)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_attn = self.attn(self.norm1(x), cu_seqlens, cos, sin)
        # The FUSED two-argument RMSNorm, as upstream writes it
        # (qwen2_5_vl.py:522-523): `residual` is `x + x_attn` and the block's
        # output is that residual plus the MLP of its norm. Written this way and
        # not as two separate adds so the residual stream is threaded exactly as
        # the text decoder threads it.
        normed, residual = self.norm2(x, residual=x_attn)
        return residual + self.mlp(normed)


class Qwen2_5_VisionPatchMerger(nn.Module):
    """RMSNorm, regroup ``merge**2`` patches into one row, then a 2-layer MLP.

    ``mlp`` is an ``nn.Sequential`` and the GELU sits at index 1, which is why
    the checkpoint's keys are ``visual.merger.mlp.0.*`` and
    ``visual.merger.mlp.2.*`` with nothing at 1. The indices are the contract:
    ``canonical_param_key`` normalises ``mlp_0`` to ``mlp.0``, so a flax port may
    name its two Dense layers ``mlp_0``/``mlp_2`` -- but it may NOT renumber them
    to 0 and 1.

    THE GELU IS THE EXACT ONE. ``nn.GELU()`` is the erf form; the tanh
    approximation (``nn.GELU(approximate="tanh")``, which several ports reach
    for because other Qwen models use it) differs by up to ~1e-3 and is the
    single likeliest silent numeric divergence in this module. Upstream is
    ``nn.GELU()`` in both vLLM (:597) and HF.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=eps)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            RowParallelLinear(self.hidden_size, d_model, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class Qwen2_5_VisionTransformer(nn.Module):
    """The tower: patch embed -> 32 windowed/full blocks -> patch merger.

    ``forward(pixel_values, grid_thw)`` returns ``[num_merged_tokens,
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
         to single patches (:1092-1096);
      2. the rotary cos/sin are reordered with the patches, not left in grid
         order (:806-809);
      3. the inverse permutation is applied AFTER the merger, at merged-token
         granularity, not before it (:1126);
      4. blocks in ``fullatt_block_indexes`` see the whole image's
         ``cu_seqlens``; every other block sees the window one (:1100-1108).

    Get 1 and 3 both wrong in the same way and the output is still a permutation
    of the right rows; get only one wrong and the image is scrambled with no
    shape change anywhere.
    """

    def __init__(self, vision_config: VisionConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
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
        )
        # `head_dim // 2`, per upstream's partial_rotary_factor of 0.5. See
        # Qwen2_5_VisionRotaryEmbedding's docstring.
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(
            vision_config.head_dim // 2)
        self.blocks = nn.ModuleList([
            Qwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                eps=norm_eps,
            )
            for _ in range(vision_config.depth)
        ])
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            eps=norm_eps,
        )

    # -- geometry ---------------------------------------------------------- #

    def pos_ids_thw(self, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
        """``[grid_t * grid_h * grid_w, 2]`` of (h, w) per patch, MERGE-ORDERED.

        The reshape/permute pair (:720-739) is not a flatten: it puts the
        ``spatial_merge_size ** 2`` patches that will become ONE llm token
        adjacent to each other, which is the order ``pixel_values`` itself
        arrives in from the processor. Reading it as a plain row-major
        ``arange`` gives the right shape and rotates every patch by the wrong
        position.
        """
        merge = self.spatial_merge_size
        hpos = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
        wpos = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
        blocks = (grid_h // merge, merge, grid_w // merge, merge)
        hpos = hpos.reshape(*blocks).permute(0, 2, 1, 3).flatten()
        wpos = wpos.reshape(*blocks).permute(0, 2, 1, 3).flatten()
        return torch.stack([hpos, wpos], dim=-1).repeat(grid_t, 1)

    def window_index_thw(self, grid_t: int, grid_h: int,
                         grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(window_index, cu_window_seqlens)`` for one image (:763-798).

        ``window_index`` is a permutation of the MERGED token indices; the
        ``-100`` padding is how a window that runs off the edge of the grid is
        made rectangular and then dropped, so windows at the right and bottom
        edges are legitimately shorter than the rest. ``cu_window_seqlens``
        counts RAW patches (hence the ``* spatial_merge_unit``), because that is
        the granularity attention runs at.
        """
        merge_window = self.window_size // self.spatial_merge_size // self.patch_size
        llm_h = grid_h // self.spatial_merge_size
        llm_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
        pad_h = merge_window - llm_h % merge_window
        pad_w = merge_window - llm_w % merge_window
        num_windows_h = (llm_h + pad_h) // merge_window
        num_windows_w = (llm_w + pad_w) // merge_window
        padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        padded = padded.reshape(grid_t, num_windows_h, merge_window,
                                num_windows_w, merge_window)
        padded = padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, merge_window, merge_window)
        seqlens = (padded != -100).sum([2, 3]).reshape(-1)
        flat = padded.reshape(-1)
        window_index = flat[flat != -100]
        cu = (seqlens.cumsum(0) * self.spatial_merge_unit).to(torch.int32)
        # `unique_consecutive` drops the zero-length windows a fully padded
        # window would contribute; upstream does the same (:796) and the
        # attention loop's empty-chunk guard is the belt to this braces.
        return window_index, torch.unique_consecutive(cu)

    @staticmethod
    def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
        """The inverse of a permutation, in O(n) (:835-840).

        ``torch.argsort(perm)`` -- which HF uses -- returns the same tensor for
        a genuine permutation. The scatter is upstream's and is kept because it
        is also a cheap assertion: a non-permutation leaves holes, which the
        caller's shape checks then catch.
        """
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
        return inverse

    def encoder_metadata(self, grid_thw) -> dict:
        """Everything ``forward`` needs that depends only on the GRID.

        The residue of upstream's ``prepare_encoder_metadata`` (:874-1065) once
        CUDA-graph padding and per-backend ``cu_seqlens`` rewriting are gone:
        the loop at :915-958 and nothing else. Separated from ``forward`` for
        the same reason upstream separates it -- it is pure geometry, it is what
        the de-vendoring tests pin, and it never touches a parameter.
        """
        cos_parts: list[torch.Tensor] = []
        sin_parts: list[torch.Tensor] = []
        window_parts: list[torch.Tensor] = []
        cu_window_parts: list[torch.Tensor] = [torch.zeros(1, dtype=torch.int32)]
        seq_parts: list[torch.Tensor] = []

        window_offset = 0
        cu_window_last = torch.zeros((), dtype=torch.int32)
        for entry in grid_thw:
            grid_t, grid_h, grid_w = (int(v) for v in entry)
            cos, sin = self.rotary_pos_emb(self.pos_ids_thw(grid_t, grid_h, grid_w))
            window_index, cu_window = self.window_index_thw(grid_t, grid_h, grid_w)

            # The rope travels WITH the patches through the window permutation
            # (:806-809): regroup into merge-units, gather, flatten back.
            unit = self.spatial_merge_unit
            cos = cos.reshape(cos.shape[0] // unit, unit, -1)[window_index]
            sin = sin.reshape(sin.shape[0] // unit, unit, -1)[window_index]
            cos_parts.append(cos.flatten(0, 1))
            sin_parts.append(sin.flatten(0, 1))

            window_parts.append(window_index + window_offset)
            window_offset += grid_t * (grid_h // self.spatial_merge_size) * (
                grid_w // self.spatial_merge_size)

            cu_window = cu_window + cu_window_last
            cu_window_last = cu_window[-1]
            cu_window_parts.append(cu_window)

            # Full attention is per FRAME, not per image: a t-frame video
            # contributes t sequences of h*w patches each (:811-813).
            seq_parts.append(torch.full((grid_t,), grid_h * grid_w,
                                        dtype=torch.int32))

        window_index = torch.cat(window_parts)
        cu_seqlens = torch.cat(seq_parts).cumsum(0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
        return {
            "cos": torch.cat(cos_parts),
            "sin": torch.cat(sin_parts),
            "window_index": window_index,
            "reverse_indices": self.invert_permutation(window_index),
            "cu_seqlens": cu_seqlens,
            "cu_window_seqlens": torch.unique_consecutive(
                torch.cat(cu_window_parts)),
        }

    # -- forward ----------------------------------------------------------- #

    def forward(self, pixel_values: torch.Tensor, grid_thw) -> torch.Tensor:
        """``[num_patches, C*tps*ps*ps]`` + grid -> ``[num_merged, out_hidden]``.

        ``grid_thw`` is any iterable of ``(t, h, w)`` triples -- a
        ``[num_images, 3]`` tensor or a list of lists both work, because
        upstream takes a list here (:1070) and the processor produces a tensor.
        """
        hidden = self.patch_embed(pixel_values.to(self.dtype))
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
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype
