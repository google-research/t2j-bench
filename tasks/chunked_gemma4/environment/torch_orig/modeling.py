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

"""Inference-only gemma-4, de-vendored from vllm/model_executor/models/gemma4.py.

TEXT PATH ONLY. `google/gemma-4-E2B-it` declares
`architectures: ["Gemma4ForConditionalGeneration"]` and both upstream registries
route that string to the MULTIMODAL module, so this port deliberately ignores
the checkpoint's own declared architecture. `Gemma4ForCausalLM` exists upstream
for exactly this, in both frameworks, and imports nothing from `gemma4_mm.py` or
from any tower. `tests/instance_context.md` states the choice and the rule
behind it; the 472 M parameters of vision and audio tower simply go unread,
because `model/loader.py::convert_checkpoint` is a PULL model.

THE CLASS NAMES ARE VLLM'S. `Gemma4MLP` (:225), `Gemma4Attention` (:375),
`Gemma4DecoderLayer` (:554), `Gemma4Model` (:949) and `Gemma4ForCausalLM`
(:1494). They also satisfy `core_llama3`'s `{Family}MLP` / `{Family}Attention`
convention, and that is a COINCIDENCE rather than a derivation: `core_gpt_oss`
is the standing counter-example (`OAIAttention`, `MLPBlock`,
`TransformerBlock`), and `test_reference_units` grades a port on mirroring THIS
file's submodule attribute names, so the rule is "mirror vLLM" and the names
were read off the file rather than built from the family string.

THE SUBMODULE NAMES ARE THE CHECKPOINT'S TOO, which is why -- unlike gpt_oss --
there is NO `CHECKPOINT_RENAMES` entry for the attention block: vLLM calls it
`self_attn` and so does the checkpoint. The one naming difference is a PREFIX,
handled in `model/loader.py`: the checkpoint stores the text stack under
`model.language_model.*` (it is shared with the conditional-generation wrapper)
and this tree is `model.*`. vLLM does the identical strip at `gemma4.py:1605`.

Structural differences from upstream, all of them removals, exactly as
`core_llama3` states them: TP=1 so every *ParallelLinear collapses to a plain
linear while KEEPING the fused qkv_proj and gate_up_proj layouts; no
@support_torch_compile, no VllmConfig, no LoRA, no pipeline parallel, no YOCO
self/cross decoder split (`Gemma4SelfDecoderLayers` / `Gemma4CrossDecoderLayers`
are a torch.compile partition of the same 35 layers, not a second model); and
vLLM's `Attention` layer is replaced by an explicit write_kv + paged_attention
pair so the cache write is visible in the model rather than hidden inside a
backend.

===========================================================================
WHAT IS NEW HERE. FOUR PER-LAYER HETEROGENEITIES, NOT ONE.
===========================================================================

`config.py` derives all four and this file applies them. Two of the four are
shape-visible and fail loudly; two are VALUES and fail silently.

  1. **The interleaved sliding window.** `layer_types` is `ssssF` seven times,
     window 512. NOT NEW to this class -- `core_gpt_oss` ships the same
     mechanism at a 1:1 pattern and a 128 window, and `attention/paged.py` is
     byte-identical to the `core` mode's copy because it already takes
     `num_kv_heads` per call and derives `head_dim` from the cache slice. For
     gemma-4 this axis is CONFIGURATION.

  2. **Heterogeneous KV geometry.** `head_dim` is 256 on the 28 sliding layers
     and 512 on the 7 full ones. One rectangular cache cannot express that, so
     the container gets TWO groups. Shape-visible.

  3. **KV-cache sharing.** `num_kv_shared_layers: 20`, so only 15 of 35 layers
     allocate a slot. A shared layer:
       * still projects q, k and v -- the checkpoint stores full Q/K/V weights
         for it and upstream loads and runs them (`gemma4.py:462` says so in as
         many words), so this port does too, and BOTH SIDES make the same
         choice. Its k and v are then discarded;
       * gets **no `k_norm` and no `v_norm` and no RoPE on K or V**. Only Q is
         normed and rotated;
       * carries `kv_writes = False`, so it never scatters into the slot it is
         about to read.
     The source slot holds the source layer's already-normed, already-rotated
     K/V for every position, written EARLIER IN THE SAME FORWARD -- the source
     is always `< first_kv_shared_layer()`, so the ordering is an invariant of
     the map and not a coincidence of this checkpoint.

  4. **The double-wide MLP.** `use_double_wide_mlp: true` doubles
     `intermediate_size` on exactly the KV-shared layers, so the declared 6144
     is wrong for 20 of 35. Shape-visible -- `model/loader.py` refuses the
     mismatch -- and coupled to axis 3 through one config key, which is why
     `config.intermediate_size_for_layer` is derived from
     `config.is_kv_shared_layer` and not from a second `>= 15` test.

PER-LAYER EMBEDDINGS (PLE) are the fifth new thing and are not a heterogeneity:
every layer has them. 46.6 % of the checkpoint is one `[262144, 8960]` table,
`262144 x (35 x 256)`. `Gemma4Model.compute_per_layer_inputs` builds a
`[T, 35, 256]` tensor once per forward from two tracks -- an embedding lookup
scaled by `sqrt(P)`, and a projection of the POST-embedding-scale residual
stream scaled by `hidden_size ** -0.5` and RMSNorm'd over P -- combines them at
`1/sqrt(2)`, and each layer consumes its own `[:, layer_idx, :]` slice at the
END of the layer, after both residual adds.

===========================================================================
THREE TRAPS. EACH PRODUCES A PLAUSIBLE WRONG ANSWER RATHER THAN A CRASH.
===========================================================================

**`layer_scalar` is a Parameter here and a BUFFER upstream.** Every one of the
35 layers carries `layers.{i}.layer_scalar`, shape (1,), and the values are
neither 1.0 nor uniform -- measured from the checkpoint: L0 0.01782, L1 0.22266,
L2 0.79297, L30 0.87109, L34 0.16699. vLLM registers it with
`register_buffer("layer_scalar", torch.ones(1))` and then has to say
`params_dict.update(dict(self.named_buffers()))  # Include buffers (e.g.
layer_scalar) so they can be loaded too` (`gemma4.py:1399`). A buffer here would
be invisible to `model/loader.py`'s coverage rule, which walks
`named_parameters()`, and would silently stay at 1.0 -- a 0.017x-to-0.87x
rescale of every layer's output, finite and wrong. `core_gpt_oss` made the
identical call for `sinks` and for the identical reason.

**The rope pairs `(i, i+256)`, not `(i, i+64)`.** `layers/rotary.py`, whose
module docstring is the long form. `partial_rotary_factor: 0.25` on head_dim 512
rotates 64 pairs either way, so nothing about the shapes says which 64.

**`sm_scale` is 1.0, and it is the most falsifiable number in this port.**
Both upstreams say so independently -- vLLM `gemma4.py:407-410` ("Gemma4 uses
scaling=1.0 ... Q/K norms with learnable weights handle scaling implicitly"),
tpu-inference `gemma4.py:245-246`, which HEDGES ("Assuming Gemma 4 also uses a
custom scalar, not 1/sqrt(head_dim)"). Three further checks were run before it
was frozen: `config.json`
carries no `query_pre_attn_scalar` and no `attn_logit_softcapping` (so there is
no third scale hiding in the config, unlike gemma-2/3); q and k are BOTH
RMS-normalised to unit scale immediately before the dot product, which is what
makes an un-normalised 1.0 sane where it would not be for an unnormed model; and
a greedy continuation at 1.0 is fluent English while the same continuation at
`head_dim ** -0.5` is not. That last one is the only DECISIVE check available --
torch-vs-JAX parity cannot see a scale both sides share.

===========================================================================

Unchanged from core_llama3, because they are the graded contract:
  * K is written to the cache AFTER `rotary_emb`; V is written un-rotated.
  * The forward is token-flat: [T, H], never [B, L, H].

THE RESIDUAL PATTERN IS GEMMA'S SANDWICH, not llama's fused-residual RMSNorm.
Four norms per layer, two residual adds, and the norm is applied to the BLOCK
OUTPUT before the add rather than to the block input only::

    residual = x;  x = input_layernorm(x);          x = self_attn(x)
    x = post_attention_layernorm(x);                x = x + residual
    residual = x;  x = pre_feedforward_layernorm(x); x = mlp(x)
    x = post_feedforward_layernorm(x);              x = x + residual

so `layers/layernorm.RMSNorm`'s two-argument fused-residual form -- which every
other family in this class uses -- is deliberately NOT used here. It computes
`norm(x + residual)`, and that is a different graph.

AND THE NORM IS THE PLAIN ONE, not gemma-3's `(1 + weight)` form. vLLM
constructs `RMSNorm(hidden_size, eps=...)` from the shared layer and comments
"Layer norms: output = norm(x) * weight" at every site (`gemma4.py:611`,
`:432`). A port that carries gemma-3's off-by-one weight convention across is
finite, fluent and wrong at every layer.

TIED. `tie_word_embeddings: true` and the checkpoint ships no `lm_head.weight`
-- confirmed by censusing all 600 text tensors -- so the head decodes through
the embedding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from attention.metadata import AttentionMetadata
from attention.paged import paged_attention
from cache.kv_cache import _require_slot, write_kv
from config import ModelConfig
from layers.activation import GeluAndMul
from layers.layernorm import RMSNorm
from layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from layers.rotary import RotaryEmbedding
from layers.vocab_embed import ParallelLMHead, VocabParallelEmbedding


class RMSNormNoWeight(nn.Module):
    """RMS normalisation with NO learned scale. gemma-4's `v_norm`, and only that.

    Upstream spells this `RMSNorm(self.head_dim, eps=..., has_weight=False)`
    (`gemma4.py:436`; tpu-inference `JaxRmsNorm(..., use_scale=False)`), reusing
    the shared norm with a flag. THIS PORT CANNOT, and the reason is a property
    of the class rather than of the model: `layers/layernorm.py` is in the
    reference tree's SHARED column and is byte-identical across every instance
    in this class, so adding a flag for one family would be a fourteen-instance
    edit. It lives here
    instead, at its only call site, and it is reported as a class-wide gap
    rather than fixed from inside one family.

    IT IS ALSO NOT A PARAMETER-BEARING MODULE, and that is independently
    forced: the checkpoint ships `q_norm.weight` and `k_norm.weight` on every
    layer and NO `v_norm.weight` at all (measured -- 17 tensors per layer, and
    that is not one of them), so an `RMSNorm` here would add 35 parameters the
    loader could never fill and `load_weights` would refuse the model.

    The arithmetic is `layers/layernorm.RMSNorm`'s with the final multiply
    removed, CAST ORDER INCLUDED: the reciprocal-sqrt accumulates in fp32 and
    the result is cast back to the input dtype. Getting that order wrong is
    bit-identical in fp32 and is not in bfloat16, which is this class's serving
    dtype.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(orig_dtype)


class Gemma4MLP(nn.Module):
    """vLLM's `Gemma4MLP` (:225).

    `intermediate_size` IS A CONSTRUCTOR ARGUMENT AND NOT A CONFIG READ, which
    is the whole shape of axis 4. Both upstreams do the same and both say why:
    tpu-inference `gemma4.py:66-72` ("KV-shared layers use 2x
    config.intermediate_size ... The caller computes this in Gemma4DecoderLayer
    and passes it in explicitly").
    """

    def __init__(self, config: ModelConfig, intermediate_size: int) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [intermediate_size] * 2, bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, config.hidden_size, bias=False,
        )
        self.act_fn = GeluAndMul(config.hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Gemma4Attention(nn.Module):
    """vLLM's `Gemma4Attention` (:375). The KV-share branch is at :534-546."""

    def __init__(self, config: ModelConfig, layer_idx: int,
                 rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # WHERE THIS LAYER'S KV LIVES IN THE CACHE CONTAINER, resolved once here
        # rather than re-derived at every forward -- and for this family it is
        # NOT the identity in either component. `config.layer_slot()` puts the
        # 15 allocating layers into two groups (sliding, head_dim 256; full,
        # head_dim 512) and points the 20 shared ones back at slot 11 of group 0
        # or slot 2 of group 1; `config.layer_writes()` is False on exactly
        # those 20.
        #
        # THOSE ARE SLOT INDICES AND NOT LAYER INDICES. Upstream's
        # `kv_share_map` speaks in layers -- {15: 13, 19: 14, ...} -- and the
        # container speaks in (group, slot). Source layer 13 is the twelfth
        # allocating SLIDING layer, so slot 11 of group 0; source layer 14 is
        # the third allocating FULL one, so slot 2 of group 1. An implementer
        # who carries 13 and 14 across asks for slot 13 of a 12-slot group,
        # which `_require_slot` refuses by name below.
        #
        # THIS IS WHERE THE MAP IS APPLIED; engine.py declares the container's
        # GEOMETRY (`config.kv_groups()`), and config.py holds the one
        # definition both spell it from.
        self.kv_group, self.kv_slot = config.layer_slot()[layer_idx]
        self.kv_writes = config.layer_writes()[layer_idx]
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads_for_layer(layer_idx)
        # PER LAYER TYPE: 256 sliding, 512 full. NOT `config.head_dim`, which is
        # only the sliding one, and not `hidden_size // num_attention_heads`,
        # which is 192 and matches no layer in this model.
        self.head_dim = config.head_dim_for_layer(layer_idx)
        # 1.0, NOT head_dim ** -0.5 -- see the module docstring's third trap.
        self.scaling = 1.0
        self.sliding_window = config.sliding_window_for_layer(layer_idx)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.num_heads,
            self.num_kv_heads, bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias,
        )
        # Q AND K NORMS HAVE A LEARNED SCALE AND V'S DOES NOT. All three
        # normalise over the HEAD dimension, so their width is this layer's
        # head_dim and differs between the two layer types.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoWeight(eps=config.rms_norm_eps)
        # SHARED BETWEEN EVERY LAYER OF THIS TYPE, and handed in rather than
        # constructed here as upstream does it. `layers/rotary.py` carries the
        # arithmetic: at max_position 131072, 35 private cos/sin tables are
        # 5.6 GB per model copy and the grader holds three, against 402 MB for
        # two shared ones. A table is a pure function of (base, head_size,
        # max_position), so no value changes -- only how many copies exist.
        self.rotary_emb = rotary_emb

    def forward(self, hidden: torch.Tensor,
                kv_cache: tuple[torch.Tensor, ...],
                attn_metadata: AttentionMetadata,
                slot_mapping: torch.Tensor) -> torch.Tensor:
        t = hidden.shape[0]
        q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)

        # Q IS ALWAYS NORMED AND ALWAYS ROTATED, on every one of the 35 layers.
        q = self.q_norm(q)
        if not self.is_kv_shared_layer:
            k = self.k_norm(k)
            q, k = self.rotary_emb(attn_metadata.input_positions, q, k)
            v = self.v_norm(v)
        else:
            # THE SHARED BRANCH. No k_norm, no v_norm, no RoPE on K or V; the
            # slot this layer reads already holds the SOURCE layer's normed and
            # rotated K/V for every position, written earlier in this same
            # forward. `k` is passed to the rope only so that both branches call
            # the same op with the same signature -- upstream writes it
            # identically, `q = self.rotary_emb(positions, q, k)[0]` -- and the
            # rotated `k` is discarded along with `v` a few lines below, because
            # `kv_writes` is False and `paged_attention` reads the cache.
            q, _ = self.rotary_emb(attn_metadata.input_positions, q, k)

        # BEFORE the index, so that BOTH sides refuse a bad (group, slot) the
        # same way: torch would raise IndexError here anyway, but JAX clamps the
        # read and drops the write, and the two references have to agree about
        # what a broken map does. THIS IS THE FAMILY THE GUARD WAS WRITTEN FOR.
        _require_slot(kv_cache, self.kv_group, self.kv_slot)

        # Post-RoPE K and pre-RoPE V go into the cache -- the order matters and
        # is the classic port bug: caching pre-RoPE K silently breaks decode.
        group = kv_cache[self.kv_group]
        if self.kv_writes:
            write_kv(group[self.kv_slot], k, v, slot_mapping)

        attn = paged_attention(
            q, group[self.kv_slot], attn_metadata,
            num_kv_heads=self.num_kv_heads, sm_scale=self.scaling,
            sliding_window=self.sliding_window,
        )
        return self.o_proj(attn.reshape(t, self.num_heads * self.head_dim))


class Gemma4DecoderLayer(nn.Module):
    """vLLM's `Gemma4DecoderLayer` (:554). Four norms, PLE, and `layer_scalar`."""

    def __init__(self, config: ModelConfig, layer_idx: int,
                 rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        h, eps = config.hidden_size, config.rms_norm_eps
        p = config.hidden_size_per_layer_input

        self.self_attn = Gemma4Attention(config, layer_idx, rotary_emb)
        # DOUBLE WIDE ON EXACTLY THE KV-SHARED LAYERS -- 12288 rather than the
        # declared 6144 on 20 of the 35. Derived from `config`, never from a
        # second copy of the `>= 15` test; see the module docstring, axis 4.
        self.mlp = Gemma4MLP(config, config.intermediate_size_for_layer(layer_idx))

        # FOUR norms, not two. See the module docstring's residual pattern.
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)

        # PLE, per layer. `per_layer_input_gate` is 1536 -> 256 and
        # `per_layer_projection` is 256 -> 1536; the norm that follows is over
        # HIDDEN SIZE, not over P, unlike the model-level
        # `per_layer_projection_norm`. The two are easy to transpose and the
        # checkpoint's shapes are the arbiter: `post_per_layer_input_norm.weight`
        # is [1536] and `per_layer_projection_norm.weight` is [256].
        self.per_layer_input_gate = ReplicatedLinear(h, p, bias=False)
        self.per_layer_projection = ReplicatedLinear(p, h, bias=False)
        self.post_per_layer_input_norm = RMSNorm(h, eps=eps)

        # A PARAMETER, NOT A BUFFER -- see the module docstring's first trap.
        # `torch.empty`, like every other weight in this tree: `load_weights`
        # refuses to return a model with any parameter left unwritten, so an
        # initialiser that looked plausible (ones, as upstream uses) would only
        # make a failure to load it harder to see.
        self.layer_scalar = nn.Parameter(torch.empty(1))

    def forward(self, hidden: torch.Tensor,
                kv_cache: tuple[torch.Tensor, ...],
                attn_metadata: AttentionMetadata,
                slot_mapping: torch.Tensor,
                per_layer_input: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, kv_cache, attn_metadata, slot_mapping)
        hidden = self.post_attention_layernorm(hidden)
        hidden = hidden + residual

        residual = hidden
        hidden = self.pre_feedforward_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)
        hidden = hidden + residual

        # PLE, at the END of the layer and after BOTH residual adds
        # (`gemma4.py:733-741`). The gate is a tanh GELU of the layer's own
        # output -- not of its input, and not of the per-layer embedding.
        gate = F.gelu(self.per_layer_input_gate(hidden), approximate="tanh")
        contribution = self.post_per_layer_input_norm(
            self.per_layer_projection(gate * per_layer_input))
        hidden = hidden + contribution

        # LAST. After PLE, after both residual adds, and before the next layer
        # sees anything (`gemma4.py:745`).
        return hidden * self.layer_scalar


class Gemma4Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        p = config.hidden_size_per_layer_input
        ell = config.num_hidden_layers
        if p <= 0:
            raise ValueError(
                "hidden_size_per_layer_input is 0, so this checkpoint has no "
                "per-layer embeddings. This reference implements the PLE "
                "variants (E2B/E4B) and refuses the others by name rather than "
                "running a model that is missing 46.6% of its parameters.")

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, h)
        # 46.6 % OF THE CHECKPOINT IS THIS ONE TABLE: [262144, 8960], which is
        # `vocab_size_per_layer_input x (num_hidden_layers x P)`. It is an
        # embedding and not a stack of them, so the layer axis only exists after
        # the reshape in `compute_per_layer_inputs`.
        self.embed_tokens_per_layer = VocabParallelEmbedding(
            config.vocab_size_per_layer_input, ell * p)
        self.per_layer_model_projection = ColumnParallelLinear(
            h, ell * p, bias=False)
        # OVER P, not over hidden_size. See the per-layer norm above for the
        # pair that is easy to transpose.
        self.per_layer_projection_norm = RMSNorm(p, eps=config.rms_norm_eps)

        # TWO ROTARY TABLES FOR THIRTY-FIVE LAYERS -- see `Gemma4Attention` and
        # `layers/rotary.py`. Constructed here, where "one per layer TYPE" can
        # be stated once, and handed down.
        rotary: dict[str, RotaryEmbedding] = {}
        for i in range(ell):
            kind = config.layer_type(i)
            if kind in rotary:
                continue
            rope = config.rope_for_layer(i)
            rotary[kind] = RotaryEmbedding(
                head_size=config.head_dim_for_layer(i),
                # rotary_dim == head_size. NOT a truncated rotary_dim -- see
                # `layers/rotary.py`, which is where this family's sharpest
                # trap lives.
                rotary_dim=config.head_dim_for_layer(i),
                max_position=config.max_position_embeddings,
                base=float(rope.get("rope_theta", 10000.0)),
                is_neox_style=True,
                rope_scaling=rope,
                partial_rotary_factor=float(
                    rope.get("partial_rotary_factor", 1.0)),
            )
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i, rotary[config.layer_type(i)])
             for i in range(ell)]
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)

        # THE EMBEDDING SCALE AND THE THREE PLE SCALES, as Python floats.
        # Upstream registers all four as non-persistent buffers so they follow
        # the model onto the device and survive torch.compile's AOT cache
        # (`gemma4.py:988`, `:1012`, `:1019`, `:1047`); neither reason applies
        # to a TP=1 eager reference, and a buffer would be a tensor in
        # `state_dict()` with no counterpart in the JAX twin's parameter pytree.
        #
        # A PYTHON FLOAT TIMES A bfloat16 TENSOR IS COMPUTED IN bfloat16 ON BOTH
        # FRAMEWORKS -- the scalar is converted to the tensor's dtype rather
        # than promoting it -- so `sqrt(1536) = 39.191836` is really 39.25 at
        # the serving dtype, on both sides, and vLLM's explicit
        # `torch.tensor(..., dtype=model_dtype)` cast is reproduced rather than
        # skipped. That agreement is verified rather than assumed.
        self.embedding_scale = float(h) ** 0.5
        self.embed_scale_per_layer = float(p) ** 0.5
        self.per_layer_input_scale = 1.0 / (2.0 ** 0.5)
        # NOTE THE SIGN. gemma-4 uses `hidden_size ** -0.5` here where the
        # embedding scale above is `hidden_size ** +0.5`, and gemma-3n flipped
        # the other one. tpu-inference calls the difference out at
        # `gemma4.py:837`.
        self.per_layer_projection_scale = float(h) ** -0.5

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Track A: the PLE lookup, `[T, L, P]`. vLLM `gemma4.py:828-850`.

        THE MASK IS A NO-OP ON THIS CHECKPOINT AND IS KEPT ANYWAY.
        `vocab_size_per_layer_input == vocab_size == 262144` on E2B, so no id can
        fall outside it -- but the general contract is that the PLE table has its
        own, smaller vocabulary and out-of-range ids read slot 0, and the E4B and
        12B siblings differ. Deleting a no-op guard because this one checkpoint
        does not need it is how a reference stops being a reference.
        """
        mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.config.vocab_size_per_layer_input)
        ple_ids = torch.where(mask, input_ids, torch.zeros_like(input_ids))
        embeds = self.embed_tokens_per_layer(ple_ids) * self.embed_scale_per_layer
        return embeds.reshape(*input_ids.shape,
                              self.config.num_hidden_layers,
                              self.config.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor,
                                 per_layer_inputs: torch.Tensor) -> torch.Tensor:
        """Track B, and the combination. vLLM `gemma4.py:852-878`.

        `inputs_embeds` is the POST-embedding-scale residual stream, not the raw
        lookup -- the projection sees `embed_tokens(ids) * sqrt(hidden_size)`.
        Feeding it the pre-scale embedding is a uniform 39.19x error on one of
        the two tracks and is invisible to every shape check.
        """
        projection = self.per_layer_model_projection(inputs_embeds)
        projection = projection * self.per_layer_projection_scale
        projection = projection.reshape(*inputs_embeds.shape[:-1],
                                        self.config.num_hidden_layers,
                                        self.config.hidden_size_per_layer_input)
        # OVER P. The reshape happens BEFORE the norm, so the norm sees a
        # [T, L, P] tensor and normalises the last axis; normalising the
        # [T, L*P] tensor instead is one line earlier and a different model.
        projection = self.per_layer_projection_norm(projection)
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def compute_per_layer_inputs(self, input_ids: torch.Tensor,
                                 inputs_embeds: torch.Tensor) -> torch.Tensor:
        """`[T, L, P]`, computed ONCE per forward and sliced per layer."""
        return self.project_per_layer_inputs(inputs_embeds,
                                             self.get_per_layer_inputs(input_ids))

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping,
                output_hidden_states=False):
        hidden = self.embed_tokens(input_ids) * self.embedding_scale
        per_layer_inputs = self.compute_per_layer_inputs(input_ids, hidden)
        collected = [hidden] if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, kv_cache, attn_metadata, slot_mapping,
                           per_layer_inputs[:, i, :])
            if output_hidden_states:
                collected.append(hidden)
        hidden = self.norm(hidden)
        if output_hidden_states:
            # The last entry is the POST-final-norm stream, exactly as every
            # other instance in this class freezes it.
            collected[-1] = hidden
            return hidden, collected
        return hidden


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # TIED, and unlike gpt_oss this arm really fires: the checkpoint ships
        # no `lm_head.weight` at all. The conditional is kept rather than
        # collapsed because it is the one line that states the difference, and
        # because a config that did NOT ask for a tie must not silently get the
        # embedding as its unembedding.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, kv_cache, attn_metadata, slot_mapping=None,
                output_hidden_states=False):
        if slot_mapping is None:
            slot_mapping = torch.arange(input_ids.shape[0], dtype=torch.long)
        return self.model(input_ids, kv_cache, attn_metadata, slot_mapping,
                          output_hidden_states=output_hidden_states)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head.project(hidden)
        # FINAL LOGIT SOFT-CAPPING. `final_logit_softcapping: 30.0`, so the
        # whole vocabulary is squashed into (-30, 30) before anything samples
        # from it. vLLM applies it inside `LogitsProcessor.forward`
        # (`layers/logits_processor.py:112-115`) as exactly these three steps and
        # IN THE HEAD'S OWN DTYPE -- there is no upcast there and there is none
        # here, so the compression happens at bfloat16 on the serving tier.
        #
        # THAT IS A REAL PRECISION COST AND IT IS UPSTREAM'S, NOT THIS PORT'S:
        # `tanh` is contractive, so two logits that differed by d before the cap
        # differ by `d * sech^2(x / 30)` after it, and near the top of the
        # distribution that factor is small. The instance's `argmax_margin` and
        # `decisive_margin` are measured against the CAPPED logits for exactly
        # this reason -- see tests/instance.yaml.
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits
