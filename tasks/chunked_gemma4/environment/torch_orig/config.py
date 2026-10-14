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

"""Instance configuration: model shape (ModelConfig) and runtime knobs (RunConfig).

THIS FILE IS THE ONE PLACE gemma-4's FOUR PER-LAYER HETEROGENEITIES ARE DERIVED,
and that concentration is deliberate rather than tidy. The four axes are:

  1. ``layer_types`` -- ``ssssF`` x 7, so 28 sliding layers and 7 full ones.
     Shared with `core_gpt_oss`; the only new thing is the pattern and the
     window (512, not 128).
  2. **KV geometry.** ``head_dim`` is **256 on sliding layers and 512 on full
     ones**. ``config.json`` declares BOTH -- ``head_dim: 256`` and
     ``global_head_dim: 512`` -- and the per-layer ``q_proj`` out-dims agree with
     ``layer_types`` on all 35 layers (measured against the checkpoint header).
  3. **KV sharing.** ``num_kv_shared_layers: 20``, so layers 15-34 attend against
     an earlier layer's cache slot and allocate none of their own. **15 slots for
     35 layers.**
  4. **The double-wide MLP.** ``use_double_wide_mlp: true`` doubles
     ``intermediate_size`` on exactly the KV-shared layers, so the declared 6144
     is WRONG for 20 of 35 layers.

Axes 3 and 4 are DERIVED FROM THE SAME BOUNDARY and are therefore derived from
the same helper here. Both upstreams compute them independently (vLLM
``gemma4.py:592-601``, tpu-inference ``gemma4.py:568-572``) and a port that
derives one from ``layer_types`` and the other from a hardcoded ``layer_idx >=
15`` will load 20 layers' MLPs at the wrong shape the moment a sibling
checkpoint moves the boundary.

TWO SCALARS ARE REPORTED FOR A MODEL THAT HAS NEITHER, and a reader must not
take them for the whole picture:

  * ``head_dim`` is 256 -- ``config.json``'s own value, which describes the
    SLIDING layers only. It is a field rather than a derivation because
    ``hidden_size // num_attention_heads`` is 1536 / 8 = 192 and matches no layer
    in this model. ``global_head_dim`` carries the other one.
  * ``num_key_value_heads`` is 1 for BOTH layer types on this checkpoint
    (``num_global_key_value_heads`` is ``null``), so the single scalar is honest
    here and would not be on E4B/12B.

They are kept as scalars because the frozen cache-spec derivation
reads exactly these two off ModelConfig and
``tests/test_spec.py`` compares them unconditionally, in the ``groups`` branch
too. Everything that actually needs a per-layer answer must call
:meth:`ModelConfig.head_dim_for_layer` / :meth:`num_kv_heads_for_layer`.

THE CACHE CONTAINER'S GEOMETRY AND ITS INDEX MAP ARE BOTH SPELLED HERE, by
:meth:`kv_groups`, :meth:`layer_slot` and :meth:`layer_writes` -- because
``cache/kv_cache.py`` is byte-identical across the whole class and carries only
the IDENTITY maps, which gemma-4 is the first family not to use. ``engine.py``
consumes :meth:`kv_groups`; ``modeling.py`` consumes the other two. Neither file
re-derives the other's value, which is what
``cache/kv_cache._require_slot`` refuses the half-done version of.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: The two ``layer_types`` values this reference implements. Anything else is
#: refused by name rather than silently treated as full attention.
_SLIDING = "sliding_attention"
_FULL = "full_attention"


@dataclass(frozen=True)
class ModelConfig:
    """gemma-4-E2B's TEXT shape, read from ``config.json`` and never inferred.

    READ OUT OF ``config.json``'s ``text_config``, not out of its top level.
    Every gemma-4 checkpoint is a nested config: the top level describes the
    conditional-generation wrapper (vision tower, audio tower, the modality
    token ids) and ``text_config`` describes the decoder this instance ports.
    :meth:`from_hf` dereferences it and falls back to the top level, so a
    hypothetical flat text-only sibling still loads.

    NINE FIELDS HERE ARE NOT IN ``core_gpt_oss``'s ModelConfig, and each one is a
    fact this family has and the dense-plus-window families do not:
    ``global_head_dim``, ``num_global_key_value_heads``, ``num_kv_shared_layers``,
    ``use_double_wide_mlp``, ``final_logit_softcapping``,
    ``hidden_size_per_layer_input``, ``vocab_size_per_layer_input``,
    ``hidden_activation`` and the two-entry ``rope_parameters`` dict.

    ``layer_types`` is a TUPLE so the dataclass stays frozen in substance as well
    as in name. ``rope_parameters`` is left as the nested dict the JSON carries,
    exactly as ``core_llama3`` leaves ``rope_scaling``.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: dict
    tie_word_embeddings: bool
    attention_bias: bool
    sliding_window: int | None
    layer_types: tuple[str, ...]
    #: The FULL layers' head_dim. `head_dim` above is the sliding one.
    global_head_dim: int
    #: The FULL layers' KV head count, or None to reuse `num_key_value_heads`.
    #: `null` on E2B; a real number on the checkpoints that set
    #: `attention_k_eq_v`.
    num_global_key_value_heads: int | None
    #: How many of the LAST layers share an earlier layer's KV slot. 20 of 35.
    num_kv_shared_layers: int
    #: Doubles `intermediate_size` on exactly the KV-shared layers.
    use_double_wide_mlp: bool
    #: `tanh(logits / cap) * cap`, applied in `compute_logits`. 30.0.
    final_logit_softcapping: float | None
    #: Per-layer-embedding width, P. 256, and 0 disables PLE entirely (26B/31B).
    hidden_size_per_layer_input: int
    #: The PLE table's own vocabulary. Equal to `vocab_size` on E2B, so the
    #: out-of-range mask in `modeling.py` is a no-op HERE and is not on E4B.
    vocab_size_per_layer_input: int
    #: `gelu_pytorch_tanh`. Not a default -- `layers/activation.py::GeluAndMul`
    #: refuses anything else, because a SiLU gate on a GELU-trained MLP is
    #: finite, fluent and wrong.
    hidden_activation: str

    # ----------------------------------------------------------------- #
    # Axis 1 -- the interleaved window.
    # ----------------------------------------------------------------- #
    def layer_type(self, layer_idx: int) -> str:
        """``'sliding_attention'`` or ``'full_attention'``, off ``layer_types``.

        Read off the config rather than off ``layer_idx % 5 == 4``. The two agree
        on this checkpoint -- 35 entries, ``ssssF`` seven times -- but the config
        is the authority and a derivative that happens to agree is a derivative
        that can stop agreeing. `core_gpt_oss` makes the same argument about the
        1:1 pattern its own checkpoint has.
        """
        if layer_idx >= len(self.layer_types):
            raise IndexError(
                f"layer {layer_idx} has no entry in layer_types "
                f"({len(self.layer_types)} entries); config.json declares one per "
                "layer and a missing one would silently become full attention")
        kind = self.layer_types[layer_idx]
        if kind not in (_SLIDING, _FULL):
            raise ValueError(f"unknown layer_type {kind!r} at layer {layer_idx}")
        return kind

    def sliding_window_for_layer(self, layer_idx: int) -> int | None:
        """The window this layer attends over, or None for full attention."""
        if self.layer_type(layer_idx) == _FULL:
            return None
        if self.sliding_window is None:
            raise ValueError(
                f"layer {layer_idx} is {_SLIDING!r} but config.json declares no "
                "sliding_window")
        return self.sliding_window

    # ----------------------------------------------------------------- #
    # Axis 2 -- the per-layer KV geometry.
    # ----------------------------------------------------------------- #
    def head_dim_for_layer(self, layer_idx: int) -> int:
        """256 on a sliding layer, 512 on a full one. MEASURED, not assumed.

        The checkpoint's own ``q_proj`` out-dims are 2048 on the 28 sliding
        layers (8 heads x 256) and 4096 on the 7 full ones (8 x 512), and they
        agree with ``layer_types`` on all 35 with zero mismatches.
        """
        return (self.global_head_dim if self.layer_type(layer_idx) == _FULL
                else self.head_dim)

    def num_kv_heads_for_layer(self, layer_idx: int) -> int:
        """1 for both layer types on E2B; the two can differ on the siblings.

        ``num_global_key_value_heads`` is ``null`` here, so the fallback is what
        actually runs. It is kept because the field exists in the config schema
        and the checkpoints that set ``attention_k_eq_v`` populate it -- reading
        ``num_key_value_heads`` unconditionally would be right for E2B and wrong
        for the family.
        """
        if (self.layer_type(layer_idx) == _FULL
                and self.num_global_key_value_heads is not None):
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    # ----------------------------------------------------------------- #
    # Axes 3 and 4 -- the KV-share boundary, and the two things that hang
    # off it. ONE derivation, two consumers, on purpose.
    # ----------------------------------------------------------------- #
    def first_kv_shared_layer(self) -> int:
        """The index at which sharing starts: ``35 - 20 = 15``.

        ``num_hidden_layers`` when nothing is shared, which makes every
        predicate below false without a second branch.
        """
        return self.num_hidden_layers - max(self.num_kv_shared_layers, 0)

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        return (self.num_kv_shared_layers > 0
                and layer_idx >= self.first_kv_shared_layer())

    def kv_share_map(self) -> dict[int, int]:
        """``{shared layer -> the layer whose slot it reads}``, in LAYER indices.

        Upstream's rule, verbatim (``tpu_inference/models/common/kv_share.py:28``
        and the identical derivation in vLLM ``gemma4.py:471-476``): the source is
        THE LAST PRECEDING LAYER OF THE SAME ATTENTION TYPE, searched among the
        allocating prefix only. For E2B that is layer 13 for every shared sliding
        layer and layer 14 for every shared full one.

        THESE ARE LAYER INDICES AND NOT CACHE SLOTS, and conflating the two is
        the mistake this family invites: in the container, source layer 13 is the
        TWELFTH allocating sliding layer, i.e. **slot 11 of group 0**, and source
        layer 14 is the THIRD allocating full one, i.e. **slot 2 of group 1**.
        :meth:`layer_slot` does that translation and is the only thing
        ``modeling.py`` reads.

        ORDERING IS AN INVARIANT, NOT A COINCIDENCE: the source is always
        ``< first_kv_shared_layer()``, so it has already run -- and already
        written its slot -- by the time any sharer reads it in the same forward.
        """
        first = self.first_kv_shared_layer()
        if self.num_kv_shared_layers <= 0:
            return {}
        prefix = [self.layer_type(i) for i in range(first)]
        out: dict[int, int] = {}
        for i in range(first, self.num_hidden_layers):
            kind = self.layer_type(i)
            if kind not in prefix:
                raise ValueError(
                    f"KV-share: layer {i} is {kind!r} and no layer before "
                    f"{first} is, so there is nothing for it to share with. "
                    f"num_kv_shared_layers={self.num_kv_shared_layers}, "
                    f"num_hidden_layers={self.num_hidden_layers}.")
            out[i] = len(prefix) - 1 - prefix[::-1].index(kind)
        return out

    def intermediate_size_for_layer(self, layer_idx: int) -> int:
        """6144, DOUBLED to 12288 on the 20 KV-shared layers.

        ``config.intermediate_size`` is therefore wrong for 20 of 35 layers, and
        it is wrong by a factor of two -- which the loader's shape check catches
        loudly, so this is the one of gemma-4's four heterogeneities that cannot
        become a silent wrong answer. It is derived from
        :meth:`is_kv_shared_layer` and NOT from a second copy of the ``>= 15``
        test, because the two axes are coupled through one config key and must
        move together if a sibling checkpoint moves it.
        """
        double = self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx)
        return self.intermediate_size * (2 if double else 1)

    # ----------------------------------------------------------------- #
    # The cache container: geometry (engine.py) and index map (modeling.py).
    # ----------------------------------------------------------------- #
    def kv_groups(self) -> tuple[tuple[int, int, int], ...]:
        """``((n_slots, num_kv_heads, head_dim), ...)`` for ``make_kv_cache``.

        ``((12, 1, 256), (3, 1, 512))`` on E2B: one array per DISTINCT geometry,
        sized by how many ALLOCATING layers carry it. A shared layer contributes
        no slot, which is why the slot counts sum to 15 and not to 35.

        GROUP ORDER IS FIRST APPEARANCE among the allocating layers, which puts
        sliding first (layer 0) and full second (layer 4). That order is frozen
        into ``cache_spec.json``'s ``groups`` list and into ``kv_cache_state``,
        so it is a contract and not an implementation detail -- sorting the
        geometries instead would be equally deterministic and would silently
        renumber every frozen artifact.
        """
        counts: dict[tuple[int, int], int] = {}
        for i in range(self.num_hidden_layers):
            if self.is_kv_shared_layer(i):
                continue
            geom = (self.num_kv_heads_for_layer(i), self.head_dim_for_layer(i))
            counts[geom] = counts.get(geom, 0) + 1
        if not counts:
            raise ValueError(
                "every layer shares a KV slot, so no layer allocates one: "
                "make_kv_cache would be asked for an empty container")
        return tuple((n, heads, dim) for (heads, dim), n in counts.items())

    def layer_slot(self) -> tuple[tuple[int, int], ...]:
        """``layer_idx -> (group, slot)``, for all 35 layers.

        The generalisation of ``cache/kv_cache.identity_layer_slot``, which every
        other family in this class uses and which is ``(0, layer_idx)``. Here it
        is neither: 15 allocating layers take the 15 slots of two groups in
        order, and the 20 shared ones repeat their SOURCE layer's entry -- slot
        11 of group 0 for the sliding ones and slot 2 of group 1 for the full
        ones. See :meth:`kv_share_map` for why those are 11 and 2 rather than 13
        and 14.
        """
        order = {(heads, dim): g
                 for g, (_, heads, dim) in enumerate(self.kv_groups())}
        used: dict[int, int] = {}
        out: list[tuple[int, int]] = []
        for i in range(self.num_hidden_layers):
            if self.is_kv_shared_layer(i):
                out.append((-1, -1))                 # patched below
                continue
            group = order[(self.num_kv_heads_for_layer(i),
                           self.head_dim_for_layer(i))]
            slot = used.get(group, 0)
            used[group] = slot + 1
            out.append((group, slot))
        for shared, source in self.kv_share_map().items():
            out[shared] = out[source]
        return tuple(out)

    def layer_writes(self) -> tuple[bool, ...]:
        """``True`` where the layer OWNS the slot it reads. 15 True, then 20 False.

        The generalisation of ``cache/kv_cache.identity_layer_writes``. A shared
        layer must not write: its K and V are never normed and never rotated (see
        ``modeling.py``), so scattering them into the source layer's slot would
        corrupt the very tensor it is about to read.
        """
        return tuple(not self.is_kv_shared_layer(i)
                     for i in range(self.num_hidden_layers))

    # ----------------------------------------------------------------- #
    # Rope, per layer type.
    # ----------------------------------------------------------------- #
    def rope_for_layer(self, layer_idx: int) -> dict:
        """This layer's ``rope_parameters`` entry, as a plain dict.

        ``config.json`` carries a DICT OF DICTS keyed by layer type -- the
        transformers-v5 shape -- and the two entries genuinely differ::

            sliding_attention  rope_theta    10000.0  rope_type default
                               (no partial_rotary_factor -> 1.0)
            full_attention     rope_theta  1000000.0  rope_type proportional
                               partial_rotary_factor 0.25

        A flat ``rope_theta`` fallback is kept for the v4 config shape, where the
        sliding layers read ``rope_local_base_freq``.
        """
        entry = self.rope_parameters.get(self.layer_type(layer_idx))
        if entry is not None:
            return dict(entry)
        return dict(self.rope_parameters)

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "ModelConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        # THE NESTED CONFIG. Every published gemma-4 checkpoint declares
        # `architectures: ["Gemma4ForConditionalGeneration"]` and puts the
        # decoder under `text_config`. This instance is the TEXT PATH ONLY and
        # says so in tests/instance_context.md; here it just means the shape is
        # one level down.
        text = raw.get("text_config", raw)
        num_layers = text["num_hidden_layers"]
        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=num_layers,
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            # NOT hidden_size // num_attention_heads: that is 1536 / 8 = 192 and
            # matches no layer in this model. See the class docstring.
            head_dim=text.get("head_dim")
            or text["hidden_size"] // text["num_attention_heads"],
            vocab_size=text["vocab_size"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_parameters=dict(text.get("rope_parameters") or {}),
            tie_word_embeddings=text.get(
                "tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            attention_bias=text.get("attention_bias", False),
            sliding_window=text.get("sliding_window"),
            layer_types=tuple(text.get("layer_types")
                              or [_FULL] * num_layers),
            global_head_dim=text.get("global_head_dim")
            or text.get("head_dim")
            or text["hidden_size"] // text["num_attention_heads"],
            num_global_key_value_heads=text.get("num_global_key_value_heads"),
            num_kv_shared_layers=int(text.get("num_kv_shared_layers") or 0),
            use_double_wide_mlp=bool(text.get("use_double_wide_mlp", False)),
            final_logit_softcapping=text.get("final_logit_softcapping"),
            hidden_size_per_layer_input=int(
                text.get("hidden_size_per_layer_input") or 0),
            vocab_size_per_layer_input=int(
                text.get("vocab_size_per_layer_input") or text["vocab_size"]),
            hidden_activation=text.get("hidden_activation", "gelu_pytorch_tanh"),
        )


_DTYPES = frozenset({"float32", "bfloat16", "float16"})


@dataclass(frozen=True)
class RunConfig:
    """Runtime knobs. Mirrors tests/instance.yaml; both are copied into
    golden/meta.json so nothing per-instance is hardcoded in the grader."""

    page_size: int
    num_pages: int
    max_num_seqs: int
    pages_per_seq: int
    max_model_len: int
    dtype: str
    seed: int
    num_decode_steps: int

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}")
        if self.pages_per_seq * self.page_size < self.max_model_len:
            raise ValueError(
                f"pages_per_seq * page_size ({self.pages_per_seq * self.page_size}) "
                f"< max_model_len ({self.max_model_len}): a full-length sequence "
                "would not fit in its block table"
            )
        # The block table promises each of max_num_seqs requests pages_per_seq
        # pages; a pool smaller than that promise can never honour it, so the
        # config is unrunnable at full occupancy however the scheduler behaves.
        # Caught here rather than as a mid-decode "out of KV pages".
        if self.num_pages < self.max_num_seqs * self.pages_per_seq:
            raise ValueError(
                f"num_pages ({self.num_pages}) < max_num_seqs * pages_per_seq "
                f"({self.max_num_seqs} * {self.pages_per_seq} = "
                f"{self.max_num_seqs * self.pages_per_seq}): the pool cannot fill "
                "the block table it advertises"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        # DELIBERATELY LAZY -- do not hoist this to module scope.
        #
        # from_yaml is an authoring/engine convenience: the torch reference and
        # the offline freezing step call it, the verifier never does. But this file is
        # byte-copied to solution/ref_jax_repo/config.py (so test_config_spec
        # compares rather than restates it), and ref_jax_repo IS imported inside
        # the verifier container -- whose environment,
        # base_image/requirements/inference-grade.txt, is torch-free and has no
        # pyyaml. At module scope this import would therefore fail the entire
        # grade at import time, with a traceback pointing at the candidate's
        # config instead of at a missing wheel.
        #
        # The right fix is for the dependency not to exist on that path, not to
        # add pyyaml to the grade image.
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        return cls(**{f: raw[f] for f in cls.__dataclass_fields__})
