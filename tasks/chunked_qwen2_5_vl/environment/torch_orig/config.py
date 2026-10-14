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

Byte-identical to ``solution/ref_jax_repo/config.py`` -- it imports neither
torch nor jax, so the two sides read one file rather than two that have to be
kept in step, and ``test_config_spec`` compares rather than restates it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    """The ViT tower's shape, read from ``config.json``'s ``vision_config``.

    A SEPARATE dataclass rather than a prefixed block of fields on
    ``ModelConfig``, because every one of these names collides with a text-stack
    name that means something else: ``hidden_size`` is 1280 here and 2048 there,
    ``num_heads`` 16 and 16 but over different widths, ``intermediate_size``
    3420 and 11008. A single flat config would have to invent
    ``vision_hidden_size``, which is a name no upstream uses and no checkpoint
    key contains.

    ``in_channels`` is vLLM's spelling; the checkpoint's ``config.json`` calls
    the same key ``in_chans``. Both are read, vLLM's wins, and the fallback
    exists because Qwen2.5-VL's own file uses the short form -- see
    :meth:`from_raw`.
    """

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    window_size: int
    #: Block indices that run FULL attention; every other block runs window
    #: attention. A tuple, not a list, so ModelConfig stays hashable/frozen.
    fullatt_block_indexes: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def spatial_merge_unit(self) -> int:
        """How many raw patches collapse into one LLM token: ``merge ** 2``.

        Named because it appears three times in the tower for three different
        reasons -- the pre-block regroup, the window-index stride, and the
        merger's input width -- and a port that gets one of them right and
        another wrong produces a correctly-shaped, wrongly-ordered tower.
        """
        return self.spatial_merge_size * self.spatial_merge_size

    @classmethod
    def from_raw(cls, raw: dict) -> "VisionConfig":
        return cls(
            depth=raw["depth"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_heads=raw["num_heads"],
            # `in_channels` is vLLM's field name (qwen2_5_vl.py:638) and
            # `in_chans` is what Qwen/Qwen2.5-VL-3B-Instruct's config.json
            # actually ships -- measured, in-image. Reading only one of them
            # would either KeyError on the real checkpoint or silently diverge
            # from upstream's accessor, so both are read and the default is the
            # only value any Qwen2.5-VL checkpoint has ever carried.
            in_channels=raw.get("in_channels", raw.get("in_chans", 3)),
            out_hidden_size=raw["out_hidden_size"],
            patch_size=raw["patch_size"],
            spatial_merge_size=raw["spatial_merge_size"],
            temporal_patch_size=raw["temporal_patch_size"],
            window_size=raw["window_size"],
            fullatt_block_indexes=tuple(raw["fullatt_block_indexes"]),
        )


@dataclass(frozen=True)
class ModelConfig:
    """qwen2_5_vl's shape, read from config.json and never inferred.

    The text stack is Qwen2.5's: vLLM builds it by instantiating
    ``Qwen2ForCausalLM`` outright (qwen2_5_vl.py:1361-1366), so every field
    above ``vision_config`` is core_qwen2's field list unchanged, with
    core_qwen3's EXPLICIT ``head_dim`` (a checkpoint that omits the key -- and
    Qwen2.5-VL-3B does omit it -- falls back to the derived value, so this stays
    correct for both).

    ``rope_scaling`` is left as the dict the JSON carries, exactly as
    core_llama3 does. For this family it is
    ``{"type": "mrope", "mrope_section": [16, 24, 24]}``; :attr:`mrope_section`
    is the accessor the model uses, so no caller indexes the raw dict.

    THE FOUR TOKEN IDS ARE ARCHITECTURE, NOT TOKENIZER TRIVIA. ``image_token_id``
    is the placeholder the vision embeddings are scattered onto, and
    ``vision_start_token_id`` is what delimits an image span when M-RoPE
    positions are built. A port that hardcodes 151655 works on this checkpoint
    and silently breaks on any other, which is the same class of defect as
    deriving ``head_dim``.
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
    rope_theta: float
    rope_scaling: dict | None
    tie_word_embeddings: bool
    attention_bias: bool
    vision_config: VisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int

    @property
    def mrope_section(self) -> tuple[int, ...] | None:
        """``rope_scaling["mrope_section"]`` as a tuple, or None if absent.

        ``layers/rotary.py`` takes exactly this: a partition of
        ``rotary_dim // 2`` between the rope's sections, or None for the
        single-section text-only default. Qwen2.5-VL ships ``[16, 24, 24]``,
        which sums to 64 == head_dim (128) // 2.

        Returned as a TUPLE because ``ModelConfig`` is frozen and a list would
        make it unhashable, and because ``RotaryEmbedding`` stores it as a plain
        attribute that must never reach ``state_dict()``.
        """
        if not self.rope_scaling:
            return None
        section = self.rope_scaling.get("mrope_section")
        if not section:
            return None
        return tuple(int(width) for width in section)

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "ModelConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            # EXPLICIT, never hidden_size // num_attention_heads. Qwen3-0.6B is
            # 1024 / 16 heads with head_dim 128, so the derived value is 64 and a
            # port that computes it builds a model whose q_proj is half the width
            # the checkpoint ships. The `or` fallback keeps families that omit
            # the key (Qwen2.5, and Qwen2.5-VL with it) working unchanged.
            head_dim=raw.get("head_dim")
            or raw["hidden_size"] // raw["num_attention_heads"],
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            rope_scaling=raw.get("rope_scaling"),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", False),
            vision_config=VisionConfig.from_raw(raw["vision_config"]),
            image_token_id=raw["image_token_id"],
            video_token_id=raw["video_token_id"],
            vision_start_token_id=raw["vision_start_token_id"],
            vision_end_token_id=raw["vision_end_token_id"],
        )


_DTYPES = frozenset({"float32", "bfloat16", "float16"})


@dataclass(frozen=True)
class RunConfig:
    """Runtime knobs. Mirrors tests/instance.yaml; the freezer copies both into
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
        # from_yaml is an authoring/engine convenience: the freezer and the
        # torch reference call it, the verifier never does. But this file is
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
