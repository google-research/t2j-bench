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

"""Instance configuration: model shape (ModelConfig) and runtime knobs (RunConfig)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """gpt-oss's shape, read from config.json and never inferred.

    FIVE FIELDS HERE ARE NOT IN core_llama3's ModelConfig, and each is a fact the
    dense families let you derive and this one does not:

    * ``head_dim`` is an EXPLICIT FIELD, not ``hidden_size //
      num_attention_heads``. gpt-oss is 2880 / 64 heads with head_dim 64, so the
      derived form gives 45 -- a shape error late enough to read as a porting bug.
      A checkpoint that omits the key falls back to the derived value, so this
      stays correct for the families that do not declare one.
    * ``num_local_experts`` / ``num_experts_per_tok`` -- 32 and 4, the routed MoE.
    * ``swiglu_limit`` -- 7.0, and NOT a default: `layers/activation.py`'s
      ``SiluAndMul`` has no limit at all.
    * ``sliding_window`` + ``layer_types`` -- 128, on every other layer. The tuple
      is stored rather than the rule, because it is what the checkpoint states;
      ``sliding_window_for_layer`` reads it, and vLLM's ``layer_idx % 2 == 0``
      shorthand is a consequence, not the source.

    ``layer_types`` is a TUPLE so the dataclass stays hashable and frozen in
    substance as well as in name; ``rope_scaling`` is left as the dict the JSON
    carries, exactly as core_llama3 does.
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
    num_local_experts: int
    num_experts_per_tok: int
    swiglu_limit: float
    sliding_window: int | None
    layer_types: tuple[str, ...]

    def sliding_window_for_layer(self, layer_idx: int) -> int | None:
        """The window this layer attends over, or None for full attention.

        Read off ``layer_types`` rather than off ``layer_idx % 2``. The two agree
        on this checkpoint -- 24 entries alternating, sliding first, which is how
        vLLM writes it -- but the config is the authority and a derivative that
        happens to agree is a derivative that can stop agreeing.
        """
        if layer_idx >= len(self.layer_types):
            raise IndexError(
                f"layer {layer_idx} has no entry in layer_types "
                f"({len(self.layer_types)} entries); config.json declares one per "
                "layer and a missing one would silently become full attention")
        kind = self.layer_types[layer_idx]
        if kind == "sliding_attention":
            if self.sliding_window is None:
                raise ValueError(
                    f"layer {layer_idx} is 'sliding_attention' but config.json "
                    "declares no sliding_window")
            return self.sliding_window
        if kind != "full_attention":
            raise ValueError(f"unknown layer_type {kind!r} at layer {layer_idx}")
        return None

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "ModelConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        num_layers = raw["num_hidden_layers"]
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=num_layers,
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim")
            or raw["hidden_size"] // raw["num_attention_heads"],
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            rope_scaling=raw.get("rope_scaling"),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", False),
            num_local_experts=raw["num_local_experts"],
            num_experts_per_tok=raw["num_experts_per_tok"],
            swiglu_limit=float(raw["swiglu_limit"]),
            sliding_window=raw.get("sliding_window"),
            layer_types=tuple(raw.get("layer_types")
                              or ["full_attention"] * num_layers),
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
