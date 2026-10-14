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
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    tie_word_embeddings: bool
    attention_bias: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "ModelConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            rope_scaling=raw.get("rope_scaling"),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", False),
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
