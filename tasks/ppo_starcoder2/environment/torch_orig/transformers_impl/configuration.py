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

"""Starcoder2 model configuration."""

import json
import logging
from pathlib import Path


class _WarnOnceLogger:
    """Stdlib logger exposing ``warning_once`` (transformers loggers provide it; the config code calls it for advisory, numerically-irrelevant notices)."""

    def __init__(self, name):
        self._logger = logging.getLogger(name)

    def warning_once(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._logger, item)


logger = _WarnOnceLogger(__name__)

_UNSET = object()

RopeParameters = dict


class PreTrainedConfig:
    """Minimal implementation of ``transformers.PreTrainedConfig``."""

    model_type: str = ""
    attribute_map: dict = {}
    keys_to_ignore_at_inference: list = []

    default_theta = 10_000.0
    ignore_keys_at_rope_validation = set()

    def __init__(self, **kwargs):
        self._attn_implementation_internal = (
            kwargs.pop("_attn_implementation", None) or kwargs.pop("attn_implementation", None)
        )

        defaults = dict(
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
            torchscript=False,
            use_bfloat16=False,
            tie_word_embeddings=True,
            is_encoder_decoder=False,
            is_decoder=False,
            num_labels=2,
            problem_type=None,
            architectures=None,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            chunk_size_feed_forward=0,
            pruned_heads={},
        )
        for key, default in defaults.items():
            provided = kwargs.pop(key, _UNSET)
            if provided is not _UNSET:
                setattr(self, key, provided)
            elif not hasattr(self, key):
                setattr(self, key, default)

        if hasattr(self, "rope_parameters"):
            kwargs = self.convert_rope_params_to_dict(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value

    @property
    def use_return_dict(self) -> bool:
        return self.return_dict and not self.torchscript

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
        self.rope_parameters.setdefault("rope_theta", rope_theta)

        partial_rotary_factor = kwargs.get("partial_rotary_factor", getattr(self, "partial_rotary_factor", None))
        if partial_rotary_factor is not None:
            self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
            self.ignore_keys_at_rope_validation = self.ignore_keys_at_rope_validation | {"partial_rotary_factor"}

        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self):
        """Helper to standardize the config's rope params field by ensuring the params are defined for each later type."""
        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        layer_types = getattr(self, "layer_types", None)

        if not (rope_parameters or rope_theta):
            logger.warning("`standardize_rope_params` was called but no RoPE parameters were found.")
            return
        elif layer_types is None or rope_parameters == {} or not set(rope_parameters.keys()).issubset(layer_types):
            rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
            rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters["partial_rotary_factor"] = partial_rotary_factor

            if rope_parameters["rope_type"] in ["llama3", "yarn", "longrope"]:
                if hasattr(self, "original_max_position_embeddings"):
                    self.rope_parameters["original_max_position_embeddings"] = self.original_max_position_embeddings
                else:
                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)

        else:
            for layer_type in set(layer_types):
                rope_parameters[layer_type].setdefault("rope_type", rope_parameters[layer_type].get("type", "default"))
                rope_parameters[layer_type].setdefault("rope_theta", rope_theta)
                if partial_rotary_factor is not None:
                    rope_parameters[layer_type]["partial_rotary_factor"] = partial_rotary_factor

                if rope_parameters[layer_type]["rope_type"] in ["llama3", "yarn", "longrope"]:
                    self.rope_parameters[layer_type].setdefault(
                        "original_max_position_embeddings", self.max_position_embeddings
                    )

        self.rope_parameters = rope_parameters

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        for junk in ("local_files_only", "cache_dir", "revision", "token", "trust_remote_code"):
            kwargs.pop(junk, None)
        config_file = Path(pretrained_model_name_or_path) / "config.json"
        with open(config_file, "r") as handle:
            config_dict = json.load(handle)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        config_dict.update(kwargs)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        output = dict(self.__dict__)
        output.pop("_attn_implementation_internal", None)
        output["model_type"] = self.__class__.model_type
        output["_attn_implementation"] = self._attn_implementation
        return output

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n"


class Starcoder2Config(PreTrainedConfig):
    """This is the configuration class to store the configuration of a [`Starcoder2Model`]."""

    model_type = "starcoder2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.c_fc": "colwise",
        "layers.*.mlp.c_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 49152,
        hidden_size: int | None = 3072,
        intermediate_size: int | None = 12288,
        num_hidden_layers: int | None = 30,
        num_attention_heads: int | None = 24,
        num_key_value_heads: int | None = 2,
        hidden_act: str | None = "gelu_pytorch_tanh",
        max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.018042,
        norm_epsilon: int | None = 1e-5,
        use_cache: bool | None = True,
        bos_token_id: int | None = 50256,
        eos_token_id: int | None = 50256,
        pad_token_id: int | None = None,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        sliding_window: int | None = None,
        attention_dropout: float | None = 0.0,
        residual_dropout: float | None = 0.0,
        embedding_dropout: float | None = 0.0,
        use_bias: bool | None = True,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.use_bias = use_bias
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_epsilon = norm_epsilon
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.rope_parameters = rope_parameters

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Starcoder2Config"]
