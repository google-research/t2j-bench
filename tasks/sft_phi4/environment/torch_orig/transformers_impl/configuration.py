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

"""Phi3 model configuration."""

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

    def validate_rope(self):
        """Validate the RoPE config arguments, given a `"PreTrainedConfig"` object"""
        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if not rope_parameters_dict:
            return

        if getattr(self, "layer_types", None) is not None and set(rope_parameters_dict.keys()).issubset(
            self.layer_types
        ):
            pass
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}

        for rope_parameters in rope_parameters_dict.values():
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            rope_parameters["rope_type"] = rope_type

            if validation_fn is not None:
                validation_fn(rope_parameters, ignore_keys=self.ignore_keys_at_rope_validation)
            else:
                logger.warning(
                    f"Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='{rope_type}'"
                )

    @staticmethod
    def _check_received_keys(
        rope_type: str,
        received_keys: set,
        required_keys: set,
        optional_keys: set | None = None,
        ignore_keys: set | None = None,
    ):
        """Compare the received keys in `config.rope_parameters` against the expected and optional keys"""
        if "type" in received_keys:
            received_keys -= {"type"}
            required_keys.add("rope_type")

        optional_keys = optional_keys or set()
        if "partial_rotary_factor" not in optional_keys:
            optional_keys.add("partial_rotary_factor")

        if ignore_keys is not None:
            received_keys -= ignore_keys

        missing_keys = required_keys - received_keys
        if missing_keys:
            raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")

        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")

    def _validate_default_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    def _validate_longrope_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "short_factor", "long_factor", "rope_theta", "original_max_position_embeddings"}
        optional_keys = {"attention_factor", "factor"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        short_factor = rope_parameters.get("short_factor")
        if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
            logger.warning(f"`rope_parameters`'s short_factor field must be a list of numbers, got {short_factor}")
        if len(short_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s short_factor field must have length {dim // 2}, got {len(short_factor)}"
            )

        long_factor = rope_parameters.get("long_factor")
        if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
            logger.warning(f"`rope_parameters`'s long_factor field must be a list of numbers, got {long_factor}")
        if len(long_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s long_factor field must have length {dim // 2}, got {len(long_factor)}"
            )

        factor = rope_parameters.get("factor")
        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]

        if factor is None and original_max_position_embeddings is not None:
            logger.warning_once(
                "This model config has set a `rope_parameters['original_max_position_embeddings']` field, to be used together with "
                "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`"
                "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
                "as it is compatible with most model architectures."
            )
        elif factor is None and original_max_position_embeddings is None:
            logger.warning("Missing required keys in `rope_parameters`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0.0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )

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


class Phi3Config(PreTrainedConfig):
    """This is the configuration class to store the configuration of a [`Phi3Model`]."""

    model_type = "phi3"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise_rep",
        "layers.*.self_attn.o_proj": "rowwise_rep",
        "layers.*.mlp.gate_up_proj": "colwise_rep",
        "layers.*.mlp.down_proj": "rowwise_rep",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 32064,
        hidden_size: int | None = 3072,
        intermediate_size: int | None = 8192,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        resid_pdrop: float | None = 0.0,
        embd_pdrop: float | None = 0.0,
        attention_dropout: float | None = 0.0,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        original_max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 32000,
        pad_token_id: int | None = 32000,
        sliding_window: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 1.0)
        self.sliding_window = sliding_window

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

    def convert_rope_params_to_dict(
        self, default_theta: int | float = 10_000.0, ignore_keys: set | None = None, **kwargs
    ):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", default_theta))
        self.rope_parameters.setdefault("partial_rotary_factor", kwargs["partial_rotary_factor"])
        self.standardize_rope_params()

        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        if rope_parameters_type is not None and rope_parameters_type in ["su", "yarn"]:
            self.rope_parameters["rope_type"] = "longrope"
        self.validate_rope(ignore_keys=ignore_keys)
        return kwargs

    def validate_rope(self, ignore_keys: set | None = None):
        """Validate the `rope_parameters` configuration."""
        if ignore_keys:
            self.ignore_keys_at_rope_validation = (
                set(self.ignore_keys_at_rope_validation) | set(ignore_keys)
            )
        super().validate_rope()

        if not isinstance(self.rope_parameters, dict):
            raise ValueError(f"`rope_parameters` must be a dictionary but got {self.rope_parameters}")
        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        rope_parameters_short_factor = self.rope_parameters.get("short_factor", None)
        rope_parameters_long_factor = self.rope_parameters.get("long_factor", None)
        rotary_ndims = int(
            self.hidden_size // self.num_attention_heads * self.rope_parameters["partial_rotary_factor"]
        )
        if rope_parameters_type not in ["default", "longrope"]:
            raise ValueError(f"`rope_parameters`'s type field must be one of ['longrope'], got {rope_parameters_type}")

        if rope_parameters_short_factor is not None:
            if not (
                isinstance(rope_parameters_short_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_short_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must be a list of numbers, got {rope_parameters_short_factor}"
                )
            if not len(rope_parameters_short_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_short_factor)}"
                )

        if rope_parameters_long_factor is not None:
            if not (
                isinstance(rope_parameters_long_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_long_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must be a list of numbers, got {rope_parameters_long_factor}"
                )
            if not len(rope_parameters_long_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_long_factor)}"
                )


__all__ = ["Phi3Config"]
