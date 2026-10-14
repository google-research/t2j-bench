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

"""InternVL model configuration."""

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

    def get_text_config(self, decoder=None, **kwargs):
        text_config = getattr(self, "text_config", None)
        return text_config if text_config is not None else self


ALLOWED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "linear_attention",
    "conv",
    "mamba",
    "attention",
    "sparse",
    "dense",
)


def layer_type_validation(layer_types: list[str], num_hidden_layers: int | None = None, attention: bool = True):
    """Based on ``transformers.configuration_utils``."""
    logger.warning(
        "`layer_type_validation` is deprecated and will be removed in v5.20. "
        "Use `PreTrainedConfig.validate_layer_type` instead"
    )

    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES}")
    if num_hidden_layers is not None and num_hidden_layers != len(layer_types):
        raise ValueError(
            f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
            f"({len(layer_types)})"
        )


class Qwen2Config(PreTrainedConfig):
    """This is the configuration class to store the configuration of a [`Qwen2Model`]."""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 22016,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


class InternVLVisionConfig(PreTrainedConfig):
    """Configuration for the InternVL vision encoder (``model_type = "internvl_vision"``)."""

    model_type = "internvl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        attention_bias: bool = False,
        use_qk_norm: bool = False,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        initializer_range: float = 0.02,
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-06,
        image_size=(448, 448),
        patch_size=(14, 14),
        num_channels: int = 3,
        use_mask_token: bool = False,
        use_absolute_position_embeddings: bool = True,
        layer_scale_init_value: float = 0.1,
        use_mean_pooling: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.initializer_range = initializer_range
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.layer_scale_init_value = layer_scale_init_value
        self.use_mean_pooling = use_mean_pooling

        super().__init__(**kwargs)


class AutoConfig:
    """Implementation of ``transformers.AutoConfig`` covering the supported model types."""

    @classmethod
    def for_model(cls, model_type, **kwargs):
        return CONFIG_MAPPING[model_type](**kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        model_type = config_dict.get("model_type")
        return CONFIG_MAPPING[model_type].from_dict(config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_file = Path(pretrained_model_name_or_path) / "config.json"
        with open(config_file, "r") as handle:
            config_dict = json.load(handle)
        return cls.from_dict(config_dict, **kwargs)


class InternVLConfig(PreTrainedConfig):
    """Configuration for the full InternVL vision-language model (``model_type = "internvl"``)."""

    model_type = "internvl"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternVLVisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id: int = 151667,
        image_seq_length: int = 256,
        downsample_ratio: float = 0.5,
        projector_hidden_act: str = "gelu",
        vision_feature_layer=-1,
        vision_feature_select_strategy: str = "default",
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = InternVLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = InternVLVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.downsample_ratio = downsample_ratio
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, **kwargs):
        return self.text_config


CONFIG_MAPPING = {
    "qwen2": Qwen2Config,
    "internvl": InternVLConfig,
    "internvl_vision": InternVLVisionConfig,
}


__all__ = [
    "PreTrainedConfig",
    "Qwen2Config",
    "InternVLVisionConfig",
    "InternVLConfig",
    "AutoConfig",
    "CONFIG_MAPPING",
]
