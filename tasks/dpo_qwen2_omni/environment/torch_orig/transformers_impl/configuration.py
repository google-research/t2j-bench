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

def auto_docstring(obj=None, *args, **kwargs):
    if callable(obj) and not args and not kwargs:
        return obj
    def _decorator(inner):
        return inner
    return _decorator

def strict(cls):
    return cls

import copy
import json
import os
from dataclasses import MISSING, dataclass, fields
from functools import wraps

import torch




class _NoopLogger:
    """Implementation of ``transformers.utils.logging``."""

    def get_logger(self, *args, **kwargs):
        return self

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warning_once(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


logging = _NoopLogger()

RopeParameters = dict

CONFIG_NAME = "config.json"

_GENERATION_PARAM_NAMES = (
    "max_length", "min_length", "do_sample", "use_cache", "early_stopping", "num_beams",
    "temperature", "top_k", "top_p", "typical_p", "repetition_penalty", "length_penalty",
    "no_repeat_ngram_size", "encoder_no_repeat_ngram_size", "bad_words_ids",
    "num_return_sequences", "output_scores", "return_dict_in_generate",
    "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values",
    "exponential_decay_length_penalty", "suppress_tokens", "begin_suppress_tokens",
    "epsilon_cutoff", "eta_cutoff", "encoder_repetition_penalty", "num_assistant_tokens",
    "num_assistant_tokens_schedule", "assistant_confidence_threshold",
    "assistant_lookbehind", "target_lookbehind", "num_beam_groups", "diversity_penalty",
)


def _wrap_init_to_accept_kwargs(cls):
    """Replica of transformers' ``wrap_init_to_accept_kwargs`` (5.4.0)."""
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        dataclass_fields = {f.name for f in fields(cls)}
        standard_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_fields}
        if len(args) > 0:
            raise ValueError(
                f"{cls.__name__} accepts only keyword arguments, but found `{len(args)}` positional args."
            )
        for f in fields(cls):
            if f.name in standard_kwargs:
                setattr(self, f.name, standard_kwargs[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                raise TypeError(f"Missing required field - '{f.name}'")
        additional_kwargs = {}
        for name, value in kwargs.items():
            if name not in dataclass_fields:
                additional_kwargs[name] = value
        self.__post_init__(**additional_kwargs)

    cls.__init__ = __init__
    return cls


class RotaryEmbeddingConfigMixin:
    """Rope standardization, verbatim from transformers 5.4.0 ``modeling_rope_utils``."""

    default_theta = 10_000.0
    ignore_keys_at_rope_validation = set()

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
        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        layer_types = getattr(self, "layer_types", None)

        if not (rope_parameters or rope_theta):
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


@dataclass(repr=False)
class PreTrainedConfig(RotaryEmbeddingConfigMixin):
    """Torch-only replica of transformers 5.4.0 declarative ``PreTrainedConfig``."""

    model_type = ""
    sub_configs = {}
    attribute_map = {}
    base_config_key = ""
    keys_to_ignore_at_inference = []

    transformers_version: str | None = None
    architectures: list | None = None
    output_hidden_states: bool | None = False
    return_dict: bool | None = True
    dtype: object = None
    chunk_size_feed_forward: int = 0
    is_encoder_decoder: bool = False
    id2label: dict | None = None
    label2id: dict | None = None
    problem_type: str | None = None

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls_has_custom_init = "__init__" in cls.__dict__
        dataclass(cls, repr=False)
        if not cls_has_custom_init:
            _wrap_init_to_accept_kwargs(cls)

    def __setattr__(self, key, value):
        attribute_map = super().__getattribute__("attribute_map")
        if key in attribute_map:
            key = attribute_map[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map":
            attribute_map = super().__getattribute__("attribute_map")
            if key in attribute_map:
                key = attribute_map[key]
        return super().__getattribute__(key)

    @property
    def name_or_path(self):
        return getattr(self, "_name_or_path", "")

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)

    @property
    def num_labels(self):
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels):
        if getattr(self, "id2label", None) is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def output_attentions(self):
        return getattr(self, "_output_attentions", False)

    @output_attentions.setter
    def output_attentions(self, value):
        if value and getattr(self, "_attn_implementation", None) not in (None, "eager"):
            raise ValueError(
                "The `output_attentions` attribute is not supported when using the "
                f"`attn_implementation` set to {self._attn_implementation}."
            )
        if value and getattr(self, "_attn_implementation", None) is None:
            self._attn_implementation = "eager"
        self._output_attentions = value

    @property
    def _attn_implementation(self):
        return getattr(self, "_attn_implementation_internal", None)

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        """Set recursively on the sub-configs as well."""
        current_attn = getattr(self, "_attn_implementation", None)
        attn_implementation = value if not isinstance(value, dict) else value.get("", current_attn)
        self._attn_implementation_internal = attn_implementation
        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_subconfig_attn = getattr(subconfig, "_attn_implementation", None)
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, current_subconfig_attn)
                )
                subconfig._attn_implementation = sub_implementation

    @property
    def _experts_implementation(self):
        return getattr(self, "_experts_implementation_internal", None)

    @_experts_implementation.setter
    def _experts_implementation(self, value):
        current = getattr(self, "_experts_implementation", None)
        experts_implementation = value if not isinstance(value, dict) else value.get("", current)
        self._experts_implementation_internal = experts_implementation
        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_sub = getattr(subconfig, "_experts_implementation", None)
                sub_impl = value if not isinstance(value, dict) else value.get(subconfig_key, current_sub)
                subconfig._experts_implementation = sub_impl

    @property
    def torch_dtype(self):
        return self.dtype

    @torch_dtype.setter
    def torch_dtype(self, value):
        self.dtype = value

    @property
    def rope_scaling(self):
        return getattr(self, "rope_parameters", None)

    @rope_scaling.setter
    def rope_scaling(self, value):
        self.rope_parameters = value

    def __post_init__(self, **kwargs):
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            self.dtype = self.dtype if self.dtype is not None else torch_dtype
        if self.dtype is not None and isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

        if self.id2label is None:
            self.num_labels = kwargs.get("num_labels", 2)
        else:
            self.id2label = {int(key): value for key, value in self.id2label.items()}

        if hasattr(self, "rope_parameters"):
            kwargs = self.convert_rope_params_to_dict(**kwargs)

        for parameter_name in _GENERATION_PARAM_NAMES:
            kwargs.pop(parameter_name, None)

        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        self._commit_hash = kwargs.pop("_commit_hash", None)

        self._output_attentions = kwargs.pop("output_attentions", False)
        self._attn_implementation = kwargs.pop("attn_implementation", None)
        self._experts_implementation = kwargs.pop("experts_implementation", None)

        for key, value in kwargs.items():
            if key not in ("_attn_implementation_internal", "_experts_implementation_internal"):
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict = cls._load_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def _load_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_path = str(pretrained_model_name_or_path)
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, CONFIG_NAME)
        elif os.path.isfile(config_path):
            config_file = config_path
        else:
            config_file = os.path.join("/data", config_path, CONFIG_NAME)
        with open(config_file, "r") as handle:
            return json.load(handle)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        config_dict.pop("return_unused_kwargs", None)
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        output.pop("kwargs", None)
        if "_output_attentions" in output:
            output["output_attentions"] = output.pop("_output_attentions")
        output.pop("_attn_implementation_internal", None)
        output.pop("_experts_implementation_internal", None)
        for key, value in list(output.items()):
            if isinstance(value, PreTrainedConfig):
                value = value.to_dict()
                value.pop("transformers_version", None)
            elif isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            output[key] = value
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as handle:
            handle.write(self.to_json_string())

    def get_text_config(self, decoder=None, encoder=None):
        decoder_names = ("decoder", "generator", "text_config")
        encoder_names = ("text_encoder",)
        if decoder:
            possible_names = decoder_names
        elif encoder:
            possible_names = encoder_names
        else:
            possible_names = encoder_names + decoder_names
        valid_names = [name for name in possible_names if getattr(self, name, None) is not None]
        if len(valid_names) > 1:
            raise ValueError(
                f"Multiple valid text configs found in {self.__class__.__name__}: {valid_names}."
            )
        elif len(valid_names) == 1:
            return getattr(self, valid_names[0])
        return self


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniVisionEncoderConfig(PreTrainedConfig):
    """window_size (`int`, *optional*, defaults to 11): Size of windows."""

    model_type = "qwen2_5_omni_vision_encoder"
    base_config_key = "vision_config"

    depth: int = 32
    hidden_size: int = 3584
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: list[int] | tuple[int, ...] = (7, 15, 23, 31)
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniAudioEncoderConfig(PreTrainedConfig):
    """max_source_positions (`int`, *optional*, defaults to 1500): The maximum sequence length of log-mel filter-bank features that this model might ever be used with."""

    model_type = "qwen2_5_omni_audio_encoder"
    attribute_map = {"num_hidden_layers": "encoder_layers"}

    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_function: str = "gelu"
    activation_dropout: float | int = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500

    n_window: int = 100
    output_dim: int = 3584


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniTextConfig(PreTrainedConfig):
    """Example:"""

    model_type = "qwen2_5_omni_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

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
    ignore_keys_at_rope_validation = {"mrope_section"}

    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 32768
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniThinkerConfig(PreTrainedConfig):
    """position_id_per_seconds (`int`, *optional*, defaults to 25): The increment of position id per second."""

    model_type = "qwen2_5_omni_thinker"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "audio_config": Qwen2_5OmniAudioEncoderConfig,
        "vision_config": Qwen2_5OmniVisionEncoderConfig,
        "text_config": Qwen2_5OmniTextConfig,
    }

    audio_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 151646
    image_token_index: int = 151655
    video_token_index: int = 151656
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    user_token_id: int = 872
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen2_5OmniVisionEncoderConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Qwen2_5OmniVisionEncoderConfig()

        if isinstance(self.audio_config, dict):
            self.audio_config = Qwen2_5OmniAudioEncoderConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = Qwen2_5OmniAudioEncoderConfig()

        if isinstance(self.text_config, dict):
            self.text_config = Qwen2_5OmniTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen2_5OmniTextConfig()

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniTalkerConfig(PreTrainedConfig):
    """tts_text_start_token_id (`int`, *optional*, defaults to 151860): The tts text start token index to encode the start of tts text."""

    model_type = "qwen2_5_omni_talker"
    default_theta = 1000000.0
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
        "audio_token_id": "audio_token_index",
    }
    ignore_keys_at_rope_validation = {"mrope_section"}

    audio_token_index: int = 151646
    image_token_index: int = 151655
    video_token_index: int = 151656
    vocab_size: int = 8448
    tts_text_start_token_id: int = 151860
    tts_text_end_token_id: int = 151861
    tts_text_pad_token_id: int = 151859
    tts_codec_start_token_id: int = 8293
    tts_codec_end_token_id: int = 8294
    tts_codec_pad_token_id: int = 8292
    tts_codec_mask_token_id: int = 8296
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    embedding_size: int = 3584
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-06
    head_dim: int = 128
    use_cache: bool = True
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 32768
    max_window_layers: int = 28
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    initializer_range: float = 0.02
    spatial_merge_size: int = 2
    layer_types: list[str] | None = None
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniDiTConfig(PreTrainedConfig):
    """ff_mult (`int`, *optional*, defaults to 2): The multiplier for the feedforward layer in each transformer block."""

    model_type = "qwen2_5_omni_dit"

    hidden_size: int = 1024
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    ff_mult: int = 2
    emb_dim: int = 512
    head_dim: int = 64
    rope_parameters: RopeParameters | dict | None = None
    max_position_embeddings: int = 32768
    block_size: int = 24
    look_ahead_layers: list[int] | tuple[int, ...] = (10,)
    look_backward_layers: list[int] | tuple[int, ...] = (0, 20)
    repeats: int = 2
    num_embeds: int = 8193
    mel_dim: int = 80
    dropout: float | int = 0.1
    enc_emb_dim: int = 192
    enc_dim: int = 128
    enc_channels: list[int] | tuple[int, ...] = (256, 256, 256, 256, 768)
    enc_kernel_sizes: list[int] | tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: list[int] | tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 64
    enc_res2net_scale: int = 2
    enc_se_channels: int = 64


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniBigVGANConfig(PreTrainedConfig):
    """mel_dim (`int`, *optional*, defaults to 80): The dimension of the mel-spectrogram."""

    model_type = "qwen2_5_omni_bigvgan"

    mel_dim: int = 80
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: list[int] | tuple[int, ...] = (5, 3, 2, 2, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (11, 7, 4, 4, 4, 4)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniToken2WavConfig(PreTrainedConfig):
    """dit_config ([`DiT_Args`], *optional*): Configuration class for the Diffusion Transformer (DiT) module responsible for generating mel-spectrograms."""

    model_type = "qwen2_5_omni_token2wav"
    sub_configs = {
        "dit_config": Qwen2_5OmniDiTConfig,
        "bigvgan_config": Qwen2_5OmniBigVGANConfig,
    }

    dit_config: dict | PreTrainedConfig | None = None
    bigvgan_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.dit_config is None:
            self.dit_config = Qwen2_5OmniDiTConfig()
        elif isinstance(self.dit_config, dict):
            self.dit_config = Qwen2_5OmniDiTConfig(**self.dit_config)

        if self.bigvgan_config is None:
            self.bigvgan_config = Qwen2_5OmniBigVGANConfig()
        elif isinstance(self.bigvgan_config, dict):
            self.bigvgan_config = Qwen2_5OmniBigVGANConfig(**self.bigvgan_config)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniConfig(PreTrainedConfig):
    """thinker_config (`dict`, *optional*): Configuration of the underlying thinker sub-model."""

    model_type = "qwen2_5_omni"
    sub_configs = {
        "thinker_config": Qwen2_5OmniThinkerConfig,
        "talker_config": Qwen2_5OmniTalkerConfig,
        "token2wav_config": Qwen2_5OmniToken2WavConfig,
    }

    thinker_config: dict | PreTrainedConfig | None = None
    talker_config: dict | PreTrainedConfig | None = None
    token2wav_config: dict | PreTrainedConfig | None = None
    enable_audio_output: bool = True

    def __post_init__(self, **kwargs):
        if self.thinker_config is None:
            self.thinker_config = Qwen2_5OmniThinkerConfig()
            logger.info("thinker_config is None. Initializing thinker model with default values")
        elif isinstance(self.thinker_config, dict):
            self.thinker_config = Qwen2_5OmniThinkerConfig(**self.thinker_config)

        if self.talker_config is None:
            self.talker_config = Qwen2_5OmniTalkerConfig()
            logger.info("talker_config is None. Initializing talker model with default values")
        elif isinstance(self.talker_config, dict):
            self.talker_config = Qwen2_5OmniTalkerConfig(**self.talker_config)

        if self.token2wav_config is None:
            self.token2wav_config = Qwen2_5OmniToken2WavConfig()
            logger.info("token2wav_config is None. Initializing token2wav model with default values")
        elif isinstance(self.token2wav_config, dict):
            self.token2wav_config = Qwen2_5OmniToken2WavConfig(**self.token2wav_config)

        super().__post_init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        """Returns the config that is meant to be used with text IO."""
        return self.thinker_config.get_text_config(*args, **kwargs)


__all__ = ["Qwen2_5OmniConfig", "Qwen2_5OmniThinkerConfig", "Qwen2_5OmniTalkerConfig", "Qwen2_5OmniToken2WavConfig"]
