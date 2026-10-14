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

"""Qwen2-VL model configuration."""

import inspect
import json
import os
from typing import Any, Optional, Union  # noqa: F401 (kept for interface/type parity)

import torch


class PreTrainedConfig:
    """Minimal PreTrainedConfig base class for loading model configurations."""

    model_type: str = ""

    def __init__(self, **kwargs):
        self.torch_dtype = kwargs.pop("torch_dtype", None)
        self._attn_implementation = kwargs.pop("_attn_implementation", None)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.return_dict = kwargs.pop("return_dict", True)
        self.use_cache = kwargs.pop("use_cache", True)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.architectures = kwargs.pop("architectures", None)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        for key, value in kwargs.items():
            if not key.startswith("_"):
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "PreTrainedConfig":
        config_dict = cls._load_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def _load_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> dict:
        config_path = pretrained_model_name_or_path
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, "config.json")
        elif os.path.isfile(config_path):
            config_file = config_path
        else:
            config_file = os.path.join("/data", config_path, "config.json")
        with open(config_file, "r") as f:
            return json.load(f)

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "PreTrainedConfig":
        return cls(**config_dict)

    def to_dict(self) -> dict:
        output = {}
        for key, value in self.__dict__.items():
            if isinstance(value, PreTrainedConfig):
                value = value.to_dict()
            elif isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            output[key] = value
        return output

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_text_config(self, decoder=None, **kwargs):
        return self


class Qwen2VLVisionConfig(PreTrainedConfig):
    model_type = "qwen2_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(initializer_range=initializer_range, **kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size


class Qwen2VLTextConfig(PreTrainedConfig):
    model_type = "qwen2_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        layer_types=None,
        attention_dropout=0.0,
        rope_parameters=None,
        rope_scaling=None,
        rope_theta=1000000.0,
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=None,
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
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        if rope_parameters is not None:
            rp = dict(rope_parameters)
        elif isinstance(rope_scaling, dict):
            rp = dict(rope_scaling)
        else:
            rp = {}
        rp.setdefault("rope_theta", rope_theta if rope_theta is not None else self.default_theta)
        if rp.get("rope_type", rp.get("type")) == "mrope":
            rp["rope_type"] = "default"
        rp.setdefault("rope_type", "default")
        self.rope_parameters = rp

        if layer_types is None:
            layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= max_window_layers
                else "full_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        super().__init__(use_cache=use_cache, initializer_range=initializer_range, **kwargs)


class Qwen2VLConfig(PreTrainedConfig):
    model_type = "qwen2_vl"
    sub_configs = {"vision_config": Qwen2VLVisionConfig, "text_config": Qwen2VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()
        else:
            self.vision_config = vision_config

        text_params = list(inspect.signature(Qwen2VLTextConfig.__init__).parameters.keys())
        text_params += ["rope_parameters", "rope_scaling", "rope_theta"]
        text_kwargs = {key: kwargs.pop(key) for key in text_params if key in kwargs}

        if isinstance(text_config, dict):
            self.text_config = Qwen2VLTextConfig(**text_config)
        elif text_config is None:
            text_kwargs["dtype"] = kwargs.get("torch_dtype", kwargs.get("dtype"))
            self.text_config = Qwen2VLTextConfig(**text_kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "_attn_implementation":
            for name in ("text_config", "vision_config"):
                sub = self.__dict__.get(name)
                if sub is not None:
                    object.__setattr__(sub, "_attn_implementation", value)

    def get_text_config(self, decoder=None, **kwargs):
        return self.text_config


_CONFIG_REGISTRY = {
    "qwen2_vl": Qwen2VLConfig,
    "qwen2_vl_text": Qwen2VLTextConfig,
}


class AutoConfig:
    """Auto config that detects model type from config.json."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> PreTrainedConfig:
        config_dict = PreTrainedConfig._load_config_dict(pretrained_model_name_or_path, **kwargs)
        model_type = config_dict.get("model_type", "")
        config_cls = _CONFIG_REGISTRY.get(model_type, PreTrainedConfig)
        return config_cls.from_dict(config_dict)


__all__ = ["Qwen2VLConfig", "Qwen2VLTextConfig"]
