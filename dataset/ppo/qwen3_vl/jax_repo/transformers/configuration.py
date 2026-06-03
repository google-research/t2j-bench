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

"""Qwen3-VL configuration classes - standalone implementation."""

import json
import os
from typing import Any, Optional, Union

import torch

from huggingface_hub import hf_hub_download


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
        # Store any extra kwargs
        for key, value in kwargs.items():
            if not key.startswith("_"):
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "PreTrainedConfig":
        config_dict = cls._load_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def _load_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> dict:
        cache_dir = kwargs.pop("cache_dir", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        config_path = pretrained_model_name_or_path
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, "config.json")
        elif os.path.isfile(config_path):
            config_file = config_path
        else:
            # Download from HF Hub
            config_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
                cache_dir=cache_dir,
                token=token,
                revision=revision,
            )
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "PreTrainedConfig":
        return cls(**config_dict)

    def to_dict(self) -> dict:
        output = {}
        for key, value in self.__dict__.items():
            if isinstance(value, PreTrainedConfig):
                value = value.to_dict()
            elif isinstance(value, torch.dtype):
                # Convert torch.dtype to its string name for JSON serialization
                value = str(value).replace("torch.", "")
            output[key] = value
        return output

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Qwen3VLVisionConfig(PreTrainedConfig):
    model_type = "qwen3_vl"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes or [8, 16, 24]




class Qwen3VLTextConfig(PreTrainedConfig):
    model_type = "qwen3_vl_text"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_parameters=None,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

        # Resolve rope_parameters: merge rope_theta + rope_scaling into rope_parameters dict
        if rope_parameters is not None:
            self.rope_parameters = rope_parameters
        else:
            rp = rope_scaling.copy() if isinstance(rope_scaling, dict) else {}
            rp.setdefault("rope_type", rp.pop("type", "default"))
            rp.setdefault("rope_theta", rope_theta)
            rp.setdefault("mrope_section", [24, 20, 20])
            self.rope_parameters = rp

        super().__init__(**kwargs)


class Qwen3VLConfig(PreTrainedConfig):
    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}

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
            self.vision_config = Qwen3VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen3VLVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = Qwen3VLTextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen3VLTextConfig()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


# Registry for AutoConfig
_CONFIG_REGISTRY = {
    "qwen3_vl": Qwen3VLConfig,
}


class AutoConfig:
    """Auto config that detects model type from config.json."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> PreTrainedConfig:
        config_dict = PreTrainedConfig._load_config_dict(pretrained_model_name_or_path, **kwargs)
        model_type = config_dict.get("model_type", "")
        config_cls = _CONFIG_REGISTRY.get(model_type, PreTrainedConfig)
        return config_cls.from_dict(config_dict)
