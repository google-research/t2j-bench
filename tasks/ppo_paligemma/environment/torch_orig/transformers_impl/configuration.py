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

"""Standalone configuration classes for PaliGemma - decoupled from HuggingFace transformers."""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class PreTrainedConfig:
    """Minimal pretrained config base class."""
    model_type = ""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        output = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"

    def get_text_config(self, decoder=None, **kwargs):
        return self


class SiglipVisionConfig(PreTrainedConfig):
    """Configuration for the SigLIP vision encoder used in PaliGemma."""
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_use_head=False,
        projection_dim=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.vision_use_head = vision_use_head
        self.projection_dim = projection_dim


class GemmaConfig(PreTrainedConfig):
    """Configuration for the Gemma (v1) text backbone used in PaliGemma."""
    model_type = "gemma"

    def __init__(
        self,
        vocab_size=257216,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_bidirectional_attention=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_bidirectional_attention = use_bidirectional_attention


class PaliGemmaConfig(PreTrainedConfig):
    """Configuration for the full PaliGemma vision-language model."""
    model_type = "paligemma"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            vision_config = {k: v for k, v in vision_config.items() if k != "model_type"}
            vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = SiglipVisionConfig(
                intermediate_size=4304,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vision_use_head=False,
            )

        if isinstance(text_config, dict):
            text_config = {k: v for k, v in text_config.items() if k != "model_type"}
            text_config = GemmaConfig(**text_config)
        elif text_config is None:
            text_config = GemmaConfig(
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                vocab_size=vocab_size,
            )

        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_index = image_token_index
        self.image_token_id = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings

        if getattr(self.text_config, "use_bidirectional_attention", None) is None:
            self.text_config.use_bidirectional_attention = True

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = self.projection_dim

    def get_text_config(self, decoder=None, **kwargs):
        return self.text_config


def load_config(model_path: str) -> PreTrainedConfig:
    """Load config from a model directory, auto-detecting the model type."""
    resolved_path = model_path
    if not os.path.isdir(model_path):
        baked = os.path.join("/data", model_path)
        if os.path.isdir(baked):
            resolved_path = baked
    config_file = os.path.join(resolved_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type", "")

    if model_type == "gemma":
        return GemmaConfig.from_dict(config_dict)
    if model_type == "siglip_vision_model":
        return SiglipVisionConfig.from_dict(config_dict)
    return PaliGemmaConfig.from_dict(config_dict)


__all__ = ["PaliGemmaConfig", "GemmaConfig", "SiglipVisionConfig", "load_config"]
