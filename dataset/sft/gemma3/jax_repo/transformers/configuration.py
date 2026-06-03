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

"""Standalone configuration classes for Gemma3 - decoupled from HuggingFace transformers."""
import json
import logging
import os
from typing import Any, Optional

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


class SiglipVisionConfig(PreTrainedConfig):
    """Configuration for the SigLip vision encoder used in Gemma3."""
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


class Gemma3TextConfig(PreTrainedConfig):
    """Configuration for the Gemma3 text model."""
    model_type = "gemma3_text"

    def __init__(
        self,
        vocab_size=262208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_parameters=None,
        use_bidirectional_attention=False,
        tie_word_embeddings=True,
        sliding_window_pattern=6,
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
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters

        # Build layer types
        self._sliding_window_pattern = sliding_window_pattern
        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types


class Gemma3Config(PreTrainedConfig):
    """Configuration for the full Gemma3 vision-language model."""
    model_type = "gemma3"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        mm_tokens_per_image=256,
        boi_token_index=255999,
        eoi_token_index=256000,
        image_token_index=262144,
        initializer_range=0.02,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = Gemma3TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma3TextConfig(**text_config)

        if vision_config is None:
            vision_config = SiglipVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings


def load_config(model_path: str) -> PreTrainedConfig:
    """Load config from a model directory, auto-detecting the model type."""
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type", "")

    if model_type == "gemma3":
        return Gemma3Config.from_dict(config_dict)
    elif model_type == "gemma3_text":
        return Gemma3TextConfig.from_dict(config_dict)
    else:
        # Try Gemma3Config for composite model
        if "text_config" in config_dict or "vision_config" in config_dict:
            return Gemma3Config.from_dict(config_dict)
        return PreTrainedConfig.from_dict(config_dict)
