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

"""Configuration classes for LLaVA-Next standalone implementation."""
import json
import logging
import os
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception:
        return model_path


class PreTrainedConfig:
    """Base config class supporting from_pretrained and serialization."""
    model_type: str = ""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        pretrained_model_name_or_path = _resolve_model_path(pretrained_model_name_or_path)
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict, **kwargs)
        raise FileNotFoundError(f"Config file not found: {config_file}")

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        config_dict.update(kwargs)
        return cls(**config_dict)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, PreTrainedConfig):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


class CLIPVisionConfig(PreTrainedConfig):
    """Configuration for CLIP Vision Model."""
    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        projection_dim: int = 768,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 336,
        patch_size: int = 14,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class LlamaConfig(PreTrainedConfig):
    """Configuration for Llama language model."""
    model_type = "llama"

    def __init__(
        self,
        vocab_size: int = 32064,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class LlavaNextConfig(PreTrainedConfig):
    """Configuration for LLaVA-Next model."""
    model_type = "llava_next"

    def __init__(
        self,
        vision_config: Optional[Union[dict, CLIPVisionConfig]] = None,
        text_config: Optional[Union[dict, LlamaConfig]] = None,
        image_token_index: int = 32000,
        projector_hidden_act: str = "gelu",
        vision_feature_select_strategy: str = "default",
        vision_feature_layer: Union[int, list] = -2,
        image_grid_pinpoints: Optional[list] = None,
        tie_word_embeddings: bool = False,
        image_seq_length: int = 576,
        multimodal_projector_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length
        self.multimodal_projector_bias = multimodal_projector_bias
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.tie_word_embeddings = tie_word_embeddings

        self.image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )

        # Build vision config
        if isinstance(vision_config, dict):
            self.vision_config = CLIPVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = CLIPVisionConfig()
        else:
            self.vision_config = vision_config

        # Build text config
        if isinstance(text_config, dict):
            self.text_config = LlamaConfig(**text_config)
        elif text_config is None:
            self.text_config = LlamaConfig()
        else:
            self.text_config = text_config

    # Alias for compatibility
    @property
    def image_token_id(self):
        return self.image_token_index


def load_config(model_path: str) -> LlavaNextConfig:
    """Load config from model directory, auto-detecting type."""
    model_path = _resolve_model_path(model_path)
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type", "llava_next")

    if model_type == "llava_next":
        return LlavaNextConfig.from_dict(config_dict)
    else:
        logger.warning(f"Unknown model_type '{model_type}', treating as LlavaNextConfig")
        return LlavaNextConfig.from_dict(config_dict)
