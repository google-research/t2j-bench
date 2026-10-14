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

"""Bloom configuration."""

import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

_UNSET = object()


class PreTrainedConfig:
    """Minimal implementation of ``transformers.PreTrainedConfig``."""

    model_type: str = ""
    attribute_map: dict = {}
    keys_to_ignore_at_inference: list = []

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

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal or "eager"

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value

    @property
    def use_return_dict(self) -> bool:
        return self.return_dict and not self.torchscript

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


class BloomConfig(PreTrainedConfig):
    """This is the configuration class to store the configuration of a [`BloomModel`]."""

    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,
        slow_but_exact=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.slow_but_exact = slow_but_exact
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


__all__ = ["BloomConfig"]
