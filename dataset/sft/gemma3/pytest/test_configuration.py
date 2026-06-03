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

"""Tests verifying jax_repo configuration classes match torch_orig."""
import json
import os
import tempfile

import pytest


class TestPreTrainedConfig:
    """Verify PreTrainedConfig base class works identically."""

    def test_from_dict_creates_identical_objects(self, torch_configuration, jax_configuration):
        d = {"model_type": "test", "hidden_size": 128, "extra_field": True}
        torch_cfg = torch_configuration.PreTrainedConfig.from_dict(d)
        jax_cfg = jax_configuration.PreTrainedConfig.from_dict(d)
        assert torch_cfg.to_dict() == jax_cfg.to_dict()

    def test_to_dict_round_trips(self, torch_configuration, jax_configuration):
        d = {"model_type": "test", "a": 1, "b": "hello"}
        torch_cfg = torch_configuration.PreTrainedConfig(**d)
        jax_cfg = jax_configuration.PreTrainedConfig(**d)
        assert torch_cfg.to_dict() == jax_cfg.to_dict()

    def test_from_pretrained_with_file(self, torch_configuration, jax_configuration):
        config_dict = {"model_type": "gemma3_text", "hidden_size": 256}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            torch_cfg = torch_configuration.PreTrainedConfig.from_pretrained(tmpdir)
            jax_cfg = jax_configuration.PreTrainedConfig.from_pretrained(tmpdir)
            assert torch_cfg.to_dict() == jax_cfg.to_dict()

    def test_from_pretrained_missing_file_raises(self, torch_configuration, jax_configuration):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                torch_configuration.PreTrainedConfig.from_pretrained(tmpdir)
            with pytest.raises(FileNotFoundError):
                jax_configuration.PreTrainedConfig.from_pretrained(tmpdir)


class TestSiglipVisionConfig:
    """Verify SiglipVisionConfig defaults and overrides match."""

    def test_defaults_match(self, torch_configuration, jax_configuration):
        torch_cfg = torch_configuration.SiglipVisionConfig()
        jax_cfg = jax_configuration.SiglipVisionConfig()
        assert torch_cfg.to_dict() == jax_cfg.to_dict()

    def test_custom_values_match(self, torch_configuration, jax_configuration, tiny_vision_config_dict):
        torch_cfg = torch_configuration.SiglipVisionConfig(**tiny_vision_config_dict)
        jax_cfg = jax_configuration.SiglipVisionConfig(**tiny_vision_config_dict)
        assert torch_cfg.hidden_size == jax_cfg.hidden_size
        assert torch_cfg.image_size == jax_cfg.image_size
        assert torch_cfg.patch_size == jax_cfg.patch_size
        assert torch_cfg.to_dict() == jax_cfg.to_dict()


class TestGemma3TextConfig:
    """Verify Gemma3TextConfig defaults and layer_types generation match."""

    def test_defaults_match(self, torch_configuration, jax_configuration):
        torch_cfg = torch_configuration.Gemma3TextConfig()
        jax_cfg = jax_configuration.Gemma3TextConfig()
        assert torch_cfg.to_dict() == jax_cfg.to_dict()

    def test_layer_types_auto_generated(self, torch_configuration, jax_configuration, tiny_text_config_dict):
        torch_cfg = torch_configuration.Gemma3TextConfig(**tiny_text_config_dict)
        jax_cfg = jax_configuration.Gemma3TextConfig(**tiny_text_config_dict)
        assert torch_cfg.layer_types == jax_cfg.layer_types

    def test_layer_types_sliding_window_pattern(self, torch_configuration, jax_configuration):
        """Layer at index i is full_attention when (i+1) % pattern == 0."""
        for pattern in [2, 3, 6]:
            torch_cfg = torch_configuration.Gemma3TextConfig(
                num_hidden_layers=12, sliding_window_pattern=pattern
            )
            jax_cfg = jax_configuration.Gemma3TextConfig(
                num_hidden_layers=12, sliding_window_pattern=pattern
            )
            assert torch_cfg.layer_types == jax_cfg.layer_types
            # Verify the pattern itself
            for i, lt in enumerate(torch_cfg.layer_types):
                if (i + 1) % pattern == 0:
                    assert lt == "full_attention"
                else:
                    assert lt == "sliding_attention"

    def test_explicit_layer_types_passthrough(self, torch_configuration, jax_configuration):
        explicit = ["full_attention", "sliding_attention", "full_attention"]
        torch_cfg = torch_configuration.Gemma3TextConfig(
            num_hidden_layers=3, layer_types=explicit
        )
        jax_cfg = jax_configuration.Gemma3TextConfig(
            num_hidden_layers=3, layer_types=explicit
        )
        assert torch_cfg.layer_types == jax_cfg.layer_types == explicit


class TestGemma3Config:
    """Verify Gemma3Config (composite VLM config) matches."""

    def test_defaults_match(self, torch_configuration, jax_configuration):
        torch_cfg = torch_configuration.Gemma3Config()
        jax_cfg = jax_configuration.Gemma3Config()
        torch_d = torch_cfg.to_dict()
        jax_d = jax_cfg.to_dict()
        assert torch_d == jax_d

    def test_nested_configs_from_dicts(self, torch_configuration, jax_configuration, tiny_gemma3_config_dict):
        torch_cfg = torch_configuration.Gemma3Config(**tiny_gemma3_config_dict)
        jax_cfg = jax_configuration.Gemma3Config(**tiny_gemma3_config_dict)

        assert torch_cfg.mm_tokens_per_image == jax_cfg.mm_tokens_per_image
        assert torch_cfg.image_token_index == jax_cfg.image_token_index
        assert torch_cfg.text_config.to_dict() == jax_cfg.text_config.to_dict()
        assert torch_cfg.vision_config.to_dict() == jax_cfg.vision_config.to_dict()

    def test_load_config_auto_detect(self, torch_configuration, jax_configuration):
        config_dict = {
            "model_type": "gemma3",
            "text_config": {"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 2,
                            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                            "intermediate_size": 64},
            "vision_config": {"hidden_size": 32},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            torch_cfg = torch_configuration.load_config(tmpdir)
            jax_cfg = jax_configuration.load_config(tmpdir)
            assert type(torch_cfg).__name__ == type(jax_cfg).__name__ == "Gemma3Config"
            assert torch_cfg.to_dict() == jax_cfg.to_dict()

    def test_load_config_text_only(self, torch_configuration, jax_configuration):
        config_dict = {"model_type": "gemma3_text", "vocab_size": 100, "hidden_size": 32,
                        "num_hidden_layers": 2, "num_attention_heads": 2,
                        "num_key_value_heads": 1, "head_dim": 16, "intermediate_size": 64}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            torch_cfg = torch_configuration.load_config(tmpdir)
            jax_cfg = jax_configuration.load_config(tmpdir)
            assert type(torch_cfg).__name__ == type(jax_cfg).__name__ == "Gemma3TextConfig"
