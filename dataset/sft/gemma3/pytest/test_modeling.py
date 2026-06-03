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

"""Tests verifying jax_repo model components produce identical forward passes to torch_orig."""
import pytest
import torch


def _seed_all(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TestGemma3RMSNorm:
    def test_forward_identical(self, torch_modeling, jax_modeling):
        _seed_all()
        dim = 32
        torch_norm = torch_modeling.Gemma3RMSNorm(dim, eps=1e-6)
        jax_norm = jax_modeling.Gemma3RMSNorm(dim, eps=1e-6)

        # Copy weights
        jax_norm.load_state_dict(torch_norm.state_dict())

        x = torch.randn(2, 10, dim)
        torch_out = torch_norm(x)
        jax_out = jax_norm(x)

        assert torch.allclose(torch_out, jax_out, atol=1e-6)

    def test_zero_weights_init(self, torch_modeling, jax_modeling):
        dim = 16
        torch_norm = torch_modeling.Gemma3RMSNorm(dim)
        jax_norm = jax_modeling.Gemma3RMSNorm(dim)

        assert torch.allclose(torch_norm.weight, torch.zeros(dim))
        assert torch.allclose(jax_norm.weight, torch.zeros(dim))

    def test_numerical_stability(self, torch_modeling, jax_modeling):
        """RMSNorm should handle very small inputs without NaN."""
        _seed_all()
        dim = 16
        torch_norm = torch_modeling.Gemma3RMSNorm(dim)
        jax_norm = jax_modeling.Gemma3RMSNorm(dim)
        jax_norm.load_state_dict(torch_norm.state_dict())

        x = torch.randn(1, 5, dim) * 1e-10
        torch_out = torch_norm(x)
        jax_out = jax_norm(x)

        assert not torch.isnan(torch_out).any()
        assert not torch.isnan(jax_out).any()
        assert torch.allclose(torch_out, jax_out, atol=1e-6)


class TestGemma3TextScaledWordEmbedding:
    def test_scaling_matches(self, torch_modeling, jax_modeling):
        _seed_all()
        num_embeddings, embedding_dim = 128, 32
        scale = 32 ** 0.5

        torch_emb = torch_modeling.Gemma3TextScaledWordEmbedding(
            num_embeddings, embedding_dim, padding_idx=0, embed_scale=scale
        )
        jax_emb = jax_modeling.Gemma3TextScaledWordEmbedding(
            num_embeddings, embedding_dim, padding_idx=0, embed_scale=scale
        )
        jax_emb.load_state_dict(torch_emb.state_dict())

        input_ids = torch.tensor([[1, 5, 10, 0]])
        torch_out = torch_emb(input_ids)
        jax_out = jax_emb(input_ids)

        assert torch.allclose(torch_out, jax_out, atol=1e-6)
        # Padding idx embedding should be zero (before scaling too)
        assert torch.allclose(torch_out[0, 3], torch.zeros(embedding_dim))


class TestRotaryEmbedding:
    def test_cos_sin_match(self, torch_modeling, jax_modeling, tiny_text_config_dict):
        _seed_all()
        torch_cfg = torch_modeling.Gemma3RMSNorm.__module__.replace(".modeling", ".configuration")
        # Build configs directly
        import importlib
        torch_config_mod = importlib.import_module(torch_cfg)
        jax_config_mod = importlib.import_module(jax_modeling.Gemma3RMSNorm.__module__.replace(".modeling", ".configuration"))

        t_text_cfg = torch_config_mod.Gemma3TextConfig(**tiny_text_config_dict)
        j_text_cfg = jax_config_mod.Gemma3TextConfig(**tiny_text_config_dict)

        torch_rope = torch_modeling.Gemma3RotaryEmbedding(t_text_cfg)
        jax_rope = jax_modeling.Gemma3RotaryEmbedding(j_text_cfg)

        x = torch.randn(2, 10, 32)
        position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)

        for layer_type in set(t_text_cfg.layer_types):
            torch_cos, torch_sin = torch_rope(x, position_ids, layer_type)
            jax_cos, jax_sin = jax_rope(x, position_ids, layer_type)
            assert torch.allclose(torch_cos, jax_cos, atol=1e-6)
            assert torch.allclose(torch_sin, jax_sin, atol=1e-6)


class TestRotateHalfAndApplyRoPE:
    def test_rotate_half(self, torch_modeling, jax_modeling):
        x = torch.randn(2, 4, 10, 8)
        assert torch.allclose(
            torch_modeling.rotate_half(x),
            jax_modeling.rotate_half(x),
        )

    def test_apply_rotary_pos_emb(self, torch_modeling, jax_modeling):
        _seed_all()
        q = torch.randn(2, 4, 10, 8)
        k = torch.randn(2, 4, 10, 8)
        cos = torch.randn(2, 10, 8)
        sin = torch.randn(2, 10, 8)

        t_q, t_k = torch_modeling.apply_rotary_pos_emb(q, k, cos, sin)
        j_q, j_k = jax_modeling.apply_rotary_pos_emb(q, k, cos, sin)

        assert torch.allclose(t_q, j_q, atol=1e-6)
        assert torch.allclose(t_k, j_k, atol=1e-6)


class TestRepeatKV:
    def test_no_repeat(self, torch_modeling, jax_modeling):
        x = torch.randn(2, 2, 10, 8)
        assert torch.equal(torch_modeling.repeat_kv(x, 1), jax_modeling.repeat_kv(x, 1))

    def test_repeat_2x(self, torch_modeling, jax_modeling):
        x = torch.randn(2, 2, 10, 8)
        t_out = torch_modeling.repeat_kv(x, 2)
        j_out = jax_modeling.repeat_kv(x, 2)
        assert t_out.shape == j_out.shape == (2, 4, 10, 8)
        assert torch.equal(t_out, j_out)


class TestCausalMask:
    def test_basic_causal_mask(self, torch_modeling, jax_modeling):
        t_mask = torch_modeling.create_causal_mask((2, 5), torch.float32, torch.device("cpu"))
        j_mask = jax_modeling.create_causal_mask((2, 5), torch.float32, torch.device("cpu"))
        assert torch.equal(t_mask, j_mask)
        # Upper triangle should be very negative
        assert t_mask[0, 0, 0, 1] < -1e30
        # Lower triangle / diagonal should be 0
        assert t_mask[0, 0, 1, 0] == 0.0
        assert t_mask[0, 0, 2, 2] == 0.0

    def test_with_past_kv(self, torch_modeling, jax_modeling):
        t_mask = torch_modeling.create_causal_mask((1, 3), torch.float32, torch.device("cpu"), past_key_values_length=5)
        j_mask = jax_modeling.create_causal_mask((1, 3), torch.float32, torch.device("cpu"), past_key_values_length=5)
        assert t_mask.shape == j_mask.shape == (1, 1, 3, 8)
        assert torch.equal(t_mask, j_mask)

    def test_with_sliding_window(self, torch_modeling, jax_modeling):
        t_mask = torch_modeling.create_causal_mask((1, 10), torch.float32, torch.device("cpu"), sliding_window=4)
        j_mask = jax_modeling.create_causal_mask((1, 10), torch.float32, torch.device("cpu"), sliding_window=4)
        assert torch.equal(t_mask, j_mask)
        # Token at position 5 should not attend to position 0 (distance > 4)
        assert t_mask[0, 0, 5, 0] < -1e30
        # Token at position 5 should attend to position 2 (distance = 3 < 4)
        assert t_mask[0, 0, 5, 2] == 0.0


class TestSiglipComponents:
    def _make_vision_config(self, config_mod):
        return config_mod.SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=28,
            patch_size=14,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            vision_use_head=False,
        )

    def test_siglip_embeddings(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_vision_config(torch_configuration)
        j_cfg = self._make_vision_config(jax_configuration)

        t_emb = torch_modeling.SiglipVisionEmbeddings(t_cfg)
        j_emb = jax_modeling.SiglipVisionEmbeddings(j_cfg)
        j_emb.load_state_dict(t_emb.state_dict())

        pixel_values = torch.randn(2, 3, 28, 28)
        t_out = t_emb(pixel_values)
        j_out = j_emb(pixel_values)

        assert t_out.shape == j_out.shape
        assert torch.allclose(t_out, j_out, atol=1e-5)

    def test_siglip_attention(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_vision_config(torch_configuration)
        j_cfg = self._make_vision_config(jax_configuration)

        t_attn = torch_modeling.SiglipAttention(t_cfg)
        j_attn = jax_modeling.SiglipAttention(j_cfg)
        j_attn.load_state_dict(t_attn.state_dict())

        hidden_states = torch.randn(2, 4, 32)
        t_out = t_attn(hidden_states)
        j_out = j_attn(hidden_states)

        assert torch.allclose(t_out, j_out, atol=1e-5)

    def test_siglip_mlp(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_vision_config(torch_configuration)
        j_cfg = self._make_vision_config(jax_configuration)

        t_mlp = torch_modeling.SiglipMLP(t_cfg)
        j_mlp = jax_modeling.SiglipMLP(j_cfg)
        j_mlp.load_state_dict(t_mlp.state_dict())

        x = torch.randn(2, 4, 32)
        assert torch.allclose(t_mlp(x), j_mlp(x), atol=1e-5)

    def test_siglip_encoder_layer(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_vision_config(torch_configuration)
        j_cfg = self._make_vision_config(jax_configuration)

        t_layer = torch_modeling.SiglipEncoderLayer(t_cfg)
        j_layer = jax_modeling.SiglipEncoderLayer(j_cfg)
        j_layer.load_state_dict(t_layer.state_dict())

        x = torch.randn(2, 4, 32)
        t_out = t_layer(x)
        j_out = j_layer(x)

        assert torch.allclose(t_out, j_out, atol=1e-5)

    def test_siglip_vision_model(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_vision_config(torch_configuration)
        j_cfg = self._make_vision_config(jax_configuration)

        t_model = torch_modeling.SiglipVisionModel(t_cfg)
        j_model = jax_modeling.SiglipVisionModel(j_cfg)
        j_model.load_state_dict(t_model.state_dict())

        pixel_values = torch.randn(2, 3, 28, 28)
        t_out = t_model(pixel_values)
        j_out = j_model(pixel_values)

        assert torch.allclose(t_out.last_hidden_state, j_out.last_hidden_state, atol=1e-4)


class TestGemma3TextComponents:
    def _make_text_config(self, config_mod):
        return config_mod.Gemma3TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            pad_token_id=0,
            attention_bias=False,
            attention_dropout=0.0,
            query_pre_attn_scalar=8,
            sliding_window=16,
            sliding_window_pattern=6,
        )

    def test_gemma3_mlp(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_text_config(torch_configuration)
        j_cfg = self._make_text_config(jax_configuration)

        t_mlp = torch_modeling.Gemma3MLP(t_cfg)
        j_mlp = jax_modeling.Gemma3MLP(j_cfg)
        j_mlp.load_state_dict(t_mlp.state_dict())

        x = torch.randn(2, 10, 32)
        assert torch.allclose(t_mlp(x), j_mlp(x), atol=1e-5)

    def test_gemma3_attention(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_text_config(torch_configuration)
        j_cfg = self._make_text_config(jax_configuration)

        t_attn = torch_modeling.Gemma3Attention(t_cfg, layer_idx=0)
        j_attn = jax_modeling.Gemma3Attention(j_cfg, layer_idx=0)
        j_attn.load_state_dict(t_attn.state_dict())

        bsz, seq_len = 2, 10
        hidden_states = torch.randn(bsz, seq_len, 32)
        cos = torch.randn(bsz, seq_len, 8)
        sin = torch.randn(bsz, seq_len, 8)
        position_embeddings = (cos, sin)

        t_out, t_weights = t_attn(hidden_states, position_embeddings=position_embeddings)
        j_out, j_weights = j_attn(hidden_states, position_embeddings=position_embeddings)

        assert torch.allclose(t_out, j_out, atol=1e-5)

    def test_gemma3_decoder_layer(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_text_config(torch_configuration)
        j_cfg = self._make_text_config(jax_configuration)

        t_layer = torch_modeling.Gemma3DecoderLayer(t_cfg, layer_idx=0)
        j_layer = jax_modeling.Gemma3DecoderLayer(j_cfg, layer_idx=0)
        j_layer.load_state_dict(t_layer.state_dict())

        bsz, seq_len = 2, 10
        hidden_states = torch.randn(bsz, seq_len, 32)
        cos = torch.randn(bsz, seq_len, 8)
        sin = torch.randn(bsz, seq_len, 8)

        t_out = t_layer(hidden_states, position_embeddings=(cos, sin))
        j_out = j_layer(hidden_states, position_embeddings=(cos, sin))

        assert torch.allclose(t_out, j_out, atol=1e-5)

    def test_gemma3_text_model(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = self._make_text_config(torch_configuration)
        j_cfg = self._make_text_config(jax_configuration)

        t_model = torch_modeling.Gemma3TextModel(t_cfg)
        j_model = jax_modeling.Gemma3TextModel(j_cfg)
        j_model.load_state_dict(t_model.state_dict())

        t_model.eval()
        j_model.eval()

        input_ids = torch.randint(0, 128, (2, 10))
        t_out = t_model(input_ids=input_ids)
        j_out = j_model(input_ids=input_ids)

        assert torch.allclose(t_out.last_hidden_state, j_out.last_hidden_state, atol=1e-4)


class TestGemma3MultiModalProjector:
    def test_forward(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = torch_configuration.Gemma3Config(
            text_config={"hidden_size": 32, "vocab_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "intermediate_size": 64},
            vision_config={"hidden_size": 32, "image_size": 28, "patch_size": 14,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "intermediate_size": 64, "layer_norm_eps": 1e-6},
            mm_tokens_per_image=4,
        )
        j_cfg = jax_configuration.Gemma3Config(
            text_config={"hidden_size": 32, "vocab_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "intermediate_size": 64},
            vision_config={"hidden_size": 32, "image_size": 28, "patch_size": 14,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "intermediate_size": 64, "layer_norm_eps": 1e-6},
            mm_tokens_per_image=4,
        )

        t_proj = torch_modeling.Gemma3MultiModalProjector(t_cfg)
        j_proj = jax_modeling.Gemma3MultiModalProjector(j_cfg)
        j_proj.load_state_dict(t_proj.state_dict())

        # patches_per_image = 28 // 14 = 2, so num_patches = 4
        vision_outputs = torch.randn(2, 4, 32)
        t_out = t_proj(vision_outputs)
        j_out = j_proj(vision_outputs)

        assert t_out.shape == j_out.shape
        assert torch.allclose(t_out, j_out, atol=1e-5)


class TestGemma3ForConditionalGeneration:
    def test_forward_text_only(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = torch_configuration.Gemma3Config(
            text_config={"hidden_size": 32, "vocab_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "intermediate_size": 64, "pad_token_id": 0, "sliding_window_pattern": 6},
            vision_config={"hidden_size": 32, "image_size": 28, "patch_size": 14,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "intermediate_size": 64, "vision_use_head": False},
            mm_tokens_per_image=4,
            image_token_index=127,
        )
        j_cfg = jax_configuration.Gemma3Config(
            text_config={"hidden_size": 32, "vocab_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "intermediate_size": 64, "pad_token_id": 0, "sliding_window_pattern": 6},
            vision_config={"hidden_size": 32, "image_size": 28, "patch_size": 14,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "intermediate_size": 64, "vision_use_head": False},
            mm_tokens_per_image=4,
            image_token_index=127,
        )

        t_model = torch_modeling.Gemma3ForConditionalGeneration(t_cfg)
        j_model = jax_modeling.Gemma3ForConditionalGeneration(j_cfg)
        j_model.load_state_dict(t_model.state_dict())

        t_model.eval()
        j_model.eval()

        input_ids = torch.randint(0, 100, (1, 8))  # avoid image token 127
        attention_mask = torch.ones_like(input_ids)

        t_out = t_model(input_ids=input_ids, attention_mask=attention_mask)
        j_out = j_model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(t_out.logits, j_out.logits, atol=1e-4)

    def test_loss_computation(self, torch_modeling, jax_modeling, torch_configuration, jax_configuration):
        _seed_all()
        t_cfg = torch_configuration.Gemma3Config(
            text_config={"hidden_size": 32, "vocab_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "intermediate_size": 64, "pad_token_id": 0, "sliding_window_pattern": 6},
            vision_config={"hidden_size": 32, "image_size": 28, "patch_size": 14,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "intermediate_size": 64, "vision_use_head": False},
            mm_tokens_per_image=4,
            image_token_index=127,
        )
        j_cfg = jax_configuration.Gemma3Config(
            text_config={"hidden_size": 32, "vocab_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "intermediate_size": 64, "pad_token_id": 0, "sliding_window_pattern": 6},
            vision_config={"hidden_size": 32, "image_size": 28, "patch_size": 14,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "intermediate_size": 64, "vision_use_head": False},
            mm_tokens_per_image=4,
            image_token_index=127,
        )

        t_model = torch_modeling.Gemma3ForConditionalGeneration(t_cfg)
        j_model = jax_modeling.Gemma3ForConditionalGeneration(j_cfg)
        j_model.load_state_dict(t_model.state_dict())

        t_model.eval()
        j_model.eval()

        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 128, (1, 8))
        labels[0, :3] = -100  # mask first 3 tokens

        t_out = t_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        j_out = j_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert t_out.loss is not None
        assert j_out.loss is not None
        assert torch.allclose(t_out.loss, j_out.loss, atol=1e-4)


class TestDynamicCache:
    def test_cache_update_and_retrieval(self, torch_modeling, jax_modeling):
        t_cache = torch_modeling.DynamicCache()
        j_cache = jax_modeling.DynamicCache()

        key = torch.randn(1, 4, 5, 8)
        value = torch.randn(1, 4, 5, 8)

        t_k, t_v = t_cache.update(key, value, layer_idx=0)
        j_k, j_v = j_cache.update(key, value, layer_idx=0)

        assert torch.equal(t_k, j_k)
        assert torch.equal(t_v, j_v)
        assert t_cache.get_seq_length(0) == j_cache.get_seq_length(0) == 5

    def test_cache_concatenation(self, torch_modeling, jax_modeling):
        t_cache = torch_modeling.DynamicCache()
        j_cache = jax_modeling.DynamicCache()

        k1 = torch.randn(1, 4, 3, 8)
        v1 = torch.randn(1, 4, 3, 8)
        k2 = torch.randn(1, 4, 2, 8)
        v2 = torch.randn(1, 4, 2, 8)

        t_cache.update(k1, v1, layer_idx=0)
        j_cache.update(k1, v1, layer_idx=0)

        t_k, t_v = t_cache.update(k2, v2, layer_idx=0)
        j_k, j_v = j_cache.update(k2, v2, layer_idx=0)

        assert t_cache.get_seq_length(0) == j_cache.get_seq_length(0) == 5
        assert torch.equal(t_k, j_k)

    def test_cache_empty_state(self, torch_modeling, jax_modeling):
        t_cache = torch_modeling.DynamicCache()
        j_cache = jax_modeling.DynamicCache()

        assert t_cache.is_initialized == j_cache.is_initialized == False
        assert t_cache.get_seq_length(0) == j_cache.get_seq_length(0) == 0
