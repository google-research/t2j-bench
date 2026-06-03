# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests verifying jax_repo template system matches torch_orig."""
from unittest.mock import MagicMock

import pytest


def _make_mock_tokenizer():
    """Create a mock tokenizer that deterministically encodes strings to ints."""
    tok = MagicMock()
    tok.bos_token = "<bos>"
    tok.bos_token_id = 2
    tok.eos_token = "<eos>"
    tok.eos_token_id = 1
    tok.pad_token_id = 0

    def mock_encode(text, add_special_tokens=False):
        # Deterministic encoding: each char -> its ordinal
        return [ord(c) for c in text]

    tok.encode = mock_encode
    tok.convert_tokens_to_ids = lambda x: hash(x) % 10000
    return tok


class TestTemplateRegistry:
    def test_gemma3_template_registered(self, torch_template, jax_template):
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")
        assert torch_t is not None
        assert jax_t is not None

    def test_gemma2_template_registered(self, torch_template, jax_template):
        torch_t = torch_template.get_template("gemma2")
        jax_t = jax_template.get_template("gemma2")
        assert torch_t is not None
        assert jax_t is not None

    def test_unknown_template_raises(self, torch_template, jax_template):
        with pytest.raises(ValueError, match="not found"):
            torch_template.get_template("nonexistent")
        with pytest.raises(ValueError, match="not found"):
            jax_template.get_template("nonexistent")


class TestGemma3TemplateProperties:
    def test_stop_words_match(self, torch_template, jax_template):
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")
        assert torch_t.stop_words == jax_t.stop_words == ["<end_of_turn>"]

    def test_replace_eos_match(self, torch_template, jax_template):
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")
        assert torch_t.replace_eos == jax_t.replace_eos == True

    def test_efficient_eos_gemma2(self, torch_template, jax_template):
        torch_t = torch_template.get_template("gemma2")
        jax_t = jax_template.get_template("gemma2")
        assert torch_t.efficient_eos == jax_t.efficient_eos == True

    def test_efficient_eos_gemma3(self, torch_template, jax_template):
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")
        assert torch_t.efficient_eos == jax_t.efficient_eos == False


class TestTemplateEncoding:
    def test_encode_oneturn_simple(self, torch_template, jax_template):
        tokenizer = _make_mock_tokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")

        torch_prompt, torch_resp = torch_t.encode_oneturn(tokenizer, messages)
        jax_prompt, jax_resp = jax_t.encode_oneturn(tokenizer, messages)

        assert torch_prompt == jax_prompt
        assert torch_resp == jax_resp

    def test_encode_multiturn(self, torch_template, jax_template):
        tokenizer = _make_mock_tokenizer()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")

        torch_pairs = torch_t.encode_multiturn(tokenizer, messages)
        jax_pairs = jax_t.encode_multiturn(tokenizer, messages)

        assert len(torch_pairs) == len(jax_pairs)
        for (tp, tr), (jp, jr) in zip(torch_pairs, jax_pairs):
            assert tp == jp
            assert tr == jr

    def test_encode_with_system(self, torch_template, jax_template):
        tokenizer = _make_mock_tokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        system = "You are helpful."

        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")

        torch_prompt, torch_resp = torch_t.encode_oneturn(tokenizer, messages, system=system)
        jax_prompt, jax_resp = jax_t.encode_oneturn(tokenizer, messages, system=system)

        assert torch_prompt == jax_prompt
        assert torch_resp == jax_resp

    def test_encode_user_only_no_response(self, torch_template, jax_template):
        tokenizer = _make_mock_tokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
        ]
        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")

        torch_pairs = torch_t.encode_multiturn(tokenizer, messages)
        jax_pairs = jax_t.encode_multiturn(tokenizer, messages)

        assert len(torch_pairs) == len(jax_pairs) == 1
        assert torch_pairs[0][1] == jax_pairs[0][1] == []

    def test_llama2_template_fuses_system(self, torch_template, jax_template):
        """Gemma3 uses Llama2Template which fuses system into first user turn."""
        tokenizer = _make_mock_tokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        system = "Be concise."

        torch_t = torch_template.get_template("gemma3")
        jax_t = jax_template.get_template("gemma3")

        torch_pairs = torch_t.encode_multiturn(tokenizer, messages, system=system)
        jax_pairs = jax_t.encode_multiturn(tokenizer, messages, system=system)

        # System should be fused into first user prompt
        assert torch_pairs[0][0] == jax_pairs[0][0]
        # The system content chars should be in the prompt IDs
        system_ords = [ord(c) for c in system]
        # Check that the system content appears somewhere in the prompt IDs
        prompt_ids = torch_pairs[0][0]
        for o in system_ords:
            assert o in prompt_ids


class TestGetTemplateAndFixTokenizer:
    def test_eos_replacement(self, torch_template, jax_template):
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids = lambda x: 107 if x == "<end_of_turn>" else 0

        data_args = MagicMock()
        data_args.template = "gemma3"

        torch_t = torch_template.get_template_and_fix_tokenizer(tokenizer, data_args)
        # Reset tokenizer for jax
        tokenizer2 = MagicMock()
        tokenizer2.convert_tokens_to_ids = lambda x: 107 if x == "<end_of_turn>" else 0
        data_args2 = MagicMock()
        data_args2.template = "gemma3"

        jax_t = jax_template.get_template_and_fix_tokenizer(tokenizer2, data_args2)

        assert tokenizer.eos_token == tokenizer2.eos_token == "<end_of_turn>"
        assert tokenizer.eos_token_id == tokenizer2.eos_token_id == 107


class TestRoleEnum:
    def test_roles_match(self, torch_template, jax_template):
        for role_name in ["USER", "ASSISTANT", "SYSTEM", "FUNCTION", "OBSERVATION"]:
            torch_role = getattr(torch_template.Role, role_name)
            jax_role = getattr(jax_template.Role, role_name)
            assert torch_role.value == jax_role.value
