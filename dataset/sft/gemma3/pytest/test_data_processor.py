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

"""Tests verifying jax_repo data processor matches torch_orig."""
from unittest.mock import MagicMock

import pytest


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.bos_token = "<bos>"
    tok.bos_token_id = 2
    tok.eos_token = "<eos>"
    tok.eos_token_id = 1
    tok.pad_token_id = 0

    def mock_encode(text, add_special_tokens=False):
        return [ord(c) for c in text]

    tok.encode = mock_encode
    tok.convert_tokens_to_ids = lambda x: hash(x) % 10000
    tok.decode = lambda ids: "".join(chr(i) if 32 <= i < 127 else "?" for i in ids)
    return tok


def _make_mock_template(torch_template_mod):
    """Get the real gemma3 template from the module."""
    return torch_template_mod.get_template("gemma3")


def _make_data_args(cutoff_len=2048, train_on_prompt=False, mask_history=False):
    args = MagicMock()
    args.cutoff_len = cutoff_len
    args.train_on_prompt = train_on_prompt
    args.mask_history = mask_history
    return args


class TestInferSeqlen:
    def test_fits_within_cutoff(self, torch_processor, jax_processor):
        assert torch_processor.infer_seqlen(10, 20, 100) == jax_processor.infer_seqlen(10, 20, 100) == (10, 20)

    def test_source_exceeds_cutoff(self, torch_processor, jax_processor):
        assert torch_processor.infer_seqlen(200, 20, 100) == jax_processor.infer_seqlen(200, 20, 100) == (100, 0)

    def test_total_exceeds_cutoff(self, torch_processor, jax_processor):
        assert torch_processor.infer_seqlen(60, 80, 100) == jax_processor.infer_seqlen(60, 80, 100) == (60, 40)

    def test_exact_fit(self, torch_processor, jax_processor):
        assert torch_processor.infer_seqlen(50, 50, 100) == jax_processor.infer_seqlen(50, 50, 100) == (50, 50)

    def test_zero_source(self, torch_processor, jax_processor):
        assert torch_processor.infer_seqlen(0, 50, 100) == jax_processor.infer_seqlen(0, 50, 100) == (0, 50)


class TestSupervisedDatasetProcessor:
    def _make_processor(self, mod, template_mod, **kwargs):
        tokenizer = _make_mock_tokenizer()
        template = template_mod.get_template("gemma3")
        data_args = _make_data_args(**kwargs)
        return mod.SupervisedDatasetProcessor(
            template=template,
            tokenizer=tokenizer,
            data_args=data_args,
        )

    def test_encode_single_turn(self, torch_processor, jax_processor, torch_template, jax_template):
        tp = self._make_processor(torch_processor, torch_template)
        jp = self._make_processor(jax_processor, jax_template)

        example = {
            "_prompt": [{"role": "user", "content": "Hi"}],
            "_response": [{"role": "assistant", "content": "Hello"}],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
        }

        t_result = tp._encode_data_example(example)
        j_result = jp._encode_data_example(example)

        assert t_result["input_ids"] == j_result["input_ids"]
        assert t_result["labels"] == j_result["labels"]
        assert t_result["attention_mask"] == j_result["attention_mask"]

    def test_labels_mask_prompt(self, torch_processor, jax_processor, torch_template, jax_template):
        """By default, prompt tokens should have IGNORE_INDEX labels."""
        IGNORE_INDEX = -100
        tp = self._make_processor(torch_processor, torch_template)
        jp = self._make_processor(jax_processor, jax_template)

        example = {
            "_prompt": [{"role": "user", "content": "Q"}],
            "_response": [{"role": "assistant", "content": "A"}],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
        }

        t_result = tp._encode_data_example(example)
        j_result = jp._encode_data_example(example)

        # Response tokens should NOT be IGNORE_INDEX
        t_non_ignored = [l for l in t_result["labels"] if l != IGNORE_INDEX]
        j_non_ignored = [l for l in j_result["labels"] if l != IGNORE_INDEX]
        assert len(t_non_ignored) > 0
        assert t_non_ignored == j_non_ignored

    def test_train_on_prompt(self, torch_processor, jax_processor, torch_template, jax_template):
        IGNORE_INDEX = -100
        tp = self._make_processor(torch_processor, torch_template, train_on_prompt=True)
        jp = self._make_processor(jax_processor, jax_template, train_on_prompt=True)

        example = {
            "_prompt": [{"role": "user", "content": "Q"}],
            "_response": [{"role": "assistant", "content": "A"}],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
        }

        t_result = tp._encode_data_example(example)
        j_result = jp._encode_data_example(example)

        assert t_result["input_ids"] == j_result["input_ids"]
        assert t_result["labels"] == j_result["labels"]
        # All labels should equal input_ids (no masking)
        assert t_result["labels"] == t_result["input_ids"]

    def test_cutoff_truncation(self, torch_processor, jax_processor, torch_template, jax_template):
        tp = self._make_processor(torch_processor, torch_template, cutoff_len=50)
        jp = self._make_processor(jax_processor, jax_template, cutoff_len=50)

        example = {
            "_prompt": [{"role": "user", "content": "A" * 100}],
            "_response": [{"role": "assistant", "content": "B" * 100}],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
        }

        t_result = tp._encode_data_example(example)
        j_result = jp._encode_data_example(example)

        assert len(t_result["input_ids"]) <= 50
        assert len(j_result["input_ids"]) <= 50
        assert t_result["input_ids"] == j_result["input_ids"]

    def test_multiturn_conversation(self, torch_processor, jax_processor, torch_template, jax_template):
        tp = self._make_processor(torch_processor, torch_template)
        jp = self._make_processor(jax_processor, jax_template)

        example = {
            "_prompt": [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ],
            "_response": [
                {"role": "assistant", "content": "Hello!"},
                {"role": "assistant", "content": "I'm good."},
            ],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
        }

        t_result = tp._encode_data_example(example)
        j_result = jp._encode_data_example(example)

        assert t_result["input_ids"] == j_result["input_ids"]
        assert t_result["labels"] == j_result["labels"]

    def test_mask_history(self, torch_processor, jax_processor, torch_template, jax_template):
        IGNORE_INDEX = -100
        tp = self._make_processor(torch_processor, torch_template, mask_history=True)
        jp = self._make_processor(jax_processor, jax_template, mask_history=True)

        example = {
            "_prompt": [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "What?"},
            ],
            "_response": [
                {"role": "assistant", "content": "Hello!"},
                {"role": "assistant", "content": "Nothing."},
            ],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
        }

        t_result = tp._encode_data_example(example)
        j_result = jp._encode_data_example(example)

        assert t_result["input_ids"] == j_result["input_ids"]
        assert t_result["labels"] == j_result["labels"]

    def test_preprocess_dataset_batch(self, torch_processor, jax_processor, torch_template, jax_template):
        tp = self._make_processor(torch_processor, torch_template)
        jp = self._make_processor(jax_processor, jax_template)

        examples = {
            "_prompt": [
                [{"role": "user", "content": "Q1"}],
                [{"role": "user", "content": "Q2"}],
            ],
            "_response": [
                [{"role": "assistant", "content": "A1"}],
                [{"role": "assistant", "content": "A2"}],
            ],
            "_system": ["", ""],
            "_tools": ["", ""],
            "_images": [None, None],
            "_videos": [None, None],
        }

        t_result = tp.preprocess_dataset(examples)
        j_result = jp.preprocess_dataset(examples)

        assert len(t_result["input_ids"]) == len(j_result["input_ids"]) == 2
        for i in range(2):
            assert t_result["input_ids"][i] == j_result["input_ids"][i]
            assert t_result["labels"][i] == j_result["labels"][i]
