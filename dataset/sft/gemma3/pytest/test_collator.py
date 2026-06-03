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

"""Tests verifying jax_repo data collators match torch_orig."""
from unittest.mock import MagicMock

import pytest
import torch


def _make_mock_tokenizer(pad_token_id=0):
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    return tok


class TestPrepare4DAttentionMask:
    def test_basic_shape(self, torch_collator, jax_collator):
        attn_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        torch_result = torch_collator.prepare_4d_attention_mask(attn_mask, torch.float32)
        jax_result = jax_collator.prepare_4d_attention_mask(attn_mask, torch.float32)

        assert torch_result.shape == jax_result.shape == (2, 1, 4, 4)

    def test_causal_structure(self, torch_collator, jax_collator):
        """Upper triangle should be masked (very negative values)."""
        attn_mask = torch.tensor([[1, 1, 1, 1]])
        torch_result = torch_collator.prepare_4d_attention_mask(attn_mask, torch.float32)
        jax_result = jax_collator.prepare_4d_attention_mask(attn_mask, torch.float32)

        assert torch.allclose(torch_result, jax_result)
        # Position (0,0,0,1) should be masked (future token)
        assert torch_result[0, 0, 0, 1] < -1e30
        # Position (0,0,1,0) should be unmasked
        assert torch_result[0, 0, 1, 0] == 0.0

    def test_padding_masked(self, torch_collator, jax_collator):
        attn_mask = torch.tensor([[1, 1, 0]])
        torch_result = torch_collator.prepare_4d_attention_mask(attn_mask, torch.float32)
        jax_result = jax_collator.prepare_4d_attention_mask(attn_mask, torch.float32)

        assert torch.allclose(torch_result, jax_result)
        # Padding position should be masked
        assert torch_result[0, 0, 0, 2] < -1e30
        assert torch_result[0, 0, 2, 0] < -1e30

    def test_packed_sequence_cross_mask(self, torch_collator, jax_collator):
        """Different sequence indices should be masked from each other."""
        attn_mask = torch.tensor([[1, 1, 2, 2]])
        torch_result = torch_collator.prepare_4d_attention_mask(attn_mask, torch.float32)
        jax_result = jax_collator.prepare_4d_attention_mask(attn_mask, torch.float32)

        assert torch.allclose(torch_result, jax_result)
        # Cross-sequence: position from seq 2 attending to seq 1
        assert torch_result[0, 0, 2, 0] < -1e30
        # Same sequence: position 1 attending to position 0
        assert torch_result[0, 0, 1, 0] == 0.0

    def test_bfloat16_dtype(self, torch_collator, jax_collator):
        attn_mask = torch.tensor([[1, 1, 1]])
        torch_result = torch_collator.prepare_4d_attention_mask(attn_mask, torch.bfloat16)
        jax_result = jax_collator.prepare_4d_attention_mask(attn_mask, torch.bfloat16)

        assert torch_result.dtype == jax_result.dtype == torch.bfloat16
        assert torch.allclose(torch_result.float(), jax_result.float())


class TestMultiModalDataCollatorForSeq2Seq:
    def _make_collator(self, mod, pad_to_multiple_of=None):
        tokenizer = _make_mock_tokenizer()
        return mod.MultiModalDataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=-100,
        )

    def test_padding_to_max_length(self, torch_collator, jax_collator):
        tc = self._make_collator(torch_collator)
        jc = self._make_collator(jax_collator)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [40, 50]},
        ]

        t_batch = tc(features)
        j_batch = jc(features)

        assert torch.equal(t_batch["input_ids"], j_batch["input_ids"])
        assert torch.equal(t_batch["attention_mask"], j_batch["attention_mask"])
        assert torch.equal(t_batch["labels"], j_batch["labels"])

        # Check shapes
        assert t_batch["input_ids"].shape == (2, 3)
        assert t_batch["labels"][1, 2] == -100  # padded label

    def test_pad_to_multiple_of(self, torch_collator, jax_collator):
        tc = self._make_collator(torch_collator, pad_to_multiple_of=8)
        jc = self._make_collator(jax_collator, pad_to_multiple_of=8)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
        ]

        t_batch = tc(features)
        j_batch = jc(features)

        assert t_batch["input_ids"].shape[1] % 8 == 0
        assert j_batch["input_ids"].shape[1] % 8 == 0
        assert torch.equal(t_batch["input_ids"], j_batch["input_ids"])

    def test_images_extracted(self, torch_collator, jax_collator):
        tc = self._make_collator(torch_collator)
        jc = self._make_collator(jax_collator)

        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [10, 20],
             "_images": ["img1.jpg"]},
        ]

        t_batch = tc(features)
        j_batch = jc(features)

        # _images should be removed from the batch
        assert "_images" not in t_batch
        assert "_images" not in j_batch

    def test_token_type_ids_for_image_tokens(self, torch_collator, jax_collator):
        tc = self._make_collator(torch_collator)
        jc = self._make_collator(jax_collator)

        # Set up processor with image_token_id
        tc.processor = MagicMock()
        tc.processor.image_token_id = 99
        jc.processor = MagicMock()
        jc.processor.image_token_id = 99

        features = [
            {"input_ids": [1, 99, 99, 2], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, -100, 20]},
        ]

        t_batch = tc(features)
        j_batch = jc(features)

        assert "token_type_ids" in t_batch
        assert "token_type_ids" in j_batch
        assert torch.equal(t_batch["token_type_ids"], j_batch["token_type_ids"])
        # Image tokens should have type_id 1
        expected = torch.tensor([[0, 1, 1, 0]])
        assert torch.equal(t_batch["token_type_ids"], expected)


class TestSFTDataCollatorWith4DAttentionMask:
    def _make_collator(self, mod, block_diag=False):
        tokenizer = _make_mock_tokenizer()
        return mod.SFTDataCollatorWith4DAttentionMask(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            block_diag_attn=block_diag,
            compute_dtype=torch.float32,
        )

    def test_without_block_diag(self, torch_collator, jax_collator):
        tc = self._make_collator(torch_collator, block_diag=False)
        jc = self._make_collator(jax_collator, block_diag=False)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
        ]

        t_batch = tc(features)
        j_batch = jc(features)

        # Attention mask should remain 2D
        assert t_batch["attention_mask"].dim() == 2
        assert torch.equal(t_batch["attention_mask"], j_batch["attention_mask"])

    def test_with_block_diag(self, torch_collator, jax_collator):
        tc = self._make_collator(torch_collator, block_diag=True)
        jc = self._make_collator(jax_collator, block_diag=True)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
        ]

        t_batch = tc(features)
        j_batch = jc(features)

        # Attention mask should be 4D
        assert t_batch["attention_mask"].dim() == 4
        assert torch.allclose(t_batch["attention_mask"], j_batch["attention_mask"])
