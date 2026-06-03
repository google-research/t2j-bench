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

"""Data collators for LLaVA-Next SFT training with multimodal support."""
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import torch

from .constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def prepare_4d_attention_mask(attention_mask_with_indices, dtype):
    """Prepare 4D attention mask from 2D attention mask with packing indices."""
    bsz, seq_len = attention_mask_with_indices.shape
    min_value = torch.finfo(dtype).min

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_mask_with_indices.device), diagonal=1)
    causal_mask = causal_mask.bool()

    # For packed sequences, mask across different sequences
    indices_expanded_row = attention_mask_with_indices[:, None, :, None].expand(bsz, 1, seq_len, seq_len)
    indices_expanded_col = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    cross_seq_mask = indices_expanded_row != indices_expanded_col

    # Combine masks
    final_mask = torch.zeros(bsz, 1, seq_len, seq_len, device=attention_mask_with_indices.device, dtype=dtype)
    final_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), min_value)
    final_mask.masked_fill_(cross_seq_mask, min_value)

    # Mask padding positions
    padding_mask = (attention_mask_with_indices == 0)
    final_mask.masked_fill_(padding_mask[:, None, None, :], min_value)
    final_mask.masked_fill_(padding_mask[:, None, :, None], min_value)

    return final_mask


@dataclass
class MultiModalDataCollatorForSeq2Seq:
    """Data collator that handles multimodal inputs (images)."""
    template: Any = None
    model: Any = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX
    tokenizer: Any = None
    processor: Any = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Extract images from features
        images = []
        for feature in features:
            if "_images" in feature:
                feature_images = feature.pop("_images")
                if feature_images:
                    images.extend(feature_images)

            # Also remove video/audio placeholders
            feature.pop("_videos", None)
            feature.pop("_audios", None)

        # Pad sequences
        batch = self._pad_features(features)

        # Process multimodal inputs
        if images and self.template and hasattr(self.template, "mm_plugin"):
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                images=images, processor=self.processor,
            )
            batch.update(mm_inputs)

        return batch

    def _pad_features(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad features to the same length."""
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())

        skip_keys = {"_images", "_videos", "_audios"}
        tensor_keys = all_keys - skip_keys

        batch = {}
        for key in tensor_keys:
            values = [f.get(key) for f in features if key in f]
            if not values:
                continue

            if isinstance(values[0], list):
                max_len = max(len(v) for v in values)
                if self.pad_to_multiple_of is not None:
                    max_len = (
                        (max_len + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                if key == "labels":
                    pad_value = self.label_pad_token_id
                elif key == "attention_mask":
                    pad_value = 0
                elif key == "input_ids":
                    pad_value = self.tokenizer.pad_token_id if self.tokenizer else 0
                else:
                    pad_value = 0

                padded = []
                for v in values:
                    remainder = [pad_value] * (max_len - len(v))
                    padded.append(v + remainder)
                batch[key] = torch.tensor(padded, dtype=torch.long)
            elif isinstance(values[0], torch.Tensor):
                if all(v.shape == values[0].shape for v in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
            else:
                batch[key] = values

        return batch


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    """SFT data collator with optional 4D attention mask support."""
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: torch.dtype = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)

        if self.block_diag_attn and "attention_mask" in batch:
            if self.attn_implementation != "flash_attention_2":
                batch["attention_mask"] = prepare_4d_attention_mask(
                    batch["attention_mask"], self.compute_dtype
                )

        return batch
