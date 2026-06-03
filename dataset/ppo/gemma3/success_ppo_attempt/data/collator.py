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

"""Data collator for Gemma3 PPO with multimodal support."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER

logger = logging.getLogger(__name__)


@dataclass
class MultiModalDataCollatorForSeq2Seq:
    """Data collator that handles multimodal inputs (images) for Gemma3 PPO."""
    template: Any = None
    model: Any = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX
    tokenizer: Any = None
    processor: Any = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
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

            input_ids = feature.get("input_ids")
            labels = feature.get("labels")
            if input_ids is not None and (labels is None or len(labels) != len(input_ids)):
                feature["labels"] = [self.label_pad_token_id] * len(input_ids)

        # Pad sequences
        batch = self._pad_features(features)

        # Process multimodal inputs
        if images and self.template and hasattr(self.template, "mm_plugin"):
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                images=images, processor=self.processor,
            )
            batch.update(mm_inputs)

        # Compute token_type_ids for Gemma3
        image_token_id = getattr(self.processor, "image_token_id",
                                 getattr(self.tokenizer, "image_token_id", None))
        if image_token_id is not None and "input_ids" in batch:
            token_type_ids = torch.zeros_like(batch["input_ids"])
            token_type_ids[batch["input_ids"] == image_token_id] = 1
            batch["token_type_ids"] = token_type_ids

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
                    # Left-pad for generation (PPO uses left-padding)
                    if self.tokenizer and getattr(self.tokenizer, "padding_side", "right") == "left":
                        padded.append(remainder + v)
                    else:
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
