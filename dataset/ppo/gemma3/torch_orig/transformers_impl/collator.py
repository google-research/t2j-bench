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

"""Standalone DataCollatorForSeq2Seq implementation."""
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class DataCollatorForSeq2Seq:
    """Data collator for sequence-to-sequence models with padding."""
    tokenizer: Any = None
    model: Any = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Separate labels
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None

        # Pad input sequences
        batch = self._pad(features)

        # Pad labels
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            padded_labels = []
            for label in labels:
                remainder = [self.label_pad_token_id] * (max_label_length - len(label))
                padded_labels.append(label + remainder)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch

    def _pad(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad features to the same length."""
        first = features[0]
        batch = {}

        for key in first:
            if key == "labels":
                continue
            values = [f[key] for f in features]
            if isinstance(values[0], list):
                max_length = max(len(v) for v in values)
                if self.pad_to_multiple_of is not None:
                    max_length = (
                        (max_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )
                pad_value = 0
                if key == "attention_mask":
                    pad_value = 0
                elif key == "input_ids":
                    pad_value = self.tokenizer.pad_token_id if self.tokenizer else 0

                padded = []
                for v in values:
                    remainder = [pad_value] * (max_length - len(v))
                    padded.append(v + remainder)
                batch[key] = torch.tensor(padded, dtype=torch.long)
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values

        return batch
