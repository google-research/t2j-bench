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

"""Standalone DataCollatorForSeq2Seq - no transformers dependency."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class DataCollatorForSeq2Seq:
    """Data collator that dynamically pads inputs and labels for seq2seq models.

    This is a standalone reimplementation of transformers.DataCollatorForSeq2Seq.
    """

    tokenizer: Any  # Our Tokenizer class
    model: Any = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]], return_tensors: Optional[str] = None) -> dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name, None) for feature in features]
        if all(l is None for l in labels):
            labels = None

        # Determine max length
        max_len = max(len(f["input_ids"]) for f in features)
        if self.max_length is not None and self.padding:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        # Pad features
        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]

            if key in ("input_ids", "attention_mask", "token_type_ids"):
                padded = []
                for v in values:
                    if isinstance(v, list):
                        v = list(v)
                    else:
                        v = v.tolist() if hasattr(v, "tolist") else list(v)
                    pad_val = pad_token_id if key == "input_ids" else 0
                    pad_len = max_len - len(v)
                    if pad_len > 0 and self.padding:
                        if padding_side == "right":
                            v = v + [pad_val] * pad_len
                        else:
                            v = [pad_val] * pad_len + v
                    padded.append(v)

                if return_tensors == "pt":
                    batch[key] = torch.tensor(padded, dtype=torch.long)
                else:
                    batch[key] = np.array(padded, dtype=np.int64)
            else:
                # For non-standard keys, try to stack or keep as list
                try:
                    if return_tensors == "pt":
                        if isinstance(values[0], torch.Tensor):
                            batch[key] = torch.stack(values)
                        else:
                            batch[key] = torch.tensor(values)
                    else:
                        batch[key] = np.array(values)
                except (ValueError, RuntimeError):
                    batch[key] = values

        # Handle labels separately with label_pad_token_id
        if labels is not None:
            max_label_len = max_len
            padded_labels = []
            for label in labels:
                if isinstance(label, list):
                    label = list(label)
                else:
                    label = label.tolist() if hasattr(label, "tolist") else list(label)

                pad_len = max_label_len - len(label)
                if pad_len > 0 and self.padding:
                    if padding_side == "right":
                        label = label + [self.label_pad_token_id] * pad_len
                    else:
                        label = [self.label_pad_token_id] * pad_len + label
                padded_labels.append(label)

            if return_tensors == "pt":
                batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            else:
                batch["labels"] = np.array(padded_labels, dtype=np.int64)

        return batch

