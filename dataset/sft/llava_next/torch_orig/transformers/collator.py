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

"""Data collator for LLaVA-Next sequence-to-sequence tasks."""
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class DataCollatorForSeq2Seq:
    """Minimal data collator with padding."""

    def __init__(
        self,
        tokenizer=None,
        model=None,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features):
        # Find max length
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0

        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for f in features:
            input_ids = f["input_ids"]
            labels = f.get("labels", f["input_ids"])
            attention_mask = f.get("attention_mask", [1] * len(input_ids))

            pad_len = max_len - len(input_ids)
            batch["input_ids"].append(input_ids + [pad_token_id] * pad_len)
            batch["attention_mask"].append(attention_mask + [0] * pad_len)
            batch["labels"].append(labels + [self.label_pad_token_id] * pad_len)

        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch
