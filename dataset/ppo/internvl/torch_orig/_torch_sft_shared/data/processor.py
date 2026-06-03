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

"""Generic supervised dataset processor."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from .constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / max(source_len + target_len, 1)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


@dataclass
class SupervisedDatasetProcessor:
    template: Any
    tokenizer: Any
    processor: Optional[Any]
    data_args: Any

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.build_chat_messages(prompt, response, system or "", tools or "", images, videos, audios)
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, processor=self.processor)
        input_ids: list[int] = []
        labels: list[int] = []
        total_length = 0

        for source_ids, target_ids in encoded_pairs:
            if total_length >= self.data_args.cutoff_len:
                break
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            source_labels = source_ids if self.data_args.train_on_prompt else [IGNORE_INDEX] * source_len
            input_ids.extend(source_ids + target_ids)
            labels.extend(source_labels + target_ids)

        if getattr(self.tokenizer, "eos_token_id", None) is not None and input_ids:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_response"][i]) != 1:
                logger.warning("Dropped invalid example at index %s", i)
                continue
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
        return model_inputs

