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

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from .constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**accepted)


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

    def _plugin(self):
        return getattr(self.template, "mm_plugin", None)

    def _process_messages(self, messages, images, videos, audios):
        plugin = self._plugin()
        if plugin is None:
            return messages
        try:
            return _call_with_supported_kwargs(
                plugin.process_messages,
                messages=messages,
                images=images,
                videos=videos,
                audios=audios,
                processor=self.processor,
            )
        except TypeError:
            try:
                return plugin.process_messages(messages, images, videos, audios, self.processor)
            except TypeError:
                return plugin.process_messages(messages, images=images, videos=videos, processor=self.processor)

    def _process_token_ids(self, input_ids, labels, images, videos, audios):
        plugin = self._plugin()
        if plugin is None:
            return input_ids, labels
        try:
            return _call_with_supported_kwargs(
                plugin.process_token_ids,
                input_ids=input_ids,
                labels=labels,
                images=images,
                videos=videos,
                audios=audios,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        except TypeError:
            try:
                return plugin.process_token_ids(
                    input_ids, labels, images, videos, audios, self.tokenizer, self.processor
                )
            except TypeError:
                return plugin.process_token_ids(
                    input_ids, labels, images=images, tokenizer=self.tokenizer, processor=self.processor
                )

    def _image_token_id(self) -> Optional[int]:
        plugin = self._plugin()
        image_token = getattr(plugin, "image_token", None) if plugin is not None else None
        if image_token is None:
            return None
        token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        return None if token_id == getattr(self.tokenizer, "unk_token_id", None) else token_id

    def _reject_truncated_images(self, required_len: int) -> None:
        raise ValueError(
            f"cutoff_len={self.data_args.cutoff_len} is too small for this multimodal example: "
            f"truncation would drop image tokens, leaving the model's image-token count out of "
            f"step with its image features. Raise cutoff_len to at least {required_len}."
        )

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
        turns = self._process_messages(prompt + response, images, videos, audios)
        messages = self.template.build_chat_messages(
            turns[: len(prompt)], turns[len(prompt) :], system or "", tools or "", images, videos, audios
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, processor=self.processor)

        seed_ids, seed_labels = self._process_token_ids([], [], images, videos, audios)
        input_ids: list[int] = list(seed_ids)
        labels: list[int] = list(seed_labels or [])
        total_length = len(input_ids)

        image_token_id = self._image_token_id() if images else None
        required_len = total_length + sum(len(source) + len(target) for source, target in encoded_pairs) + 1
        if image_token_id is not None and total_length >= self.data_args.cutoff_len:
            self._reject_truncated_images(required_len)

        for source_ids, target_ids in encoded_pairs:
            if total_length >= self.data_args.cutoff_len:
                break
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length)
            if image_token_id is not None and image_token_id in source_ids[source_len:]:
                self._reject_truncated_images(required_len)
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

