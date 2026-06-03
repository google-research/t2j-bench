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

"""Generic prompt-only dataset processor for PPO training."""

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .constants import IGNORE_INDEX

if TYPE_CHECKING:
    from .template import Template

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
    if source_len + target_len <= cutoff_len:
        return source_len, target_len
    if source_len > cutoff_len:
        return cutoff_len, 0
    return source_len, cutoff_len - source_len


def _common_prefix(left: list[int], right: list[int]) -> list[int]:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return list(left[:index])


@dataclass
class UnsupervisedDatasetProcessor:
    template: "Template"
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

    def _encode_oneturn(self, messages, system, tools):
        if hasattr(self.template, "encode_oneturn"):
            return self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        if hasattr(self.template, "encode_multiturn"):
            encoded = self.template.encode_multiturn(self.tokenizer, messages, system=system, tools=tools)
            if not encoded:
                return [], []
            prompt_ids = []
            for source_ids, target_ids in encoded[:-1]:
                prompt_ids.extend(source_ids)
                prompt_ids.extend(target_ids)
            last_source, last_target = encoded[-1]
            prompt_ids.extend(last_source)
            return prompt_ids, last_target
        raise AttributeError("Template must provide encode_oneturn() or encode_multiturn().")

    def _encode_generation_prompt(self, messages, system, tools):
        if hasattr(self.template, "encode_multiturn"):
            encoded = self.template.encode_multiturn(self.tokenizer, messages, system=system, tools=tools)
            if not encoded:
                return []
            prompt_ids = []
            for source_ids, target_ids in encoded[:-1]:
                prompt_ids.extend(source_ids)
                prompt_ids.extend(target_ids)
            last_source, last_target = encoded[-1]
            prompt_ids.extend(last_source)
            prompt_ids.extend(last_target)
            return prompt_ids
        prompt_ids, _ = self._encode_oneturn(messages, system, tools)
        return prompt_ids

    def _encode_target_labels(self, messages, system, tools):
        if hasattr(self.template, "encode_multiturn"):
            encoded = self.template.encode_multiturn(self.tokenizer, messages, system=system, tools=tools)
            if not encoded:
                return []
            return list(encoded[-1][1])
        _, target_ids = self._encode_oneturn(messages, system, tools)
        return target_ids

    def _build_chat_messages(self, prompt, response, system, tools, images, videos, audios):
        builder = getattr(self.template, "build_chat_messages", None)
        if callable(builder):
            return builder(prompt, response, system or "", tools or "", images, videos, audios)
        return None

    def _encode_data_example(self, prompt, response, system, tools, images, videos, audios):
        messages = self._build_chat_messages(
            prompt,
            response if len(response) == 1 else [{"role": "assistant", "content": ""}],
            system,
            tools,
            images,
            videos,
            audios,
        )

        if messages is not None and hasattr(self.template, "encode_multiturn"):
            encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, processor=self.processor)
            input_ids = []
            labels = []
            total_length = 0

            for source_ids, target_ids in encoded_pairs:
                if total_length >= self.data_args.cutoff_len:
                    break
                source_len, target_len = infer_seqlen(
                    len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
                )
                source_ids = source_ids[:source_len]
                target_ids = target_ids[:target_len]
                total_length += source_len + target_len

                input_ids.extend(source_ids + target_ids)
                labels.extend([IGNORE_INDEX] * source_len + target_ids)

            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and input_ids:
                input_ids.append(eos_token_id)
                labels.append(eos_token_id)
            return input_ids, labels

        # Fallback path for templates that do not expose the higher-level
        # SFT chat-building helpers.
        prompt_input_ids = None
        target_labels = None
        prompt_messages = prompt + [{"role": "assistant", "content": ""}]
        prompt_messages = self._process_messages(prompt_messages, images, videos, audios)
        prompt_input_ids = self._encode_generation_prompt(prompt_messages, system, tools)

        if len(response) == 1:
            label_messages = prompt + response
        else:
            label_messages = prompt_messages
        label_messages = self._process_messages(label_messages, images, videos, audios)
        target_labels = self._encode_target_labels(label_messages, system, tools)
        prompt_input_ids, target_labels = self._process_token_ids(prompt_input_ids, target_labels, images, videos, audios)

        if getattr(self.template, "efficient_eos", False):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                target_labels += [eos_token_id]

        input_ids = prompt_input_ids
        labels = target_labels
        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i] or []
            response = examples["_response"][i] or []
            if len(prompt) % 2 != 1:
                logger.warning("Dropped invalid example: %s", prompt + response)
                continue

            images = examples.get("_images", [None])[i] or []
            videos = examples.get("_videos", [None])[i] or []
            audios = examples.get("_audios", [None])[i] or []
            input_ids, labels = self._encode_data_example(
                prompt=prompt,
                response=response,
                system=examples.get("_system", [""])[i],
                tools=examples.get("_tools", [""])[i],
                images=images,
                videos=videos,
                audios=audios,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(images if images else None)
            model_inputs["videos"].append(videos if videos else None)
            model_inputs["audios"].append(audios if audios else None)
        return model_inputs
