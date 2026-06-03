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

"""Generic pairwise dataset processor for DPO training."""

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
class PairwiseDatasetProcessor:
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

    def _process_prompt_token_ids(self, prompt_ids, images, videos, audios):
        plugin = self._plugin()
        if plugin is None:
            return prompt_ids
        try:
            prompt_ids, _ = _call_with_supported_kwargs(
                plugin.process_token_ids,
                input_ids=prompt_ids,
                labels=None,
                images=images,
                videos=videos,
                audios=audios,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        except TypeError:
            try:
                prompt_ids, _ = plugin.process_token_ids(
                    prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
                )
            except TypeError:
                prompt_ids, _ = plugin.process_token_ids(
                    prompt_ids, None, images=images, tokenizer=self.tokenizer, processor=self.processor
                )
        return prompt_ids

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

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list,
        videos: list,
        audios: list,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        chosen_messages = self._process_messages(prompt + [response[0]], images, videos, audios)
        rejected_messages = self._process_messages(prompt + [response[1]], images, videos, audios)

        prompt_ids, chosen_ids = self._encode_oneturn(chosen_messages, system, tools)
        _, rejected_ids = self._encode_oneturn(rejected_messages, system, tools)

        if getattr(self.template, "efficient_eos", False):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                chosen_ids += [eos_token_id]
                rejected_ids += [eos_token_id]

        prompt_ids = self._process_prompt_token_ids(prompt_ids, images, videos, audios)

        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i] or []
            response = examples["_response"][i] or []
            if len(prompt) % 2 != 1 or len(response) < 2:
                logger.warning("Dropped invalid example: %s", prompt + response)
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=prompt,
                response=response,
                system=examples.get("_system", [""])[i],
                tools=examples.get("_tools", [""])[i],
                images=(examples.get("_images", [None])[i] or []),
                videos=(examples.get("_videos", [None])[i] or []),
                audios=(examples.get("_audios", [None])[i] or []),
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples.get("_images", [None])[i])
            model_inputs["videos"].append(examples.get("_videos", [None])[i])
            model_inputs["audios"].append(examples.get("_audios", [None])[i])
        return model_inputs
