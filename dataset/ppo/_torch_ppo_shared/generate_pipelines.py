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

"""Generate PPO torch_orig repos by extending the validated SFT repos."""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "sft"
PPO_ROOT = ROOT / "ppo"
EVAL_ROOT = ROOT / "eval"
SFT_CONFIG_ROOT = EVAL_ROOT / "generated_sft_reference_configs"
SFT_SHARED_ROOT = SFT_ROOT / "_torch_sft_shared"
PPO_TEMPLATE_ROOT = PPO_ROOT / "gemma3" / "torch_orig"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "output", "outputs")


PPO_TRAIN_TEXT = textwrap.dedent(
    """
    \"\"\"Main entry point for PPO training.\"\"\"

    import logging
    from typing import Optional

    from config import get_train_args
    from data.collator import MultiModalDataCollatorForSeq2Seq
    from data.loader import get_dataset
    from data.template import get_template_and_fix_tokenizer
    from model.loader import load_model, load_tokenizer
    from ppo.trainer import StandalonePPOTrainer


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)


    def run_ppo(config: Optional[dict] = None):
        \"\"\"Run PPO training.\"\"\"
        model_args, data_args, training_args, finetuning_args, ppo_args, generating_args = get_train_args(config)

        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        processor = tokenizer_module.get("processor")
        template = get_template_and_fix_tokenizer(tokenizer, data_args)

        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
            processor=processor,
        )

        model = load_model(
            tokenizer=tokenizer,
            model_args=model_args,
            finetuning_args=finetuning_args,
            is_trainable=training_args.do_train,
            add_valuehead=True,
        )

        data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            model=model,
            pad_to_multiple_of=training_args.pad_to_multiple_of if training_args.do_train else None,
            label_pad_token_id=-100,
            tokenizer=tokenizer,
            processor=processor,
            compute_dtype=model_args.compute_dtype,
        )

        trainer = StandalonePPOTrainer(
            model=model,
            ref_model=None,
            reward_model=ppo_args.reward_model,
            tokenizer=tokenizer,
            processor=processor,
            data_collator=data_collator,
            train_dataset=dataset_module["train_dataset"],
            training_args=training_args,
            ppo_args=ppo_args,
            generating_args=generating_args,
            finetuning_args=finetuning_args,
        )

        if training_args.do_train:
            logger.info("*** Starting PPO training ***")
            trainer.ppo_train()
            trainer.save_model()
            logger.info("*** PPO training complete ***")


    if __name__ == "__main__":
        run_ppo()
    """
).lstrip()


UNSUPERVISED_PROCESSOR_TEXT = textwrap.dedent(
    """
    \"\"\"Generic prompt-only dataset processor for PPO training.\"\"\"

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
    """
).lstrip()


PPO_COLLATOR_TEXT = textwrap.dedent(
    """
    \"\"\"Generic multimodal collator for PPO training.\"\"\"

    from dataclasses import dataclass
    import inspect
    from typing import TYPE_CHECKING, Any, Optional

    import torch

    from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER

    if TYPE_CHECKING:
        from .template import Template


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


    @dataclass
    class MultiModalDataCollatorForSeq2Seq:
        template: Optional["Template"] = None
        model: Any = None
        pad_to_multiple_of: Optional[int] = None
        label_pad_token_id: int = IGNORE_INDEX
        tokenizer: Any = None
        processor: Any = None
        compute_dtype: "torch.dtype" = torch.float32

        def __post_init__(self):
            if self.template is None:
                raise ValueError("Template is required for MultiModalDataCollator.")
            actual_model = self.model
            if actual_model is not None and hasattr(actual_model, "base_model") and hasattr(actual_model, "peft_config"):
                actual_model = actual_model.base_model.model
            if actual_model is not None and hasattr(actual_model, "pretrained_model"):
                actual_model = actual_model.pretrained_model
                if hasattr(actual_model, "base_model") and hasattr(actual_model, "peft_config"):
                    actual_model = actual_model.base_model.model
            self.get_rope_func = None
            if actual_model is not None:
                if hasattr(actual_model, "get_rope_index"):
                    self.get_rope_func = actual_model.get_rope_index
                elif hasattr(actual_model, "model") and hasattr(actual_model.model, "get_rope_index"):
                    self.get_rope_func = actual_model.model.get_rope_index

        def _pad_features(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            all_keys = set()
            for feature in features:
                all_keys.update(feature.keys())
            skip_keys = {"images", "videos", "audios"}
            tensor_keys = all_keys - skip_keys
            batch = {}
            for key in tensor_keys:
                values = [feature.get(key) for feature in features if key in feature]
                if not values:
                    continue
                if isinstance(values[0], list):
                    max_len = max(len(value) for value in values)
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
                        pad_value = getattr(self.tokenizer, "pad_token_id", 0) or 0
                    elif key == "token_type_ids":
                        pad_value = 0
                    else:
                        pad_value = 0
                    padded = []
                    for value in values:
                        remainder = [pad_value] * (max_len - len(value))
                        if getattr(self.tokenizer, "padding_side", "right") == "left":
                            padded.append(remainder + value)
                        else:
                            padded.append(value + remainder)
                    batch[key] = torch.tensor(padded, dtype=torch.long)
                elif isinstance(values[0], torch.Tensor):
                    if all(tuple(value.shape) == tuple(values[0].shape) for value in values):
                        batch[key] = torch.stack(values)
                    else:
                        batch[key] = values
                else:
                    batch[key] = values
            return batch

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
            features = [dict(feature) for feature in features]
            batch_images, batch_videos, batch_audios = [], [], []
            batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
            for feature in features:
                images = feature.pop("images", None) or []
                videos = feature.pop("videos", None) or []
                audios = feature.pop("audios", None) or []
                batch_images.extend(images)
                batch_videos.extend(videos)
                batch_audios.extend(audios)
                batch_imglens.append(len(images))
                batch_vidlens.append(len(videos))
                batch_audlens.append(len(audios))
                batch_input_ids.append(feature["input_ids"])

            plugin = getattr(self.template, "mm_plugin", None)
            fake_input_ids = []
            if (
                plugin is not None
                and getattr(plugin, "image_token", None) is not None
                and sum(batch_imglens) == 0
                and sum(batch_vidlens) == 0
            ):
                from PIL import Image
                fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
                fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
                try:
                    fake_messages = _call_with_supported_kwargs(
                        plugin.process_messages,
                        messages=fake_messages,
                        images=fake_images,
                        videos=[],
                        audios=[],
                        processor=self.processor,
                    )
                except TypeError:
                    fake_messages = plugin.process_messages(
                        fake_messages, images=fake_images, videos=[], processor=self.processor
                    )
                fake_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
                try:
                    fake_ids, _ = _call_with_supported_kwargs(
                        plugin.process_token_ids,
                        input_ids=fake_ids,
                        labels=None,
                        images=fake_images,
                        videos=[],
                        audios=[],
                        tokenizer=self.tokenizer,
                        processor=self.processor,
                    )
                except TypeError:
                    fake_ids, _ = plugin.process_token_ids(
                        fake_ids, None, images=fake_images, tokenizer=self.tokenizer, processor=self.processor
                    )
                fake_input_ids.extend(fake_ids)
                batch_images = fake_images
                batch_imglens[0] = 1

            if fake_input_ids:
                if getattr(self.tokenizer, "padding_side", "right") == "right":
                    features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                    features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                    features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
                else:
                    features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                    features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                    features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]
                batch_input_ids[0] = features[0]["input_ids"]

            mm_inputs = {}
            if plugin is not None:
                mm_inputs = _call_with_supported_kwargs(
                    plugin.get_mm_inputs,
                    images=batch_images,
                    videos=batch_videos,
                    audios=batch_audios,
                    batch_images=batch_images,
                    batch_videos=batch_videos,
                    batch_audios=batch_audios,
                    imglens=batch_imglens,
                    vidlens=batch_vidlens,
                    audlens=batch_audlens,
                    batch_imglens=batch_imglens,
                    batch_vidlens=batch_vidlens,
                    batch_audlens=batch_audlens,
                    batch_ids=batch_input_ids,
                    batch_input_ids=batch_input_ids,
                    processor=self.processor,
                ) or {}

            if "token_type_ids" in mm_inputs:
                token_type_ids = mm_inputs.pop("token_type_ids")
                for i, feature in enumerate(features):
                    feature["token_type_ids"] = token_type_ids[i]

            batch = self._pad_features(features)
            if "token_type_ids" not in batch:
                multimodal_token_ids = []
                for name in ("image_token_id", "video_token_id"):
                    token_id = getattr(self.processor, name, getattr(self.tokenizer, name, None))
                    if token_id is not None:
                        multimodal_token_ids.append(token_id)
                if multimodal_token_ids and "input_ids" in batch:
                    token_type_ids = torch.zeros_like(batch["input_ids"])
                    for token_id in multimodal_token_ids:
                        token_type_ids[batch["input_ids"] == token_id] = 1
                    if torch.any(token_type_ids != 0):
                        batch["token_type_ids"] = token_type_ids

            if self.get_rope_func is not None:
                rope_kwargs = {
                    "input_ids": batch["input_ids"],
                    "image_grid_thw": mm_inputs.get("image_grid_thw"),
                    "video_grid_thw": mm_inputs.get("video_grid_thw"),
                    "attention_mask": (batch["attention_mask"] >= 1).float(),
                }
                if "second_per_grid_ts" in mm_inputs:
                    rope_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
                try:
                    position_ids, rope_deltas = self.get_rope_func(**rope_kwargs)
                    batch["position_ids"] = position_ids
                    batch["rope_deltas"] = rope_deltas
                except TypeError:
                    pass
            batch.update(mm_inputs)

            if self.compute_dtype is not None:
                for key, value in batch.items():
                    if torch.is_tensor(value) and torch.is_floating_point(value):
                        batch[key] = value.to(self.compute_dtype)
            return batch
    """
).lstrip()


UNSUPERVISED_LOADER_TEXT = textwrap.dedent(
    """
    \"\"\"Generic dataset loading and preprocessing for PPO.\"\"\"

    import json
    import logging
    import os
    from dataclasses import dataclass
    from typing import Any, Literal, Optional

    import numpy as np
    from datasets import concatenate_datasets, load_dataset

    from .constants import DATA_CONFIG, FILEEXT2TYPE
    from .processor import UnsupervisedDatasetProcessor

    logger = logging.getLogger(__name__)


    @dataclass
    class DatasetAttr:
        load_from: Literal["hf_hub", "file"] = "file"
        dataset_name: str = ""
        formatting: Literal["alpaca", "sharegpt"] = "alpaca"
        subset: Optional[str] = None
        split: str = "train"
        folder: Optional[str] = None
        num_samples: Optional[int] = None
        system: Optional[str] = None
        tools: Optional[str] = None
        images: Optional[str] = None
        videos: Optional[str] = None
        audios: Optional[str] = None
        prompt: Optional[str] = "instruction"
        query: Optional[str] = "input"
        response: Optional[str] = "output"
        history: Optional[str] = None
        messages: Optional[str] = "conversations"
        role_tag: Optional[str] = "from"
        content_tag: Optional[str] = "value"
        user_tag: Optional[str] = "human"
        assistant_tag: Optional[str] = "gpt"
        observation_tag: Optional[str] = "observation"
        function_tag: Optional[str] = "function_call"
        system_tag: Optional[str] = "system"

        def set_attr(self, key: str, obj: dict, default=None):
            setattr(self, key, obj.get(key, default))

        def join(self, attr: dict):
            self.set_attr("formatting", attr, default="alpaca")
            self.set_attr("subset", attr)
            self.set_attr("split", attr, default="train")
            self.set_attr("folder", attr)
            self.set_attr("num_samples", attr)
            if "columns" in attr:
                for col in [
                    "prompt", "query", "response", "history", "messages",
                    "system", "tools", "images", "videos", "audios",
                ]:
                    self.set_attr(col, attr["columns"])
            if "tags" in attr:
                for tag in [
                    "role_tag", "content_tag", "user_tag", "assistant_tag",
                    "observation_tag", "function_tag", "system_tag",
                ]:
                    self.set_attr(tag, attr["tags"])


    def get_dataset_list(dataset_names: list[str], dataset_dir: str) -> list[DatasetAttr]:
        config_path = os.path.join(dataset_dir, DATA_CONFIG)
        dataset_info = {}
        if os.path.isfile(config_path):
            with open(config_path) as handle:
                dataset_info = json.load(handle)
        result = []
        for name in dataset_names:
            if name not in dataset_info:
                result.append(DatasetAttr(load_from="hf_hub", dataset_name=name))
                continue
            info = dataset_info[name]
            if "hf_hub_url" in info:
                attr = DatasetAttr(load_from="hf_hub", dataset_name=info["hf_hub_url"])
            else:
                attr = DatasetAttr(load_from="file", dataset_name=info["file_name"])
            attr.join(info)
            result.append(attr)
        return result


    def _find_medias(medias, dataset_attr, data_args):
        if medias is None:
            return None
        if not isinstance(medias, list):
            medias = [medias]
        if len(medias) == 0:
            return None
        medias = medias[:]
        if dataset_attr.load_from == "file" and isinstance(medias[0], str):
            media_dir = getattr(data_args, "media_dir", getattr(data_args, "dataset_dir", "data"))
            for i in range(len(medias)):
                candidate = os.path.join(media_dir, medias[i])
                if os.path.isfile(candidate):
                    medias[i] = candidate
        return medias


    def _convert_alpaca(example: dict, dataset_attr: DatasetAttr, data_args) -> dict:
        prompt = []
        if dataset_attr.history and isinstance(example.get(dataset_attr.history), list):
            for old_p, old_r in example[dataset_attr.history]:
                prompt.append({"role": "user", "content": old_p})
                prompt.append({"role": "assistant", "content": old_r})

        query_parts = []
        if dataset_attr.prompt and example.get(dataset_attr.prompt):
            query_parts.append(example[dataset_attr.prompt])
        if dataset_attr.query and example.get(dataset_attr.query):
            query_parts.append(example[dataset_attr.query])
        prompt.append({"role": "user", "content": "\\n".join(query_parts)})

        response = []
        if dataset_attr.response and isinstance(example.get(dataset_attr.response), str):
            response = [{"role": "assistant", "content": example[dataset_attr.response]}]

        return {
            "_prompt": prompt,
            "_response": response,
            "_system": example.get(dataset_attr.system, "") if dataset_attr.system else "",
            "_tools": example.get(dataset_attr.tools, "") if dataset_attr.tools else "",
            "_images": _find_medias(example.get(dataset_attr.images) if dataset_attr.images else None, dataset_attr, data_args),
            "_videos": _find_medias(example.get(dataset_attr.videos) if dataset_attr.videos else None, dataset_attr, data_args),
            "_audios": _find_medias(example.get(dataset_attr.audios) if dataset_attr.audios else None, dataset_attr, data_args),
        }


    def _convert_sharegpt(example: dict, dataset_attr: DatasetAttr, data_args) -> dict:
        tag_mapping = {
            dataset_attr.user_tag: "user",
            dataset_attr.assistant_tag: "assistant",
            dataset_attr.observation_tag: "observation",
            dataset_attr.function_tag: "function",
            dataset_attr.system_tag: "system",
        }
        odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
        even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        messages = example[dataset_attr.messages]
        system = ""
        if dataset_attr.system_tag and len(messages) > 0 and messages[0].get(dataset_attr.role_tag) == dataset_attr.system_tag:
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        elif dataset_attr.system and example.get(dataset_attr.system):
            system = example[dataset_attr.system]

        aligned = []
        broken = False
        for turn_idx, msg in enumerate(messages):
            if msg[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in %s", messages)
                broken = True
                break
            aligned.append({"role": tag_mapping[msg[dataset_attr.role_tag]], "content": msg[dataset_attr.content_tag]})

        if len(aligned) % 2 == 0:
            prompt = aligned
            response = []
        elif broken:
            prompt, response = [], []
        else:
            prompt = aligned[:-1]
            response = aligned[-1:]

        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example.get(dataset_attr.tools, "") if dataset_attr.tools else "",
            "_images": _find_medias(example.get(dataset_attr.images) if dataset_attr.images else None, dataset_attr, data_args),
            "_videos": _find_medias(example.get(dataset_attr.videos) if dataset_attr.videos else None, dataset_attr, data_args),
            "_audios": _find_medias(example.get(dataset_attr.audios) if dataset_attr.audios else None, dataset_attr, data_args),
        }


    def _load_single_dataset(dataset_attr: DatasetAttr, data_args):
        if dataset_attr.load_from == "hf_hub":
            return load_dataset(
                path=dataset_attr.dataset_name,
                name=dataset_attr.subset,
                data_dir=dataset_attr.folder,
                split=dataset_attr.split,
                cache_dir=None,
            )

        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_files = []
        if os.path.isdir(local_path):
            for fn in os.listdir(local_path):
                data_files.append(os.path.join(local_path, fn))
        elif os.path.isfile(local_path):
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        file_type = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if file_type is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
        return load_dataset(path=file_type, data_files=data_files, split=dataset_attr.split)


    def get_dataset(template, model_args, data_args, training_args, tokenizer, processor=None) -> dict:
        if data_args.dataset is None:
            raise ValueError("No dataset specified.")
        dataset_names = [name.strip() for name in data_args.dataset.split(",")]
        dataset_list = get_dataset_list(dataset_names, data_args.dataset_dir)
        all_datasets = []
        for dataset_attr in dataset_list:
            dataset = _load_single_dataset(dataset_attr, data_args)
            if dataset_attr.num_samples is not None:
                target = dataset_attr.num_samples
                idxs = np.random.permutation(len(dataset))[:target]
                if len(idxs) < target:
                    idxs = np.concatenate([idxs, np.random.choice(len(dataset), target - len(idxs))])
                dataset = dataset.select(idxs)
            converter = _convert_alpaca if dataset_attr.formatting == "alpaca" else _convert_sharegpt
            column_names = list(next(iter(dataset)).keys())
            dataset = dataset.map(
                lambda ex: converter(ex, dataset_attr, data_args),
                batched=False,
                remove_columns=column_names,
                num_proc=getattr(data_args, "preprocessing_num_workers", 1),
                desc="Converting format of dataset",
            )
            all_datasets.append(dataset)

        dataset = all_datasets[0] if len(all_datasets) == 1 else concatenate_datasets(all_datasets)
        if getattr(data_args, "max_samples", None) is not None:
            dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))
        eval_dataset = None
        if data_args.val_size > 0:
            split = dataset.train_test_split(test_size=data_args.val_size, seed=training_args.seed)
            dataset = split["train"]
            eval_dataset = split["test"]

        proc = UnsupervisedDatasetProcessor(
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        column_names = list(next(iter(dataset)).keys())
        kwargs = dict(
            num_proc=getattr(data_args, "preprocessing_num_workers", 1),
            load_from_cache_file=not getattr(data_args, "overwrite_cache", False),
            desc="Running tokenizer on dataset",
        )
        dataset = dataset.map(proc.preprocess_dataset, batched=True, batch_size=1000, remove_columns=column_names, **kwargs)

        result = {"train_dataset": dataset}
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(proc.preprocess_dataset, batched=True, batch_size=1000, remove_columns=column_names, **kwargs)
            result["eval_dataset"] = eval_dataset
        return result
    """
).lstrip()


SHARED_BASE_PPO_CONFIG = textwrap.dedent(
    """
    from pathlib import Path
    import importlib.util
    import sys
    from dataclasses import asdict, dataclass
    from typing import Any, Literal, Optional

    _REPO_ROOT = Path(__file__).resolve().parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    _SFT_ROOT = Path(__file__).resolve().parents[2]
    if str(_SFT_ROOT) not in sys.path:
        sys.path.insert(0, str(_SFT_ROOT))

    _MODEL_META_PATH = _REPO_ROOT / "model_meta.py"
    _MODEL_META_SPEC = importlib.util.spec_from_file_location(f"{__name__}_model_meta", _MODEL_META_PATH)
    _MODEL_META = importlib.util.module_from_spec(_MODEL_META_SPEC)
    _MODEL_META_SPEC.loader.exec_module(_MODEL_META)

    from _torch_sft_shared.config_base import _populate_dataclass, build_get_train_args, read_args

    DEFAULT_TEMPLATE = _MODEL_META.DEFAULT_TEMPLATE
    get_train_args = build_get_train_args(DEFAULT_TEMPLATE)
    _SFT_GET_TRAIN_ARGS = get_train_args


    @dataclass
    class PPOArguments:
        ppo_epochs: int = 4
        ppo_buffer_size: int = 1
        cliprange: float = 0.2
        cliprange_value: float = 0.2
        vf_coef: float = 0.1
        gamma: float = 1.0
        lam: float = 0.95
        kl_coef: float = 0.05
        ppo_target: float = 6.0
        ppo_score_norm: bool = False
        ppo_whiten_rewards: bool = False
        reward_model: Optional[str] = None
        reward_model_type: Literal["api", "full"] = "api"
        ref_model: Optional[str] = None


    @dataclass
    class GeneratingArguments:
        do_sample: bool = True
        temperature: float = 0.95
        top_p: float = 0.7
        top_k: int = 50
        max_new_tokens: int = 256
        repetition_penalty: float = 1.0

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)


    def get_train_args(args: Optional[dict[str, Any]] = None):
        config = read_args(args)
        if isinstance(config, list):
            raise ValueError("Please provide a YAML or JSON configuration file.")

        model_args, data_args, training_args, finetuning_args = _SFT_GET_TRAIN_ARGS(config)
        ppo_args = _populate_dataclass(PPOArguments, config)
        generating_args = _populate_dataclass(GeneratingArguments, config)
        return model_args, data_args, training_args, finetuning_args, ppo_args, generating_args
    """
).lstrip()


PPO_CONFIG_APPEND = textwrap.dedent(
    """


    _SFT_GET_TRAIN_ARGS = get_train_args


    @dataclass
    class PPOArguments:
        ppo_epochs: int = 4
        ppo_buffer_size: int = 1
        cliprange: float = 0.2
        cliprange_value: float = 0.2
        vf_coef: float = 0.1
        gamma: float = 1.0
        lam: float = 0.95
        kl_coef: float = 0.05
        ppo_target: float = 6.0
        ppo_score_norm: bool = False
        ppo_whiten_rewards: bool = False
        reward_model: Optional[str] = None
        reward_model_type: Literal["api", "full"] = "api"
        ref_model: Optional[str] = None


    @dataclass
    class GeneratingArguments:
        do_sample: bool = True
        temperature: float = 0.95
        top_p: float = 0.7
        top_k: int = 50
        max_new_tokens: int = 256
        repetition_penalty: float = 1.0

        def to_dict(self) -> dict[str, Any]:
            from dataclasses import asdict
            return asdict(self)


    def get_train_args(args: Optional[dict[str, Any]] = None):
        \"\"\"Parse and return all PPO training arguments from YAML/CLI.\"\"\"
        config = read_args(args)
        if isinstance(config, list):
            raise ValueError("Please provide a YAML or JSON configuration file.")

        model_args, data_args, training_args, finetuning_args = _SFT_GET_TRAIN_ARGS(config)
        ppo_args = _populate_dataclass(PPOArguments, config)
        generating_args = _populate_dataclass(GeneratingArguments, config)
        return model_args, data_args, training_args, finetuning_args, ppo_args, generating_args
    """
).strip()


PPO_LOAD_MODEL_APPEND = textwrap.dedent(
    """


    def _ppo_call_with_supported_kwargs(fn, **kwargs):
        import inspect

        signature = inspect.signature(fn)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return fn(**kwargs)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return fn(**accepted)


    def _ppo_value_head_hidden_size(model):
        config = getattr(model, "config", None)
        if config is None:
            return None
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            hidden_size = getattr(text_config, "hidden_size", None)
            if hidden_size is not None:
                return hidden_size
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_config = get_text_config(decoder=True)
            except TypeError:
                text_config = get_text_config()
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
                if hidden_size is not None:
                    return hidden_size
        return None


    def _ppo_find_lm_head(model):
        direct = getattr(model, "lm_head", None)
        if direct is not None:
            return direct
        for name, module in model.named_modules():
            if name.endswith("lm_head"):
                return module
        return None


    def _ppo_output_has_values(output):
        value = None
        if isinstance(output, dict):
            value = output.get("values")
        else:
            value = getattr(output, "values", None)
        return value is not None and not callable(value) and hasattr(value, "shape")


    def _ppo_inject_values(output, values):
        if isinstance(output, dict):
            result = dict(output)
            result["values"] = values
            return result
        try:
            setattr(output, "values", values)
            return output
        except Exception:
            pass
        if hasattr(output, "logits"):
            return {"logits": output.logits, "values": values, "raw_output": output}
        return {"values": values, "raw_output": output}


    def _ppo_patch_forward(model):
        if getattr(model, "_ppo_values_forward_patched", False):
            return model
        lm_head = _ppo_find_lm_head(model)
        if lm_head is None:
            return model
        original_forward = model.forward

        def _wrapped_forward(*args, **kwargs):
            captured = {}

            def _capture(module, inputs):
                if inputs:
                    captured["hidden"] = inputs[0]

            hook = lm_head.register_forward_pre_hook(_capture)
            try:
                output = original_forward(*args, **kwargs)
            finally:
                hook.remove()

            if _ppo_output_has_values(output):
                return output
            v_head = getattr(model, "v_head", None)
            hidden = captured.get("hidden")
            if v_head is None or hidden is None or not hasattr(hidden, "shape"):
                return output
            values = v_head(hidden.to(v_head.weight.device)).squeeze(-1)
            return _ppo_inject_values(output, values)

        model.forward = _wrapped_forward
        model._ppo_values_forward_patched = True
        return model


    def _ppo_attach_value_head(model, model_args):
        import torch

        if hasattr(model, "v_head"):
            return model
        hidden_size = _ppo_value_head_hidden_size(model)
        if hidden_size is None:
            return model
        v_head = torch.nn.Linear(hidden_size, 1, bias=False)
        v_head.weight.data.zero_()
        try:
            first_param = next(model.parameters())
            v_head = v_head.to(device=first_param.device, dtype=first_param.dtype)
        except Exception:
            compute_dtype = getattr(model_args, "compute_dtype", None)
            if compute_dtype is not None:
                v_head = v_head.to(dtype=compute_dtype)
        model.v_head = v_head
        return model


    _BASE_LOAD_MODEL = load_model


    def load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=False, backend_device=None, **kwargs):
        model = _ppo_call_with_supported_kwargs(
            _BASE_LOAD_MODEL,
            tokenizer=tokenizer,
            model_args=model_args,
            finetuning_args=finetuning_args,
            is_trainable=is_trainable,
            **kwargs,
        )
        if add_valuehead:
            model = _ppo_attach_value_head(model, model_args)
            model = _ppo_patch_forward(model)
        return model
    """
).rstrip() + "\n"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _kept_families() -> list[str]:
    return sorted(path.stem for path in SFT_CONFIG_ROOT.glob("*.json"))


def _ignore_copy(src: str, names: list[str]) -> set[str]:
    ignored = set(IGNORE(src, names))
    src_path = Path(src)
    for name in names:
        try:
            if (src_path / name).is_symlink():
                ignored.add(name)
        except OSError:
            ignored.add(name)
    return ignored


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_shared_sft_config(family: str) -> dict:
    path = SFT_CONFIG_ROOT / f"{family}.json"
    return _read_json(path)["shared_config"]


def _uses_sft_shared(repo_root: Path) -> bool:
    for path in repo_root.glob("**/*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            if "_torch_sft_shared" in _read_text(path):
                return True
        except Exception:
            continue
    return False


def _render_ppo_config_text(base_text: str) -> str:
    if "_torch_sft_shared.config_base" in base_text:
        return SHARED_BASE_PPO_CONFIG
    if "class PPOArguments" in base_text and "class GeneratingArguments" in base_text:
        return base_text
    return base_text.rstrip() + "\n\n" + PPO_CONFIG_APPEND + "\n"


def _append_load_model_wrapper(base_text: str) -> str:
    if "_BASE_LOAD_MODEL = load_model" in base_text:
        return base_text
    return base_text.rstrip() + "\n\n" + PPO_LOAD_MODEL_APPEND


def _copy_template_file(src: Path, dst: Path) -> None:
    _write_text(dst, _read_text(src))


def _render_trainer_text() -> str:
    text = _read_text(PPO_TEMPLATE_ROOT / "ppo" / "trainer.py")
    text = text.replace('/ "transformers" / "ppo_compat.py"', '/ "transformers_impl" / "ppo_compat.py"')
    text = text.replace("_gemma3_ppo_torch_orig_ppo_compat", "_generic_ppo_torch_orig_ppo_compat")
    text = text.replace("for Gemma3", "for generic torch_orig models")
    text = text.replace("for Gemma3,", "for generic torch_orig models,")
    text = text.replace("Gemma3 PPO", "Generic PPO")
    return text


def _render_example_yaml(family: str, shared_config: dict) -> str:
    model_name = shared_config["model_name_or_path"]
    template = shared_config.get("template", "default")
    return textwrap.dedent(
        f"""
        model_name_or_path: {model_name}
        template: {template}
        output_dir: output
        do_train: true
        do_eval: false
        per_device_train_batch_size: 1
        gradient_accumulation_steps: 1
        learning_rate: 1.0e-6
        num_train_epochs: 1
        bf16: true
        fp16: false
        reward_model_type: api
        reward_model: http://localhost:8000
        """
    ).lstrip()


def build_family(family: str) -> None:
    sft_repo = SFT_ROOT / family / "torch_orig"
    ppo_repo = PPO_ROOT / family / "torch_orig"
    if not sft_repo.is_dir():
        raise FileNotFoundError(sft_repo)

    ppo_repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        sft_repo,
        ppo_repo,
        ignore=_ignore_copy,
        dirs_exist_ok=True,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    if _uses_sft_shared(sft_repo):
        shutil.copytree(
            SFT_SHARED_ROOT,
            ppo_repo / "_torch_sft_shared",
            ignore=_ignore_copy,
            dirs_exist_ok=True,
            symlinks=True,
            ignore_dangling_symlinks=True,
        )

    _write_text(ppo_repo / "config.py", _render_ppo_config_text(_read_text(sft_repo / "config.py")))
    _write_text(ppo_repo / "train.py", PPO_TRAIN_TEXT)
    _write_text(ppo_repo / "data" / "loader.py", UNSUPERVISED_LOADER_TEXT)
    _write_text(ppo_repo / "data" / "processor.py", UNSUPERVISED_PROCESSOR_TEXT)
    _write_text(ppo_repo / "data" / "collator.py", PPO_COLLATOR_TEXT)
    _write_text(ppo_repo / "model" / "loader.py", _append_load_model_wrapper(_read_text(sft_repo / "model" / "loader.py")))

    _copy_template_file(PPO_TEMPLATE_ROOT / "ppo" / "__init__.py", ppo_repo / "ppo" / "__init__.py")
    _copy_template_file(PPO_TEMPLATE_ROOT / "ppo" / "ppo_utils.py", ppo_repo / "ppo" / "ppo_utils.py")
    _write_text(ppo_repo / "ppo" / "trainer.py", _render_trainer_text())
    _copy_template_file(PPO_TEMPLATE_ROOT / "mock_reward_server.py", ppo_repo / "mock_reward_server.py")
    _copy_template_file(PPO_TEMPLATE_ROOT / "transformers_impl" / "ppo_compat.py", ppo_repo / "transformers_impl" / "ppo_compat.py")

    examples_dir = ppo_repo / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    _write_text(examples_dir / f"{family}_ppo.yaml", _render_example_yaml(family, _load_shared_sft_config(family)))


def build_reference_config(family: str) -> dict:
    shared = dict(_load_shared_sft_config(family))
    shared["learning_rate"] = shared.get("learning_rate", 1.0e-6)
    shared["reward_model_type"] = "api"
    shared["reward_model"] = "http://localhost:8000"
    repo_dir = str((PPO_ROOT / family / "torch_orig").resolve())
    output_dir = str((EVAL_ROOT / "generated_ppo_reference_configs" / f"{family}_output").resolve())
    return {
        "training_method": "ppo",
        "reference_impl": "torch_orig",
        "candidate_impl": "torch_orig_shadow",
        "precision_profile": "bf16_compare",
        "runtime_profile": "single_gpu",
        "output_dir": output_dir,
        "batch_size": 1,
        "max_eval_examples": 1,
        "loss_trajectory_steps": 2,
        "generation_max_new_tokens": 8,
        "compare_hidden_states": False,
        "shared_config": shared,
        "implementations": [
            {
                "name": "torch_orig",
                "framework": "pytorch",
                "repo_dir": repo_dir,
                "device": "cuda:0",
                "backend_device": "cuda:1",
            },
            {
                "name": "torch_orig_shadow",
                "framework": "pytorch",
                "repo_dir": repo_dir,
                "device": "cuda:0",
            },
        ],
    }


def write_reference_configs(families: list[str]) -> None:
    config_root = EVAL_ROOT / "generated_ppo_reference_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for family in families:
        path = config_root / f"{family}.json"
        path.write_text(json.dumps(build_reference_config(family), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    families = _kept_families()
    for family in families:
        build_family(family)
    write_reference_configs(families)
    print(json.dumps({"built_families": families}, indent=2))


if __name__ == "__main__":
    main()
