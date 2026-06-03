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

"""Generate DPO torch_orig repos by extending the validated SFT repos."""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "sft"
DPO_ROOT = ROOT / "dpo"
EVAL_ROOT = ROOT / "eval"
SFT_CONFIG_ROOT = EVAL_ROOT / "generated_sft_reference_configs"
SFT_SHARED_ROOT = SFT_ROOT / "_torch_sft_shared"

TEXT_TEMPLATE_ROOT = DPO_ROOT / "gemma3" / "torch_orig"
QWEN_TEMPLATE_ROOT = DPO_ROOT / "qwen3_vl" / "torch_orig"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "output", "outputs")


DPO_CONFIG_APPEND = textwrap.dedent(
    """


    @dataclass
    class DPOArguments:
        \"\"\"Arguments for DPO training.\"\"\"
        pref_loss: Literal["sigmoid", "orpo", "simpo"] = "sigmoid"
        pref_beta: float = 0.1
        pref_ftx: float = 0.0
        dpo_label_smoothing: float = 0.0
        simpo_gamma: float = 0.5
        use_ref_model: Optional[bool] = None
        ref_model: Optional[str] = None
        disable_dropout: bool = True


    def get_train_args(args: Optional[dict[str, Any]] = None):
        \"\"\"Parse and return all DPO training arguments from YAML/CLI.\"\"\"
        config = read_args(args)
        if isinstance(config, list):
            raise ValueError("Please provide a YAML or JSON configuration file.")

        model_args = _populate_dataclass(ModelArguments, config)
        data_args = _populate_dataclass(DataArguments, config)
        training_args = _populate_dataclass(TrainingArguments, config)
        finetuning_args = _populate_dataclass(FinetuningArguments, config)
        dpo_args = _populate_dataclass(DPOArguments, config)

        if getattr(finetuning_args, "lora_alpha", None) is None:
            finetuning_args.lora_alpha = finetuning_args.lora_rank * 2

        if dpo_args.use_ref_model is None:
            dpo_args.use_ref_model = dpo_args.pref_loss not in ("orpo", "simpo")

        import torch
        if model_args.compute_dtype is not None and isinstance(model_args.compute_dtype, str):
            model_args.compute_dtype = getattr(torch, model_args.compute_dtype)
        elif model_args.compute_dtype is None:
            if getattr(training_args, "bf16", False):
                model_args.compute_dtype = torch.bfloat16
            elif getattr(training_args, "fp16", False):
                model_args.compute_dtype = torch.float16
            else:
                model_args.compute_dtype = torch.float32

        return model_args, data_args, training_args, finetuning_args, dpo_args
    """
).strip()


DPO_LOAD_REF_APPEND = textwrap.dedent(
    """


    def load_ref_model(tokenizer, model_args, finetuning_args, dpo_args):
        \"\"\"Load the DPO reference model using the same repo-local loader path.\"\"\"
        if not getattr(dpo_args, "use_ref_model", True):
            return None
        import copy

        ref_model_args = copy.deepcopy(model_args)
        ref_finetuning_args = copy.deepcopy(finetuning_args)
        ref_model_name = getattr(dpo_args, "ref_model", None)
        if ref_model_name:
            ref_model_args.model_name_or_path = ref_model_name
        return load_model(tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False)
    """
).strip()


DPO_TRAIN_TEXT = textwrap.dedent(
    """
    \"\"\"Main entry point for DPO training.\"\"\"

    import inspect
    import logging
    import os
    import sys
    from typing import Optional

    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from config import get_train_args
    from data.collator import PairwiseDataCollatorWithPadding
    from data.constants import IGNORE_INDEX
    from data.loader import get_dataset
    from data.template import get_template_and_fix_tokenizer
    from dpo.trainer import DPOTrainer
    from model.loader import load_model, load_ref_model, load_tokenizer

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
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


    def run_dpo(config: Optional[dict] = None):
        \"\"\"Run DPO training.\"\"\"
        model_args, data_args, training_args, finetuning_args, dpo_args = get_train_args(config)

        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        processor = tokenizer_module["processor"]

        template = get_template_and_fix_tokenizer(tokenizer, data_args)

        dataset_module = get_dataset(
            template=template,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
            processor=processor,
        )

        model = _call_with_supported_kwargs(
            load_model,
            tokenizer=tokenizer,
            model_args=model_args,
            finetuning_args=finetuning_args,
            is_trainable=training_args.do_train,
            disable_dropout=dpo_args.disable_dropout,
        )
        ref_model = _call_with_supported_kwargs(
            load_ref_model,
            tokenizer=tokenizer,
            model_args=model_args,
            finetuning_args=finetuning_args,
            dpo_args=dpo_args,
        )

        data_collator = PairwiseDataCollatorWithPadding(
            template=template,
            model=model,
            pad_to_multiple_of=training_args.pad_to_multiple_of if training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX,
            compute_dtype=model_args.compute_dtype or torch.bfloat16,
            tokenizer=tokenizer,
            processor=processor,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            dpo_args=dpo_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            **dataset_module,
        )

        if training_args.do_train:
            logger.info("*** Starting DPO training ***")
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint
            )
            trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

        if training_args.do_eval:
            logger.info("*** Starting evaluation ***")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


    if __name__ == "__main__":
        run_dpo()
    """
).lstrip()


PAIRWISE_PROCESSOR_TEXT = textwrap.dedent(
    """
    \"\"\"Generic pairwise dataset processor for DPO training.\"\"\"

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
    """
).lstrip()


PAIRWISE_COLLATOR_TEXT = textwrap.dedent(
    """
    \"\"\"Generic multimodal pairwise collator for DPO.\"\"\"

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
            if (
                self.model is not None
                and hasattr(self.model, "base_model")
                and hasattr(self.model.base_model, "model")
                and "Peft" in self.model.__class__.__name__
            ):
                self.model = self.model.base_model.model
            self.get_rope_func = None
            if self.model is not None:
                if hasattr(self.model, "get_rope_index"):
                    self.get_rope_func = self.model.get_rope_index
                elif hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
                    self.get_rope_func = self.model.model.get_rope_index

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


    @dataclass
    class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
            concatenated_features = []
            for key in ("chosen", "rejected"):
                for feature in features:
                    concatenated_features.append(
                        {
                            "input_ids": feature[f"{key}_input_ids"],
                            "attention_mask": feature[f"{key}_attention_mask"],
                            "labels": feature[f"{key}_labels"],
                            "images": feature.get("images"),
                            "videos": feature.get("videos"),
                            "audios": feature.get("audios"),
                        }
                    )
            return super().__call__(concatenated_features)
    """
).lstrip()


PAIRWISE_LOADER_TEXT = textwrap.dedent(
    """
    \"\"\"Generic DPO dataset loading and preprocessing.\"\"\"

    import json
    import logging
    import os
    from dataclasses import dataclass
    from typing import Any, Literal, Optional

    from .constants import DATA_CONFIG, FILEEXT2TYPE
    from .processor import PairwiseDatasetProcessor

    logger = logging.getLogger(__name__)


    @dataclass
    class DatasetAttr:
        load_from: Literal["hf_hub", "file"] = "file"
        dataset_name: str = ""
        formatting: Literal["alpaca", "sharegpt"] = "alpaca"
        ranking: bool = True
        split: str = "train"
        subset: Optional[str] = None
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
        chosen: Optional[str] = "chosen"
        rejected: Optional[str] = "rejected"
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
            self.set_attr("ranking", attr, default=True)
            self.set_attr("subset", attr)
            self.set_attr("split", attr, default="train")
            self.set_attr("folder", attr)
            self.set_attr("num_samples", attr)
            if "columns" in attr:
                for col in [
                    "prompt", "query", "response", "chosen", "rejected", "history",
                    "messages", "system", "tools", "images", "videos", "audios",
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
        chosen_value = example.get(dataset_attr.chosen) if dataset_attr.chosen else None
        rejected_value = example.get(dataset_attr.rejected) if dataset_attr.rejected else None
        if isinstance(chosen_value, str) and isinstance(rejected_value, str):
            response = [
                {"role": "assistant", "content": chosen_value},
                {"role": "assistant", "content": rejected_value},
            ]
        elif dataset_attr.response and isinstance(example.get(dataset_attr.response), list):
            raw_response = example[dataset_attr.response]
            if len(raw_response) >= 2:
                response = raw_response[:2]

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
        messages = example.get(dataset_attr.messages, [])
        system = ""
        if (
            dataset_attr.system_tag
            and len(messages) > 0
            and messages[0].get(dataset_attr.role_tag) == dataset_attr.system_tag
        ):
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        elif dataset_attr.system and example.get(dataset_attr.system):
            system = example[dataset_attr.system]

        prompt = []
        broken = False
        for turn_idx, msg in enumerate(messages):
            expected_roles = (
                (dataset_attr.user_tag, dataset_attr.observation_tag)
                if turn_idx % 2 == 0
                else (dataset_attr.assistant_tag, dataset_attr.function_tag)
            )
            if msg.get(dataset_attr.role_tag) not in expected_roles:
                broken = True
                break
            prompt.append(
                {
                    "role": tag_mapping.get(msg.get(dataset_attr.role_tag), "user"),
                    "content": msg.get(dataset_attr.content_tag, ""),
                }
            )

        response = []
        chosen = example.get(dataset_attr.chosen) if dataset_attr.chosen else None
        rejected = example.get(dataset_attr.rejected) if dataset_attr.rejected else None
        if isinstance(chosen, dict) and isinstance(rejected, dict):
            response = [
                {
                    "role": tag_mapping.get(chosen.get(dataset_attr.role_tag, dataset_attr.assistant_tag), "assistant"),
                    "content": chosen[dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping.get(rejected.get(dataset_attr.role_tag, dataset_attr.assistant_tag), "assistant"),
                    "content": rejected[dataset_attr.content_tag],
                },
            ]
        elif isinstance(chosen, str) and isinstance(rejected, str):
            response = [
                {"role": "assistant", "content": chosen},
                {"role": "assistant", "content": rejected},
            ]

        return {
            "_prompt": [] if broken else prompt,
            "_response": response,
            "_system": system,
            "_tools": example.get(dataset_attr.tools, "") if dataset_attr.tools else "",
            "_images": _find_medias(example.get(dataset_attr.images) if dataset_attr.images else None, dataset_attr, data_args),
            "_videos": _find_medias(example.get(dataset_attr.videos) if dataset_attr.videos else None, dataset_attr, data_args),
            "_audios": _find_medias(example.get(dataset_attr.audios) if dataset_attr.audios else None, dataset_attr, data_args),
        }


    def _load_single_dataset(dataset_attr: DatasetAttr, data_args):
        from datasets import load_dataset

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
        from datasets import concatenate_datasets

        if data_args.dataset is None:
            raise ValueError("No dataset specified.")
        dataset_names = [name.strip() for name in data_args.dataset.split(",")]
        dataset_list = get_dataset_list(dataset_names, data_args.dataset_dir)
        all_datasets = []
        for dataset_attr in dataset_list:
            dataset = _load_single_dataset(dataset_attr, data_args)
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
        eval_dataset = None
        if data_args.val_size > 0:
            split = dataset.train_test_split(test_size=data_args.val_size, seed=training_args.seed)
            dataset = split["train"]
            eval_dataset = split["test"]

        proc = PairwiseDatasetProcessor(
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


SHARED_BASE_DPO_CONFIG = textwrap.dedent(
    """
    from pathlib import Path
    import importlib.util
    import json
    import sys
    from dataclasses import dataclass
    from typing import Any, Literal, Optional

    _REPO_ROOT = Path(__file__).resolve().parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    _MODEL_META_PATH = _REPO_ROOT / "model_meta.py"
    _MODEL_META_SPEC = importlib.util.spec_from_file_location(f"{__name__}_model_meta", _MODEL_META_PATH)
    _MODEL_META = importlib.util.module_from_spec(_MODEL_META_SPEC)
    _MODEL_META_SPEC.loader.exec_module(_MODEL_META)
    DEFAULT_TEMPLATE = _MODEL_META.DEFAULT_TEMPLATE


    def read_args(args: Optional[dict[str, Any]] = None) -> dict[str, Any] | list[str]:
        from omegaconf import OmegaConf

        if args is not None:
            return args

        if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
            override_config = OmegaConf.from_cli(sys.argv[2:])
            dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
            return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
        if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
            override_config = OmegaConf.from_cli(sys.argv[2:])
            with open(Path(sys.argv[1]).absolute()) as f:
                dict_config = OmegaConf.create(json.load(f))
            return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
        return sys.argv[1:]


    @dataclass
    class ModelArguments:
        model_name_or_path: str = ""
        trust_remote_code: bool = True
        cache_dir: Optional[str] = None
        model_revision: str = "main"
        hf_hub_token: Optional[str] = None
        use_fast_tokenizer: bool = True
        resize_vocab: bool = False
        image_max_pixels: int = 768 * 768
        image_min_pixels: int = 32 * 32
        image_do_pan_and_scan: bool = False
        compute_dtype: Optional[str] = None
        block_diag_attn: bool = False
        device_map: Optional[str] = None
        low_cpu_mem_usage: bool = True
        attn_implementation: str = "eager"


    @dataclass
    class DataArguments:
        dataset: Optional[str] = None
        dataset_dir: str = "data"
        template: str = "default"
        cutoff_len: int = 2048
        train_on_prompt: bool = False
        overwrite_cache: bool = False
        preprocessing_num_workers: int = 1
        max_samples: Optional[int] = None
        val_size: float = 0.0
        default_system: Optional[str] = None
        media_dir: Optional[str] = None


    @dataclass
    class FinetuningArguments:
        finetuning_type: str = "lora"
        lora_rank: int = 8
        lora_alpha: Optional[int] = None
        lora_dropout: float = 0.0
        lora_target: str = "all"
        additional_target: Optional[str] = None
        use_rslora: bool = False
        use_dora: bool = False
        freeze_vision_tower: bool = True
        freeze_multi_modal_projector: bool = True
        pure_bf16: bool = False


    @dataclass
    class DPOArguments:
        pref_loss: Literal["sigmoid", "orpo", "simpo"] = "sigmoid"
        pref_beta: float = 0.1
        pref_ftx: float = 0.0
        dpo_label_smoothing: float = 0.0
        simpo_gamma: float = 0.5
        use_ref_model: Optional[bool] = None
        ref_model: Optional[str] = None
        disable_dropout: bool = True


    def _populate_dataclass(cls, config_dict: dict[str, Any]):
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


    def get_train_args(args: Optional[dict[str, Any]] = None):
        from transformers import TrainingArguments as HfTrainingArguments
        import torch

        config = read_args(args)
        if isinstance(config, list):
            raise ValueError("Please provide a YAML or JSON configuration file.")

        merged_config = dict(config)
        merged_config.setdefault("template", DEFAULT_TEMPLATE)

        model_args = _populate_dataclass(ModelArguments, merged_config)
        data_args = _populate_dataclass(DataArguments, merged_config)
        finetuning_args = _populate_dataclass(FinetuningArguments, merged_config)
        dpo_args = _populate_dataclass(DPOArguments, merged_config)

        if finetuning_args.lora_alpha is None:
            finetuning_args.lora_alpha = finetuning_args.lora_rank * 2
        if dpo_args.use_ref_model is None:
            dpo_args.use_ref_model = dpo_args.pref_loss not in ("orpo", "simpo")

        if model_args.compute_dtype is not None:
            model_args.compute_dtype = getattr(torch, model_args.compute_dtype)
        elif merged_config.get("bf16", True):
            model_args.compute_dtype = torch.bfloat16
        elif merged_config.get("fp16", False):
            model_args.compute_dtype = torch.float16
        else:
            model_args.compute_dtype = torch.float32

        training_field_names = set(HfTrainingArguments.__dataclass_fields__.keys())
        training_kwargs = {k: v for k, v in merged_config.items() if k in training_field_names}
        training_kwargs.setdefault("output_dir", merged_config.get("output_dir", "output"))
        training_kwargs.setdefault("per_device_train_batch_size", merged_config.get("per_device_train_batch_size", 1))
        training_kwargs.setdefault("per_device_eval_batch_size", merged_config.get("per_device_eval_batch_size", 1))
        training_kwargs.setdefault("gradient_accumulation_steps", merged_config.get("gradient_accumulation_steps", 8))
        training_kwargs.setdefault("learning_rate", merged_config.get("learning_rate", 5e-6))
        training_kwargs.setdefault("logging_steps", merged_config.get("logging_steps", 10))
        training_kwargs.setdefault("save_steps", merged_config.get("save_steps", 500))
        training_kwargs.setdefault("num_train_epochs", merged_config.get("num_train_epochs", 3.0))
        training_kwargs.setdefault("remove_unused_columns", False)
        training_kwargs.setdefault("report_to", merged_config.get("report_to", []))
        training_kwargs.setdefault("bf16", merged_config.get("bf16", True))
        training_kwargs.setdefault("fp16", merged_config.get("fp16", False))
        training_kwargs.setdefault("gradient_checkpointing", merged_config.get("gradient_checkpointing", True))
        training_kwargs.setdefault("do_train", merged_config.get("do_train", True))
        training_kwargs.setdefault("do_eval", merged_config.get("do_eval", False))
        training_kwargs.setdefault("seed", merged_config.get("seed", 42))
        training_args = HfTrainingArguments(**training_kwargs)

        return model_args, data_args, training_args, finetuning_args, dpo_args
    """
).lstrip()


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


def _needs_rope_collator(sft_repo: Path) -> bool:
    for folder in ("model", "transformers_impl", "data"):
        root = sft_repo / folder
        if not root.is_dir():
            continue
        for path in root.glob("*.py"):
            try:
                text = _read_text(path)
            except Exception:
                continue
            if "get_rope_index" in text:
                return True
    return False


def _load_shared_sft_config(family: str) -> dict:
    path = SFT_CONFIG_ROOT / f"{family}.json"
    return json.loads(path.read_text(encoding="utf-8"))["shared_config"]


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
        learning_rate: 5.0e-6
        num_train_epochs: 1
        bf16: true
        fp16: false
        pref_loss: sigmoid
        pref_beta: 0.1
        """
    ).lstrip()


def _append_dpo_config(base_text: str) -> str:
    if "class DPOArguments" in base_text:
        return base_text
    return base_text.rstrip() + "\n\n" + DPO_CONFIG_APPEND + "\n"


def _render_dpo_config_text(base_text: str) -> str:
    if "_torch_sft_shared.config_base" in base_text:
        return SHARED_BASE_DPO_CONFIG
    return _append_dpo_config(base_text)


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


def _append_load_ref_model(base_text: str) -> str:
    if "def load_ref_model(" in base_text:
        return base_text
    return base_text.rstrip() + "\n\n" + DPO_LOAD_REF_APPEND + "\n"


def _copy_template_file(src: Path, dst: Path) -> None:
    _write_text(dst, _read_text(src))


def build_family(family: str) -> None:
    sft_repo = SFT_ROOT / family / "torch_orig"
    dpo_repo = DPO_ROOT / family / "torch_orig"
    if not sft_repo.is_dir():
        raise FileNotFoundError(sft_repo)

    dpo_repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        sft_repo,
        dpo_repo,
        ignore=_ignore_copy,
        dirs_exist_ok=True,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    if _uses_sft_shared(sft_repo):
        shutil.copytree(
            SFT_SHARED_ROOT,
            dpo_repo / "_torch_sft_shared",
            ignore=_ignore_copy,
            dirs_exist_ok=True,
            symlinks=True,
            ignore_dangling_symlinks=True,
        )

    text_template = TEXT_TEMPLATE_ROOT
    qwen_template = QWEN_TEMPLATE_ROOT
    collator_template = qwen_template if _needs_rope_collator(sft_repo) else text_template

    # Generic DPO overlay.
    _write_text(dpo_repo / "config.py", _render_dpo_config_text(_read_text(sft_repo / "config.py")))
    _write_text(dpo_repo / "train.py", DPO_TRAIN_TEXT)
    _write_text(dpo_repo / "data" / "loader.py", PAIRWISE_LOADER_TEXT)
    _write_text(dpo_repo / "data" / "processor.py", PAIRWISE_PROCESSOR_TEXT)
    _write_text(dpo_repo / "data" / "collator.py", PAIRWISE_COLLATOR_TEXT)
    _copy_template_file(text_template / "datasets" / "dataset.py", dpo_repo / "datasets" / "dataset.py")
    _copy_template_file(text_template / "dpo" / "trainer.py", dpo_repo / "dpo" / "trainer.py")
    _copy_template_file(text_template / "dpo" / "__init__.py", dpo_repo / "dpo" / "__init__.py")
    _write_text(dpo_repo / "model" / "loader.py", _append_load_ref_model(_read_text(sft_repo / "model" / "loader.py")))

    examples_dir = dpo_repo / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    _write_text(examples_dir / f"{family}_dpo.yaml", _render_example_yaml(family, _load_shared_sft_config(family)))


def build_reference_config(family: str) -> dict:
    shared = dict(_load_shared_sft_config(family))
    shared["learning_rate"] = shared.get("learning_rate", 5.0e-6)
    shared["pref_loss"] = "orpo"
    shared["pref_beta"] = shared.get("pref_beta", 0.1)
    shared["eval_examples"] = [
        {
            "_prompt": [{"role": "user", "content": "Say hello in one short sentence."}],
            "_response": [
                {"role": "assistant", "content": "Hello there!"},
                {"role": "assistant", "content": "Goodbye forever."},
            ],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
            "_audios": None,
        }
    ]
    repo_dir = str((DPO_ROOT / family / "torch_orig").resolve())
    output_dir = str((EVAL_ROOT / "generated_dpo_reference_configs" / f"{family}_output").resolve())
    return {
        "training_method": "dpo",
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
    config_root = EVAL_ROOT / "generated_dpo_reference_configs"
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
