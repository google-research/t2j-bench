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

"""Dataset loading and preprocessing for SFT training."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from _dataset_utils import concatenate_datasets, load_dataset
from .constants import DATA_CONFIG, FILEEXT2TYPE
from .processor import SupervisedDatasetProcessor

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
            for col in ["prompt", "query", "response", "history", "messages", "system", "tools", "images", "videos", "audios"]:
                self.set_attr(col, attr["columns"])
        if "tags" in attr:
            for tag in ["role_tag", "content_tag", "user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]:
                self.set_attr(tag, attr["tags"])


def get_dataset_list(dataset_names: list[str], dataset_dir: str) -> list[DatasetAttr]:
    config_path = os.path.join(dataset_dir, DATA_CONFIG)
    dataset_info = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            dataset_info = json.load(f)

    result = []
    for name in dataset_names:
        if name not in dataset_info:
            result.append(DatasetAttr(load_from="file", dataset_name=name))
            continue
        info = dataset_info[name]
        attr = DatasetAttr(load_from="hf_hub" if "hf_hub_url" in info else "file", dataset_name=info.get("hf_hub_url", info.get("file_name", name)))
        attr.join(info)
        result.append(attr)
    return result


def _find_medias(medias, dataset_attr, data_args):
    if medias is None:
        return None
    if not isinstance(medias, list):
        medias = [medias]
    if not medias:
        return None
    resolved = medias[:]
    if dataset_attr.load_from == "file" and isinstance(resolved[0], str):
        media_dir = getattr(data_args, "media_dir", None) or getattr(data_args, "dataset_dir", "data")
        for idx, media in enumerate(resolved):
            path = os.path.join(media_dir, media)
            if os.path.isfile(path):
                resolved[idx] = path
    return resolved


def _convert_alpaca(example: dict, dataset_attr: DatasetAttr, data_args) -> dict:
    prompt = []
    if dataset_attr.history and isinstance(example.get(dataset_attr.history), list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({"role": "user", "content": old_prompt})
            prompt.append({"role": "assistant", "content": old_response})
    query_parts = []
    if dataset_attr.prompt and example.get(dataset_attr.prompt):
        query_parts.append(example[dataset_attr.prompt])
    if dataset_attr.query and example.get(dataset_attr.query):
        query_parts.append(example[dataset_attr.query])
    prompt.append({"role": "user", "content": "\n".join(query_parts)})
    response = [{"role": "assistant", "content": example[dataset_attr.response]}] if dataset_attr.response and isinstance(example.get(dataset_attr.response), str) else []
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
    messages = example[dataset_attr.messages]
    system = ""
    if dataset_attr.system_tag and messages and messages[0].get(dataset_attr.role_tag) == dataset_attr.system_tag:
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    aligned = [{"role": tag_mapping.get(message[dataset_attr.role_tag], "user"), "content": message[dataset_attr.content_tag]} for message in messages]
    prompt = aligned[:-1] if len(aligned) >= 2 else []
    response = aligned[-1:] if len(aligned) >= 1 else []
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
        raise ValueError("Generated SFT repos expect local or exported datasets, not HF dataset runtime loading.")
    local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    data_files: list[str] = []
    if os.path.isdir(local_path):
        for file_name in sorted(os.listdir(local_path)):
            data_files.append(os.path.join(local_path, file_name))
    elif os.path.isfile(local_path):
        data_files.append(local_path)
    else:
        raise ValueError(f"Dataset path {local_path} not found.")

    data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:])
    if data_path is None:
        raise ValueError(f"Allowed file types: {','.join(FILEEXT2TYPE.keys())}.")
    dataset = load_dataset(path=data_path, data_files=data_files, split=dataset_attr.split)
    if dataset_attr.num_samples is not None:
        indices = np.random.permutation(len(dataset))[: dataset_attr.num_samples]
        dataset = dataset.select(indices.tolist())
    if getattr(data_args, "max_samples", None) is not None:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    converter = _convert_alpaca if dataset_attr.formatting == "alpaca" else _convert_sharegpt
    return dataset.map(lambda example: converter(example, dataset_attr, data_args), batched=False)


def get_dataset(template, model_args, data_args, training_args, tokenizer, processor):
    if not data_args.dataset:
        raise ValueError("`dataset` must be provided in the SFT config.")

    dataset_names = [name.strip() for name in data_args.dataset.split(",") if name.strip()]
    dataset_list = [_load_single_dataset(dataset_attr, data_args) for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir)]
    merged_dataset = dataset_list[0] if len(dataset_list) == 1 else concatenate_datasets(dataset_list)

    processor_fn = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)
    processed_dataset = merged_dataset.map(
        processor_fn.preprocess_dataset,
        batched=True,
        remove_columns=merged_dataset.column_names,
    )
    if len(processed_dataset) == 0:
        raise ValueError("No usable samples remained after preprocessing.")

    if getattr(training_args, "do_eval", False) and getattr(data_args, "val_size", 0.0):
        split = processed_dataset.train_test_split(test_size=data_args.val_size, seed=getattr(training_args, "seed", 42))
        return {"train_dataset": split["train"], "eval_dataset": split["test"]}
    return {"train_dataset": processed_dataset}

