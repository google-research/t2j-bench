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

"""Generic DPO dataset loading and preprocessing."""

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
    prompt.append({"role": "user", "content": "\n".join(query_parts)})

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
