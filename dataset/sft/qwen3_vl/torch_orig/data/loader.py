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

"""Dataset loading, conversion, and preprocessing for Qwen3-VL SFT."""

import json
import os
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset

from .constants import DATA_CONFIG, FILEEXT2TYPE

if TYPE_CHECKING:
    from .template import Template

logger = logging.getLogger(__name__)


# ============================================================
# Dataset Attribute
# ============================================================

@dataclass
class DatasetAttr:
    """Simplified dataset attributes (file or hf_hub only)."""
    load_from: Literal["hf_hub", "file"] = "file"
    dataset_name: str = ""
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    audios: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    # sharegpt columns
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
            for col in ["prompt", "query", "response", "history", "messages",
                        "system", "tools", "images", "videos", "audios"]:
                self.set_attr(col, attr["columns"])
        if "tags" in attr:
            for tag in ["role_tag", "content_tag", "user_tag", "assistant_tag",
                        "observation_tag", "function_tag", "system_tag"]:
                self.set_attr(tag, attr["tags"])


def get_dataset_list(dataset_names: list[str], dataset_dir: str) -> list[DatasetAttr]:
    """Parse dataset_info.json to get dataset attributes."""
    config_path = os.path.join(dataset_dir, DATA_CONFIG)
    dataset_info = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            dataset_info = json.load(f)

    result = []
    for name in dataset_names:
        if name not in dataset_info:
            # Try as HF Hub dataset
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


# ============================================================
# Dataset Converters
# ============================================================

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
            path = os.path.join(media_dir, medias[i])
            if os.path.isfile(path):
                medias[i] = path
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
            logger.warning(f"Invalid role tag in {messages}.")
            broken = True
            break
        aligned.append({"role": tag_mapping[msg[dataset_attr.role_tag]], "content": msg[dataset_attr.content_tag]})

    if len(aligned) % 2 != 0:
        logger.warning(f"Invalid message count in {messages}.")
        broken = True

    if broken:
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


# ============================================================
# Dataset Loading
# ============================================================

def _load_single_dataset(dataset_attr: DatasetAttr, data_args) -> Dataset:
    """Load a single dataset and align it to the standard format."""
    logger.info(f"Loading dataset {dataset_attr.dataset_name}...")
    if dataset_attr.load_from == "hf_hub":
        dataset = load_dataset(
            path=dataset_attr.dataset_name,
            name=dataset_attr.subset,
            data_dir=dataset_attr.folder,
            split=dataset_attr.split,
            cache_dir=None,
        )
    elif dataset_attr.load_from == "file":
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_files = []
        if os.path.isdir(local_path):
            for fn in os.listdir(local_path):
                data_files.append(os.path.join(local_path, fn))
        elif os.path.isfile(local_path):
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
        dataset = load_dataset(path=data_path, data_files=data_files, split=dataset_attr.split)
    else:
        raise ValueError(f"Unknown load type: {dataset_attr.load_from}")

    if dataset_attr.num_samples is not None:
        target = dataset_attr.num_samples
        idxs = np.random.permutation(len(dataset))[:target]
        if len(idxs) < target:
            idxs = np.concatenate([idxs, np.random.choice(len(dataset), target - len(idxs))])
        dataset = dataset.select(idxs)

    if getattr(data_args, "max_samples", None) is not None:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    # Align dataset
    converter_fn = _convert_alpaca if dataset_attr.formatting == "alpaca" else _convert_sharegpt
    column_names = list(next(iter(dataset)).keys())
    dataset = dataset.map(
        lambda ex: converter_fn(ex, dataset_attr, data_args),
        batched=False,
        remove_columns=column_names,
        num_proc=min(getattr(data_args, "preprocessing_num_workers", 1), 1),
        desc="Converting format of dataset",
    )
    return dataset


def get_dataset(
    template: "Template",
    model_args,
    data_args,
    training_args,
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> dict:
    """Load and preprocess datasets. Returns dict with train_dataset and optionally eval_dataset."""
    from .processor import SupervisedDatasetProcessor

    if data_args.dataset is None:
        raise ValueError("No dataset specified.")

    dataset_names = [n.strip() for n in data_args.dataset.split(",")]
    dataset_list = get_dataset_list(dataset_names, data_args.dataset_dir)

    all_datasets = []
    for ds_attr in dataset_list:
        all_datasets.append(_load_single_dataset(ds_attr, data_args))

    if len(all_datasets) == 1:
        dataset = all_datasets[0]
    else:
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(all_datasets)

    # Split into train/eval
    eval_dataset = None
    if data_args.val_size > 0:
        split = dataset.train_test_split(test_size=data_args.val_size, seed=training_args.seed)
        dataset = split["train"]
        eval_dataset = split["test"]

    # Tokenize
    proc = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )
    column_names = list(next(iter(dataset)).keys())
    num_proc = min(getattr(data_args, 'preprocessing_num_workers', 1), 1)
    kwargs = dict(
        num_proc=num_proc,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    dataset = dataset.map(proc.preprocess_dataset, batched=True, batch_size=1000, remove_columns=column_names, **kwargs)

    try:
        next(iter(dataset))
    except StopIteration:
        raise RuntimeError("Cannot find valid samples.")

    result = {"train_dataset": dataset}
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            proc.preprocess_dataset, batched=True, batch_size=1000, remove_columns=column_names, **kwargs
        )
        result["eval_dataset"] = eval_dataset

    return result
