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

"""Dataset loading and preprocessing for Gemma3 SFT."""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .constants import DATA_CONFIG, FILEEXT2TYPE
from .processor import SupervisedDatasetProcessor

logger = logging.getLogger(__name__)


@dataclass
class DatasetAttr:
    """Attributes for a single dataset."""
    dataset_name: str = ""
    file_name: Optional[str] = None
    formatting: str = "alpaca"
    split: str = "train"
    prompt: str = "instruction"
    query: str = "input"
    response: str = "output"
    history: Optional[str] = None
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    messages: str = "conversations"
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    system_tag: str = "system"
    observation_tag: str = "observation"


def get_dataset_list(dataset_dir: str, dataset_names: str) -> list[DatasetAttr]:
    """Parse dataset names and load their attributes from dataset_info.json."""
    dataset_info_path = os.path.join(dataset_dir, DATA_CONFIG)
    dataset_info = {}
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)

    dataset_list = []
    names = [n.strip() for n in dataset_names.split(",")]

    for name in names:
        if name in dataset_info:
            info = dataset_info[name]
            attr = DatasetAttr(dataset_name=name)
            if "file_name" in info:
                attr.file_name = info["file_name"]
            if "formatting" in info:
                attr.formatting = info["formatting"]
            if "columns" in info:
                cols = info["columns"]
                for key in ["prompt", "query", "response", "history", "system",
                            "tools", "images", "videos", "messages"]:
                    if key in cols:
                        setattr(attr, key, cols[key])
            if "tags" in info:
                tags = info["tags"]
                for key in ["role_tag", "content_tag", "user_tag", "assistant_tag",
                            "system_tag", "observation_tag"]:
                    if key in tags:
                        setattr(attr, key, tags[key])
            dataset_list.append(attr)
        else:
            attr = DatasetAttr(dataset_name=name, file_name=name)
            dataset_list.append(attr)

    return dataset_list


def _convert_alpaca(example: dict, attr: DatasetAttr) -> dict:
    """Convert alpaca-format example to standard format."""
    prompt_text = str(example.get(attr.prompt, ""))
    query_text = str(example.get(attr.query, ""))
    response_text = str(example.get(attr.response, ""))

    if query_text:
        prompt_text = prompt_text + "\n" + query_text if prompt_text else query_text

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]

    result = {
        "_prompt": [messages[0]],
        "_response": [messages[1]],
        "_system": str(example.get(attr.system, "")) if attr.system else "",
        "_tools": str(example.get(attr.tools, "")) if attr.tools else "",
    }

    if attr.images and attr.images in example:
        imgs = example[attr.images]
        if isinstance(imgs, str):
            imgs = [imgs]
        result["_images"] = imgs
    else:
        result["_images"] = None

    if attr.videos and attr.videos in example:
        vids = example[attr.videos]
        if isinstance(vids, str):
            vids = [vids]
        result["_videos"] = vids
    else:
        result["_videos"] = None

    return result


def _resolve_media_paths(medias, dataset_dir: str):
    if medias is None:
        return None
    if not isinstance(medias, list):
        medias = [medias]
    resolved = []
    media_root = os.path.join(dataset_dir, "mllm_demo_data")
    for media in medias:
        if isinstance(media, str) and not os.path.isabs(media):
            candidates = [os.path.join(dataset_dir, media), os.path.join(media_root, media)]
            matched = next((path for path in candidates if os.path.exists(path)), candidates[0])
            resolved.append(matched)
        else:
            resolved.append(media)
    return resolved


def _convert_sharegpt(example: dict, attr: DatasetAttr) -> dict:
    """Convert ShareGPT-format example to standard format."""
    conversations = example.get(attr.messages, [])

    prompt_msgs = []
    response_msgs = []
    system = ""
    images = []
    videos = []

    for i, turn in enumerate(conversations):
        role = turn.get(attr.role_tag, "")
        content = turn.get(attr.content_tag, "")

        if role == attr.system_tag:
            system = content
        elif role == attr.user_tag:
            prompt_msgs.append({"role": "user", "content": content})
        elif role == attr.assistant_tag:
            response_msgs.append({"role": "assistant", "content": content})

    if attr.images and attr.images in example:
        imgs = example[attr.images]
        if isinstance(imgs, str):
            imgs = [imgs]
        images = imgs

    result = {
        "_prompt": prompt_msgs,
        "_response": response_msgs,
        "_system": system,
        "_tools": "",
        "_images": images if images else None,
        "_videos": videos if videos else None,
    }

    return result


def _load_single_dataset(dataset_dir: str, attr: DatasetAttr):
    """Load a single dataset from file."""
    from datasets import load_dataset, Dataset

    file_path = attr.file_name
    if file_path and not os.path.isabs(file_path):
        file_path = os.path.join(dataset_dir, file_path)

    if file_path and os.path.exists(file_path):
        data_path = FILEEXT2TYPE.get(os.path.splitext(file_path)[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
        return load_dataset(path=data_path, data_files=[file_path], split=attr.split)

    raise FileNotFoundError(f"Dataset file not found: {file_path}")


def get_dataset(template, model_args, data_args, training_args, tokenizer, processor=None):
    """Load and preprocess datasets for SFT training."""
    from datasets import Dataset, concatenate_datasets

    dataset_names = getattr(data_args, "dataset", None)
    if not dataset_names:
        raise ValueError("No dataset specified. Set 'dataset' in config.")

    dataset_dir = getattr(data_args, "dataset_dir", "data")
    max_samples = getattr(data_args, "max_samples", None)

    dataset_list = get_dataset_list(dataset_dir, dataset_names)
    all_datasets = []

    for attr in dataset_list:
        try:
            ds = _load_single_dataset(dataset_dir, attr)
            logger.info(f"Loaded dataset '{attr.dataset_name}' with {len(ds)} examples")

            if attr.formatting == "sharegpt":
                converter = lambda ex: _convert_sharegpt(ex, attr)
            else:
                converter = lambda ex: _convert_alpaca(ex, attr)

            ds = ds.map(converter)
            ds = ds.map(
                lambda ex: {
                    **ex,
                    "_images": _resolve_media_paths(ex.get("_images"), dataset_dir),
                    "_videos": _resolve_media_paths(ex.get("_videos"), dataset_dir),
                }
            )
            all_datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load dataset '{attr.dataset_name}': {e}")

    if not all_datasets:
        raise ValueError("No datasets were loaded successfully.")

    dataset = concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(list(range(max_samples)))

    sft_processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )

    column_names = list(dataset.column_names) if hasattr(dataset, "column_names") else list(next(iter(dataset)).keys())
    dataset = dataset.map(
        sft_processor.preprocess_dataset,
        batched=True,
        batch_size=100,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )

    val_size = getattr(data_args, "val_size", 0.0)
    if val_size > 0 and val_size < 1:
        split = dataset.train_test_split(test_size=val_size)
        return {"train_dataset": split["train"], "eval_dataset": split["test"]}
    else:
        return {"train_dataset": dataset}
