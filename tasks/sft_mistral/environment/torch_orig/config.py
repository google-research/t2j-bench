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

"""Training configuration helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from omegaconf import OmegaConf
import dataclasses
import torch


def read_args(args: Optional[dict[str, Any]] = None) -> dict[str, Any] | list[str]:
    """Read a YAML/JSON config from the CLI; otherwise return an empty dict (dataclass field defaults apply)."""

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
    return {}


@dataclass
class ModelArguments:
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.3"
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    hf_hub_token: Optional[str] = None
    use_fast_tokenizer: bool = True
    resize_vocab: bool = False
    image_max_pixels: int = 768 * 768
    image_min_pixels: int = 32 * 32
    image_do_pan_and_scan: bool = False
    compute_dtype: Optional[str] = "float32"
    block_diag_attn: bool = False
    device_map: Optional[str] = None
    low_cpu_mem_usage: bool = True
    attn_implementation: str = "eager"


@dataclass
class DataArguments:
    dataset: Optional[str] = "hh_rlhf_sft"
    dataset_dir: str = "/data"
    template: str = "default"
    cutoff_len: int = 512
    train_on_prompt: bool = False
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 1
    max_samples: Optional[int] = 200
    val_size: float = 0.1
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
class TrainingArguments:
    """The training hyperparameters the trainer and runtime read."""

    output_dir: str = "/tmp/"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    num_train_epochs: float = 3.0
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    save_steps: int = 1000000000
    save_total_limit: Optional[int] = None
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    pad_to_multiple_of: Optional[int] = None
    remove_unused_columns: bool = False
    report_to: Any = None
    bf16: bool = True
    fp16: bool = False
    do_train: bool = True
    do_eval: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


def _populate_dataclass(cls, config_dict: dict[str, Any]):

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return cls(**filtered)


def build_get_train_args(default_template: str):
    """Return a get_train_args with model-specific defaults."""

    def get_train_args(args: Optional[dict[str, Any]] = None):
        config = read_args(args)
        if isinstance(config, list):
            raise ValueError("Please provide a YAML or JSON configuration file.")

        merged_config = dict(config)
        merged_config.setdefault("template", default_template)

        model_args = _populate_dataclass(ModelArguments, merged_config)
        data_args = _populate_dataclass(DataArguments, merged_config)
        finetuning_args = _populate_dataclass(FinetuningArguments, merged_config)

        if finetuning_args.lora_alpha is None:
            finetuning_args.lora_alpha = finetuning_args.lora_rank * 2

        if model_args.compute_dtype is not None:
            model_args.compute_dtype = getattr(torch, model_args.compute_dtype)
        elif merged_config.get("bf16", True):
            model_args.compute_dtype = torch.bfloat16
        elif merged_config.get("fp16", False):
            model_args.compute_dtype = torch.float16
        else:
            model_args.compute_dtype = torch.float32

        training_field_names = {f.name for f in dataclasses.fields(TrainingArguments)}
        training_kwargs = {k: v for k, v in merged_config.items() if k in training_field_names}
        training_kwargs.setdefault("output_dir", merged_config.get("output_dir", "/tmp/"))
        training_kwargs.setdefault("per_device_train_batch_size", merged_config.get("per_device_train_batch_size", 1))
        training_kwargs.setdefault("per_device_eval_batch_size", merged_config.get("per_device_eval_batch_size", 1))
        training_kwargs.setdefault("gradient_accumulation_steps", merged_config.get("gradient_accumulation_steps", 1))
        training_kwargs.setdefault("logging_steps", merged_config.get("logging_steps", 1))
        training_kwargs.setdefault("save_steps", merged_config.get("save_steps", 1000000000))
        training_kwargs.setdefault("remove_unused_columns", False)
        training_kwargs.setdefault("report_to", merged_config.get("report_to", []))
        training_kwargs.setdefault("bf16", merged_config.get("bf16", True))
        training_kwargs.setdefault("fp16", merged_config.get("fp16", False))
        training_kwargs.setdefault("gradient_checkpointing", merged_config.get("gradient_checkpointing", False))
        training_kwargs.setdefault("do_train", merged_config.get("do_train", True))
        training_kwargs.setdefault("seed", merged_config.get("seed", 42))
        training_args = _populate_dataclass(TrainingArguments, training_kwargs)

        return model_args, data_args, training_args, finetuning_args

    return get_train_args



DEFAULT_TEMPLATE = "mistral"

get_train_args = build_get_train_args(DEFAULT_TEMPLATE)
