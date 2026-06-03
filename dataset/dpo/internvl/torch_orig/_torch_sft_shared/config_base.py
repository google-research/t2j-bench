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

"""Config helpers for generated pure-Torch SFT repos."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def read_args(args: Optional[dict[str, Any]] = None) -> dict[str, Any] | list[str]:
    """Read YAML/JSON config or raw CLI args."""
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


def _populate_dataclass(cls, config_dict: dict[str, Any]):
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return cls(**filtered)


def build_get_train_args(default_template: str):
    """Return a repo-local get_train_args with model-specific defaults."""
    from transformers import TrainingArguments as HfTrainingArguments
    import torch

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

        training_field_names = set(HfTrainingArguments.__dataclass_fields__.keys())
        training_kwargs = {k: v for k, v in merged_config.items() if k in training_field_names}
        training_kwargs.setdefault("output_dir", merged_config.get("output_dir", "output"))
        training_kwargs.setdefault("per_device_train_batch_size", merged_config.get("per_device_train_batch_size", 1))
        training_kwargs.setdefault("per_device_eval_batch_size", merged_config.get("per_device_eval_batch_size", 1))
        training_kwargs.setdefault("gradient_accumulation_steps", merged_config.get("gradient_accumulation_steps", 8))
        training_kwargs.setdefault("learning_rate", merged_config.get("learning_rate", 1e-4))
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

        return model_args, data_args, training_args, finetuning_args

    return get_train_args

