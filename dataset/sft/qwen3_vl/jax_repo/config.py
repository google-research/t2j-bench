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

"""
Configuration system for Qwen3-VL SFT framework.
Supports YAML configuration files with CLI overrides using OmegaConf.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


def read_args(args: Optional[dict[str, Any]] = None) -> dict[str, Any] | list[str]:
    """Read arguments from YAML/JSON file or command line."""
    from omegaconf import OmegaConf

    if args is not None:
        return args

    if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        with open(Path(sys.argv[1]).absolute()) as f:
            dict_config = OmegaConf.create(json.load(f))
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    else:
        return sys.argv[1:]


@dataclass
class ModelArguments:
    """Arguments for model loading and configuration."""
    model_name_or_path: str = ""
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    hf_hub_token: Optional[str] = None
    use_fast_tokenizer: bool = True
    resize_vocab: bool = False
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = "auto"
    # Processor arguments for multimodal
    image_max_pixels: int = 768 * 768
    image_min_pixels: int = 32 * 32
    video_max_pixels: int = 256 * 256
    video_min_pixels: int = 16 * 16
    video_fps: float = 2.0
    video_maxlen: int = 128
    # Compute dtype
    compute_dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    block_diag_attn: bool = False
    # Device
    device_map: Optional[str] = "auto"
    low_cpu_mem_usage: bool = True


@dataclass
class DataArguments:
    """Arguments for dataset loading and processing."""
    dataset: Optional[str] = None  # Comma-separated dataset names
    dataset_dir: str = "data"
    template: Optional[str] = "qwen3_vl_nothink"
    cutoff_len: int = 2048
    train_on_prompt: bool = False
    mask_history: bool = False
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 4
    max_samples: Optional[int] = None
    val_size: float = 0.0
    streaming: bool = False
    # Tool and system message
    tool_format: Optional[str] = None
    default_system: Optional[str] = None
    enable_thinking: Optional[bool] = None


@dataclass
class TrainingArguments:
    """Arguments for training configuration."""
    output_dir: str = "output"
    do_train: bool = True
    do_eval: bool = False
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    report_to: str = "none"
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    resume_from_checkpoint: Optional[str] = None
    pad_to_multiple_of: Optional[int] = 8
    predict_with_generate: bool = False
    # Eval / generation
    generation_max_length: Optional[int] = None


@dataclass
class FinetuningArguments:
    """Arguments for finetuning configuration."""
    finetuning_type: Literal["full", "lora"] = "lora"
    # LoRA arguments
    lora_rank: int = 8
    lora_alpha: Optional[int] = None  # defaults to lora_rank * 2
    lora_dropout: float = 0.0
    lora_target: str = "all"  # comma-separated or "all"
    additional_target: Optional[str] = None
    use_rslora: bool = False
    use_dora: bool = False
    # Vision tower freezing
    freeze_vision_tower: bool = True
    freeze_multi_modal_projector: bool = True
    # Full tuning
    pure_bf16: bool = False


def _populate_dataclass(cls, config_dict: dict[str, Any]):
    """Populate a dataclass from a dictionary, ignoring unknown fields."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return cls(**filtered)


def get_train_args(args: Optional[dict[str, Any]] = None):
    """Parse and return all training arguments from YAML/CLI."""
    config = read_args(args)
    if isinstance(config, list):
        raise ValueError("Please provide a YAML or JSON configuration file.")

    model_args = _populate_dataclass(ModelArguments, config)
    data_args = _populate_dataclass(DataArguments, config)
    training_args = _populate_dataclass(TrainingArguments, config)
    finetuning_args = _populate_dataclass(FinetuningArguments, config)

    # Post-processing
    if finetuning_args.lora_alpha is None:
        finetuning_args.lora_alpha = finetuning_args.lora_rank * 2

    # Resolve compute dtype
    import torch
    if model_args.compute_dtype is not None:
        model_args.compute_dtype = getattr(torch, model_args.compute_dtype)
    elif training_args.bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
    else:
        model_args.compute_dtype = torch.float32

    return model_args, data_args, training_args, finetuning_args

