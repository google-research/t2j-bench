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

"""Argument parsing + config dataclasses — mirror of ``torch_orig/config.py`` (SFT).

Same public surface: ``read_args``, ``ModelArguments``, ``DataArguments``,
``TrainingArguments``, ``FinetuningArguments``, ``_populate_dataclass``,
``get_train_args`` — which returns the SFT 4-tuple
``(model, data, training, finetuning)``.

JAX adjustments:
- ``compute_dtype`` is resolved to a **string** (``"bfloat16"``/``"float16"``/
  ``"float32"``) — the flax modules turn it into a ``jnp`` dtype — instead of a
  ``torch.dtype``.
- ``TrainingArguments`` is a plain dataclass exposing exactly the fields the trainer
  reads, instead of HF ``transformers.TrainingArguments`` (torch-free).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_TEMPLATE = "qwen3_vl_nothink"


def read_args(args: Optional[dict[str, Any]] = None) -> dict[str, Any] | list[str]:
    """Return a config dict (mirror of torch ``read_args``)."""
    if args is not None:
        return args
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        from omegaconf import OmegaConf

        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        from omegaconf import OmegaConf

        override_config = OmegaConf.from_cli(sys.argv[2:])
        with open(Path(sys.argv[1]).absolute()) as f:
            dict_config = OmegaConf.create(json.load(f))
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    return sys.argv[1:]


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    hf_hub_token: Optional[str] = None
    use_fast_tokenizer: bool = True
    resize_vocab: bool = False
    image_max_pixels: int = 224 * 224
    image_min_pixels: int = 32 * 32
    image_do_pan_and_scan: bool = False
    compute_dtype: Optional[str] = "float32"
    block_diag_attn: bool = False
    device_map: Optional[str] = "auto"
    low_cpu_mem_usage: bool = True
    attn_implementation: str = "eager"


@dataclass
class DataArguments:
    dataset: Optional[str] = "geometry3k"
    dataset_dir: str = "/data"
    template: str = "qwen3_vl_nothink"
    cutoff_len: int = 512
    train_on_prompt: bool = False
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 4
    max_samples: Optional[int] = 200
    val_size: float = 0.1
    default_system: Optional[str] = None
    media_dir: Optional[str] = None
    tool_format: Optional[str] = None


@dataclass
class TrainingArguments:
    """The training hyperparameters the trainer reads (JAX stand-in for HF
    ``transformers.TrainingArguments``). Defaults match HF/torch_orig where the
    goldens rely on them (``max_grad_norm=1.0``, ``lr_scheduler_type='cosine'``)."""

    output_dir: str = "/tmp/"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 3.0
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    save_steps: int = 1000000000
    save_total_limit: Optional[int] = 3
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    pad_to_multiple_of: Optional[int] = 8
    predict_with_generate: bool = False
    bf16: bool = True
    fp16: bool = False
    do_train: bool = True
    do_eval: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


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
    """Filter ``config_dict`` to ``cls``'s fields and construct it (mirror of torch).
    Keys mapped to an explicit ``None`` are dropped so the dataclass default wins."""
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields and v is not None}
    return cls(**filtered)


def get_train_args(args: Optional[dict[str, Any]] = None):
    """Build ``(model, data, training, finetuning)`` args (mirror of torch)."""
    config = read_args(args)
    if isinstance(config, list):
        raise ValueError("Please provide a YAML or JSON configuration file.")

    merged_config = dict(config)
    merged_config.setdefault("template", DEFAULT_TEMPLATE)

    model_args = _populate_dataclass(ModelArguments, merged_config)
    data_args = _populate_dataclass(DataArguments, merged_config)
    training_args = _populate_dataclass(TrainingArguments, merged_config)
    finetuning_args = _populate_dataclass(FinetuningArguments, merged_config)

    if finetuning_args.lora_alpha is None:
        finetuning_args.lora_alpha = finetuning_args.lora_rank * 2

    # JAX adjustment: keep compute_dtype as a string (resolved to a jnp dtype in the
    # flax modeling / loader), not a torch dtype.
    if model_args.compute_dtype is None:
        if merged_config.get("bf16", True):
            model_args.compute_dtype = "bfloat16"
        elif merged_config.get("fp16", False):
            model_args.compute_dtype = "float16"
        else:
            model_args.compute_dtype = "float32"

    return model_args, data_args, training_args, finetuning_args
