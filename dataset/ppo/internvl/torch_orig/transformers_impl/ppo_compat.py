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

"""PPO compatibility utilities: GenerationConfig and get_scheduler.

These are lightweight implementations that avoid importing the real
HuggingFace transformers package (which conflicts with the local shadow).
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class GenerationConfig:
    """Minimal generation configuration for PPO rollouts.

    Compatible with HuggingFace's model.generate() interface.
    """
    # Token IDs
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int | list[int]] = None
    bos_token_id: Optional[int] = None
    # Sampling
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    # Length
    max_new_tokens: Optional[int] = 256
    max_length: Optional[int] = None
    min_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, num_warmup_steps: int, num_training_steps: int,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, num_warmup_steps: int, num_training_steps: int,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
    )


def _get_constant_schedule_with_warmup_lr_lambda(
    current_step: int, num_warmup_steps: int,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int = 0,
    num_training_steps: int = 0,
) -> LambdaLR:
    """Create learning rate scheduler.

    Supports: cosine, linear, constant, constant_with_warmup.
    """
    if name == "cosine":
        lr_lambda = lambda step: _get_cosine_schedule_with_warmup_lr_lambda(
            step, num_warmup_steps, num_training_steps
        )
    elif name == "linear":
        lr_lambda = lambda step: _get_linear_schedule_with_warmup_lr_lambda(
            step, num_warmup_steps, num_training_steps
        )
    elif name == "constant":
        lr_lambda = lambda step: 1.0
    elif name == "constant_with_warmup":
        lr_lambda = lambda step: _get_constant_schedule_with_warmup_lr_lambda(
            step, num_warmup_steps
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Supported: cosine, linear, constant, constant_with_warmup"
        )
    return LambdaLR(optimizer, lr_lambda)
