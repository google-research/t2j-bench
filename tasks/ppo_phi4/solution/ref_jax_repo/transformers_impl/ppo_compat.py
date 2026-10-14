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

"""PPO compatibility utilities: GenerationConfig and get_scheduler.

Mirror of ``torch_orig/transformers_impl/ppo_compat.py``. JAX adjustment: torch's
``get_scheduler`` returns a ``torch.optim.lr_scheduler.LambdaLR`` wrapping an
optimizer; here it returns an **optax schedule** — a ``step -> learning_rate``
callable — that the trainer folds into ``optax.adamw``. The per-step multipliers are
the same cosine/linear/constant lambdas as torch, scaled by the base learning rate.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import optax


@dataclass
class GenerationConfig:
    """Minimal generation configuration for PPO rollouts (torch-free)."""

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


def _cosine_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _linear_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def _constant_warmup_lambda(current_step: int, num_warmup_steps: int) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0


def get_scheduler(
    name: str,
    num_warmup_steps: int = 0,
    num_training_steps: int = 0,
    base_lr: float = 1.0,
) -> Callable[[int], float]:
    """Create an optax learning-rate schedule (``step -> lr``).

    Built from optax primitives so the schedule is **jit-traceable** (the step count is a
    tracer inside a jitted optimizer update). Supports cosine / linear / constant /
    constant_with_warmup — the same shapes as torch's ``ppo_compat.get_scheduler`` (linear
    warmup then decay), scaled by ``base_lr``.
    """
    import optax

    nt = max(1, num_training_steps)
    nw = max(0, num_warmup_steps)
    decay_steps = max(1, nt - nw)
    if name == "cosine":
        decay = optax.cosine_decay_schedule(base_lr, decay_steps)
    elif name == "linear":
        decay = optax.linear_schedule(base_lr, 0.0, decay_steps)
    elif name in ("constant", "constant_with_warmup"):
        decay = optax.constant_schedule(base_lr)
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. Supported: cosine, linear, constant, constant_with_warmup"
        )
    if nw > 0:
        warmup = optax.linear_schedule(0.0, base_lr, nw)
        return optax.join_schedules([warmup, decay], [nw])
    return decay
