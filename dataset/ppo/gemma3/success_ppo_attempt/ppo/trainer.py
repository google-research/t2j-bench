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

"""PPO trainer utilities for JAX/Flax Gemma3 pipeline.

Contains scheduler helpers, parameter splitting, and deep merge functions.
"""

import math


def create_cosine_schedule(learning_rate, num_warmup, total_steps):
    def schedule_fn(current_step):
        current_step = int(current_step)
        if current_step < num_warmup:
            return learning_rate * float(current_step) / float(max(1, num_warmup))
        progress = float(current_step - num_warmup) / float(max(1, total_steps - num_warmup))
        return learning_rate * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return schedule_fn


def create_linear_schedule(learning_rate, num_warmup, total_steps):
    def schedule_fn(current_step):
        current_step = int(current_step)
        if current_step < num_warmup:
            return learning_rate * float(current_step) / float(max(1, num_warmup))
        progress = float(current_step - num_warmup) / float(max(1, total_steps - num_warmup))
        return learning_rate * max(0.0, 1.0 - progress)
    return schedule_fn


SCHEDULER_MAP = {"cosine": create_cosine_schedule, "linear": create_linear_schedule}


def _split_by_prefixes(params, frozen_prefixes):
    """Split param tree into (trainable, frozen).
    LoRA params (path contains '/lora_A/' or '/lora_B/') and v_head params
    are always trainable even if they match a frozen prefix."""
    trainable, frozen = {}, {}
    def _walk(node, path_parts, t_dst, f_dst):
        for key, value in node.items():
            full_path = "/".join(path_parts + [key])
            is_lora = "/lora_A/" in ("/" + full_path + "/") or "/lora_B/" in ("/" + full_path + "/")
            is_vhead = full_path.startswith("v_head")
            is_frozen = any(full_path.startswith(pfx) for pfx in frozen_prefixes) and not is_lora and not is_vhead
            if isinstance(value, dict):
                t_child, f_child = {}, {}
                _walk(value, path_parts + [key], t_child, f_child)
                if t_child: t_dst[key] = t_child
                if f_child: f_dst[key] = f_child
            else:
                (f_dst if is_frozen else t_dst)[key] = value
    _walk(params, [], trainable, frozen)
    return trainable, frozen


def _deep_merge(frozen, trainable):
    merged = {}
    for key in set(frozen) | set(trainable):
        f_val, t_val = frozen.get(key), trainable.get(key)
        if isinstance(f_val, dict) and isinstance(t_val, dict):
            merged[key] = _deep_merge(f_val, t_val)
        elif t_val is not None:
            merged[key] = t_val
        else:
            merged[key] = f_val
    return merged
