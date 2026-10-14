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

"""Utility classes and functions replacing transformers internals."""

import math
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def gelu(x):
    return F.gelu(x)

def silu(x):
    return F.silu(x)

def relu(x):
    return F.relu(x)

def gelu_pytorch_tanh(x):
    return F.gelu(x, approximate="tanh")

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

ACT2FN = {
    "gelu": gelu,
    "relu": relu,
    "silu": silu,
    "swish": silu,
    "gelu_new": gelu,
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "quick_gelu": quick_gelu,
}



class ModelOutput(OrderedDict):
    """Base class for model outputs, similar to transformers.ModelOutput."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if not hasattr(self, key):
                object.__setattr__(self, key, value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

    def __post_init__(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                self[f.name] = val


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

    def __post_init__(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                self[f.name] = val



class BatchFeature(dict):
    """A dict subclass that can convert values to tensors."""

    def __init__(self, data: Optional[dict] = None, tensor_type: Optional[str] = None):
        super().__init__(data or {})
        if tensor_type is not None:
            self.convert_to_tensors(tensor_type)

    def convert_to_tensors(self, tensor_type: str = "pt"):
        if tensor_type == "pt":
            import torch
            for key, value in self.items():
                if isinstance(value, np.ndarray):
                    self[key] = torch.tensor(value)
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        self[key] = torch.tensor(np.array(value))
        return self

    def to(self, device):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self



OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def is_valid_image(img) -> bool:
    """Check if img is a valid image (PIL, numpy, or torch tensor)."""
    try:
        from PIL.Image import Image as PILImage
        if isinstance(img, PILImage):
            return True
    except ImportError:
        pass
    if isinstance(img, np.ndarray):
        return True
    if isinstance(img, torch.Tensor):
        return True
    return False

