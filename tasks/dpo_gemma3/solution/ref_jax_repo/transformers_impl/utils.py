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

"""Utility classes and functions for Gemma3 standalone implementation."""
import math
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



# Activation functions
def gelu(x):
    return 0.5*x*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))

def silu(x):
    return x*_np_sigmoid(x)

def relu(x):
    return np.maximum(x,0)

def gelu_pytorch_tanh(x):
    return 0.5*x*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))

def quick_gelu(x):
    return x * _np_sigmoid(1.702 * x)


ACT2FN = {
    "gelu": gelu,
    "silu": silu,
    "swish": silu,
    "relu": relu,
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "gelu_new": gelu_pytorch_tanh,
    "quick_gelu": quick_gelu,
}


class ModelOutput:
    """Base class for model outputs supporting both attribute and index access."""

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        elif isinstance(key, int):
            return list(self.__dict__.values())[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[Any] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: Optional[Any] = None
    pooler_output: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[Any] = None
    logits: Optional[Any] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class BatchFeature(dict):
    """Container for batch features with tensor conversion."""

    def __init__(self, data=None, tensor_type=None):
        super().__init__(data or {})
        if tensor_type == "pt":
            self.convert_to_tensors("pt")

    def convert_to_tensors(self, tensor_type="pt"):
        for key, value in self.items():
            if isinstance(value, list):
                try:
                    if len(value) > 0 and isinstance(value[0], (list, np.ndarray)):
                        self[key] = np.asarray(value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(value, np.ndarray):
                self[key] = np.asarray(value)
        return self

    def to(self, device=None, dtype=None):
        new_data = {}
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                new_data[k] = v.astype(dtype) if dtype else v
            else:
                new_data[k] = v
        return BatchFeature(new_data)
