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

"""Torch-only helpers shared by the image/video processors."""

from typing import Optional

import numpy as np
import torch

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


class BatchFeature(dict):
    """Implementation of ``transformers.BatchFeature``."""

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
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    self[key] = torch.tensor(np.array(value))
            return self
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                continue
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                self[key] = np.asarray(np.array(value))
        return self

    def to(self, device):

        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self
