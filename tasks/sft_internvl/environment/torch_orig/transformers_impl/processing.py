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

"""Standalone InternVL processor -- no transformers dependency."""

import json
import os
from typing import Optional

from .image_processing import InternVLImageProcessor


class InternVLProcessor:
    """Bundle of the image processor, tokenizer, and the fixed image-token budget."""

    image_token: str = "<IMG_CONTEXT>"
    start_image_token: str = "<img>"
    end_image_token: str = "</img>"

    def __init__(self, image_processor=None, tokenizer=None, image_seq_length: int = 256, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_length = image_seq_length

    def __call__(self, images=None, return_tensors=None, **kwargs):
        if images is None:
            raise ValueError("This processor only handles images; text is tokenized upstream.")
        return self.image_processor(images, return_tensors=return_tensors, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, tokenizer: Optional[object] = None, **kwargs):
        image_processor = InternVLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_seq_length = 256
        processor_config_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        if os.path.isfile(processor_config_path):
            with open(processor_config_path, "r") as handle:
                image_seq_length = json.load(handle).get("image_seq_length", image_seq_length)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=image_seq_length,
        )


__all__ = ["InternVLProcessor"]
