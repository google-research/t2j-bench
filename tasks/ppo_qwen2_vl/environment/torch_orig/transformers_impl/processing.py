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

"""Standalone Qwen2-VL processor - no transformers dependency."""

from typing import Optional

from .image_processing import Qwen2VLImageProcessor


class Qwen2VLProcessor:
    """Bundle of the image processor and tokenizer, as the data pipeline expects it."""

    image_token: str = "<|image_pad|>"
    video_token: str = "<|video_pad|>"
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer


    def _token_id(self, token: str):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return None
        return tokenizer.convert_tokens_to_ids(token)

    @property
    def image_token_id(self):
        return self._token_id(self.image_token)

    @property
    def video_token_id(self):
        return self._token_id(self.video_token)

    def __call__(self, images=None, return_tensors=None, **kwargs):
        if images is None:
            raise ValueError("This processor only handles images; text is tokenized upstream.")
        return self.image_processor(images, return_tensors=return_tensors, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, tokenizer: Optional[object] = None, **kwargs):
        image_processor = Qwen2VLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer)
