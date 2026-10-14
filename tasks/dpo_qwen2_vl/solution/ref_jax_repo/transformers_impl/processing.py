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

"""Standalone Qwen2-VL processor - no transformers dependency.

Only the surface ``data/mm_plugin.py`` actually touches: the image processor (for
``merge_size`` and the ``pixel_values`` / ``image_grid_thw`` pair) and the tokenizer.
Text goes through ``tokenizer.apply_chat_template`` in ``data/template.py``, never
through a processor ``__call__``, so no placeholder expansion lives here.
"""

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

    # --- multimodal token ids ------------------------------------------------ #
    # torch's Qwen2VLProcessor resolves these off the tokenizer in __init__
    # (``image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)``). The PPO
    # data collator reads them to derive ``token_type_ids``, so a processor that exposes
    # only the token STRINGS silently drops that key from the batch. Resolved lazily
    # rather than in __init__ so the class stays a thin bundle and a processor built
    # without a tokenizer just has no ids -- which is what ``getattr(..., None)`` at the
    # call site already expects.

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
