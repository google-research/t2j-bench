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

"""Standalone Qwen3-VL processor - no transformers dependency."""

import json
import logging
import os
from typing import Optional, Union

import numpy as np

from huggingface_hub import hf_hub_download

from .utils import BatchFeature
from .image_processing import Qwen2VLImageProcessor
from .video_processing import Qwen3VLVideoProcessor
from .tokenization import Tokenizer

logger = logging.getLogger(__name__)


class Qwen3VLProcessor:
    """Processor combining tokenizer, image processor, and video processor for Qwen3-VL."""

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor

        # Token strings and ids
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or (
            tokenizer.convert_tokens_to_ids(self.image_token) if tokenizer else None
        )
        self.video_token_id = getattr(tokenizer, "video_token_id", None) or (
            tokenizer.convert_tokens_to_ids(self.video_token) if tokenizer else None
        )
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.vision_start_token_id = getattr(tokenizer, "vision_start_token_id", None) or (
            tokenizer.convert_tokens_to_ids(self.vision_start_token) if tokenizer else None
        )
        self.vision_end_token_id = getattr(tokenizer, "vision_end_token_id", None) or (
            tokenizer.convert_tokens_to_ids(self.vision_end_token) if tokenizer else None
        )

    def __call__(self, images=None, text=None, videos=None, return_tensors=None, **kwargs):
        """Process images, videos, and text for Qwen3-VL."""
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor.preprocess(images, return_tensors=return_tensors)
            image_grid_thw = image_inputs.get("image_grid_thw")

        videos_inputs = {}
        video_grid_thw = None
        if videos is not None:
            videos_inputs = self.video_processor.preprocess(videos, return_tensors=return_tensors)
            video_grid_thw = videos_inputs.get("video_grid_thw")

        if not isinstance(text, list):
            text = [text]
        text = list(text)  # copy

        # Replace image placeholders with correct number of tokens
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    thw = image_grid_thw[index]
                    if isinstance(thw, np.ndarray):
                        num_image_tokens = int(np.prod(thw)) // merge_length
                    else:
                        num_image_tokens = int(thw[0] * thw[1] * thw[2]) // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Replace video placeholders with timestamps and correct number of tokens
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    thw = video_grid_thw[index]
                    if isinstance(thw, np.ndarray):
                        frame_seqlen = int(np.prod(thw[1:])) // merge_length
                        num_t = int(thw[0])
                    else:
                        frame_seqlen = int(thw[1] * thw[2]) // merge_length
                        num_t = int(thw[0])

                    video_placeholder = ""
                    for frame_idx in range(num_t):
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )

                    full_token = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                    if full_token in text[i]:
                        text[i] = text[i].replace(full_token, video_placeholder, 1)
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors=return_tensors)

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        """Calculate timestamps for video frames."""
        if not isinstance(indices, list):
            indices = list(indices)
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load processor from pretrained model path or HF hub."""
        image_processor = Qwen2VLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        video_processor = Qwen3VLVideoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor)


class AutoProcessor:
    """Auto processor class."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return Qwen3VLProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

