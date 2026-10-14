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

"""LLaVA-Next processor combining image processor and tokenizer."""
import json
import logging
import os
from typing import Optional, Union

import numpy as np

from .image_processing import LlavaNextImageProcessor, _resolve_model_path, select_best_resolution

logger = logging.getLogger(__name__)

# The subset of ``processor_config.json`` this processor understands.
_PROCESSOR_CONFIG_KEYS = (
    "patch_size",
    "vision_feature_select_strategy",
    "image_token",
    "num_additional_image_tokens",
)


def _read_json(model_path: str, file_name: str) -> Optional[dict]:
    path = os.path.join(_resolve_model_path(model_path), file_name)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def load_processor_config(model_path: str) -> dict:
    """Read the processor knobs HF stores alongside the weights.

    ``patch_size`` here is the *vision* patch size (14 for the CLIP tower), not the 336-px
    crop the image processor works in. ``_get_number_of_features`` divides the crop by it
    to get the feature grid, so reading it from the checkpoint is what keeps the expanded
    token count equal to what the vision tower emits. ``config.json`` is the fallback
    because ``vision_config.patch_size`` is the same number and is always present.
    """
    raw = _read_json(model_path, "processor_config.json") or {}
    config = {key: raw[key] for key in _PROCESSOR_CONFIG_KEYS if raw.get(key) is not None}
    if "patch_size" not in config:
        vision_config = (_read_json(model_path, "config.json") or {}).get("vision_config") or {}
        if vision_config.get("patch_size") is not None:
            config["patch_size"] = int(vision_config["patch_size"])
    return config


class LlavaNextProcessor:
    """Processor combining image processor and tokenizer for LLaVA-Next."""

    def __init__(
        self,
        image_processor: Optional[LlavaNextImageProcessor] = None,
        tokenizer=None,
        patch_size: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = "default",
        image_token: str = "<image>",
        num_additional_image_tokens: int = 0,
    ):
        self.image_processor = image_processor or LlavaNextImageProcessor()
        self.tokenizer = tokenizer
        if patch_size is None:
            # Not the vision patch size, so the feature grid collapses to 1x1 and the
            # expansion silently under-counts; load_processor_config avoids this path.
            patch_size = self.image_processor.crop_size.get("height", 224)
            logger.warning("No patch_size given; falling back to the crop size (%s).", patch_size)
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        self.num_additional_image_tokens = num_additional_image_tokens

        # Get image token ID from tokenizer
        if tokenizer is not None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token_id = None

        # For compatibility with LlamaFactory's mm_plugin
        self.image_seq_length = None  # Dynamic, computed per image

    @classmethod
    def from_pretrained(cls, model_path, tokenizer=None, image_processor=None, **kwargs):
        from .tokenization import Tokenizer
        from .image_processing import LlavaNextImageProcessor

        if tokenizer is None:
            tokenizer = Tokenizer(model_path)
        if image_processor is None:
            image_processor = LlavaNextImageProcessor.from_pretrained(model_path)
        config = load_processor_config(model_path)
        config.update(kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer, **config)

    def _get_number_of_features(self, orig_height, orig_width, height, width):
        """Calculate the number of image tokens needed for a given image."""
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height = height_best_resolution // height
        scale_width = width_best_resolution // width

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size

        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )

        # Base patch features (+ CLS token)
        base_features = patches_height * patches_width + self.num_additional_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """Get number of features accounting for unpadding."""
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
        return unpadded_features, newline_features

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        """Process images and text together."""
        if images is None and text is None:
            raise ValueError("Provide at least images or text.")

        # Process images
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors)

        # Process text (expand image tokens if needed)
        if text is not None:
            if isinstance(text, str):
                text = [text]

            if image_inputs and "image_sizes" in image_inputs:
                pixel_values = image_inputs["pixel_values"]
                image_sizes = image_inputs["image_sizes"]

                # Get processed image height/width
                if isinstance(pixel_values, np.ndarray):
                    h, w = pixel_values.shape[-2], pixel_values.shape[-1]
                else:
                    h, w = pixel_values[0].shape[-2], pixel_values[0].shape[-1]

                image_sizes_iter = iter(
                    image_sizes.tolist() if isinstance(image_sizes, np.ndarray) else image_sizes
                )

                expanded_text = []
                for sample in text:
                    while self.image_token in sample:
                        image_size = next(image_sizes_iter)
                        orig_h, orig_w = image_size
                        num_tokens = self._get_number_of_features(orig_h, orig_w, h, w)
                        if self.vision_feature_select_strategy == "default":
                            num_tokens -= 1
                        sample = sample.replace(self.image_token, "<placeholder>" * num_tokens, 1)
                    expanded_text.append(sample)
                text = [s.replace("<placeholder>", self.image_token) for s in expanded_text]

            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        else:
            text_inputs = {}

        result = {**text_inputs, **image_inputs}
        return result
