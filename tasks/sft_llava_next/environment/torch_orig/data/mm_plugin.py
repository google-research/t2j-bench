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

"""Multimodal plugin for LLaVA-Next image processing in SFT."""
import logging
from typing import Any, Optional, Sequence

import numpy as np

from .constants import IMAGE_PLACEHOLDER
import torch

logger = logging.getLogger(__name__)


class BasePlugin:
    """Base multimodal plugin with no-op image/video handling."""

    def __init__(self, image_token: Optional[str] = None, **kwargs):
        self.image_token = image_token

    def process_messages(self, messages, images=None, videos=None, processor=None, **kwargs):
        """Process messages - base implementation passes through."""
        return messages

    def process_token_ids(self, input_ids, labels=None, images=None, tokenizer=None, processor=None, **kwargs):
        """Process token IDs after tokenization."""
        return input_ids, labels

    def get_mm_inputs(self, images=None, videos=None, audios=None, processor=None, **kwargs):
        """Get multimodal inputs (pixel values, etc.)."""
        return {}

    @staticmethod
    def _preprocess_image(image, max_pixels=None, min_pixels=None):
        """Basic image preprocessing."""
        try:
            from PIL import Image
        except ImportError:
            return image

        if not isinstance(image, Image.Image):
            return image

        if image.mode != "RGB":
            image = image.convert("RGB")

        if max_pixels:
            w, h = image.size
            if w * h > max_pixels:
                ratio = (max_pixels / (w * h)) ** 0.5
                new_w, new_h = int(w * ratio), int(h * ratio)
                image = image.resize((new_w, new_h))

        return image


class LlavaNextPlugin(BasePlugin):
    """LLaVA-Next-specific multimodal plugin for image handling."""

    def __init__(self, image_token="<image>", **kwargs):
        super().__init__(image_token=image_token, **kwargs)

    def process_messages(self, messages, images=None, videos=None, processor=None, **kwargs):
        """Replace <image> placeholders with expanded image token sequences."""
        if images is None or processor is None:
            processed = []
            for msg in messages:
                new_msg = dict(msg)
                content = new_msg.get("content", "")
                if isinstance(content, str):
                    content = content.replace(IMAGE_PLACEHOLDER, self.image_token)
                    new_msg["content"] = content
                processed.append(new_msg)
            return processed

        mm_inputs = self._get_mm_inputs_for_sizing(images, processor)

        image_sizes_iter = None
        processed_height = None
        processed_width = None

        if mm_inputs and "pixel_values" in mm_inputs and "image_sizes" in mm_inputs:
            import torch
            pixel_values = mm_inputs["pixel_values"]
            image_sizes = mm_inputs["image_sizes"]

            if isinstance(image_sizes, torch.Tensor):
                image_sizes_iter = iter(image_sizes.tolist())
            elif isinstance(image_sizes, (list, np.ndarray)):
                image_sizes_iter = iter(image_sizes)

            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.dim() == 5:
                    processed_height = pixel_values.shape[-2]
                    processed_width = pixel_values.shape[-1]
                elif pixel_values.dim() == 4:
                    processed_height = pixel_values.shape[-2]
                    processed_width = pixel_values.shape[-1]
            elif isinstance(pixel_values, (list, np.ndarray)):
                first_pv = pixel_values[0]
                if isinstance(first_pv, np.ndarray):
                    if first_pv.ndim == 4:
                        processed_height = first_pv.shape[-2]
                        processed_width = first_pv.shape[-1]
                    elif first_pv.ndim == 3:
                        processed_height = first_pv.shape[-2]
                        processed_width = first_pv.shape[-1]

        processed = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str):
                while IMAGE_PLACEHOLDER in content:
                    if image_sizes_iter is not None and processed_height is not None:
                        try:
                            orig_height, orig_width = next(image_sizes_iter)
                            image_seqlen = processor._get_number_of_features(
                                orig_height, orig_width, processed_height, processed_width
                            )
                            if processor.vision_feature_select_strategy == "default":
                                image_seqlen -= 1
                        except StopIteration:
                            image_seqlen = 1
                    else:
                        image_seqlen = 1

                    content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

                content = content.replace("{{image}}", self.image_token)
                new_msg["content"] = content
            processed.append(new_msg)

        return processed

    def _get_mm_inputs_for_sizing(self, images, processor):
        """Get multimodal inputs just for sizing information."""
        return self.get_mm_inputs(images=images, processor=processor)

    def process_token_ids(self, input_ids, labels=None, images=None, tokenizer=None, processor=None, **kwargs):
        """No special token ID processing needed for LLaVA-Next."""
        return input_ids, labels

    def get_mm_inputs(self, images=None, videos=None, audios=None, processor=None, **kwargs):
        """Process images and return pixel values and image_sizes."""
        if images is None or processor is None:
            return {}

        try:
            from PIL import Image
        except ImportError:
            Image = None

        processed_images = []
        for img in images:
            if isinstance(img, str):
                if Image is not None:
                    img = Image.open(img).convert("RGB")
            elif Image is not None and isinstance(img, Image.Image):
                img = img.convert("RGB")
            processed_images.append(img)

        if not processed_images:
            return {}

        image_processor = getattr(processor, "image_processor", processor)
        if hasattr(image_processor, "preprocess"):
            image_inputs = image_processor.preprocess(processed_images, return_tensors="pt")
        else:
            image_inputs = image_processor(processed_images, return_tensors="pt")

        result = {}
        if "pixel_values" in image_inputs:
            pixel_values = image_inputs["pixel_values"]
            if isinstance(pixel_values, torch.Tensor):
                result["pixel_values"] = pixel_values
            elif isinstance(pixel_values, np.ndarray):
                result["pixel_values"] = torch.from_numpy(pixel_values)
            elif isinstance(pixel_values, list):
                result["pixel_values"] = torch.tensor(np.array(pixel_values), dtype=torch.float32)

        if "image_sizes" in image_inputs:
            image_sizes = image_inputs["image_sizes"]
            if isinstance(image_sizes, torch.Tensor):
                result["image_sizes"] = image_sizes
            elif isinstance(image_sizes, (list, np.ndarray)):
                result["image_sizes"] = torch.tensor(image_sizes, dtype=torch.long)

        return result



PLUGINS = {
    "base": BasePlugin,
    "llava_next": LlavaNextPlugin,
}


def get_mm_plugin(name: str, **kwargs) -> BasePlugin:
    """Get a multimodal plugin by name."""
    plugin_cls = PLUGINS.get(name, BasePlugin)
    return plugin_cls(**kwargs)
