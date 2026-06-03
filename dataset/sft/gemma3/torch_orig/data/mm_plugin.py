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

"""Multimodal plugin for Gemma3 image processing in SFT."""
import logging
import re
from typing import Any, Optional, Sequence

from .constants import IMAGE_PLACEHOLDER

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


def _get_gemma3_token_type_ids(input_ids, image_token_id):
    """Compute token_type_ids for Gemma3: 0 for text, 1 for image tokens."""
    import torch
    if isinstance(input_ids, list):
        token_type_ids = [1 if tid == image_token_id else 0 for tid in input_ids]
    elif isinstance(input_ids, torch.Tensor):
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == image_token_id] = 1
    else:
        import numpy as np
        token_type_ids = np.zeros_like(input_ids)
        token_type_ids[input_ids == image_token_id] = 1
    return token_type_ids


class Gemma3Plugin(BasePlugin):
    """Gemma3-specific multimodal plugin for image handling."""

    def __init__(self, image_token="<image_soft_token>", **kwargs):
        super().__init__(image_token=image_token, **kwargs)

    def process_messages(self, messages, images=None, videos=None, processor=None, **kwargs):
        """Replace <image> placeholders with Gemma3 image token sequences."""
        if images is None or processor is None:
            return messages

        image_seq_length = getattr(processor, "image_seq_length", 256)
        boi_token = getattr(processor, "boi_token", "<start_of_image>")
        eoi_token = getattr(processor, "eoi_token", "<end_of_image>")
        image_token = self.image_token

        # Build full image token sequence
        image_tokens = image_token * image_seq_length
        full_image_seq = f"\n\n{boi_token}{image_tokens}{eoi_token}\n\n"

        processed = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content", "")
            if isinstance(content, str):
                # Replace <image> with full token sequence
                content = content.replace(IMAGE_PLACEHOLDER, full_image_seq)
                new_msg["content"] = content
            processed.append(new_msg)

        return processed

    def process_token_ids(self, input_ids, labels=None, images=None, tokenizer=None, processor=None, **kwargs):
        """No special token ID processing needed for Gemma3."""
        return input_ids, labels

    def get_mm_inputs(self, images=None, videos=None, audios=None, processor=None, **kwargs):
        """Process images and return pixel values and token_type_ids."""
        if images is None or processor is None:
            return {}

        import torch
        import numpy as np
        try:
            from PIL import Image
        except ImportError:
            Image = None

        # Preprocess images
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

        # Use the image processor from the processor
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
                result["pixel_values"] = torch.tensor(np.array(pixel_values))

        return result


# ===================== Plugin Registry =====================

PLUGINS = {
    "base": BasePlugin,
    "gemma3": Gemma3Plugin,
}


def get_mm_plugin(name: str, **kwargs) -> BasePlugin:
    """Get a multimodal plugin by name."""
    plugin_cls = PLUGINS.get(name, BasePlugin)
    return plugin_cls(**kwargs)
