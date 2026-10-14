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

"""Multimodal plugin for PaliGemma image processing."""

import math
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class BasePlugin:
    """Base multimodal plugin."""

    image_token: Optional[str] = None
    video_token: Optional[str] = None
    audio_token: Optional[str] = None
    expand_mm_tokens: bool = True

    def _preprocess_image(
        self, image: Any, image_max_pixels: int = 768 * 768, image_min_pixels: int = 32 * 32, **kwargs
    ) -> Any:
        """Resize one image into the [min_pixels, max_pixels] band and force RGB."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _regularize_images(self, images: list[Any], **kwargs) -> dict[str, list[Any]]:
        """Load images from paths/bytes/dicts and normalize them."""
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image.get("bytes") is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])
            results.append(self._preprocess_image(image, **kwargs))
        return {"images": results}

    def _image_kwargs(self, processor: Any) -> dict[str, int]:
        """Sizing knobs that ``model/loader.py`` copies from ModelArguments onto the processor."""
        return {
            "image_max_pixels": getattr(processor, "image_max_pixels", 768 * 768),
            "image_min_pixels": getattr(processor, "image_min_pixels", 32 * 32),
        }

    def _validate_input(self, processor: Any, images: list[Any], videos: list[Any], audios: list[Any]) -> None:
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")
        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")
        if len(audios) != 0 and self.audio_token is None:
            raise ValueError("This model does not support audio input.")
        if len(images) != 0 and processor is None:
            raise ValueError(
                "Image input needs a processor, but load_tokenizer returned none. Check that the "
                "image processor's vision backend (torchvision) is installed."
            )

    def _validate_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any]
    ) -> None:
        num_images = sum(message["content"].count(IMAGE_PLACEHOLDER) for message in messages)
        num_videos = sum(message["content"].count(VIDEO_PLACEHOLDER) for message in messages)
        if len(images) != num_images:
            raise ValueError(f"Got {len(images)} images but {num_images} {IMAGE_PLACEHOLDER} tokens.")
        if len(videos) != num_videos:
            raise ValueError(f"Got {len(videos)} videos but {num_videos} {VIDEO_PLACEHOLDER} tokens.")

    def _get_mm_inputs(self, images: list[Any], videos: list[Any], audios: list[Any], processor: Any) -> dict[str, Any]:
        """Run the image processor over the regularized images."""
        if len(images) == 0:
            return {}
        image_processor = getattr(processor, "image_processor", processor)
        images = self._regularize_images(images, **self._image_kwargs(processor))["images"]
        return dict(image_processor(images, return_tensors="pt"))

    def process_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any], processor: Any
    ) -> list[dict[str, str]]:
        """Expand media placeholders in the message text, before tokenization."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        tokenizer: Any,
        processor: Any,
    ) -> tuple[list[int], Optional[list[int]]]:
        """Adjust token ids after tokenization, for models that prepend image tokens."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Any,
    ) -> dict[str, Any]:
        """Build the batched tensor inputs (pixel_values, grids, ...) handed to the model."""
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: Any) -> list[list[int]]:
    """0 over the prefix (image tokens + prompt), 1 over the suffix."""
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seq_length", 0)
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


@dataclass
class PaliGemmaPlugin(BasePlugin):
    """PaliGemma: image tokens are *prepended* to the sequence rather than inlined."""

    def process_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any], processor: Any
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        for message in messages:
            message["content"] = message["content"].replace(IMAGE_PLACEHOLDER, "")

        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        tokenizer: Any,
        processor: Any,
    ) -> tuple[list[int], Optional[list[int]]]:
        self._validate_input(processor, images, videos, audios)
        num_images = len(images)
        image_seqlen = num_images * getattr(processor, "image_seq_length", 0) if self.expand_mm_tokens else 0
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels

        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Any,
    ) -> dict[str, Any]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


PLUGINS = {
    "base": BasePlugin,
    "paligemma": PaliGemmaPlugin,
}

_SPEC = {"name": "paligemma", "image_token": "<image>"}
PLUGIN_SPECS = {"default": _SPEC, "paligemma": _SPEC}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
    **kwargs,
) -> BasePlugin:
    """Instantiate a multimodal plugin by registry name."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
