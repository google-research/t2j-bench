# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone LLaVA-Next image processor - decoupled from HuggingFace."""
import json
import logging
import math
import os
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# CLIP normalization constants
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception:
        return model_path


def select_best_resolution(original_size, possible_resolutions):
    """Select best resolution from candidates based on input image size."""
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_patch_output_size(image, target_resolution, input_data_format="channels_last"):
    """Calculate output size after resize to target while maintaining aspect ratio."""
    if input_data_format == "channels_first":
        original_height, original_width = image.shape[1], image.shape[2]
    else:
        original_height, original_width = image.shape[0], image.shape[1]

    target_height, target_width = target_resolution
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


class LlavaNextImageProcessor:
    """Image processor for LLaVA-Next with multi-resolution support."""

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        image_grid_pinpoints: Optional[list] = None,
        do_center_crop: bool = True,
        crop_size: Optional[dict] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[list] = None,
        image_std: Optional[list] = None,
        do_pad: bool = True,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        self.do_resize = do_resize
        self.size = size or {"shortest_edge": 224}
        self.image_grid_pinpoints = image_grid_pinpoints or [
            [336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]
        ]
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size or {"height": 224, "width": 224}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Load from preprocessor_config.json."""
        model_path = _resolve_model_path(model_path)
        config_file = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            return cls(**{k: v for k, v in config.items() if k != "image_processor_type"})
        return cls(**kwargs)

    def _resize_image(self, image, target_size):
        """Resize image to target size using PIL."""
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        return image.resize((target_size[1], target_size[0]), Image.BICUBIC)

    def _center_crop(self, image, crop_h, crop_w):
        """Center crop a numpy array."""
        h, w = image.shape[:2]
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        return image[top:top + crop_h, left:left + crop_w]

    def _normalize(self, image, mean, std):
        """Normalize a numpy image array."""
        image = image.astype(np.float32)
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        if image.ndim == 3 and image.shape[2] == 3:
            image = (image - mean) / std
        return image

    def _preprocess_single(self, image):
        """Preprocess a single image patch (resize, crop, rescale, normalize)."""
        from PIL import Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Resize to shortest_edge
        if self.do_resize:
            shortest_edge = self.size.get("shortest_edge", 224)
            h, w = image.shape[:2]
            if h <= w:
                new_h = shortest_edge
                new_w = int(w * shortest_edge / h)
            else:
                new_w = shortest_edge
                new_h = int(h * shortest_edge / w)
            pil_img = self._resize_image(image, (new_h, new_w))
            image = np.array(pil_img)

        # Center crop
        if self.do_center_crop:
            crop_h = self.crop_size["height"]
            crop_w = self.crop_size["width"]
            image = self._center_crop(image, crop_h, crop_w)

        # Rescale
        if self.do_rescale:
            image = image.astype(np.float32) * self.rescale_factor

        # Normalize
        if self.do_normalize:
            image = self._normalize(image, self.image_mean, self.image_std)

        # Convert to CHW
        if image.ndim == 3 and image.shape[2] in (1, 3):
            image = np.transpose(image, (2, 0, 1))

        return image

    def _resize_for_patching(self, image, target_resolution):
        """Resize image maintaining aspect ratio to fit target resolution."""
        from PIL import Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        target_h, target_w = target_resolution
        scale_w = target_w / w
        scale_h = target_h / h

        if scale_w < scale_h:
            new_w = target_w
            new_h = min(math.ceil(h * scale_w), target_h)
        else:
            new_h = target_h
            new_w = min(math.ceil(w * scale_h), target_w)

        pil_img = self._resize_image(image, (new_h, new_w))
        return np.array(pil_img)

    def _pad_for_patching(self, image, target_resolution):
        """Pad image to exactly match target resolution."""
        h, w = image.shape[:2]
        target_h, target_w = target_resolution
        pad_h = target_h - h
        pad_w = target_w - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if image.ndim == 3:
            padded = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)
        else:
            padded = np.pad(image, ((top, bottom), (left, right)), mode="constant", constant_values=0)
        return padded

    def get_image_patches(self, image, grid_pinpoints, base_size, patch_size):
        """Divide image into patches for multi-resolution processing."""
        from PIL import Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        best_resolution = select_best_resolution((h, w), grid_pinpoints)

        # Resize and pad to best resolution
        resized = self._resize_for_patching(image, best_resolution)
        padded = self._pad_for_patching(resized, best_resolution)

        # Divide into patches
        patches = []
        ph, pw = padded.shape[:2]
        for i in range(0, ph, patch_size):
            for j in range(0, pw, patch_size):
                if padded.ndim == 3:
                    patch = padded[i:i + patch_size, j:j + patch_size]
                else:
                    patch = padded[i:i + patch_size, j:j + patch_size]
                patches.append(patch)

        # Base image (resized to base_size)
        base_img_pil = self._resize_image(image, base_size)
        base_img = np.array(base_img_pil)

        return [base_img] + patches

    def preprocess(self, images, return_tensors=None):
        """Preprocess images for LLaVA-Next."""
        import torch
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]

        # Convert to PIL and then numpy
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            if isinstance(img, Image.Image):
                if self.do_convert_rgb:
                    img = img.convert("RGB")
                img = np.array(img)
            processed_images.append(img)

        # Get original sizes
        image_sizes = [(img.shape[0], img.shape[1]) for img in processed_images]

        # Process each image into patches
        shortest_edge = self.size.get("shortest_edge", 224)
        base_size = (shortest_edge, shortest_edge)
        patch_size = self.crop_size["height"]

        all_patches = []
        for img in processed_images:
            patches = self.get_image_patches(img, self.image_grid_pinpoints, base_size, patch_size)
            # Preprocess each patch
            processed_patches = [self._preprocess_single(p) for p in patches]
            all_patches.append(np.array(processed_patches))

        # Pad to max number of patches
        if self.do_pad:
            max_patches = max(len(p) for p in all_patches)
            padded_patches = []
            for patches in all_patches:
                if len(patches) < max_patches:
                    pad_shape = ((0, max_patches - len(patches)),) + tuple((0, 0) for _ in range(patches.ndim - 1))
                    patches = np.pad(patches, pad_shape, mode="constant", constant_values=0)
                padded_patches.append(patches)
            all_patches = padded_patches

        result = {
            "pixel_values": all_patches,
            "image_sizes": image_sizes,
        }

        if return_tensors == "pt":
            result["pixel_values"] = torch.tensor(np.array(result["pixel_values"]), dtype=torch.float32)
            result["image_sizes"] = torch.tensor(result["image_sizes"], dtype=torch.long)

        return result

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)
