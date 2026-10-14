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

"""Standalone torchvision image processor for InternVL -- no transformers dependency."""

import json
import logging
import os
from functools import lru_cache
from typing import Optional

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvF
import numpy as np

logger = logging.getLogger(__name__)


_PIL_TO_TV_INTERPOLATION = {
    0: InterpolationMode.NEAREST_EXACT,
    1: InterpolationMode.LANCZOS,
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
    4: InterpolationMode.BOX,
    5: InterpolationMode.HAMMING,
}

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


class BatchFeature(dict):
    """Minimal implementation of transformers' ``BatchFeature``."""

    def __init__(self, data=None, tensor_type=None):
        super().__init__(data or {})
        if tensor_type is not None:
            self.convert_to_tensors(tensor_type)

    def convert_to_tensors(self, tensor_type="pt"):
        if tensor_type != "pt":
            return self

        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                continue
            if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                self[key] = torch.stack(list(value), dim=0)
            elif isinstance(value, np.ndarray):
                self[key] = torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                try:
                    self[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass
        return self

    def to(self, device=None, dtype=None):
        new_data = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.to(device=device, dtype=dtype) if dtype else value.to(device=device)
            else:
                new_data[key] = value
        return BatchFeature(new_data)


def _convert_to_rgb(image):
    """transformers' ``convert_to_rgb``: only touches PIL images, RGB passes through."""
    if not isinstance(image, Image.Image):
        return image
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _make_flat_list_of_images(images):
    """Normalize the input into a flat list of images (single image -> length-1 list)."""
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, (list, tuple)):
        flat = []
        for item in images:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [images]


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int):
    """All ``(width, height)`` tile grids with ``min <= w*h <= max`` (GotOcr2)."""
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles and width * height >= min_image_tiles:
                aspect_ratios.append((width, height))
    aspect_ratios = sorted(aspect_ratios, key=lambda x: x[0] * x[1])
    return aspect_ratios


@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(original_image_size, target_tile_size, min_image_tiles, max_image_tiles):
    """Pick the tile grid whose aspect ratio is closest to the image's (GotOcr2)."""
    possible_tile_arrangements = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)
    original_height, original_width = original_image_size
    target_tile_height, target_tile_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height

    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    for grid in possible_tile_arrangements:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * target_tile_height * target_tile_width * grid[0] * grid[1]:
                best_grid = grid
    return best_grid


class InternVLImageProcessor:
    """Faithful reimplementation of ``GotOcr2ImageProcessor``."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size=None,
        resample: int = 3,
        image_mean=None,
        image_std=None,
        rescale_factor: float = 1 / 255,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        crop_to_patches: bool = False,
        min_patches: int = 1,
        max_patches: int = 12,
        **kwargs,
    ):
        if size is None:
            size = {"height": 448, "width": 448}
        self.size = {"height": int(size["height"]), "width": int(size["width"])}
        self.resample = int(resample)
        self.image_mean = list(image_mean) if image_mean is not None else list(IMAGENET_STANDARD_MEAN)
        self.image_std = list(image_std) if image_std is not None else list(IMAGENET_STANDARD_STD)
        self.rescale_factor = float(rescale_factor)
        self.do_resize = bool(do_resize)
        self.do_rescale = bool(do_rescale)
        self.do_normalize = bool(do_normalize)
        self.do_convert_rgb = bool(do_convert_rgb)
        self.crop_to_patches = bool(crop_to_patches)
        self.min_patches = int(min_patches)
        self.max_patches = int(max_patches)

    def _resize(self, image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        interpolation = _PIL_TO_TV_INTERPOLATION[self.resample]
        if interpolation == InterpolationMode.LANCZOS:
            interpolation = InterpolationMode.BICUBIC
        return tvF.resize(image, [height, width], interpolation=interpolation, antialias=True)

    def _rescale_and_normalize(self, images: torch.Tensor) -> torch.Tensor:
        do_rescale = self.do_rescale
        image_mean = self.image_mean
        image_std = self.image_std
        if self.do_rescale and self.do_normalize:
            image_mean = torch.tensor(image_mean) * (1.0 / self.rescale_factor)
            image_std = torch.tensor(image_std) * (1.0 / self.rescale_factor)
            do_rescale = False
        if self.do_normalize:
            return tvF.normalize(images.to(dtype=torch.float32), image_mean, image_std)
        if do_rescale:
            return images * self.rescale_factor
        return images

    def _crop_image_to_patches(self, image: torch.Tensor) -> torch.Tensor:
        """GotOcr2 dynamic tiling (only reached when ``crop_to_patches=True``)."""
        patch_h, patch_w = self.size["height"], self.size["width"]
        original_height, original_width = image.shape[-2:]
        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (patch_h, patch_w), self.min_patches, self.max_patches
        )
        target_width = patch_w * num_columns
        target_height = patch_h * num_rows
        num_blocks = num_columns * num_rows
        resized_image = self._resize(image, target_height, target_width)
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (column * patch_w, row * patch_h, (column + 1) * patch_w, (row + 1) * patch_h)
            processed_images.append(resized_image[..., box[1]:box[3], box[0]:box[2]])
        if len(processed_images) != 1:
            processed_images.append(self._resize(image, patch_h, patch_w))
        return torch.stack(processed_images, dim=0)

    def _to_chw_uint8(self, image) -> torch.Tensor:
        """PIL/numpy/torch image -> channels-first tensor, mirroring ``process_image``."""

        if isinstance(image, Image.Image):
            return tvF.pil_to_tensor(image)
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).contiguous()
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            raise ValueError(f"Unsupported input image type {type(image)}")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor

    def preprocess(self, images, return_tensors: str = "pt", **overrides) -> BatchFeature:
        images = _make_flat_list_of_images(images)
        crop_to_patches = bool(overrides.get("crop_to_patches", self.crop_to_patches))
        height, width = self.size["height"], self.size["width"]

        pixel_values = []
        num_patches = []
        for image in images:
            if self.do_convert_rgb:
                image = _convert_to_rgb(image)
            tensor = self._to_chw_uint8(image)
            if crop_to_patches:
                tiles = self._crop_image_to_patches(tensor)
                tiles = self._resize(tiles, height, width) if self.do_resize else tiles
                tiles = self._rescale_and_normalize(tiles)
                for tile in tiles:
                    pixel_values.append(tile)
                num_patches.append(int(tiles.shape[0]))
            else:
                tensor = tensor.unsqueeze(0)
                if self.do_resize:
                    tensor = self._resize(tensor, height, width)
                tensor = self._rescale_and_normalize(tensor)
                pixel_values.append(tensor[0])
                num_patches.append(1)

        return BatchFeature(
            data={"pixel_values": pixel_values, "num_patches": num_patches},
            tensor_type=return_tensors,
        )

    def __call__(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """Number of patches for an image size (GotOcr2), 1 when tiling is off."""
        images_kwargs = images_kwargs or {}
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        patch_size = images_kwargs.get("patch_size", self.size)
        crop_to_patches = images_kwargs.get("crop_to_patches", self.crop_to_patches)
        num_patches = 1
        if crop_to_patches and max_patches > 1:
            patch_height, patch_width = patch_size["height"], patch_size["width"]
            num_columns, num_rows = get_optimal_tiled_canvas(
                (height, width), (patch_height, patch_width), min_patches, max_patches
            )
            if num_columns * num_rows > 1:
                num_patches += num_columns * num_rows
        return num_patches

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "InternVLImageProcessor":
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config: dict = {}
        if os.path.isfile(config_path):
            with open(config_path, "r") as handle:
                config = json.load(handle)
        for stale in (
            "image_processor_type",
            "processor_class",
            "auto_map",
            "data_format",
            "input_data_format",
            "device",
            "return_tensors",
            "default_to_square",
            "crop_size",
            "do_center_crop",
            "disable_grouping",
        ):
            config.pop(stale, None)
        config.update(kwargs)
        return cls(**config)


__all__ = ["InternVLImageProcessor", "BatchFeature"]
