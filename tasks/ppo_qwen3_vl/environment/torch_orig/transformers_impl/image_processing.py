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

"""Standalone Qwen2-VL image processor - no transformers dependency."""

import json
import math
import os
import logging
from typing import Optional, Union

import numpy as np
from PIL import Image


from .utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, BatchFeature

logger = logging.getLogger(__name__)

BICUBIC = Image.BICUBIC


def smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescale image so dimensions are divisible by factor and within pixel range."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _to_numpy(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize (H, W, C) or (H, W) numpy image to (new_h, new_w)."""
    h, w = size
    pil_img = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
    pil_img = pil_img.resize((w, h), BICUBIC)
    return np.array(pil_img)


def _rescale(image: np.ndarray, scale: float) -> np.ndarray:
    return image.astype(np.float32) * scale


def _normalize(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == len(mean):
        image = (image - mean) / std
    elif image.ndim == 3 and image.shape[0] == len(mean):
        image = (image - mean[:, None, None]) / std[:, None, None]
    return image


def _to_chw(image: np.ndarray) -> np.ndarray:
    """Convert HWC to CHW."""
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        return image.transpose(2, 0, 1)
    return image


class Qwen2VLImageProcessor:
    """Standalone Qwen2-VL image processor."""

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ):
        if size is None:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels

        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or list(OPENAI_CLIP_MEAN)
        self.image_std = image_std or list(OPENAI_CLIP_STD)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb


    def _preprocess(self, images, do_resize=None, size=None, do_rescale=None,
                    rescale_factor=None, do_normalize=None, image_mean=None,
                    image_std=None, patch_size=None, temporal_patch_size=None,
                    merge_size=None, do_convert_rgb=None):
        """Process a single image or list of frames into patches with grid_thw."""
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if not isinstance(images, (list, tuple)):
            images = [images]

        if do_convert_rgb:
            images = [_convert_to_rgb(img) if isinstance(img, Image.Image) else img for img in images]

        images = [_to_numpy(img) for img in images]

        height, width = images[0].shape[:2]
        resized_height, resized_width = height, width

        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height, width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = _resize_image(image, (resized_height, resized_width))

            if do_rescale:
                image = _rescale(image, rescale_factor)

            if do_normalize:
                image = _normalize(image, image_mean, image_std)

            image = _to_chw(image)
            processed_images.append(image)

        patches = np.array(processed_images)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                temporal_patch_size - (patches.shape[0] % temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.reshape(
            grid_t, temporal_patch_size, channel,
            grid_h // merge_size, merge_size, patch_size,
            grid_w // merge_size, merge_size, patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def __call__(self, images, **kwargs):
        """Make the processor callable."""
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, do_resize=None, size=None, min_pixels=None,
                   max_pixels=None, do_rescale=None, rescale_factor=None,
                   do_normalize=None, image_mean=None, image_std=None,
                   patch_size=None, temporal_patch_size=None, merge_size=None,
                   do_convert_rgb=None, return_tensors=None, **kwargs):
        """Process a batch of images. Each element can be a PIL image or list of frames."""
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        elif min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if not isinstance(images, (list, tuple)):
            images = [images]

        pixel_values, vision_grid_thws = [], []
        for image in images:
            patches, grid_thw = self._preprocess(
                image, do_resize=do_resize, size=size, do_rescale=do_rescale,
                rescale_factor=rescale_factor, do_normalize=do_normalize,
                image_mean=image_mean, image_std=image_std, patch_size=patch_size,
                temporal_patch_size=temporal_patch_size, merge_size=merge_size,
                do_convert_rgb=do_convert_rgb,
            )
            pixel_values.extend(patches)
            vision_grid_thws.append(grid_thw)

        pixel_values = np.array(pixel_values)
        vision_grid_thws = np.array(vision_grid_thws)
        data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """Return number of image patches for a given image size."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", self.size["longest_edge"])
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load image processor from a pretrained model path or HF hub."""
        cache_dir = kwargs.pop("cache_dir", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")

        path = pretrained_model_name_or_path
        if os.path.isdir(path):
            config_file = os.path.join(path, "preprocessor_config.json")
        elif os.path.isfile(path):
            config_file = path
        else:
            config_file = os.path.join("/data", path, "preprocessor_config.json")
        with open(config_file, "r") as f:
            config = json.load(f)

        config.pop("image_processor_type", None)
        config.pop("processor_class", None)
        config.pop("auto_map", None)
        config.update(kwargs)
        return cls(**config)
