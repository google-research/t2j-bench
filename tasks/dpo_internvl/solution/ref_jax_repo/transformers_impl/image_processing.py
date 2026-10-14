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

"""Standalone InternVL image processor - no transformers dependency."""

import json
import logging
import os
from typing import Optional

import numpy as np
from PIL import Image


from .utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, BatchFeature

logger = logging.getLogger(__name__)

BICUBIC = Image.BICUBIC


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
    """Resize an (H, W, C) numpy image to (new_h, new_w)."""
    height, width = size
    pil_image = Image.fromarray(image if image.dtype == np.uint8 else image.astype(np.uint8))
    return np.array(pil_image.resize((width, height), BICUBIC))


def _normalize(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    image = image.astype(np.float32)
    return (image - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)


class InternVLImageProcessor:
    """Fixed square resize + rescale + normalize, reporting one patch per image.

    ``num_patches`` is what ``InternVLPlugin`` multiplies ``image_seq_length`` by to size
    the ``<IMG_CONTEXT>`` run, so it is returned here rather than re-derived. The
    checkpoint ships ``crop_to_patches=false``, which is the only path mirrored: the
    dynamic-tiling branch is rejected outright instead of being approximated, so a config
    change surfaces as an error rather than as a silent token-count mismatch.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        resample: int = BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        do_convert_rgb: bool = True,
        crop_to_patches: bool = False,
        min_patches: int = 1,
        max_patches: int = 12,
        **kwargs,
    ):
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 448, "width": 448}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or list(IMAGENET_STANDARD_MEAN)
        self.image_std = image_std or list(IMAGENET_STANDARD_STD)
        self.do_convert_rgb = True if do_convert_rgb is None else do_convert_rgb
        self.crop_to_patches = bool(crop_to_patches)
        self.min_patches = min_patches
        self.max_patches = max_patches

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(
        self,
        images,
        do_resize=None,
        size=None,
        do_rescale=None,
        rescale_factor=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        do_convert_rgb=None,
        crop_to_patches=None,
        return_tensors=None,
        **kwargs,
    ):
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb
        crop_to_patches = self.crop_to_patches if crop_to_patches is None else crop_to_patches
        if crop_to_patches:
            raise NotImplementedError(
                "crop_to_patches is not mirrored in the JAX reference; the InternVL "
                "checkpoint ships crop_to_patches=false."
            )

        if not isinstance(images, (list, tuple)):
            images = [images]

        pixel_values = []
        for image in images:
            if do_convert_rgb and isinstance(image, Image.Image):
                image = _convert_to_rgb(image)
            image = _to_numpy(image)
            if do_resize:
                image = _resize_image(image, (size["height"], size["width"]))
            if do_rescale:
                image = image.astype(np.float32) * rescale_factor
            if do_normalize:
                image = _normalize(image, image_mean, image_std)
            pixel_values.append(image.transpose(2, 0, 1))

        data = {
            "pixel_values": np.stack(pixel_values).astype(np.float32),
            "num_patches": [1] * len(pixel_values),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load the image processor from a local checkpoint directory or the HF hub."""
        cache_dir = kwargs.pop("cache_dir", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")

        path = pretrained_model_name_or_path
        if os.path.isdir(path):
            config_file = os.path.join(path, "preprocessor_config.json")
        elif os.path.isfile(path):
            config_file = path
        else:
            # Baked into the image at /data/{repo_id} at build time
            # (base_image/prepare_model.py); the hub is never reached at run time.
            config_file = os.path.join("/data", path, "preprocessor_config.json")
        with open(config_file) as handle:
            config = json.load(handle)

        for key in ("image_processor_type", "processor_class", "auto_map"):
            config.pop(key, None)
        # Null-valued keys in the checkpoint config mean "use the default", but passing
        # them through would clobber it.
        config = {key: value for key, value in config.items() if value is not None}
        config.update(kwargs)
        return cls(**config)
