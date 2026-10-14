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

"""Standalone Gemma3 image processor - decoupled from HuggingFace."""
import itertools
import math
from typing import Any, List, Optional, Union

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


class Gemma3ImageProcessor:
    """Image processor for Gemma3 with pan-and-scan support."""

    def __init__(
        self,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        do_pan_and_scan=None,
        pan_and_scan_min_crop_size=None,
        pan_and_scan_max_num_crops=None,
        pan_and_scan_min_ratio_to_activate=None,
        **kwargs,
    ):
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 224, "width": 224}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or IMAGENET_STANDARD_MEAN
        self.image_std = image_std or IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb
        self.do_pan_and_scan = do_pan_and_scan
        self.pan_and_scan_min_crop_size = pan_and_scan_min_crop_size
        self.pan_and_scan_max_num_crops = pan_and_scan_max_num_crops
        self.pan_and_scan_min_ratio_to_activate = pan_and_scan_min_ratio_to_activate
        self.model_input_names = ["pixel_values"]

    def pan_and_scan(self, image, min_crop_size, max_num_crops, min_ratio):
        """Crop image into sub-images for aspect-ratio handling."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                # channels first
                _, height, width = image.shape
            else:
                height, width = image.shape[0], image.shape[1]
        else:
            height, width = image.shape[:2]

        if width >= height:
            if width / height < min_ratio:
                return []
            num_crops_w = int(math.floor(width / height + 0.5))
            num_crops_w = min(int(math.floor(width / min_crop_size)), num_crops_w)
            num_crops_w = max(2, min(max_num_crops, num_crops_w))
            num_crops_h = 1
        else:
            if height / width < min_ratio:
                return []
            num_crops_h = int(math.floor(height / width + 0.5))
            num_crops_h = min(int(math.floor(height / min_crop_size)), num_crops_h)
            num_crops_h = max(2, min(max_num_crops, num_crops_h))
            num_crops_w = 1

        crop_w = int(math.ceil(width / num_crops_w))
        crop_h = int(math.ceil(height / num_crops_h))

        if min(crop_w, crop_h) < min_crop_size:
            return []

        crops = []
        for pos_h, pos_w in itertools.product(
            [crop_h * i for i in range(num_crops_h)],
            [crop_w * i for i in range(num_crops_w)],
        ):
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                crops.append(image[:, pos_h:pos_h + crop_h, pos_w:pos_w + crop_w])
            else:
                crops.append(image[pos_h:pos_h + crop_h, pos_w:pos_w + crop_w])
        return crops

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
        do_pan_and_scan=None,
        pan_and_scan_min_crop_size=None,
        pan_and_scan_max_num_crops=None,
        pan_and_scan_min_ratio_to_activate=None,
        return_tensors=None,
        **kwargs,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pan_and_scan = do_pan_and_scan if do_pan_and_scan is not None else self.do_pan_and_scan
        pan_and_scan_min_crop_size = pan_and_scan_min_crop_size or self.pan_and_scan_min_crop_size or 256
        pan_and_scan_max_num_crops = pan_and_scan_max_num_crops or self.pan_and_scan_max_num_crops or 4
        pan_and_scan_min_ratio_to_activate = pan_and_scan_min_ratio_to_activate or self.pan_and_scan_min_ratio_to_activate or 1.2

        # Make flat list of images
        if not isinstance(images, list):
            images = [images]

        # Convert to numpy
        processed_images = []
        num_crops = []
        for img in images:
            if do_convert_rgb and Image is not None and isinstance(img, Image.Image):
                img = img.convert("RGB")

            if isinstance(img, Image.Image):
                img = np.array(img)

            # Pan and scan
            if do_pan_and_scan:
                pas_crops = self.pan_and_scan(
                    img, pan_and_scan_min_crop_size,
                    pan_and_scan_max_num_crops, pan_and_scan_min_ratio_to_activate,
                )
                processed_images.append(img)
                processed_images.extend(pas_crops)
                num_crops.append(len(pas_crops))
            else:
                processed_images.append(img)
                num_crops.append(0)

        # Process each image
        final_images = []
        for img in processed_images:
            if isinstance(img, Image.Image):
                img = np.array(img)

            # Ensure HWC format
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Resize
            if do_resize:
                h, w = size["height"], size["width"]
                if Image is not None:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    pil_img = pil_img.resize((w, h), Image.BILINEAR)
                    img = np.array(pil_img)

            # Convert to float
            img = img.astype(np.float32)

            # Rescale
            if do_rescale:
                img = img * rescale_factor

            # Normalize
            if do_normalize:
                mean = np.array(image_mean, dtype=np.float32)
                std = np.array(image_std, dtype=np.float32)
                img = (img - mean) / std

            # Convert to CHW
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                img = np.transpose(img, (2, 0, 1))

            final_images.append(img)

        result = {"pixel_values": final_images, "num_crops": num_crops}

        if return_tensors == "pt":
            result["pixel_values"] = np.asarray(np.stack(final_images))

        return result

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        # /data/{repo_id} is baked in at build time; nothing is fetched at run time.
        resolved_path = model_path
        if not os.path.isdir(model_path):
            baked = os.path.join("/data", model_path)
            if os.path.isdir(baked):
                resolved_path = baked
        config_file = os.path.join(resolved_path, "preprocessor_config.json")
        config = {}
        if os.path.exists(config_file):
            import json
            with open(config_file, "r") as f:
                config = json.load(f)
        config.update(kwargs)
        return cls(**config)


import os  # needed for from_pretrained
