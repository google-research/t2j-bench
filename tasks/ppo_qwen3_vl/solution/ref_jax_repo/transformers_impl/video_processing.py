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

"""Standalone Qwen3-VL video processor - no transformers dependency."""

import json
import logging
import math
import os
from typing import Optional

import numpy as np


def _no_video_resize(*a, **k):
    raise NotImplementedError("video resize not supported torch-free (DPO is image/text-only)")
from PIL import Image


from .utils import BatchFeature

logger = logging.getLogger(__name__)


def smart_resize_video(
    num_frames: int, height: int, width: int,
    temporal_factor: int = 2, factor: int = 32,
    min_pixels: int = 128 * 128, max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    """Rescale video dimensions so they are divisible by factor and within pixel range."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen3VLVideoProcessor:
    """Standalone Qwen3-VL video processor."""

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        fps: float = 2,
        min_frames: int = 4,
        max_frames: int = 768,
        do_sample_frames: bool = True,
        **kwargs,
    ):
        if size is None:
            size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
        self.size = size
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.do_sample_frames = do_sample_frames

    def sample_frames(self, total_num_frames: int, video_fps: Optional[float] = None,
                      num_frames: Optional[int] = None, fps: Optional[float] = None):
        """Sample frame indices uniformly from a video."""
        fps = fps if fps is not None else self.fps

        if num_frames is None and fps is not None:
            if video_fps is None:
                video_fps = 24
            num_frames = int(total_num_frames / video_fps * fps)
            num_frames = min(max(num_frames, self.min_frames), self.max_frames, total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)
        return indices


    def _preprocess(self, video_frames: list, do_resize=None, size=None,
                    do_rescale=None, rescale_factor=None, do_normalize=None,
                    image_mean=None, image_std=None, patch_size=None,
                    temporal_patch_size=None, merge_size=None):
        """Process video frames into patches with grid_thw.

        Args:
            video_frames: list of PIL Images or numpy arrays (T frames).
        Returns:
            flatten_patches: (grid_t * grid_h * grid_w, C * tp * ps * ps) array
            grid_thw: (grid_t, grid_h, grid_w) tuple
        """
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

        # Convert frames to numpy arrays (C, H, W)
        processed = []
        for frame in video_frames:
            if isinstance(frame, Image.Image):
                if self.do_convert_rgb and frame.mode != "RGB":
                    frame = frame.convert("RGB")
                frame = np.array(frame)
            if isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
                    frame = frame.transpose(2, 0, 1)  # HWC -> CHW
                processed.append(frame.astype(np.float32))
            elif isinstance(frame, np.ndarray):
                processed.append(frame.float().numpy())

        # Stack into (T, C, H, W)
        patches = np.stack(processed, axis=0)
        T, C, H, W = patches.shape
        num_frames = T

        if do_resize:
            resized_h, resized_w = smart_resize_video(
                num_frames=num_frames, height=H, width=W,
                temporal_factor=temporal_patch_size,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
            # Resize using torch interpolation
            t = np.asarray(patches)
            t = t.view(T * C, 1, H, W).float()
            t = _no_video_resize(t, size=(resized_h, resized_w), mode="bicubic", align_corners=False)
            patches = t.view(T, C, resized_h, resized_w).numpy()
            H, W = resized_h, resized_w

        if do_rescale:
            patches = patches * rescale_factor

        if do_normalize:
            mean = np.array(image_mean, dtype=np.float32).reshape(1, -1, 1, 1)
            std = np.array(image_std, dtype=np.float32).reshape(1, -1, 1, 1)
            patches = (patches - mean) / std

        # Pad temporal dimension
        if patches.shape[0] % temporal_patch_size != 0:
            pad_n = temporal_patch_size - (patches.shape[0] % temporal_patch_size)
            repeats = np.repeat(patches[-1:], pad_n, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)

        grid_t = patches.shape[0] // temporal_patch_size
        grid_h = H // patch_size
        grid_w = W // patch_size

        patches = patches.reshape(
            grid_t, temporal_patch_size, C,
            grid_h // merge_size, merge_size, patch_size,
            grid_w // merge_size, merge_size, patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            C * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def __call__(self, videos=None, **kwargs):
        """Make the processor callable."""
        return self.preprocess(videos, **kwargs)

    def preprocess(self, videos, return_tensors=None, **kwargs):
        """Process a batch of videos.

        Args:
            videos: list of videos, each being a list of frames (PIL/numpy/torch).
        Returns:
            BatchFeature with pixel_values_videos and video_grid_thw.
        """
        if not isinstance(videos, (list, tuple)):
            videos = [videos]

        all_patches, all_grids = [], []
        for video_frames in videos:
            patches, grid_thw = self._preprocess(video_frames, **kwargs)
            all_patches.append(patches)
            all_grids.append(grid_thw)

        pixel_values_videos = np.concatenate(all_patches, axis=0)
        video_grid_thw = np.array(all_grids)
        data = {"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load video processor from pretrained model path or HF hub."""
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
        with open(config_file, "r") as f:
            config = json.load(f)

        config.pop("video_processor_type", None)
        config.pop("processor_class", None)
        config.pop("auto_map", None)
        config.pop("image_processor_type", None)
        config.update(kwargs)
        return cls(**config)

