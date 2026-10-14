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

# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""video processor class for Qwen2-VL.

Mirror of ``torch_orig/transformers_impl/video_processing.py``. JAX adjustments:
torch's ``Qwen2VLVideoProcessor`` extends ``transformers``' ``BaseVideoProcessor`` and
batches frames through ``torchvision.transforms.v2.functional``; the torch-free repo has
neither, so videos are ``numpy`` arrays in ``(frames, channels, height, width)`` and the
base ops ``_preprocess`` calls — ``convert_to_rgb``, ``resize``, ``rescale_and_normalize``
— are defined here alongside numpy versions of ``group_videos_by_shape``/``reorder_videos``
/``VideoMetadata``. ``SizeDict`` is a plain ``dict``, ``TensorType`` a plain string, and
``valid_kwargs``/``Unpack``/``@add_start_docstrings`` — ``transformers`` registration hooks
with no behaviour — are dropped. The frame-sampling arithmetic and the patchification
(reshape → ``transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)`` → flatten) are torch's unchanged.
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from PIL import Image

from .image_processing import BICUBIC, _resize_image, smart_resize
from .utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, BatchFeature


@dataclass
class VideoMetadata:
    """The subset of ``transformers.video_utils.VideoMetadata`` ``sample_frames`` reads."""

    total_num_frames: int
    fps: Optional[float] = None
    duration: Optional[float] = None
    video_backend: Optional[str] = None


def group_videos_by_shape(videos: list[np.ndarray]) -> tuple[dict, dict]:
    """Batch same-shaped videos together so each distinct shape is processed once.

    Mirror of ``transformers.video_utils.group_videos_by_shape``: a pure batching
    optimisation with no numeric effect. Returns ``(grouped, index)`` where ``grouped``
    maps a shape to a stacked ``(n_videos, frames, channels, height, width)`` array and
    ``index`` maps each original position to its ``(shape, offset)`` in that stack.
    """
    grouped: dict[tuple, list[np.ndarray]] = {}
    index: dict[int, tuple[tuple, int]] = {}
    for i, video in enumerate(videos):
        shape = tuple(video.shape)
        index[i] = (shape, len(grouped.setdefault(shape, [])))
        grouped[shape].append(video)
    return {shape: np.stack(group) for shape, group in grouped.items()}, index


def reorder_videos(processed_videos: dict, grouped_videos_index: dict) -> list:
    """Undo ``group_videos_by_shape``, restoring the caller's original ordering."""
    return [
        processed_videos[grouped_videos_index[i][0]][grouped_videos_index[i][1]]
        for i in range(len(grouped_videos_index))
    ]


def get_image_size(image: np.ndarray) -> tuple[int, int]:
    """``(height, width)`` of a channels-first frame — torch's ``ChannelDimension.FIRST``."""
    return image.shape[-2], image.shape[-1]


class Qwen2VLVideoProcessorInitKwargs:
    """The kwargs ``Qwen2VLVideoProcessor`` accepts (torch spells this as a ``TypedDict``)."""

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


class Qwen2VLVideoProcessor:
    """Fast Qwen2-VL video processor that dynamically resizes videos based on the original videos."""

    resample = BICUBIC
    size = {"shortest_edge": 128 * 28 * 28, "longest_edge": 28 * 28 * 768}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    rescale_factor = 1 / 255
    num_frames = None
    fps = None
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = dict(type(self).size) if size is None else dict(size)
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.size = size
        self.image_mean = list(self.image_mean)
        self.image_std = list(self.image_std)

    def _standardize_kwargs(
        self,
        size: Optional[dict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        size = size if size is not None else self.size
        if not size.get("shortest_edge") or not size.get("longest_edge"):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        kwargs["size"] = size
        return kwargs

    # --- the ``BaseVideoProcessor`` ops ``_preprocess`` leans on ------------------- #

    def convert_to_rgb(self, videos: np.ndarray) -> np.ndarray:
        """Force the channel axis to 3, alpha-compositing onto white as torchvision does."""
        channels = videos.shape[-3]
        if channels == 3:
            return videos
        if channels == 1:
            return np.repeat(videos, 3, axis=-3)
        if channels == 4:
            rgb, alpha = videos[..., :3, :, :], videos[..., 3:, :, :]
            alpha = alpha.astype(np.float32) / 255.0
            return (rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(videos.dtype)
        raise ValueError(f"Unsupported number of channels: {channels}")

    def resize(self, image: np.ndarray, size: dict, resample=None) -> np.ndarray:
        """Resize every frame to ``size['height'] x size['width']`` (bicubic, per frame)."""
        height, width = size["height"], size["width"]
        if image.shape[-2:] == (height, width):
            return image
        leading, channels = image.shape[:-3], image.shape[-3]
        frames = image.reshape(-1, channels, *image.shape[-2:])
        resized = np.stack(
            # PIL drops the trailing axis of a single-channel image, hence the reshape back.
            [
                _resize_image(frame.transpose(1, 2, 0), (height, width)).reshape(height, width, channels).transpose(2, 0, 1)
                for frame in frames
            ]
        )
        return resized.reshape(*leading, channels, height, width)

    def rescale_and_normalize(
        self, videos: np.ndarray, do_rescale: bool, rescale_factor: float, do_normalize: bool, image_mean, image_std
    ) -> np.ndarray:
        """Fused rescale + normalize, matching torch's single-pass base implementation."""
        videos = videos.astype(np.float32)
        if do_rescale:
            videos = videos * rescale_factor
        if do_normalize:
            mean = np.array(image_mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(image_std, dtype=np.float32).reshape(-1, 1, 1)
            videos = (videos - mean) / std
        return videos

    # --- torch's own surface ------------------------------------------------------ #

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: Optional[int] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            temporal_patch_size (`int`, *optional*):
                The temporal patch size of the vision encoder. Number of sampled frames will be rounded to be divisible by frame factor.
            min_frames (`int`, *optional*):
                The minimum number of frames that can be sampled.
            max_frames (`int`, *optional*):
                The maximum number of frames that can be sampled.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        num_frames = num_frames if num_frames is not None else self.num_frames
        fps = fps if fps is not None else self.fps
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        min_frames = min_frames if min_frames is not None else self.min_frames
        max_frames = max_frames if max_frames is not None else self.max_frames
        total_num_frames = metadata.total_num_frames

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is not None:
            num_frames = round(num_frames / temporal_patch_size) * temporal_patch_size
        elif fps is not None:
            if metadata is None or metadata.fps is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            max_frames = math.floor(min(max_frames, total_num_frames) / temporal_patch_size) * temporal_patch_size
            num_frames = total_num_frames / metadata.fps * fps
            num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)
            num_frames = math.floor(num_frames / temporal_patch_size) * temporal_patch_size

        if num_frames is not None and num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
                "Decrease `num_frames` or `fps` for sampling."
            )

        if num_frames is not None:
            indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
        else:
            indices = np.arange(0, total_num_frames).astype(np.int32)

        return indices

    def _preprocess(
        self,
        videos: list[np.ndarray],
        do_resize: bool,
        size: dict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0])
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size={"height": resized_height, "width": resized_width},
                    resample=resample,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0])

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            T = patches.shape[1]
            if pad := -T % temporal_patch_size:
                repeats = np.repeat(patches[:, -1:], pad, axis=1)
                patches = np.concatenate((patches, repeats), axis=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.reshape(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = np.concatenate(processed_videos, axis=0)
        video_grid_thw = np.array(processed_grids, dtype=np.int64)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )

    def preprocess(self, videos, **kwargs) -> BatchFeature:
        """Normalize the input to a list of ``(F, C, H, W)`` arrays, then run ``_preprocess``.

        torch inherits this outer loop from ``BaseVideoProcessor``; it is spelled out here
        because the torch-free repo has no base class to inherit it from.
        """
        videos = [_to_video_array(video) for video in _as_video_list(videos)]
        do_convert_rgb = kwargs.pop("do_convert_rgb", self.do_convert_rgb)
        if do_convert_rgb:
            videos = [self.convert_to_rgb(video) for video in videos]
        return_tensors = kwargs.pop("return_tensors", None)
        settings = {
            "do_resize": self.do_resize,
            "resample": self.resample,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
        }
        settings.update({k: v for k, v in kwargs.items() if k not in ("size", "min_pixels", "max_pixels")})
        settings.update(self._standardize_kwargs(**{k: kwargs.get(k) for k in ("size", "min_pixels", "max_pixels")}))
        return self._preprocess(videos, return_tensors=return_tensors, **settings)

    def __call__(self, videos, **kwargs) -> BatchFeature:
        return self.preprocess(videos, **kwargs)

    def get_num_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        """
        A utility that returns number of video patches a given video size.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video.
            width (`int`):
                Width of the input video.
            videos_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the video processor.
        Returns:
            `Tuple(int, int)`: Number of placeholder tokens required and number of patches per image.
        """
        videos_kwargs = videos_kwargs or {}
        min_pixels = videos_kwargs.get("min_pixels", None) or self.size["shortest_edge"]
        max_pixels = videos_kwargs.get("max_pixels", None) or self.size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", None) or self.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.merge_size
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", None) or self.temporal_patch_size

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_t = num_frames // temporal_patch_size
        return grid_t * grid_h * grid_w


def _as_video_list(videos) -> list:
    """A single video (array or frame list) becomes a one-element batch."""
    if isinstance(videos, np.ndarray):
        return list(videos) if videos.ndim == 5 else [videos]
    if isinstance(videos, (list, tuple)) and videos and isinstance(videos[0], (Image.Image, np.ndarray)):
        first = videos[0]
        if isinstance(first, Image.Image) or first.ndim == 3:
            return [videos]  # one video given as a list of frames
    return list(videos)


def _to_video_array(video) -> np.ndarray:
    """Stack a video's frames into ``(F, C, H, W)``, accepting PIL frames or HWC arrays."""
    stacked = []
    for frame in list(video):
        array = np.array(frame) if isinstance(frame, Image.Image) else np.asarray(frame)
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = array.transpose(2, 0, 1)
        stacked.append(array)
    return np.stack(stacked)


__all__ = ["Qwen2VLVideoProcessor"]
