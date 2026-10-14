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

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Video processor class for InternVL.

Mirror of ``torch_orig/transformers_impl/video_processing.py``. JAX adjustments:
torch's ``InternVLVideoProcessor`` extends ``transformers``' ``BaseVideoProcessor`` and
batches frames through ``torchvision.transforms.v2.functional``; the torch-free repo has
neither, so videos are ``numpy`` arrays in ``(frames, channels, height, width)`` and the
four ops ``_preprocess`` calls on its base — ``convert_to_rgb``, ``resize``,
``center_crop``, ``rescale_and_normalize`` — are defined here alongside numpy versions of
``group_videos_by_shape``/``reorder_videos``/``VideoMetadata``. ``SizeDict`` is a plain
``dict``, ``TensorType`` a plain string, and ``valid_kwargs``/``Unpack`` — ``transformers``
registration hooks with no behaviour — are dropped. The shape-grouping, frame-sampling
arithmetic and the ``pixel_values_videos`` output key are torch's unchanged.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from PIL import Image

from .image_processing import BICUBIC, _resize_image
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


def reorder_videos(processed_videos: dict, grouped_videos_index: dict) -> list[np.ndarray]:
    """Undo ``group_videos_by_shape``, restoring the caller's original ordering."""
    return [
        processed_videos[grouped_videos_index[i][0]][grouped_videos_index[i][1]]
        for i in range(len(grouped_videos_index))
    ]


class InternVLVideoProcessorInitKwargs:
    """The kwargs ``InternVLVideoProcessor`` accepts (torch spells this as a ``TypedDict``)."""

    initial_shift: Union[bool, float, int]


class InternVLVideoProcessor:
    resample = BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    initial_shift = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    do_center_crop = False
    crop_size = None
    rescale_factor = 1 / 255
    num_frames = None
    fps = None
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.size = dict(self.size) if self.size is not None else None
        self.image_mean = list(self.image_mean)
        self.image_std = list(self.image_std)

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

    def resize(self, videos: np.ndarray, size: dict, resample=None) -> np.ndarray:
        """Resize every frame to ``size['height'] x size['width']`` (bicubic, per frame)."""
        height, width = size["height"], size["width"]
        if videos.shape[-2:] == (height, width):
            return videos
        leading, (channels, _, _) = videos.shape[:-3], videos.shape[-3:]
        frames = videos.reshape(-1, channels, *videos.shape[-2:])
        resized = np.stack(
            # PIL drops the trailing axis of a single-channel image, hence the reshape back.
            [
                _resize_image(frame.transpose(1, 2, 0), (height, width)).reshape(height, width, channels).transpose(2, 0, 1)
                for frame in frames
            ]
        )
        return resized.reshape(*leading, channels, height, width)

    def center_crop(self, videos: np.ndarray, crop_size: dict) -> np.ndarray:
        """Crop every frame to ``crop_size`` about its centre."""
        height, width = videos.shape[-2:]
        crop_h, crop_w = crop_size["height"], crop_size["width"]
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        return videos[..., top : top + crop_h, left : left + crop_w]

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
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        initial_shift: Optional[Union[bool, float, int]] = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
            initial_shift (`bool`, `float` or `int`, defaults to `self.initial_shift`):
                The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        num_frames = num_frames if num_frames is not None else self.num_frames
        initial_shift = initial_shift if initial_shift is not None else self.initial_shift
        total_num_frames = metadata.total_num_frames

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata is None or metadata.fps is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            num_frames = int(total_num_frames / metadata.fps * fps)

        if initial_shift is True:
            initial_shift = total_num_frames / num_frames / 2

        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
            )

        indices = np.arange(initial_shift, total_num_frames, total_num_frames / num_frames).astype(np.int32)
        return indices

    def _preprocess(
        self,
        videos: list[np.ndarray],
        do_convert_rgb: bool,
        do_resize: bool,
        size: dict,
        resample,
        do_center_crop: bool,
        crop_size: Optional[dict],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(stacked_videos, size=size, resample=resample)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=return_tensors)

    def preprocess(self, videos, **kwargs) -> BatchFeature:
        """Normalize the input to a list of ``(F, C, H, W)`` arrays, then run ``_preprocess``.

        torch inherits this outer loop from ``BaseVideoProcessor``; it is spelled out here
        because the torch-free repo has no base class to inherit it from.
        """
        videos = [_to_video_array(video) for video in _as_video_list(videos)]
        settings = {
            "do_convert_rgb": self.do_convert_rgb,
            "do_resize": self.do_resize,
            "size": self.size,
            "resample": self.resample,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }
        settings.update({k: v for k, v in kwargs.items() if v is not None or k in settings})
        return self._preprocess(videos, **settings)

    def __call__(self, videos, **kwargs) -> BatchFeature:
        return self.preprocess(videos, **kwargs)


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
    if isinstance(video, np.ndarray) and video.ndim == 4:
        frames = list(video)
    else:
        frames = list(video)
    stacked = []
    for frame in frames:
        array = np.array(frame) if isinstance(frame, Image.Image) else np.asarray(frame)
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = array.transpose(2, 0, 1)
        stacked.append(array)
    return np.stack(stacked)


__all__ = ["InternVLVideoProcessor"]
