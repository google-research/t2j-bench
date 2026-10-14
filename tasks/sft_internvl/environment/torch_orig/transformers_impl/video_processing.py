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

"""Standalone InternVL video processor -- no transformers dependency."""

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvF

from .image_processing import BatchFeature

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class InternVLVideoProcessor:
    """Minimal implementation of the InternVL video processor."""

    resample = 3
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    initial_shift = True
    do_sample_frames = False
    rescale_factor = 1 / 255

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def sample_frames(self, metadata, num_frames=None, fps=None, initial_shift=None, **kwargs):
        num_frames = num_frames if num_frames is not None else getattr(self, "num_frames", None)
        initial_shift = initial_shift if initial_shift is not None else self.initial_shift
        total_num_frames = metadata.total_num_frames
        if num_frames is None and fps is not None:
            if metadata is None or metadata.fps is None:
                raise ValueError("Sampling by fps requires video metadata with an fps.")
            num_frames = int(total_num_frames / metadata.fps * fps)
        if initial_shift is True:
            initial_shift = total_num_frames / num_frames / 2
        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`."
            )
        return torch.arange(initial_shift, total_num_frames, total_num_frames / num_frames).int()

    def _resize(self, video: torch.Tensor) -> torch.Tensor:
        return tvF.resize(
            video,
            [self.size["height"], self.size["width"]],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _rescale_and_normalize(self, video: torch.Tensor) -> torch.Tensor:
        if self.do_rescale and self.do_normalize:
            mean = torch.tensor(self.image_mean) * (1.0 / self.rescale_factor)
            std = torch.tensor(self.image_std) * (1.0 / self.rescale_factor)
            return tvF.normalize(video.to(dtype=torch.float32), mean, std)
        if self.do_normalize:
            return tvF.normalize(video.to(dtype=torch.float32), self.image_mean, self.image_std)
        if self.do_rescale:
            return video * self.rescale_factor
        return video

    def __call__(self, videos=None, return_tensors="pt", **kwargs) -> BatchFeature:
        if videos is None:
            raise ValueError("No videos provided.")
        processed = []
        for video in videos:
            frames = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
            if self.do_resize:
                frames = self._resize(frames)
            frames = self._rescale_and_normalize(frames)
            processed.append(frames)
        return BatchFeature(data={"pixel_values_videos": processed}, tensor_type=return_tensors)


__all__ = ["InternVLVideoProcessor"]
