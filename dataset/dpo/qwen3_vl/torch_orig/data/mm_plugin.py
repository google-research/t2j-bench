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

"""Multimodal plugin for Qwen3-VL image/video processing."""

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Optional, Union

import numpy as np
import torch
from transformers_impl.utils import is_valid_image

from .constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

try:
    from PIL import Image
    from PIL.Image import Image as ImageObject
except ImportError:
    ImageObject = None

try:
    import av
except ImportError:
    av = None


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray


ImageInput = Union[str, bytes, BinaryIO, "ImageObject"]
VideoInput = Union[str, BinaryIO, list[list["ImageInput"]]]
AudioInput = Union[str, BinaryIO, "NDArray"]


def _check_video_is_nested_images(video) -> bool:
    """Check if the video is nested images."""
    return isinstance(video, list) and all(
        isinstance(frame, (str, BinaryIO, dict)) or (ImageObject is not None and isinstance(frame, ImageObject))
        for frame in video
    )


@dataclass
class BasePlugin:
    """Base multimodal plugin."""
    image_token: str | None
    video_token: str | None
    audio_token: str | None = None
    expand_mm_tokens: bool = True

    def _preprocess_image(self, image: "ImageObject", image_max_pixels: int = 768*768,
                          image_min_pixels: int = 32*32, **kwargs) -> "ImageObject":
        """Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _get_video_sample_indices(self, video_stream: "Stream", video_fps: float = 2.0,
                                   video_maxlen: int = 128, **kwargs) -> list[int]:
        """Compute video sample indices."""
        total_frames = video_stream.frames
        if total_frames == 0:
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)
        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _regularize_images(self, images: list, **kwargs) -> dict:
        """Read and preprocess images."""
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

    def _regularize_videos(self, videos: list, **kwargs) -> dict:
        """Read, resize and convert videos."""
        results, durations = [], []
        for video in videos:
            frames = []
            if _check_video_is_nested_images(video):
                frames = video
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
            else:
                container = av.open(video, "r")
                video_stream = next(s for s in container.streams if s.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())
                if video_stream.duration is None:
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    durations.append(float(video_stream.duration * video_stream.time_base))
            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)
        return {"videos": results, "durations": durations}

    def _validate_input(self, processor, images, videos, audios) -> None:
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")
        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _validate_messages(self, messages, images, videos, audios):
        num_img = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in messages)
        num_vid = sum(m["content"].count(VIDEO_PLACEHOLDER) for m in messages)
        if len(images) != num_img:
            raise ValueError(f"Image count mismatch: {len(images)} images vs {num_img} placeholders.")
        if len(videos) != num_vid:
            raise ValueError(f"Video count mismatch: {len(videos)} videos vs {num_vid} placeholders.")

    def _get_mm_inputs(self, images, videos, audios, processor) -> dict:
        """Build multimodal inputs from images/videos using the processor."""
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", getattr(processor, "image_processor", None))
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(video_processor(images=None, videos=video_data["videos"], return_tensors="pt"))
        return mm_inputs

    def process_messages(self, messages, images, videos, audios, processor):
        """Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
        """Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(self, images, videos, audios, imglens, vidlens, audlens, batch_ids, processor):
        """Build batched multimodal inputs for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    """Plugin for Qwen2-VL and Qwen2.5-VL models."""
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    def _preprocess_image(self, image, **kwargs):
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))
        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))
        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))
        return image

    def _regularize_videos(self, videos, **kwargs):
        results, fps_per_video, durations = [], [], []
        for video in videos:
            frames = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
            else:
                container = av.open(video, "r")
                video_stream = next(s for s in container.streams if s.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())
                if video_stream.duration is None:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))
                    durations.append(float(video_stream.duration * video_stream.time_base))
            if len(frames) % 2 != 0:
                frames.append(frames[-1])
            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)
        return {"videos": results, "fps_per_video": fps_per_video, "durations": durations}

    def _get_mm_inputs(self, images, videos, audios, processor):
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))
        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(video_processor(videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]
        return mm_inputs

    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")
        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1
            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1
            message["content"] = content
        return messages



@dataclass
class Qwen3VLPlugin(Qwen2VLPlugin):
    """Plugin for Qwen3-VL models with timestamp and video metadata support."""

    def _get_mm_inputs(self, images, videos, audios, processor):
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))
        if len(videos) != 0:
            videos_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_metadata = [
                {"fps": getattr(processor, "video_fps", 24.0), "duration": dur, "total_num_frames": len(vid)}
                for vid, dur in zip(videos_data["videos"], videos_data["durations"])
            ]
            mm_inputs.update(
                video_processor(
                    videos=videos_data["videos"],
                    video_metadata=video_metadata,
                    fps=getattr(processor, "video_fps", 2.0),
                    return_metadata=True,
                )
            )
            temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [
                    temporal_patch_size / fps for fps in videos_data["fps_per_video"]
                ]
        return mm_inputs

    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")
        video_processor = getattr(processor, "video_processor")
        image_merge_length = getattr(image_processor, "merge_size") ** 2
        video_merge_length = getattr(video_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0
            video_metadata = mm_inputs.get("video_metadata", {})
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0

        for idx, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod() // image_merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    metadata = video_metadata[idx]
                    timestamps = processor._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        video_processor.merge_size,
                    )
                    video_structure = ""
                    for frame_index in range(num_frames):
                        video_seqlen = (
                            video_grid_thw[num_video_tokens][1:].prod() // video_merge_length
                            if self.expand_mm_tokens
                            else 1
                        )
                        timestamp_sec = timestamps[frame_index]
                        frame_structure = (
                            f"<{timestamp_sec:.1f} seconds>"
                            f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}"
                        )
                        video_structure += frame_structure
                else:
                    video_structure = f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"
                content = content.replace(VIDEO_PLACEHOLDER, video_structure, 1)
                num_video_tokens += 1
            message["content"] = content
        return messages


# ============================================================
# Plugin Registry
# ============================================================

PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: str | None = None,
    video_token: str | None = None,
    audio_token: str | None = None,
    **kwargs,
) -> "BasePlugin":
    """Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")
    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
