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

"""Multimodal plugin for Qwen2.5-Omni image and audio processing."""

import math
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np

from .constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import soundfile
except ImportError:
    soundfile = None


@dataclass
class BasePlugin:
    """Base multimodal plugin.

    Owns the encode-time/collate-time contract: ``process_messages`` and
    ``process_token_ids`` decide how many image tokens land in ``input_ids``, and
    ``get_mm_inputs`` builds the ``pixel_values`` those tokens must line up with. Both
    sides go through ``_regularize_images``, so the forward pass assertion
    ``input_ids.count(image_token) == image_features.shape[0]`` holds by construction.
    """

    image_token: Optional[str] = None
    video_token: Optional[str] = None
    audio_token: Optional[str] = None
    expand_mm_tokens: bool = True

    def _preprocess_image(
        self, image: Any, image_max_pixels: int = 768 * 768, image_min_pixels: int = 32 * 32, **kwargs
    ) -> Any:
        """Resize one image into the [min_pixels, max_pixels] band and force RGB."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _regularize_images(self, images: list[Any], **kwargs) -> dict[str, list[Any]]:
        """Load images from paths/bytes/dicts and normalize them."""
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

    def _regularize_audios(self, audios: list[Any], sampling_rate: int) -> dict[str, list[Any]]:
        """Load audio from paths/bytes/dicts/arrays and normalize to float32 mono.

        Nothing here resamples: the clips under /data were written at the feature
        extractor's rate by the task's ``environment/prepare_*.py`` at image build time,
        so a mismatch is a bug in the bake, not something to paper over per epoch --
        silent resampling would only show up as a quietly worse model.
        """
        results = []
        for audio in audios:
            if isinstance(audio, np.ndarray):
                results.append(audio.astype(np.float32))
                continue

            if isinstance(audio, str):
                source = audio
            elif isinstance(audio, bytes):
                source = BytesIO(audio)
            elif isinstance(audio, dict):
                source = BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
            else:
                raise ValueError(f"Unsupported audio input {type(audio)}.")

            if soundfile is None:
                raise ImportError("Audio input needs `soundfile`; it is not installed.")

            array, rate = soundfile.read(source, dtype="float32", always_2d=True)
            if rate != sampling_rate:
                raise ValueError(
                    f"Audio is {rate} Hz but the feature extractor wants {sampling_rate} Hz. "
                    "The clips in /data are resampled once at image build time by the task's "
                    "environment/prepare_*.py -- nothing resamples at run time, on purpose."
                )
            results.append(array.mean(axis=1) if array.shape[1] > 1 else array[:, 0])

        return {"audios": results}

    def _image_kwargs(self, processor: Any) -> dict[str, int]:
        """Sizing knobs that ``model/loader.py`` copies from ModelArguments onto the processor."""
        return {
            "image_max_pixels": getattr(processor, "image_max_pixels", 768 * 768),
            "image_min_pixels": getattr(processor, "image_min_pixels", 32 * 32),
        }

    def _validate_input(self, processor: Any, images: list[Any], videos: list[Any], audios: list[Any]) -> None:
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")
        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")
        if len(audios) != 0 and self.audio_token is None:
            raise ValueError("This model does not support audio input.")
        if len(images) != 0 and processor is None:
            raise ValueError(
                "Image input needs a processor, but load_tokenizer returned none. Check that "
                "transformers_impl.processing loaded the vendored image processor."
            )

    def _validate_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any]
    ) -> None:
        num_images = sum(message["content"].count(IMAGE_PLACEHOLDER) for message in messages)
        num_videos = sum(message["content"].count(VIDEO_PLACEHOLDER) for message in messages)
        num_audios = sum(message["content"].count(AUDIO_PLACEHOLDER) for message in messages)
        if len(images) != num_images:
            raise ValueError(f"Got {len(images)} images but {num_images} {IMAGE_PLACEHOLDER} tokens.")
        if len(videos) != num_videos:
            raise ValueError(f"Got {len(videos)} videos but {num_videos} {VIDEO_PLACEHOLDER} tokens.")
        if len(audios) != num_audios:
            raise ValueError(f"Got {len(audios)} audios but {num_audios} {AUDIO_PLACEHOLDER} tokens.")

    def _get_mm_inputs(self, images: list[Any], videos: list[Any], audios: list[Any], processor: Any) -> dict[str, Any]:
        """Run the image processor over the regularized images."""
        if len(images) == 0:
            return {}
        image_processor = getattr(processor, "image_processor", processor)
        images = self._regularize_images(images, **self._image_kwargs(processor))["images"]
        return dict(image_processor(images, return_tensors="pt"))

    def process_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any], processor: Any
    ) -> list[dict[str, str]]:
        """Expand media placeholders in the message text, before tokenization."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        tokenizer: Any,
        processor: Any,
    ) -> tuple[list[int], Optional[list[int]]]:
        """Adjust token ids after tokenization, for models that prepend image tokens."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Any,
    ) -> dict[str, Any]:
        """Build the batched tensor inputs (pixel_values, grids, ...) handed to the model."""
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    """Qwen2-VL family: inline ``<|vision_start|> <|image_pad|>*N <|vision_end|>``.

    N is ``image_grid_thw.prod() // merge_size ** 2`` -- exactly the number of vision
    embeddings ``get_image_features`` returns for that image, so the count is read off
    the same processor call the collator will make rather than re-derived.
    """

    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    def _preprocess_image(self, image: Any, **kwargs) -> Any:
        image = super()._preprocess_image(image, **kwargs)
        # The ViT needs at least one full 28x28 patch and chokes on extreme aspect ratios.
        if min(image.width, image.height) < 28:
            image = image.resize((max(image.width, 28), max(image.height, 28)))
        if image.width / image.height > 200:
            image = image.resize((image.height * 180, image.height))
        if image.height / image.width > 200:
            image = image.resize((image.width, image.width * 180))
        return image

    def process_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any], processor: Any
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            merge_length = getattr(processor.image_processor, "merge_size") ** 2
            image_grid_thw = self._get_mm_inputs(images, videos, audios, processor).get("image_grid_thw", [])

        num_image_tokens = 0
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length
                else:
                    image_seqlen = 1

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen2OmniPlugin(Qwen2VLPlugin):
    """Qwen2.5-Omni thinker: same patch/merge grid math, plus the audio tower.

    Audio mirrors the image contract exactly: ``process_messages`` expands each
    ``<audio>`` into as many ``<|AUDIO|>`` tokens as the audio tower will emit
    embeddings for, and ``_get_mm_inputs`` produces the mel features those tokens line up
    with -- both derived from the same ``feature_attention_mask``, so
    ``input_ids.count(audio_token) == audio_features.shape[0]`` holds by construction.
    """

    vision_bos_token: str = "<|vision_bos|>"
    vision_eos_token: str = "<|vision_eos|>"
    audio_bos_token: str = "<|audio_bos|>"
    audio_eos_token: str = "<|audio_eos|>"

    @staticmethod
    def _audio_seqlen(mel_length: int) -> int:
        """Mel frames -> audio embeddings, following the tower's two stride-2 convs.

        conv1 is stride 1 with padding 1 and conv2 is stride 2 (``(L - 1) // 2 + 1``),
        then the avg-pool that feeds the projector drops one frame and halves again.
        """
        return ((mel_length - 1) // 2 + 1 - 2) // 2 + 1

    def _get_mm_inputs(self, images: list[Any], videos: list[Any], audios: list[Any], processor: Any) -> dict[str, Any]:
        mm_inputs = super()._get_mm_inputs(images, videos, audios, processor)
        if len(audios) == 0:
            return mm_inputs

        feature_extractor = getattr(processor, "feature_extractor", None)
        if feature_extractor is None:
            raise ValueError(
                "Audio input needs a feature extractor, but the processor has none. Check that "
                "AutoProcessor resolved the Qwen2.5-Omni processor and not a bare tokenizer."
            )

        sampling_rate = feature_extractor.sampling_rate
        arrays = self._regularize_audios(audios, sampling_rate=sampling_rate)["audios"]
        # padding="max_length" so every clip yields the same mel width and the batch stacks;
        # return_attention_mask is what carries the pre-pad length downstream.
        features = feature_extractor(
            arrays,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        mm_inputs.update(dict(features))
        # The extractor calls it attention_mask; the model wants feature_attention_mask, and
        # leaving it under the generic name would clobber the text attention mask in the batch.
        if "attention_mask" in mm_inputs:
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")
        return mm_inputs

    def process_messages(
        self, messages: list[dict[str, str]], images: list[Any], videos: list[Any], audios: list[Any], processor: Any
    ) -> list[dict[str, str]]:
        messages = super().process_messages(messages, images, videos, audios, processor)
        if len(audios) == 0:
            return messages

        if self.expand_mm_tokens:
            feature_attention_mask = self._get_mm_inputs([], [], audios, processor)["feature_attention_mask"]

        num_audio_tokens = 0
        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    # The mask, not the padded width: every clip is padded out to the
                    # extractor's max_length, and counting that would put thousands of
                    # audio tokens in input_ids for a three-second question.
                    mel_length = int(feature_attention_mask[num_audio_tokens].sum())
                    audio_seqlen = self._audio_seqlen(mel_length)
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER,
                    f"{self.audio_bos_token}{self.audio_token * audio_seqlen}{self.audio_eos_token}",
                    1,
                )
                num_audio_tokens += 1

            message["content"] = content

        return messages


PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
}

# ``data_args.template`` -> plugin kwargs. This repo only ever serves Qwen2.5-Omni, so the
# generic "default" template name resolves to the same plugin as the canonical name.
_SPEC = {"name": "qwen2_omni", "image_token": "<|IMAGE|>", "audio_token": "<|AUDIO|>"}
PLUGIN_SPECS = {"default": _SPEC, "qwen2_omni": _SPEC}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
    **kwargs,
) -> BasePlugin:
    """Instantiate a multimodal plugin by registry name."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
