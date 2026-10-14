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

"""Standalone Qwen2.5-Omni thinker processor - no transformers dependency.

Only the surface ``data/mm_plugin.py`` actually touches: the image processor (for
``merge_size`` and the ``pixel_values`` / ``image_grid_thw`` pair), the feature extractor
(for ``sampling_rate`` and the ``input_features`` / ``feature_attention_mask`` pair) and
the tokenizer. Text goes through ``tokenizer.apply_chat_template`` in
``data/template.py``, never through a processor ``__call__``, so no placeholder expansion
lives here. Video is out of scope for these tasks.
"""

from typing import Optional

from .feature_extraction import WhisperFeatureExtractor
from .image_processing import Qwen2VLImageProcessor


class Qwen2_5OmniProcessor:
    """Bundle of the image processor and tokenizer, as the data pipeline expects it."""

    image_token: str = "<|IMAGE|>"
    video_token: str = "<|VIDEO|>"
    audio_token: str = "<|AUDIO|>"
    vision_start_token: str = "<|vision_bos|>"
    vision_end_token: str = "<|vision_eos|>"

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, **kwargs):
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    # --- multimodal token ids ------------------------------------------------ #
    # torch's Qwen2_5OmniProcessor exposes no ``*_token_id`` of its own, so the data
    # collator falls back to the tokenizer's -- ``<|IMAGE|>`` -> 151655 and
    # ``<|VIDEO|>`` -> 151656 for Qwen2.5-Omni-3B. The Rust fast tokenizer this repo
    # wraps reads the same ``tokenizer.json``, so resolving the same strings through it
    # gives the same ids. Without them the collator cannot derive ``token_type_ids`` and
    # silently drops the key from the batch.

    def _token_id(self, token: str):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return None
        return tokenizer.convert_tokens_to_ids(token)

    @property
    def image_token_id(self):
        return self._token_id(self.image_token)

    @property
    def video_token_id(self):
        return self._token_id(self.video_token)

    @property
    def audio_token_id(self):
        return self._token_id(self.audio_token)

    def __call__(self, images=None, return_tensors=None, **kwargs):
        if images is None:
            raise ValueError("This processor only handles images; text is tokenized upstream.")
        return self.image_processor(images, return_tensors=return_tensors, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, tokenizer: Optional[object] = None, **kwargs):
        image_processor = Qwen2VLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Both read the same preprocessor_config.json; each keeps only the keys it owns.
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor=image_processor, feature_extractor=feature_extractor, tokenizer=tokenizer)
