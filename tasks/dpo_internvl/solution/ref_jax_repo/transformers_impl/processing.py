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

"""Standalone InternVL processor - no transformers dependency.

Only the surface ``data/mm_plugin.py`` actually touches: the image processor, the
tokenizer, and ``image_seq_length`` -- which the plugin multiplies by the per-image
patch count to size the ``<IMG_CONTEXT>`` run. Text goes through
``tokenizer.apply_chat_template`` in ``data/template.py``, never through a processor
``__call__``.
"""

import json
import logging
import os
from typing import Optional


from .image_processing import InternVLImageProcessor

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SEQ_LENGTH = 256


def _read_image_seq_length(pretrained_model_name_or_path: str, **download_kwargs) -> int:
    """Prefer ``processor_config.json``; the same value also lives in ``config.json``."""
    for file_name in ("processor_config.json", "config.json"):
        try:
            if os.path.isdir(pretrained_model_name_or_path):
                path = os.path.join(pretrained_model_name_or_path, file_name)
                if not os.path.isfile(path):
                    continue
            else:
                # Baked into the image at /data/{repo_id} at build time
                # (base_image/prepare_model.py); the hub is never reached at run time.
                path = os.path.join("/data", pretrained_model_name_or_path, file_name)
                if not os.path.isfile(path):
                    continue
            with open(path) as handle:
                value = json.load(handle).get("image_seq_length")
            if value is not None:
                return int(value)
        except Exception as error:  # noqa: BLE001 - a missing file is not fatal here
            logger.debug("Could not read %s for image_seq_length: %s", file_name, error)
    return DEFAULT_IMAGE_SEQ_LENGTH


class InternVLProcessor:
    """Bundle of the image processor, tokenizer, and the per-patch image-token budget."""

    image_token: str = "<IMG_CONTEXT>"
    video_token: str = "<video>"
    start_image_token: str = "<img>"
    end_image_token: str = "</img>"

    def __init__(self, image_processor=None, tokenizer=None, image_seq_length: int = DEFAULT_IMAGE_SEQ_LENGTH, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_length = image_seq_length

    # --- multimodal token ids ------------------------------------------------ #
    # torch's InternVLProcessor takes these off the tokenizer in __init__
    # (``image_token_id = tokenizer.context_image_token_id``, and the collator falls
    # back to ``tokenizer.video_token_id``). The data collator reads them to derive
    # ``token_type_ids``, so a processor that exposes only the token STRINGS silently
    # drops that key from the batch. Resolved lazily rather than in __init__ so the
    # class stays a thin bundle and a processor built without a tokenizer just has no
    # ids -- which is what ``getattr(..., None)`` at the call site already expects.

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

    def __call__(self, images=None, return_tensors=None, **kwargs):
        if images is None:
            raise ValueError("This processor only handles images; text is tokenized upstream.")
        return self.image_processor(images, return_tensors=return_tensors, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, tokenizer: Optional[object] = None, **kwargs):
        download_kwargs = {key: kwargs[key] for key in ("cache_dir", "token", "revision") if key in kwargs}
        image_processor = InternVLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=_read_image_seq_length(pretrained_model_name_or_path, **download_kwargs),
        )
