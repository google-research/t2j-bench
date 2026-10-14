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

"""Standalone Gemma3 Processor combining tokenizer and image processor."""
import re
import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Gemma3Processor:
    """Processor for Gemma3 - combines image processing and tokenization."""

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_seq_length=256,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_seq_length = image_seq_length

        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token

        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"

    def __call__(
        self,
        images=None,
        text=None,
        return_tensors=None,
        padding=False,
        do_pan_and_scan=False,
        return_mm_token_type_ids=True,
        **kwargs,
    ):
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        if isinstance(text, str):
            text = [text]

        image_inputs = {}
        if images is not None:
            if not isinstance(images, list):
                images = [images]

            image_inputs = self.image_processor(
                images,
                do_pan_and_scan=do_pan_and_scan,
                return_tensors=return_tensors,
                **kwargs,
            )

            # Replace BOI tokens with full image sequence
            num_crops = image_inputs.pop("num_crops", [0] * len(images))

            if text:
                text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]

        # Tokenize
        text_inputs = self.tokenizer(
            text=text,
            padding=padding,
            return_tensors=return_tensors,
            **{k: v for k, v in kwargs.items() if k not in ("do_pan_and_scan",)},
        )

        # Add token type IDs
        if return_mm_token_type_ids and images is not None:
            if isinstance(text_inputs["input_ids"], list):
                mm_token_type_ids = []
                for ids in text_inputs["input_ids"]:
                    ttids = [1 if tid == self.image_token_id else 0 for tid in ids]
                    mm_token_type_ids.append(ttids)
                text_inputs["token_type_ids"] = mm_token_type_ids
            else:
                array_ids = text_inputs["input_ids"]
                mm_token_type_ids = np.zeros_like(array_ids)
                mm_token_type_ids[array_ids == self.image_token_id] = 1
                text_inputs["token_type_ids"] = mm_token_type_ids

        result = {**text_inputs, **image_inputs}
        return result

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        from .tokenization import Tokenizer
        from .image_processing import Gemma3ImageProcessor

        tokenizer = Tokenizer(model_path)
        image_processor = Gemma3ImageProcessor.from_pretrained(model_path)
        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)
