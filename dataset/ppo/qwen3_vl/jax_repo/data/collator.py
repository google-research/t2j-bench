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

"""Data collator for Qwen3-VL PPO with multimodal support."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import DataCollatorForSeq2Seq

from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER

if TYPE_CHECKING:
    from .template import Template


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """Data collator that supports VLMs with multimodal inputs (images, videos)."""

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        # Unwrap PeftModel if needed (detect by attribute, avoid importing peft
        # which conflicts with local transformers/ shadow)
        if self.model is not None and hasattr(self.model, "base_model") and hasattr(self.model, "peft_config"):
            self.model = self.model.base_model.model

        # Handle ModelWithValueHead wrapper
        actual_model = self.model
        if actual_model is not None and hasattr(actual_model, "pretrained_model"):
            actual_model = actual_model.pretrained_model
            # Also unwrap PeftModel inside value head wrapper
            if hasattr(actual_model, "base_model") and hasattr(actual_model, "peft_config"):
                actual_model = actual_model.base_model.model

        # Find get_rope_index for Qwen VL mrope
        self.get_rope_func = None
        if actual_model is not None:
            if hasattr(actual_model, "get_rope_index"):
                self.get_rope_func = actual_model.get_rope_index
            elif hasattr(actual_model, "model") and hasattr(actual_model.model, "get_rope_index"):
                self.get_rope_func = actual_model.model.get_rope_index

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = [dict(feature) for feature in features]
        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])

        # Handle zero-image case to avoid process hanging
        fake_input_ids = []
        if (
            self.template.mm_plugin.image_token is not None
            and sum(batch_imglens) == 0
            and sum(batch_vidlens) == 0
        ):
            from PIL import Image
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_ids)
            batch_images = fake_images
            batch_imglens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]
            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_audios,
            batch_imglens, batch_vidlens, batch_audlens,
            batch_input_ids, self.processor,
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: dict[str, torch.Tensor] = super().__call__(features)

        # Compute 3D rope position ids for Qwen VL models
        if self.get_rope_func is not None:
            rope_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (features["attention_mask"] >= 1).float(),
            }
            if "second_per_grid_ts" in mm_inputs:
                rope_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_kwargs)

        features.update(mm_inputs)
        return features
