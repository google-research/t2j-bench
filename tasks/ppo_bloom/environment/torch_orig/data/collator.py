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

"""Generic multimodal collator for PPO training."""

from dataclasses import dataclass
import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER

if TYPE_CHECKING:
    from .template import Template

logger = logging.getLogger(__name__)


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**accepted)


@dataclass
class MultiModalDataCollatorForSeq2Seq:
    template: Optional["Template"] = None
    model: Any = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX
    tokenizer: Any = None
    processor: Any = None
    compute_dtype: "torch.dtype" = torch.float32

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")
        actual_model = self.model
        if actual_model is not None and hasattr(actual_model, "base_model") and hasattr(actual_model, "peft_config"):
            actual_model = actual_model.base_model.model
        if actual_model is not None and hasattr(actual_model, "pretrained_model"):
            actual_model = actual_model.pretrained_model
            if hasattr(actual_model, "base_model") and hasattr(actual_model, "peft_config"):
                actual_model = actual_model.base_model.model
        self.get_rope_func = None
        if actual_model is not None:
            if hasattr(actual_model, "get_rope_index"):
                self.get_rope_func = actual_model.get_rope_index
            elif hasattr(actual_model, "model") and hasattr(actual_model.model, "get_rope_index"):
                self.get_rope_func = actual_model.model.get_rope_index

    def _mm_token_type_ids(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """Modality of every position -- 0 text, 1 image, 2 video, the M-RoPE convention."""
        token_type_ids = torch.zeros_like(input_ids)
        plugin = getattr(self.template, "mm_plugin", None)
        for modality, attribute in ((1, "image_token"), (2, "video_token")):
            token = getattr(plugin, attribute, None) or getattr(self.processor, attribute, None)
            token_id = self.tokenizer.convert_tokens_to_ids(token) if token else None
            if token_id is not None:
                token_type_ids[input_ids == token_id] = modality

        return token_type_ids

    def _pad_features(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())
        skip_keys = {"images", "videos", "audios"}
        tensor_keys = all_keys - skip_keys
        batch = {}
        for key in tensor_keys:
            values = [feature.get(key) for feature in features if key in feature]
            if not values:
                continue
            if isinstance(values[0], list):
                max_len = max(len(value) for value in values)
                if self.pad_to_multiple_of is not None:
                    max_len = (
                        (max_len + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )
                if key == "labels":
                    pad_value = self.label_pad_token_id
                elif key == "attention_mask":
                    pad_value = 0
                elif key == "input_ids":
                    pad_value = getattr(self.tokenizer, "pad_token_id", 0) or 0
                elif key == "token_type_ids":
                    pad_value = 0
                else:
                    pad_value = 0
                padded = []
                for value in values:
                    remainder = [pad_value] * (max_len - len(value))
                    if getattr(self.tokenizer, "padding_side", "right") == "left":
                        padded.append(remainder + value)
                    else:
                        padded.append(value + remainder)
                batch[key] = torch.tensor(padded, dtype=torch.long)
            elif isinstance(values[0], torch.Tensor):
                if all(tuple(value.shape) == tuple(values[0].shape) for value in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
            else:
                batch[key] = values
        return batch

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

        plugin = getattr(self.template, "mm_plugin", None)
        fake_input_ids = []
        if (
            plugin is not None
            and getattr(plugin, "image_token", None) is not None
            and sum(batch_imglens) == 0
            and sum(batch_vidlens) == 0
        ):
            from PIL import Image
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            try:
                fake_messages = _call_with_supported_kwargs(
                    plugin.process_messages,
                    messages=fake_messages,
                    images=fake_images,
                    videos=[],
                    audios=[],
                    processor=self.processor,
                )
            except TypeError:
                fake_messages = plugin.process_messages(
                    fake_messages, images=fake_images, videos=[], processor=self.processor
                )
            fake_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            try:
                fake_ids, _ = _call_with_supported_kwargs(
                    plugin.process_token_ids,
                    input_ids=fake_ids,
                    labels=None,
                    images=fake_images,
                    videos=[],
                    audios=[],
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )
            except TypeError:
                fake_ids, _ = plugin.process_token_ids(
                    fake_ids, None, images=fake_images, tokenizer=self.tokenizer, processor=self.processor
                )
            fake_input_ids.extend(fake_ids)
            batch_images = fake_images
            batch_imglens[0] = 1

        if fake_input_ids:
            if getattr(self.tokenizer, "padding_side", "right") == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]
            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = {}
        if plugin is not None:
            mm_inputs = _call_with_supported_kwargs(
                plugin.get_mm_inputs,
                images=batch_images,
                videos=batch_videos,
                audios=batch_audios,
                batch_images=batch_images,
                batch_videos=batch_videos,
                batch_audios=batch_audios,
                imglens=batch_imglens,
                vidlens=batch_vidlens,
                audlens=batch_audlens,
                batch_imglens=batch_imglens,
                batch_vidlens=batch_vidlens,
                batch_audlens=batch_audlens,
                batch_ids=batch_input_ids,
                batch_input_ids=batch_input_ids,
                processor=self.processor,
            ) or {}

        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        batch = self._pad_features(features)
        if "token_type_ids" not in batch:
            multimodal_token_ids = []
            for name in ("image_token_id", "video_token_id"):
                token_id = getattr(self.processor, name, getattr(self.tokenizer, name, None))
                if token_id is not None:
                    multimodal_token_ids.append(token_id)
            if multimodal_token_ids and "input_ids" in batch:
                token_type_ids = torch.zeros_like(batch["input_ids"])
                for token_id in multimodal_token_ids:
                    token_type_ids[batch["input_ids"] == token_id] = 1
                if torch.any(token_type_ids != 0):
                    batch["token_type_ids"] = token_type_ids

        if self.get_rope_func is not None:
            rope_kwargs = {
                "input_ids": batch["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (batch["attention_mask"] >= 1).float(),
            }
            if "second_per_grid_ts" in mm_inputs:
                rope_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            if "mm_token_type_ids" in inspect.signature(self.get_rope_func).parameters:
                rope_kwargs["mm_token_type_ids"] = self._mm_token_type_ids(batch["input_ids"])
            try:
                position_ids, rope_deltas = self.get_rope_func(**rope_kwargs)
                batch["position_ids"] = position_ids
                batch["rope_deltas"] = rope_deltas
            except TypeError as error:
                logger.warning("get_rope_index failed, leaving position_ids to the model: %s", error)
        batch.update(mm_inputs)

        if self.compute_dtype is not None:
            for key, value in batch.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    batch[key] = value.to(self.compute_dtype)
        return batch
