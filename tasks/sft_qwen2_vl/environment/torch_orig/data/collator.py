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

"""Shared SFT collator with optional multimodal processor support."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch

from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class _DataCollatorForSeq2Seq:
    """Implementation of ``transformers.DataCollatorForSeq2Seq`` (no transformers dep)."""

    tokenizer: Any
    model: Any = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]], return_tensors: Optional[str] = None) -> dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name, None) for feature in features]
        if all(l is None for l in labels):
            labels = None

        max_len = max(len(f["input_ids"]) for f in features)
        if self.max_length is not None and self.padding:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]

            if key in ("input_ids", "attention_mask", "token_type_ids"):
                padded = []
                for v in values:
                    if isinstance(v, list):
                        v = list(v)
                    else:
                        v = v.tolist() if hasattr(v, "tolist") else list(v)
                    pad_val = pad_token_id if key == "input_ids" else 0
                    pad_len = max_len - len(v)
                    if pad_len > 0 and self.padding:
                        if padding_side == "right":
                            v = v + [pad_val] * pad_len
                        else:
                            v = [pad_val] * pad_len + v
                    padded.append(v)

                if return_tensors == "pt":
                    batch[key] = torch.tensor(padded, dtype=torch.long)
                else:
                    batch[key] = np.array(padded, dtype=np.int64)
            else:
                try:
                    if return_tensors == "pt":
                        if isinstance(values[0], torch.Tensor):
                            batch[key] = torch.stack(values)
                        else:
                            batch[key] = torch.tensor(values)
                    else:
                        batch[key] = np.array(values)
                except (ValueError, RuntimeError):
                    batch[key] = values

        if labels is not None:
            max_label_len = max_len
            padded_labels = []
            for label in labels:
                if isinstance(label, list):
                    label = list(label)
                else:
                    label = label.tolist() if hasattr(label, "tolist") else list(label)

                pad_len = max_label_len - len(label)
                if pad_len > 0 and self.padding:
                    if padding_side == "right":
                        label = label + [self.label_pad_token_id] * pad_len
                    else:
                        label = [self.label_pad_token_id] * pad_len + label
                padded_labels.append(label)

            if return_tensors == "pt":
                batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            else:
                batch["labels"] = np.array(padded_labels, dtype=np.int64)

        return batch


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


def prepare_4d_attention_mask(attention_mask_with_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    return torch.where(attention_mask_4d, zero_tensor, min_dtype)


def _merge_media_inputs(mm_inputs: dict[str, Any], features: dict[str, torch.Tensor]):
    if not mm_inputs:
        return features
    for key, value in mm_inputs.items():
        if torch.is_tensor(value):
            features[key] = value
    return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(_DataCollatorForSeq2Seq):
    template: Optional[Any] = None
    processor: Optional[Any] = None
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: torch.dtype = torch.float32

    def _inject_fake_image(self, features, batch_images, batch_videos, batch_input_ids) -> None:
        """Append one masked-out white image when the batch carries no media at all."""
        plugin = getattr(self.template, "mm_plugin", None)
        if plugin is None or getattr(plugin, "image_token", None) is None:
            return
        if any(batch_images) or any(batch_videos):
            return


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
        if not fake_ids:
            return

        if getattr(self.tokenizer, "padding_side", "right") == "right":
            features[0]["input_ids"] = features[0]["input_ids"] + list(fake_ids)
            features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_ids)
            features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_ids)
        else:
            features[0]["input_ids"] = list(fake_ids) + features[0]["input_ids"]
            features[0]["attention_mask"] = [0] * len(fake_ids) + features[0]["attention_mask"]
            features[0]["labels"] = [IGNORE_INDEX] * len(fake_ids) + features[0]["labels"]
        batch_input_ids[0] = features[0]["input_ids"]
        batch_images[0] = fake_images

    def _build_multimodal_inputs(self, batch_images, batch_videos, batch_audios, batch_input_ids):
        merged_images = [item for sample in batch_images for item in sample]
        merged_videos = [item for sample in batch_videos for item in sample]
        merged_audios = [item for sample in batch_audios for item in sample]
        if not merged_images and not merged_videos and not merged_audios:
            return {}

        plugin = getattr(self.template, "mm_plugin", None)
        if plugin is not None:
            imglens = [len(sample) for sample in batch_images]
            vidlens = [len(sample) for sample in batch_videos]
            audlens = [len(sample) for sample in batch_audios]
            return _call_with_supported_kwargs(
                plugin.get_mm_inputs,
                images=merged_images,
                videos=merged_videos,
                audios=merged_audios,
                batch_images=merged_images,
                batch_videos=merged_videos,
                batch_audios=merged_audios,
                imglens=imglens,
                vidlens=vidlens,
                audlens=audlens,
                batch_imglens=imglens,
                batch_vidlens=vidlens,
                batch_audlens=audlens,
                batch_ids=batch_input_ids,
                batch_input_ids=batch_input_ids,
                processor=self.processor,
            ) or {}

        if self.processor is None:
            return {}

        try:
            kwargs = {"return_tensors": "pt"}
            if merged_images:
                kwargs["images"] = merged_images
            if merged_videos:
                kwargs["videos"] = merged_videos
            if merged_audios:
                kwargs["audios"] = merged_audios
            result = self.processor(**kwargs)
            if hasattr(result, "data"):
                result = result.data
            text_keys = {"input_ids", "attention_mask", "labels", "token_type_ids"}
            return {
                key: value
                for key, value in result.items()
                if torch.is_tensor(value) and key not in text_keys
            }
        except Exception:
            result = {}
            image_processor = getattr(self.processor, "image_processor", None)
            if merged_images and image_processor is not None:
                image_inputs = image_processor(merged_images, return_tensors="pt")
                if hasattr(image_inputs, "data"):
                    image_inputs = image_inputs.data
                result.update({key: value for key, value in image_inputs.items() if torch.is_tensor(value)})
            return result

    def _resolve_rope_index(self):
        """Find ``get_rope_index``, which sits several wrappers down once PEFT is in play."""
        candidate = self.model
        for _ in range(4):
            if candidate is None:
                return None
            rope_index = getattr(candidate, "get_rope_index", None)
            if callable(rope_index):
                return rope_index

            candidate = getattr(candidate, "model", None)

        return None

    def _mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Modality of every position -- 0 text, 1 image, 2 video, the M-RoPE convention."""
        token_type_ids = torch.zeros_like(input_ids)
        plugin = getattr(self.template, "mm_plugin", None)
        for modality, attribute in ((1, "image_token"), (2, "video_token")):
            token = getattr(plugin, attribute, None) or getattr(self.processor, attribute, None)
            token_id = self.tokenizer.convert_tokens_to_ids(token) if token else None
            if token_id is not None:
                token_type_ids[input_ids == token_id] = modality

        return token_type_ids

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        batch_images = [feature.pop("images", None) or [] for feature in features]
        batch_videos = [feature.pop("videos", None) or [] for feature in features]
        batch_audios = [feature.pop("audios", None) or [] for feature in features]
        batch_input_ids = [feature["input_ids"] for feature in features]

        self._inject_fake_image(features, batch_images, batch_videos, batch_input_ids)
        mm_inputs = self._build_multimodal_inputs(batch_images, batch_videos, batch_audios, batch_input_ids)
        if "token_type_ids" in mm_inputs:
            for feature, token_type_ids in zip(features, mm_inputs.pop("token_type_ids")):
                feature["token_type_ids"] = token_type_ids

        batch = super().__call__(features)
        batch = _merge_media_inputs(mm_inputs, batch)

        rope_index = self._resolve_rope_index()
        if rope_index is not None:
            rope_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": (batch["attention_mask"] >= 1).float(),
            }
            for key in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                if key in mm_inputs:
                    rope_kwargs[key] = mm_inputs[key]
            if "mm_token_type_ids" in inspect.signature(rope_index).parameters:
                rope_kwargs["mm_token_type_ids"] = self._mm_token_type_ids(batch["input_ids"])
            try:
                batch["position_ids"], batch["rope_deltas"] = rope_index(**rope_kwargs)
            except Exception as error:
                logger.warning("get_rope_index failed, leaving position_ids to the model: %s", error)

        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            batch["attention_mask"] = prepare_4d_attention_mask(batch["attention_mask"], self.compute_dtype)
        for key, value in list(batch.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(self.compute_dtype)
        return batch

