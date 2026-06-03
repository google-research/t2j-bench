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

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from transformers import DataCollatorForSeq2Seq


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
class SFTDataCollatorWith4DAttentionMask(DataCollatorForSeq2Seq):
    template: Optional[Any] = None
    processor: Optional[Any] = None
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: torch.dtype = torch.float32

    def _build_multimodal_inputs(self, batch_images, batch_videos, batch_audios):
        if self.processor is None:
            return {}

        merged_images = [item for sample in batch_images for item in sample]
        merged_videos = [item for sample in batch_videos for item in sample]
        merged_audios = [item for sample in batch_audios for item in sample]
        if not merged_images and not merged_videos and not merged_audios:
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
            return {key: value for key, value in result.items() if torch.is_tensor(value)}
        except Exception:
            result = {}
            image_processor = getattr(self.processor, "image_processor", None)
            if merged_images and image_processor is not None:
                image_inputs = image_processor(merged_images, return_tensors="pt")
                if hasattr(image_inputs, "data"):
                    image_inputs = image_inputs.data
                result.update({key: value for key, value in image_inputs.items() if torch.is_tensor(value)})
            return result

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        batch_images = [feature.pop("images", None) or [] for feature in features]
        batch_videos = [feature.pop("videos", None) or [] for feature in features]
        batch_audios = [feature.pop("audios", None) or [] for feature in features]

        batch = super().__call__(features)
        mm_inputs = self._build_multimodal_inputs(batch_images, batch_videos, batch_audios)
        batch = _merge_media_inputs(mm_inputs, batch)

        if self.model is not None and hasattr(self.model, "get_rope_index"):
            rope_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": (batch["attention_mask"] >= 1).float(),
            }
            for key in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                if key in mm_inputs:
                    rope_kwargs[key] = mm_inputs[key]
            try:
                batch["position_ids"], batch["rope_deltas"] = self.model.get_rope_index(**rope_kwargs)
            except Exception:
                pass

        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            batch["attention_mask"] = prepare_4d_attention_mask(batch["attention_mask"], self.compute_dtype)
        for key, value in list(batch.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(self.compute_dtype)
        return batch

