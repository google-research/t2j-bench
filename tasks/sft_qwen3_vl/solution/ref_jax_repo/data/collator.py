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

"""Data collator for Qwen3-VL SFT with multimodal and 4D attention mask support."""

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from transformers_impl.collator import DataCollatorForSeq2Seq

from .constants import IGNORE_INDEX, IMAGE_PLACEHOLDER

if TYPE_CHECKING:
    from .template import Template


def _np_is_tensor(x):
    return isinstance(x, np.ndarray)


def _np_is_float(x):
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating)


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
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """Data collator that supports VLMs with multimodal inputs (images, videos).

    Mirror of torch's class, base included: ``tokenizer``, ``model``, ``pad_to_multiple_of``
    and ``label_pad_token_id`` are ``DataCollatorForSeq2Seq``'s fields, and the padding
    itself is that base's ``__call__``. JAX adjustment: ``compute_dtype`` is a numpy dtype
    and is declared here rather than on the SFT subclass, because the pairwise collator
    casts through this same code path.
    """

    template: Optional["Template"] = None
    processor: Any = None
    compute_dtype: "np.dtype" = np.float32

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")
        if (
            self.model is not None
            and hasattr(self.model, "base_model")
            and hasattr(self.model.base_model, "model")
            and "Peft" in self.model.__class__.__name__
        ):
            self.model = self.model.base_model.model
        self.get_rope_func = None
        if self.model is not None:
            if hasattr(self.model, "get_rope_index"):
                self.get_rope_func = self.model.get_rope_index
            elif hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
                self.get_rope_func = self.model.model.get_rope_index

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "np.ndarray"]:
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

        batch = super().__call__(features)
        if self.get_rope_func is not None:
            rope_kwargs = {
                "input_ids": batch["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (np.asarray(batch["attention_mask"]) >= 1).astype(np.float32),
            }
            if "second_per_grid_ts" in mm_inputs:
                rope_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            try:
                position_ids, rope_deltas = self.get_rope_func(**rope_kwargs)
                batch["position_ids"] = position_ids
                batch["rope_deltas"] = rope_deltas
            except TypeError:
                pass
        batch.update(mm_inputs)

        if self.compute_dtype is not None:
            for key, value in batch.items():
                if _np_is_tensor(value) and _np_is_float(value):
                    batch[key] = value.astype(self.compute_dtype)
        return batch


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "np.ndarray"]:
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                concatenated_features.append(
                    {
                        "input_ids": feature[f"{key}_input_ids"],
                        "attention_mask": feature[f"{key}_attention_mask"],
                        "labels": feature[f"{key}_labels"],
                        "images": feature.get("images"),
                        "videos": feature.get("videos"),
                        "audios": feature.get("audios"),
                    }
                )
        return super().__call__(concatenated_features)


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    """SFT collator — mirror of torch's ``SFTDataCollatorWith4DAttentionMask``.

    JAX adjustment: the 4D block-diagonal attention mask is a torch/eager-attention
    optimization; the flax modules build their own additive causal+padding bias from
    ``attention_mask``, so the produced batch is identical to the base collator. Kept as
    a distinct class so the reference exposes the same name the SFT pipeline references."""

    block_diag_attn: bool = False
    attn_implementation: str = "eager"
