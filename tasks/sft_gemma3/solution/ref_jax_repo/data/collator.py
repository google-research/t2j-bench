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

"""Data collators for Gemma3 SFT training with multimodal support."""

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .constants import IGNORE_INDEX

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
class MultiModalDataCollatorForSeq2Seq:
    template: Optional["Template"] = None
    model: Any = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX
    tokenizer: Any = None
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

    def _pad_features(self, features: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())
        skip_keys = {"_images", "_videos", "_audios"}
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
                    padded.append(value + remainder)
                batch[key] = np.array(padded, dtype=np.int64)
            elif isinstance(values[0], np.ndarray):
                if all(tuple(value.shape) == tuple(values[0].shape) for value in values):
                    batch[key] = np.stack(values)
                else:
                    batch[key] = values
            else:
                batch[key] = values
        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "np.ndarray"]:
        # Faithful port of THIS task's torch collator (environment/torch_orig/data/
        # collator.py), which is the simplified one: images are collected off the
        # features and the mm-plugin is consulted only when there are some.
        #
        # In particular there is deliberately no fake-image synthesis. The generic
        # LLaMA-Factory collator inserts a blank 64x64 image (and its placeholder
        # expansion) whenever a multimodal batch arrives with none; this task's torch
        # reference does not, and the golden batch is text-only because of it. There is
        # also no get_rope_index call -- this task's torch collator has none.
        features = [dict(feature) for feature in features]
        images = []
        for feature in features:
            feature_images = feature.pop("_images", None)
            if feature_images:
                images.extend(feature_images)
            feature.pop("_videos", None)
            feature.pop("_audios", None)

        batch = self._pad_features(features)

        plugin = getattr(self.template, "mm_plugin", None)
        if images and plugin is not None:
            mm_inputs = _call_with_supported_kwargs(
                plugin.get_mm_inputs,
                images=images,
                videos=[],
                audios=[],
                processor=self.processor,
            ) or {}
            batch.update(mm_inputs)

        # Faithful port of the torch collator: Gemma3 requires token_type_ids during
        # training even if pixel values are absent, so they are derived from the image
        # token id unconditionally (0 = text token, 1 = image token) and overwrite
        # anything the mm-plugin supplied -- the reference does the same, and the goldens
        # were frozen from it.
        image_token_id = getattr(self.processor, "image_token_id", getattr(self.tokenizer, "image_token_id", None))
        if image_token_id is not None and "input_ids" in batch:
            _ids = np.asarray(batch["input_ids"])
            token_type_ids = np.zeros_like(_ids)
            token_type_ids[_ids == image_token_id] = 1
            batch["token_type_ids"] = token_type_ids

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
                        "_images": feature.get("_images"),
                        "_videos": feature.get("_videos"),
                        "_audios": feature.get("_audios"),
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
