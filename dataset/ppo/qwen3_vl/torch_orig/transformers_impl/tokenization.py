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

"""Standalone tokenizer for Qwen models - wraps tokenizers library directly."""

import json
import logging
import os
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as HFTokenizer

logger = logging.getLogger(__name__)


class Tokenizer:
    """Minimal tokenizer wrapper around the tokenizers library for Qwen models."""

    def __init__(
        self,
        tokenizer: HFTokenizer,
        eos_token: str = "<|endoftext|>",
        pad_token: Optional[str] = None,
        padding_side: str = "right",
        model_max_length: int = 128000,
        added_tokens_extra_ids: Optional[dict] = None,
        init_kwargs: Optional[dict] = None,
    ):
        self._tokenizer = tokenizer
        self.eos_token = eos_token
        self.pad_token = pad_token or eos_token
        self.padding_side = padding_side
        self.model_max_length = model_max_length
        self._added_tokens = added_tokens_extra_ids or {}
        self.init_kwargs = init_kwargs or {}

        # Build token-to-id lookup
        vocab = self._tokenizer.get_vocab()
        self._vocab = vocab
        self.eos_token_id = vocab.get(self.eos_token, None)
        self.pad_token_id = vocab.get(self.pad_token, self.eos_token_id)
        self.bos_token_id = None  # Qwen doesn't use BOS

        # Special tokens for Qwen3-VL
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.image_token_id = vocab.get(self.image_token, None)
        self.video_token_id = vocab.get(self.video_token, None)
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.vision_start_token_id = vocab.get(self.vision_start_token, None)
        self.vision_end_token_id = vocab.get(self.vision_end_token, None)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def all_special_tokens(self) -> list[str]:
        return [self.eos_token, self.pad_token]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(tokens, str):
            return self._vocab.get(tokens, 0)
        return [self._vocab.get(t, 0) for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        id_to_token = {v: k for k, v in self._vocab.items()}
        if isinstance(ids, int):
            return id_to_token.get(ids, "<unk>")
        return [id_to_token.get(i, "<unk>") for i in ids]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_ids: list[list[int]], skip_special_tokens: bool = False,
                     **kwargs) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def __call__(self, text: Union[str, list[str]], padding: bool = False,
                 return_tensors: Optional[str] = None, add_special_tokens: bool = True,
                 **kwargs) -> dict:
        """Tokenize text input(s)."""
        if isinstance(text, str):
            text = [text]

        all_ids = []
        for t in text:
            ids = self.encode(t, add_special_tokens=add_special_tokens)
            all_ids.append(ids)

        # Pad if requested
        attention_masks = [[1] * len(ids) for ids in all_ids]
        if padding and len(all_ids) > 1:
            max_len = max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                pad_len = max_len - len(all_ids[i])
                if pad_len > 0:
                    pad_id = self.pad_token_id or 0
                    if self.padding_side == "right":
                        all_ids[i] = all_ids[i] + [pad_id] * pad_len
                        attention_masks[i] = attention_masks[i] + [0] * pad_len
                    else:
                        all_ids[i] = [pad_id] * pad_len + all_ids[i]
                        attention_masks[i] = [0] * pad_len + attention_masks[i]

        result = {"input_ids": all_ids, "attention_mask": attention_masks}

        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def pad(
        self,
        encoded_inputs,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Pad a batch of already-tokenized inputs in a transformers-compatible way."""
        if isinstance(encoded_inputs, dict):
            keys = list(encoded_inputs.keys())
            batch_size = len(encoded_inputs[keys[0]]) if keys else 0
            features = [{key: encoded_inputs[key][idx] for key in keys} for idx in range(batch_size)]
        else:
            features = [dict(item) for item in encoded_inputs]

        if not features:
            return {}

        lengths = [len(feature["input_ids"]) for feature in features]
        if max_length is None:
            max_length = max(lengths)
        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        pad_token_id = self.pad_token_id or 0
        padded: dict[str, list] = {}
        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max(0, max_length - seq_len)
            for key, value in feature.items():
                if isinstance(value, tuple):
                    value = list(value)
                if not isinstance(value, list):
                    padded.setdefault(key, []).append(value)
                    continue
                if key == "input_ids":
                    pad_value = pad_token_id
                elif key == "attention_mask":
                    pad_value = 0
                elif key == "token_type_ids":
                    pad_value = 0
                elif key == "labels":
                    pad_value = -100
                elif key == "position_ids":
                    pad_value = 0
                else:
                    padded.setdefault(key, []).append(value)
                    continue
                if self.padding_side == "right":
                    padded_value = value + [pad_value] * pad_len
                else:
                    padded_value = [pad_value] * pad_len + value
                padded.setdefault(key, []).append(padded_value)

        if return_tensors == "pt":
            import torch

            tensor_batch = {}
            for key, value in padded.items():
                if isinstance(value, list) and value and isinstance(value[0], list):
                    dtype = torch.long
                    tensor_batch[key] = torch.tensor(value, dtype=dtype)
                else:
                    tensor_batch[key] = value
            return tensor_batch

        return padded

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """Add special tokens. Returns number of tokens added."""
        num_added = 0
        from tokenizers import AddedToken
        for key, value in special_tokens_dict.items():
            if value is not None and value not in self._vocab:
                self._tokenizer.add_special_tokens([AddedToken(value, special=True)])
                self._vocab = self._tokenizer.get_vocab()
                num_added += 1
            setattr(self, key, value)
            token_id = self._vocab.get(value, None)
            setattr(self, f"{key}_id", token_id)
        return num_added



    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "Tokenizer":
        """Load tokenizer from a pretrained model path or HF hub."""
        cache_dir = kwargs.pop("cache_dir", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        padding_side = kwargs.pop("padding_side", "right")

        path = pretrained_model_name_or_path
        if os.path.isdir(path):
            tokenizer_file = os.path.join(path, "tokenizer.json")
            config_file = os.path.join(path, "tokenizer_config.json")
        else:
            tokenizer_file = hf_hub_download(
                repo_id=path, filename="tokenizer.json",
                cache_dir=cache_dir, token=token, revision=revision,
            )
            config_file = hf_hub_download(
                repo_id=path, filename="tokenizer_config.json",
                cache_dir=cache_dir, token=token, revision=revision,
            )

        hf_tokenizer = HFTokenizer.from_file(tokenizer_file)

        # Load config for special tokens
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)

        eos_token = config.get("eos_token", "<|endoftext|>")
        if isinstance(eos_token, dict):
            eos_token = eos_token.get("content", "<|endoftext|>")
        pad_token = config.get("pad_token", None)
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content", None)
        model_max_length = config.get("model_max_length", 128000)

        return cls(
            tokenizer=hf_tokenizer,
            eos_token=eos_token,
            pad_token=pad_token,
            padding_side=padding_side,
            model_max_length=model_max_length,
            init_kwargs=config,
        )


def load_tokenizer(pretrained_model_name_or_path: str, **kwargs) -> Tokenizer:
    """Convenience function to load a tokenizer."""
    return Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoTokenizer:
    """Auto tokenizer class compatible with transformers API."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> Tokenizer:
        return Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
