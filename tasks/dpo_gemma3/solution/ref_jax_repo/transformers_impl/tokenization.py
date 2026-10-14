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

"""Standalone tokenizer for Gemma3 - wraps the tokenizers library."""
import json
import logging
import os
from typing import Any, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Tokenizer:
    """Minimal tokenizer wrapping the tokenizers library."""

    def __init__(self, tokenizer_path: str, **kwargs):
        from tokenizers import Tokenizer as HFTokenizer

        # Resolve tokenizer_path: the image bakes the model files into /data/{repo_id}
        # at build time (base_image/prepare_model.py); nothing is fetched at run time.
        resolved_path = tokenizer_path
        if not os.path.isdir(tokenizer_path):
            baked = os.path.join("/data", tokenizer_path)
            if os.path.isdir(baked):
                resolved_path = baked

        tokenizer_file = os.path.join(resolved_path, "tokenizer.json")
        if os.path.exists(tokenizer_file):
            self._tokenizer = HFTokenizer.from_file(tokenizer_file)
        else:
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

        # Load tokenizer config
        config_file = os.path.join(resolved_path, "tokenizer_config.json")
        self.init_kwargs = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.init_kwargs = json.load(f)

        # Load special tokens map
        special_tokens_file = os.path.join(resolved_path, "special_tokens_map.json")
        self._special_tokens_map = {}
        if os.path.exists(special_tokens_file):
            with open(special_tokens_file, "r") as f:
                self._special_tokens_map = json.load(f)

        # Set special tokens
        self.pad_token = self.init_kwargs.get("pad_token", "<pad>")
        self.eos_token = self.init_kwargs.get("eos_token", "<eos>")
        self.bos_token = self.init_kwargs.get("bos_token", "<bos>")
        self.unk_token = self.init_kwargs.get("unk_token", "<unk>")

        # Handle special token dicts
        if isinstance(self.pad_token, dict):
            self.pad_token = self.pad_token.get("content", "<pad>")
        if isinstance(self.eos_token, dict):
            self.eos_token = self.eos_token.get("content", "<eos>")
        if isinstance(self.bos_token, dict):
            self.bos_token = self.bos_token.get("content", "<bos>")

        # Token IDs
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.eos_token_id = self.convert_tokens_to_ids(self.eos_token)
        self.bos_token_id = self.convert_tokens_to_ids(self.bos_token)

        # Gemma3 special tokens
        self.image_token = "<image_soft_token>"
        self.boi_token = "<start_of_image>"
        self.eoi_token = "<end_of_image>"

        self.image_token_id = self.convert_tokens_to_ids(self.image_token)
        self.boi_token_id = self.convert_tokens_to_ids(self.boi_token)
        self.eoi_token_id = self.convert_tokens_to_ids(self.eoi_token)

        # Additional properties
        self.model_input_names = ["input_ids", "attention_mask"]
        self._added_tokens = {}
        self.padding_side = self.init_kwargs.get("padding_side", "right")
        self._path = tokenizer_path

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            token_id = self._tokenizer.token_to_id(tokens)
            return token_id if token_id is not None else 0
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids) or ""
        return [self.convert_ids_to_tokens(i) for i in ids]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        if isinstance(text, str):
            text = [text]

        all_input_ids = []
        for t in text:
            ids = self.encode(t, add_special_tokens=add_special_tokens)
            if truncation and max_length:
                ids = ids[:max_length]
            all_input_ids.append(ids)

        # Padding
        if padding:
            max_len = max(len(ids) for ids in all_input_ids)
            if max_length:
                max_len = min(max_len, max_length)

            attention_masks = []
            for i, ids in enumerate(all_input_ids):
                pad_len = max_len - len(ids)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
                all_input_ids[i] = ids + [self.pad_token_id] * pad_len
        else:
            attention_masks = [[1] * len(ids) for ids in all_input_ids]

        result = {
            "input_ids": all_input_ids,
            "attention_mask": attention_masks,
        }

        if return_tensors == "pt":
            result = {k: np.asarray(v) for k, v in result.items()}

        return result

    def add_special_tokens(self, special_tokens_dict):
        """Add special tokens to the tokenizer."""
        added = 0
        for key, value in special_tokens_dict.items():
            if isinstance(value, list):
                for v in value:
                    token = v if isinstance(v, str) else str(v)
                    if token not in self._added_tokens:
                        self._added_tokens[token] = self.vocab_size + len(self._added_tokens)
                        added += 1
            else:
                token = value if isinstance(value, str) else str(value)
                setattr(self, key, token)
                if key.endswith("_token"):
                    id_key = key + "_id"
                    setattr(self, id_key, self.convert_tokens_to_ids(token))
        return added

    def save_pretrained(self, save_directory):
        """Save tokenizer files."""
        os.makedirs(save_directory, exist_ok=True)
        self._tokenizer.save(os.path.join(save_directory, "tokenizer.json"))

        if self.init_kwargs:
            with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
                json.dump(self.init_kwargs, f, indent=2)


class AutoTokenizer:
    """Auto tokenizer loader."""

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        return Tokenizer(pretrained_model_name_or_path, **kwargs)
