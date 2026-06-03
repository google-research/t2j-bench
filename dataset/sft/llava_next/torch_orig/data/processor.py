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

"""Supervised dataset processor for LLaVA-Next SFT."""
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int):
    """Infer source and target lengths after cutoff."""
    if source_len + target_len <= cutoff_len:
        return source_len, target_len

    if source_len > cutoff_len:
        return cutoff_len, 0

    return source_len, cutoff_len - source_len


@dataclass
class SupervisedDatasetProcessor:
    """Processes datasets for supervised fine-tuning."""
    template: Any
    tokenizer: Any
    processor: Any = None
    data_args: Any = None

    def _encode_data_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Encode a single data example into input_ids and labels."""
        messages = example.get("_prompt", []) + example.get("_response", [])
        system = example.get("_system", "")
        tools = example.get("_tools", "")
        images = example.get("_images", None)
        videos = example.get("_videos", None)

        cutoff_len = getattr(self.data_args, "cutoff_len", 2048)
        train_on_prompt = getattr(self.data_args, "train_on_prompt", False)
        mask_history = getattr(self.data_args, "mask_history", False)

        # Process messages through mm_plugin
        if hasattr(self.template, "mm_plugin"):
            messages = self.template.mm_plugin.process_messages(
                messages, images=images, videos=videos, processor=self.processor,
            )

        # Encode messages
        encoded_pairs = self.template.encode_multiturn(
            self.tokenizer, messages, system=system, tools=tools,
        )

        # Build input_ids and labels
        input_ids = []
        labels = []

        for turn_idx, (prompt_ids, response_ids) in enumerate(encoded_pairs):
            if not response_ids:
                # Last turn with only prompt
                input_ids.extend(prompt_ids)
                labels.extend([IGNORE_INDEX] * len(prompt_ids))
                continue

            # Check cutoff
            source_len, target_len = infer_seqlen(
                len(prompt_ids), len(response_ids), cutoff_len - len(input_ids)
            )

            if source_len == 0 and target_len == 0:
                break

            prompt_ids = prompt_ids[:source_len]
            response_ids = response_ids[:target_len]

            if train_on_prompt:
                input_ids.extend(prompt_ids + response_ids)
                labels.extend(prompt_ids + response_ids)
            elif mask_history and turn_idx < len(encoded_pairs) - 1:
                input_ids.extend(prompt_ids + response_ids)
                labels.extend([IGNORE_INDEX] * (len(prompt_ids) + len(response_ids)))
            else:
                input_ids.extend(prompt_ids + response_ids)
                labels.extend([IGNORE_INDEX] * len(prompt_ids) + response_ids)

        # Truncate to cutoff_len
        input_ids = input_ids[:cutoff_len]
        labels = labels[:cutoff_len]

        # Process token IDs through mm_plugin
        if hasattr(self.template, "mm_plugin"):
            input_ids, labels = self.template.mm_plugin.process_token_ids(
                input_ids, labels=labels, images=images,
                tokenizer=self.tokenizer, processor=self.processor,
            )

        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

        # Pass through images for collator
        if images:
            result["_images"] = images

        return result

    def preprocess_dataset(self, examples: dict[str, list]) -> dict[str, list]:
        """Process a batch of examples."""
        batch_size = len(next(iter(examples.values())))
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "_images": [],
        }

        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            encoded = self._encode_data_example(example)
            if encoded and len(encoded["input_ids"]) > 0:
                for key in results:
                    if key in encoded:
                        results[key].append(encoded[key])
                    else:
                        results[key].append(None)

        return results

    def print_data_example(self, example: dict[str, Any]):
        """Print a data example for debugging."""
        input_ids = example.get("input_ids", [])
        labels = example.get("labels", [])

        logger.info(f"input_ids length: {len(input_ids)}")
        logger.info(f"labels length: {len(labels)}")

        decoded = self.tokenizer.decode(input_ids)
        logger.info(f"inputs:\n{decoded}")

        # Show which tokens are trained on
        trained_tokens = [tid for tid, lid in zip(input_ids, labels) if lid != IGNORE_INDEX]
        if trained_tokens:
            decoded_labels = self.tokenizer.decode(trained_tokens)
            logger.info(f"labels:\n{decoded_labels}")
