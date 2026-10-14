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

"""Generic chat-template handling for generated SFT repos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

try:  # text-only repos ship no mm_plugin.py, and then nothing below changes behaviour
    from .mm_plugin import PLUGIN_SPECS, get_mm_plugin
except ImportError:
    PLUGIN_SPECS, get_mm_plugin = {}, None


@dataclass
class Template:
    name: str
    efficient_eos: bool = False
    # When set, the plugin owns media handling: it expands the <image> placeholders in the
    # raw message text, so the template must not also wrap them in structured content.
    mm_plugin: Optional[Any] = None

    def _structured_content(self, text: str, images: list[Any], videos: list[Any], audios: list[Any]):
        items: list[dict[str, str]] = []
        items.extend({"type": "image"} for _ in images)
        items.extend({"type": "video"} for _ in videos)
        items.extend({"type": "audio"} for _ in audios)
        items.append({"type": "text", "text": text})
        return items

    def build_chat_messages(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str,
        tools: str,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if tools:
            tool_message = {"role": "system", "content": tools}
            messages.append(tool_message)

        media_attached = self.mm_plugin is not None
        for message in prompt + response:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not media_attached and role == "user" and (images or videos or audios):
                messages.append({"role": role, "content": self._structured_content(content, images, videos, audios)})
                media_attached = True
            else:
                messages.append({"role": role, "content": content})
        return messages

    def _fallback_string(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                text_bits = []
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        text_bits.append(item.get("text", ""))
                    elif item_type == "image":
                        text_bits.append(IMAGE_PLACEHOLDER)
                    elif item_type == "video":
                        text_bits.append(VIDEO_PLACEHOLDER)
                    elif item_type == "audio":
                        text_bits.append(AUDIO_PLACEHOLDER)
                content = " ".join(bit for bit in text_bits if bit)
            lines.append(f"{message.get('role', 'user')}: {content}")
        return "\n".join(lines)

    def _apply_template(self, tokenizer, messages: list[dict[str, Any]], processor=None) -> list[int]:
        apply_template = None
        if self.mm_plugin is not None:
            # The plugin already expanded the media tokens into the text, so the string is
            # complete: go through the tokenizer and leave the processor to the collator.
            # processor.apply_chat_template would re-expand (and returns a batched result).
            processor = None
        if processor is not None and hasattr(processor, "apply_chat_template"):
            apply_template = processor.apply_chat_template
        elif hasattr(tokenizer, "apply_chat_template"):
            apply_template = tokenizer.apply_chat_template

        if apply_template is not None:
            for kwargs in (
                {"tokenize": True, "add_generation_prompt": False},
                {"tokenize": True},
                {},
            ):
                try:
                    result = apply_template(messages, **kwargs)
                    if isinstance(result, dict) and "input_ids" in result:
                        result = result["input_ids"]
                    if hasattr(result, "tolist"):
                        result = result.tolist()
                    if isinstance(result, list):
                        return [int(token_id) for token_id in result]
                    if isinstance(result, str):
                        return tokenizer.encode(result, add_special_tokens=False)
                except TypeError:
                    continue
                except Exception:
                    break

        return tokenizer.encode(self._fallback_string(messages), add_special_tokens=True)

    def encode_multiturn(self, tokenizer, messages, system="", tools="", processor=None, **kwargs):
        encoded_pairs = []
        last_ids: list[int] = []
        for idx, message in enumerate(messages):
            role = message.get("role", "user")
            if role != "assistant":
                continue
            prompt_ids = self._apply_template(tokenizer, messages[:idx], processor=processor)
            full_ids = self._apply_template(tokenizer, messages[: idx + 1], processor=processor)
            source_ids = prompt_ids[len(last_ids) :]
            target_ids = full_ids[len(prompt_ids) :]
            encoded_pairs.append((source_ids, target_ids))
            last_ids = full_ids
        return encoded_pairs


def get_template_and_fix_tokenizer(tokenizer, data_args) -> Template:
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    name = getattr(data_args, "template", "default")
    mm_plugin = None
    if get_mm_plugin is not None:
        # A repo with an mm_plugin.py serves exactly one vision model, so an unrecognised
        # template name still gets that model's plugin rather than silently losing images.
        spec = PLUGIN_SPECS.get(name) or PLUGIN_SPECS.get("default")
        if spec is not None:
            mm_plugin = get_mm_plugin(**spec)

    return Template(name=name, mm_plugin=mm_plugin)

