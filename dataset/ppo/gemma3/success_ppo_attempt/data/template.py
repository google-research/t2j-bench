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

"""Chat template system for Gemma3 SFT training."""
import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Optional, Sequence, Union

from .formatter import EmptyFormatter, Formatter, StringFormatter, SLOT
from .mm_plugin import BasePlugin, Gemma3Plugin, get_mm_plugin

logger = logging.getLogger(__name__)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class Template:
    """Chat template for formatting messages."""
    format_user: Formatter
    format_assistant: Formatter
    format_system: Formatter
    format_function: Formatter = field(default_factory=lambda: EmptyFormatter(slots=[]))
    format_observation: Formatter = field(default_factory=lambda: EmptyFormatter(slots=[]))
    format_tools: Formatter = field(default_factory=lambda: EmptyFormatter(slots=[]))
    format_prefix: Formatter = field(default_factory=lambda: EmptyFormatter(slots=[]))
    format_separator: Formatter = field(default_factory=lambda: EmptyFormatter(slots=[]))
    default_system: str = ""
    stop_words: list[str] = field(default_factory=list)
    efficient_eos: bool = False
    replace_eos: bool = False
    replace_jinja_template: bool = False
    mm_plugin: BasePlugin = field(default_factory=BasePlugin)

    def encode_oneturn(self, tokenizer, messages, system="", tools="", **kwargs):
        """Encode a single-turn conversation into token IDs."""
        encoded_pairs = self.encode_multiturn(tokenizer, messages, system, tools, **kwargs)
        prompt_ids = []
        response_ids = []
        for i, (p, r) in enumerate(encoded_pairs):
            if i < len(encoded_pairs) - 1:
                prompt_ids.extend(p + r)
            else:
                prompt_ids.extend(p)
                response_ids.extend(r)
        return prompt_ids, response_ids

    def encode_multiturn(self, tokenizer, messages, system="", tools="", **kwargs):
        """Encode a multi-turn conversation into pairs of (prompt_ids, response_ids)."""
        system = system or self.default_system
        encoded_pairs = []

        # Build system/prefix
        system_ids = self._encode_system(tokenizer, system, tools)
        prefix_ids = self._encode_prefix(tokenizer)

        first_turn = True
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", Role.USER)
            content = msg.get("content", "")

            if role in (Role.USER, "user"):
                # Format user message
                user_slots = self.format_user.apply(content=content)
                user_ids = self._tokenize_slots(tokenizer, user_slots)

                if first_turn:
                    user_ids = prefix_ids + system_ids + user_ids
                    first_turn = False

                # Look for assistant response
                response_ids = []
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    next_role = next_msg.get("role", "")
                    if next_role in (Role.ASSISTANT, "assistant"):
                        next_content = next_msg.get("content", "")
                        assistant_slots = self.format_assistant.apply(content=next_content)
                        response_ids = self._tokenize_slots(tokenizer, assistant_slots)
                        i += 1

                encoded_pairs.append((user_ids, response_ids))

            elif role in (Role.ASSISTANT, "assistant"):
                # Standalone assistant message (shouldn't normally happen)
                assistant_slots = self.format_assistant.apply(content=content)
                response_ids = self._tokenize_slots(tokenizer, assistant_slots)
                encoded_pairs.append(([], response_ids))

            i += 1

        return encoded_pairs

    def _encode_system(self, tokenizer, system, tools=""):
        if not system and not tools:
            return []
        content = system
        if tools:
            content = content + "\n\n" + tools if content else tools
        system_slots = self.format_system.apply(content=content)
        return self._tokenize_slots(tokenizer, system_slots)

    def _encode_prefix(self, tokenizer):
        if not self.format_prefix.slots:
            return []
        prefix_slots = self.format_prefix.apply()
        return self._tokenize_slots(tokenizer, prefix_slots)

    def _tokenize_slots(self, tokenizer, slots):
        """Convert slots to token IDs."""
        token_ids = []
        for slot in slots:
            if isinstance(slot, str):
                if slot:
                    ids = tokenizer.encode(slot, add_special_tokens=False)
                    token_ids.extend(ids)
            elif isinstance(slot, set):
                # Special token reference like {"bos_token"}
                for token_name in slot:
                    token = getattr(tokenizer, token_name, None)
                    if token:
                        token_id = getattr(tokenizer, token_name + "_id", None)
                        if token_id is not None:
                            token_ids.append(token_id)
            elif isinstance(slot, dict):
                # Token with metadata
                for token_name, value in slot.items():
                    token_id = tokenizer.convert_tokens_to_ids(value)
                    if token_id is not None:
                        token_ids.append(token_id)
        return token_ids


# ===================== Template Registry =====================

TEMPLATES: dict[str, Template] = {}


def register_template(name: str, **kwargs):
    """Register a chat template."""
    template_class = kwargs.pop("template_class", Template)
    template = template_class(**kwargs)
    TEMPLATES[name] = template
    return template


def get_template(name: str) -> Template:
    """Get a registered template by name."""
    if name not in TEMPLATES:
        raise ValueError(f"Template '{name}' not found. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


# ===================== Llama2Template (used by Gemma3) =====================

class Llama2Template(Template):
    """Template that fuses system message into the first user message."""

    def encode_multiturn(self, tokenizer, messages, system="", tools="", **kwargs):
        system = system or self.default_system
        encoded_pairs = []
        prefix_ids = self._encode_prefix(tokenizer)

        first_turn = True
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", Role.USER)
            content = msg.get("content", "")

            if role in (Role.USER, "user"):
                user_slots = self.format_user.apply(content=content)
                user_ids = self._tokenize_slots(tokenizer, user_slots)

                if first_turn:
                    # Fuse system into first user turn
                    system_ids = self._encode_system(tokenizer, system, tools)
                    user_ids = prefix_ids + system_ids + user_ids
                    first_turn = False

                response_ids = []
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    next_role = next_msg.get("role", "")
                    if next_role in (Role.ASSISTANT, "assistant"):
                        next_content = next_msg.get("content", "")
                        assistant_slots = self.format_assistant.apply(content=next_content)
                        response_ids = self._tokenize_slots(tokenizer, assistant_slots)
                        i += 1

                encoded_pairs.append((user_ids, response_ids))

            elif role in (Role.ASSISTANT, "assistant"):
                assistant_slots = self.format_assistant.apply(content=content)
                response_ids = self._tokenize_slots(tokenizer, assistant_slots)
                encoded_pairs.append(([], response_ids))

            i += 1

        return encoded_pairs


# ===================== Register Gemma3 Templates =====================

def _register_gemma3_templates():
    """Register Gemma3 chat templates."""
    # gemma3 template (for multimodal models 4B+)
    register_template(
        name="gemma3",
        format_user=StringFormatter(
            slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"]
        ),
        format_system=StringFormatter(
            slots=["{{content}}\n\n"]
        ),
        format_observation=StringFormatter(
            slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<end_of_turn>"],
        replace_eos=True,
        mm_plugin=get_mm_plugin("gemma3", image_token="<image_soft_token>"),
        template_class=Llama2Template,
    )

    # gemma2 template (for small text-only models)
    register_template(
        name="gemma2",
        format_user=StringFormatter(
            slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"]
        ),
        format_system=StringFormatter(
            slots=["{{content}}\n\n"]
        ),
        format_observation=StringFormatter(
            slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<end_of_turn>"],
        replace_eos=True,
        efficient_eos=True,
        mm_plugin=get_mm_plugin("base"),
        template_class=Llama2Template,
    )


_register_gemma3_templates()


def get_template_and_fix_tokenizer(tokenizer, data_args) -> Template:
    """Get template and fix tokenizer special tokens."""
    template_name = getattr(data_args, "template", "gemma3")
    template = get_template(template_name)

    # Fix EOS token if replace_eos is set
    if template.replace_eos and template.stop_words:
        stop_word = template.stop_words[0]
        stop_word_id = tokenizer.convert_tokens_to_ids(stop_word)
        if stop_word_id and stop_word_id != 0:
            tokenizer.eos_token = stop_word
            tokenizer.eos_token_id = stop_word_id

    return template
