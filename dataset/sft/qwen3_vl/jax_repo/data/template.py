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

"""Chat template system for Qwen3-VL SFT framework."""

import logging
import re
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Optional, Union

from .formatter import SLOTS, EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .mm_plugin import BasePlugin
    from .tool_utils import FunctionCall

logger = logging.getLogger(__name__)


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    thought_words: tuple[str, str]
    tool_call_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    enable_thinking: Optional[bool]
    mm_plugin: "BasePlugin"

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        """Return multiple pairs of token ids representing prompts and responses."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        """Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        """Encode formatted inputs to pairs of token ids."""
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message["content"], thought_words=self.thought_words, tool_call_words=self.tool_call_words
                )
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))
        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        if tokenizer.eos_token == eos_token:
            return
        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})
        if is_added:
            logger.info(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info(f"Replace eos token: {tokenizer.eos_token}.")
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")
            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            try:
                num_added_tokens = tokenizer.add_special_tokens(
                    dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
                )
            except TypeError:
                num_added_tokens = tokenizer.add_special_tokens(dict(additional_special_tokens=stop_words))
            logger.info("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


# ============================================================
# Template Registry
# ============================================================

TEMPLATES: dict[str, "Template"] = {}


def _get_mm_plugin(name: str, **kwargs) -> "BasePlugin":
    """Lazy import to avoid circular dependency."""
    from .mm_plugin import get_mm_plugin
    return get_mm_plugin(name, **kwargs)


def register_template(
    name: str,
    format_user=None,
    format_assistant=None,
    format_system=None,
    format_function=None,
    format_observation=None,
    format_tools=None,
    format_prefix=None,
    default_system: str = "",
    stop_words: list[str] | None = None,
    thought_words: tuple[str, str] | None = None,
    tool_call_words: tuple[str, str] | None = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    enable_thinking: Optional[bool] = True,
    mm_plugin: "BasePlugin | None" = None,
) -> None:
    """Register a chat template."""
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    if mm_plugin is None:
        mm_plugin = _get_mm_plugin(name="base")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    if format_assistant is not None:
        default_function_formatter = FunctionFormatter(slots=format_assistant.slots, tool_format="default")
    else:
        default_function_formatter = FunctionFormatter(slots=default_slots, tool_format="default")

    default_tool_formatter = ToolFormatter(tool_format="default")
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = Template(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        tool_call_words=tool_call_words or ("<tool_call>", "</tool_call>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        enable_thinking=enable_thinking,
        mm_plugin=mm_plugin,
    )


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args) -> "Template":
    """Get chat template and fix the tokenizer."""
    if data_args.template is None:
        raise ValueError("`template` must be specified.")

    if data_args.template not in TEMPLATES:
        raise ValueError(f"Template {data_args.template} does not exist.")

    template = TEMPLATES[data_args.template]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if data_args.tool_format is not None:
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)

    if data_args.default_system is not None:
        template.default_system = data_args.default_system

    template.fix_special_tokens(tokenizer)
    return template


# ============================================================
# Register Qwen3-VL templates
# ============================================================


def _register_qwen3_vl_templates():
    """Register Qwen3-VL templates (called at module load time)."""
    register_template(
        name="qwen3_vl_nothink",
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),
        format_tools=ToolFormatter(tool_format="qwen"),
        stop_words=["<|im_end|>"],
        replace_eos=True,
        mm_plugin=_get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    )


_register_qwen3_vl_templates()

