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

"""Tool utilities for Qwen format function calling."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Union


class FunctionCall(NamedTuple):
    name: str
    arguments: str


QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)

DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)


@dataclass
class ToolUtils(ABC):
    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str: ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: list["FunctionCall"]) -> str: ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]: ...


class DefaultToolUtils(ToolUtils):
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required = ", required" if name in tool["parameters"].get("required", []) else ""
                enum = ", should be one of [{}]".format(", ".join(param["enum"])) if param.get("enum") else ""
                items = ", where each item should be {}".format(param["items"].get("type", "")) if param.get("items") else ""
                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name, type=param.get("type", ""), required=required,
                    desc=param.get("description", ""), enum=enum, items=items,
                )
            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])
        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return "\n".join([f"Action: {name}\nAction Input: {arguments}" for name, arguments in functions])

    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match = re.findall(regex, content)
        if not action_match:
            return content
        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append(FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content
        return results


class QwenToolUtils(ToolUtils):
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)
        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = [
            json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False)
            for name, arguments in functions
        ]
        return "\n".join([f"<tool_call>\n{text}\n</tool_call>" for text in function_texts])

    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match = re.findall(regex, content)
        if not tool_match:
            return content
        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content
            if "name" not in tool or "arguments" not in tool:
                return content
            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))
        return results


TOOLS = {
    "default": DefaultToolUtils(),
    "qwen": QwenToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")
    return tool_utils

