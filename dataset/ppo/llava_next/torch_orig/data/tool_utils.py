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

"""Tool utilities for function calling support."""
import json
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional, Sequence


class FunctionCall(NamedTuple):
    name: str
    arguments: str


class ToolUtils(ABC):
    @staticmethod
    @abstractmethod
    def function_formatter(functions: Sequence[FunctionCall]) -> str:
        ...

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> list[FunctionCall]:
        ...


class DefaultToolUtils(ToolUtils):
    @staticmethod
    def function_formatter(functions: Sequence[FunctionCall]) -> str:
        result = ""
        for function in functions:
            result += f"Action: {function.name}\nAction Input: {function.arguments}\n"
        return result

    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        return json.dumps(tools, ensure_ascii=False)

    @staticmethod
    def tool_extractor(content: str) -> list[FunctionCall]:
        functions = []
        lines = content.strip().split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith("Action:"):
                name = lines[i][len("Action:"):].strip()
                args = ""
                if i + 1 < len(lines) and lines[i + 1].startswith("Action Input:"):
                    args = lines[i + 1][len("Action Input:"):].strip()
                    i += 1
                functions.append(FunctionCall(name=name, arguments=args))
            i += 1
        return functions


_TOOL_UTILS = {
    "default": DefaultToolUtils,
}


def get_tool_utils(tool_format: Optional[str] = None) -> ToolUtils:
    return _TOOL_UTILS.get(tool_format or "default", DefaultToolUtils)()
