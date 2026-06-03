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

"""Chat formatters for building template slots."""
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from .tool_utils import FunctionCall, get_tool_utils


SLOT = Union[str, set[str], dict[str, str]]


@dataclass
class Formatter(ABC):
    slots: Sequence[SLOT] = field(default_factory=list)

    @abstractmethod
    def apply(self, **kwargs) -> list[SLOT]:
        ...

    def _compile(self, slot: SLOT, **kwargs) -> SLOT:
        if isinstance(slot, str):
            for k, v in kwargs.items():
                slot = slot.replace("{{" + k + "}}", str(v))
            return slot
        elif isinstance(slot, (dict, set)):
            return slot
        else:
            raise ValueError(f"Invalid slot type: {type(slot)}")


@dataclass
class EmptyFormatter(Formatter):
    """Returns slots directly without modification."""

    def apply(self, **kwargs) -> list[SLOT]:
        return list(self.slots)


@dataclass
class StringFormatter(Formatter):
    """Replaces {{content}} placeholders in slots."""

    def apply(self, **kwargs) -> list[SLOT]:
        content = kwargs.get("content", "")
        results = []
        for slot in self.slots:
            if isinstance(slot, str):
                results.append(slot.replace("{{content}}", str(content)))
            else:
                results.append(slot)
        return results


@dataclass
class FunctionFormatter(Formatter):
    """Formats function calls."""

    def apply(self, **kwargs) -> list[SLOT]:
        content = kwargs.get("content", "")
        functions: list[FunctionCall] = kwargs.get("functions", [])
        tool_utils = get_tool_utils(kwargs.get("tool_format", None))

        if functions:
            return [tool_utils.function_formatter(functions)]
        return [str(content)]


@dataclass
class ToolFormatter(Formatter):
    """Formats tool definitions."""

    def apply(self, **kwargs) -> list[SLOT]:
        tools = kwargs.get("content", "")
        tool_utils = get_tool_utils(kwargs.get("tool_format", None))

        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                return [str(tools)]

        if isinstance(tools, list):
            return [tool_utils.tool_formatter(tools)]
        return [str(tools)]
