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

from __future__ import annotations

from collections import deque
from typing import Deque


class TerminalOutputBuffer:
    def __init__(self, *, max_lines: int | None = None) -> None:
        self._lines: Deque[tuple[int, str]] = deque(maxlen=max_lines)
        self._next_seq = 1
        self._current_line = ""
        self._current_line_seq: int | None = None
        self._overwrite_pending = False

    def _append_line(self, text: str) -> int:
        seq = self._next_seq
        self._next_seq += 1
        self._lines.append((seq, text))
        return seq

    def _publish_current_line(self) -> None:
        if self._current_line_seq is not None and self._lines and self._lines[-1][0] == self._current_line_seq:
            self._lines.pop()
        self._current_line_seq = self._append_line(self._current_line)

    def feed(self, text: str) -> None:
        current_dirty = False
        for char in text:
            if char == "\r":
                self._overwrite_pending = True
                continue

            if char == "\n":
                if current_dirty:
                    self._publish_current_line()
                    current_dirty = False
                elif self._current_line_seq is None:
                    self._append_line("")
                self._current_line = ""
                self._current_line_seq = None
                self._overwrite_pending = False
                continue

            if self._overwrite_pending:
                self._current_line = ""
                self._overwrite_pending = False
                current_dirty = True

            self._current_line += char
            current_dirty = True

        if current_dirty:
            self._publish_current_line()

    def items(self) -> list[tuple[int, str]]:
        return list(self._lines)

    def text(self) -> str:
        return "\n".join(line for _, line in self._lines)


def normalize_terminal_output_text(text: str) -> str:
    had_trailing_newline = bool(text) and text.endswith(("\n", "\r"))
    buffer = TerminalOutputBuffer()
    buffer.feed(text)
    normalized = buffer.text()
    if had_trailing_newline:
        return normalized + "\n"
    return normalized
