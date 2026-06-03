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

from __future__ import annotations


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def format_codebase_output(files: list[dict[str, object]], summary: str) -> dict:
    """Select file excerpts to return from a codebase investigation.

    Use this as the final step of a read-only investigation. Provide the exact
    file slices that should be returned to the caller plus a very short summary.
    You may include multiple slices from the same file and multiple different
    files in one response, as long as they were actually inspected.
    Prefer small, targeted excerpts from files you already read so the final
    response stays token-efficient.

    Each item in `files` must include:
    - `path`: workspace-relative file path
    - `start_line`: 1-based inclusive start line
    - `end_line`: 1-based inclusive end line

    Example input:
        files=[{"path": "src/app.py", "start_line": 10, "end_line": 22}]
        summary="Loader initializes config before model setup."

    Example output:
        {"status": "success", "files": [{"path": "src/app.py", "start_line": 10, "end_line": 22}], "summary": "..."}
    """
    normalized_files: list[dict[str, object]] = []
    for item in files:
        path = _as_str(item.get("path"))
        if not path:
            raise ValueError("Each file selection must include a non-empty string 'path'.")

        start_line = max(_as_int(item.get("start_line"), 1), 1)
        end_line = max(_as_int(item.get("end_line"), start_line), start_line)
        normalized_files.append(
            {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    return {
        "status": "success",
        "files": normalized_files,
        "summary": _as_str(summary).strip(),
    }
