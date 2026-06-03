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

from pathlib import Path


_MEMORY_DIR = Path.home() / ".gemini"
_MEMORY_FILE = _MEMORY_DIR / "GEMINI.md"
_HEADER = "## Gemini Added Memories"


def _read_memory_text() -> str:
    if not _MEMORY_FILE.exists():
        return ""
    return _MEMORY_FILE.read_text(encoding="utf-8")


def _sanitize_fact(fact: str) -> str:
    cleaned = fact.replace("\r", " ").replace("\n", " ").strip()
    while cleaned.startswith("-"):
        cleaned = cleaned[1:].strip()
    return cleaned


def save_memory(fact: str) -> dict:
    """Save a concise cross-session memory for future runs.

    Store durable user preferences or facts that should persist across
    workspaces. Do not store repository-local paths, one-off commands, or
    transient task state.

    Example input:
        fact="Prefer concise final answers."

    Example output:
        {"status": "success", "saved_fact": "Prefer concise final answers."}
    """
    sanitized = _sanitize_fact(fact)
    if not sanitized:
        return {"status": "error", "error_message": "Memory fact cannot be empty."}

    current = _read_memory_text()
    entry = f"- {sanitized}"
    if _HEADER not in current:
        new_content = f"{current.rstrip()}\n\n{_HEADER}\n{entry}\n".lstrip("\n")
    else:
        before, after = current.split(_HEADER, maxsplit=1)
        section = after.strip("\n")
        new_content = f"{before.rstrip()}\n\n{_HEADER}\n"
        if section:
            new_content += f"{section}\n"
        new_content += f"{entry}\n"

    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_FILE.write_text(new_content, encoding="utf-8")
    return {
        "status": "success",
        "memory_file": str(_MEMORY_FILE),
        "saved_fact": sanitized,
    }


def read_memory() -> dict:
    """Read the current cross-session memory file.

    Example input:
        read_memory()

    Example output:
        {"status": "success", "memory_file": ".../GEMINI.md", "content": "..."}
    """
    return {
        "status": "success",
        "memory_file": str(_MEMORY_FILE),
        "content": _read_memory_text(),
    }
