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

import os
from pathlib import Path
import re
import shlex

_SUDO_PATTERN = re.compile(r"(^|[\s;&|()])sudo(?=$|[\s;&|()])")
_PATH_OPERATOR_TOKENS = {">", ">>", "<", "<<", "1>", "1>>", "2>", "2>>"}


def find_forbidden_shell_token(command: str) -> str | None:
    if _SUDO_PATTERN.search(command):
        return "sudo"
    return None


def _looks_like_path_token(token: str) -> bool:
    if not token or token in _PATH_OPERATOR_TOKENS:
        return False
    if "://" in token:
        return False
    if token.startswith("-"):
        return False
    if any(ch in token for ch in "$*?[]{}()"):
        return False
    return token.startswith(("/", "./", "../", "~")) or "/" in token


def _extract_candidate_path_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return []

    candidates: list[str] = []
    for token in tokens:
        if "=" in token and not token.startswith(("/", "./", "../", "~")):
            _, rhs = token.split("=", maxsplit=1)
            if _looks_like_path_token(rhs):
                candidates.append(rhs)
            continue
        if _looks_like_path_token(token):
            candidates.append(token)
    return candidates


def find_restricted_workspace_command_path(command: str, workspace_root: Path, *, restricted_mode: bool) -> str | None:
    if not restricted_mode:
        return None
    for token in _extract_candidate_path_tokens(command):
        raw_path = Path(os.path.expanduser(token))
        resolved = raw_path.resolve() if raw_path.is_absolute() else (workspace_root / raw_path).resolve()
        if not str(resolved).startswith(str(workspace_root)):
            return token
    return None


def restricted_workspace_command_warning(command: str, offending_path: str, workspace_root: Path) -> dict:
    message = (
        f"Restricted mode warning: command references path `{offending_path}` outside the current "
        f"workspace and has been banned."
    )
    return {
        "status": "warning",
        "warning_message": message,
        "error_message": message,
        "output": message,
        "command": command,
        "working_directory": str(workspace_root),
    }
