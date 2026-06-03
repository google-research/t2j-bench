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


_ENV_VAR = "CODING_AGENT_WORKSPACE"
_YOLO_ENV_VAR = "CODING_AGENT_YOLO"
_RESTRICTED_ENV_VAR = "CODING_AGENT_RESTRICTED_MODE"
_DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1]


def get_workspace_root() -> Path:
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return _DEFAULT_WORKSPACE
    return Path(raw).expanduser().resolve()


def is_yolo_mode() -> bool:
    return os.environ.get(_YOLO_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def is_restricted_mode() -> bool:
    return os.environ.get(_RESTRICTED_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


class RestrictedWorkspacePathError(ValueError):
    def __init__(self, original_path: str, resolved_path: Path, workspace_root: Path) -> None:
        self.original_path = original_path
        self.resolved_path = resolved_path
        self.workspace_root = workspace_root
        message = (
            f"Restricted mode banned path outside workspace: {original_path} "
            f"(resolved to {resolved_path}, workspace={workspace_root})."
        )
        super().__init__(message)

    def to_warning_payload(self) -> dict:
        message = (
            f"Restricted mode warning: path `{self.original_path}` resolves outside the current "
            f"workspace and has been banned."
        )
        return {
            "status": "warning",
            "warning_message": message,
            "error_message": message,
            "path": self.original_path,
            "resolved_path": str(self.resolved_path),
            "workspace_root": str(self.workspace_root),
        }


def resolve_workspace_path(path: str) -> Path:
    workspace = get_workspace_root()
    raw_path = Path(path).expanduser()
    candidate = raw_path.resolve() if raw_path.is_absolute() else (workspace / raw_path).resolve()
    if is_restricted_mode() and not str(candidate).startswith(str(workspace)):
        raise RestrictedWorkspacePathError(path, candidate, workspace)
    return candidate


def to_workspace_relative(path: Path) -> str:
    resolved = path.resolve()
    workspace = get_workspace_root()
    try:
        return str(resolved.relative_to(workspace))
    except ValueError:
        return str(resolved)
