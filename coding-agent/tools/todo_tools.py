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

import json
from pathlib import Path

from .workspace import get_workspace_root
_VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
_DEFAULT_AGENT_TODO_SCOPE = "default"


def _get_todo_file(workspace_root: str | None = None):
    root = Path(workspace_root).expanduser().resolve() if workspace_root else get_workspace_root()
    return root / ".adk" / "coding_agent_todos.json"


def _normalize_agent_scope(agent_name: str | None) -> str:
    normalized = (agent_name or "").strip()
    return normalized or _DEFAULT_AGENT_TODO_SCOPE


def _load_todo_store(workspace_root: str | None = None) -> dict[str, list[dict]]:
    todo_file = _get_todo_file(workspace_root)
    if not todo_file.exists():
        return {}
    raw = json.loads(todo_file.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {_DEFAULT_AGENT_TODO_SCOPE: raw}
    if isinstance(raw, dict):
        normalized: dict[str, list[dict]] = {}
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, list):
                normalized[key] = value
        return normalized
    return {}


def _save_todo_store(store: dict[str, list[dict]], workspace_root: str | None = None) -> None:
    todo_file = _get_todo_file(workspace_root)
    todo_file.parent.mkdir(parents=True, exist_ok=True)
    todo_file.write_text(json.dumps(store, indent=2, ensure_ascii=True), encoding="utf-8")


def write_todos(
    todos: list[dict],
    agent_name: str | None = None,
    workspace_root: str | None = None,
) -> dict:
    """Replace the todo list for a complex multi-step task.

    Use this to maintain a visible execution plan when the work is large enough
    to benefit from explicit progress tracking. Keep at most one item in
    `in_progress` at a time.

    Example input:
        todos=[{"description": "Inspect loader", "status": "in_progress"}], agent_name="client_agent", workspace_root="/repo"

    Example output:
        {"status": "success", "agent_name": "client_agent", "todos": [{"description": "Inspect loader", "status": "in_progress"}], "todo_file": "/repo/.adk/coding_agent_todos.json"}
    """
    if not isinstance(todos, list):
        return {"status": "error", "error_message": "`todos` must be a list."}

    in_progress_count = 0
    normalized: list[dict] = []
    for item in todos:
        if not isinstance(item, dict):
            return {"status": "error", "error_message": "Each todo must be an object."}
        description = str(item.get("description", "")).strip()
        status = str(item.get("status", "")).strip()
        if not description:
            return {"status": "error", "error_message": "Each todo needs a non-empty description."}
        if status not in _VALID_STATUSES:
            return {
                "status": "error",
                "error_message": f"Invalid status: {status}. Valid statuses: {sorted(_VALID_STATUSES)}",
            }
        if status == "in_progress":
            in_progress_count += 1
        normalized.append({"description": description, "status": status})

    if in_progress_count > 1:
        return {
            "status": "error",
            "error_message": 'Only one todo can be marked as "in_progress".',
        }

    normalized_agent_name = _normalize_agent_scope(agent_name)
    store = _load_todo_store(workspace_root)
    store[normalized_agent_name] = normalized
    todo_file = _get_todo_file(workspace_root)
    _save_todo_store(store, workspace_root)
    return {
        "status": "success",
        "agent_name": normalized_agent_name,
        "todos": normalized,
        "message": "Todo list updated.",
        "todo_file": str(todo_file),
    }


def read_todos(agent_name: str | None = None, workspace_root: str | None = None) -> dict:
    """Return the current todo list.

    Example input:
        read_todos(agent_name="planner", workspace_root="/repo")

    Example output:
        {"status": "success", "agent_name": "planner", "todos": [{"description": "Inspect loader", "status": "completed"}], "todo_file": "/repo/.adk/coding_agent_todos.json"}
    """
    normalized_agent_name = _normalize_agent_scope(agent_name)
    store = _load_todo_store(workspace_root)
    todo_file = _get_todo_file(workspace_root)
    return {
        "status": "success",
        "agent_name": normalized_agent_name,
        "todos": store.get(normalized_agent_name, []),
        "todo_file": str(todo_file),
    }
