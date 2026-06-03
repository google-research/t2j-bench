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

import jedi

from .workspace import (
    RestrictedWorkspacePathError,
    get_workspace_root,
    resolve_workspace_path,
    to_workspace_relative,
)

_SUPPORTED_OPERATIONS = {
    "goToDefinition",
    "findReferences",
    "hover",
    "documentSymbol",
    "workspaceSymbol",
}
_MAX_FILE_SIZE_BYTES = 10_000_000
_MAX_RESULTS = 50


def _format_location(name: jedi.api.classes.Name) -> str:
    module_path = getattr(name, "module_path", None)
    location_bits: list[str] = []
    if module_path:
        location_bits.append(to_workspace_relative(Path(module_path)))
    line = getattr(name, "line", None)
    column = getattr(name, "column", None)
    if line is not None:
        if column is not None:
            location_bits.append(f"{line}:{column + 1}")
        else:
            location_bits.append(str(line))
    return ":".join(location_bits) if location_bits else "<unknown location>"


def _workspace_path_filter(name: jedi.api.classes.Name, workspace_root: Path) -> bool:
    module_path = getattr(name, "module_path", None)
    if not module_path:
        return False
    path = Path(module_path).resolve()
    return str(path).startswith(str(workspace_root)) and path.suffix == ".py"


def _make_project(file_or_dir: Path) -> jedi.Project:
    root = file_or_dir if file_or_dir.is_dir() else file_or_dir.parent
    workspace_root = get_workspace_root()
    return jedi.Project(
        path=str(root),
        added_sys_path=(str(workspace_root),),
    )


def _load_script(file_path: Path) -> tuple[jedi.Script, list[str]]:
    if file_path.suffix != ".py":
        raise ValueError("lsp_tool currently supports only Python files.")
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if file_path.stat().st_size > _MAX_FILE_SIZE_BYTES:
        raise ValueError("File too large for lsp_tool analysis.")
    source = file_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    script = jedi.Script(code=source, path=str(file_path), project=_make_project(file_path))
    return script, lines


def _validate_position(lines: list[str], line: int | None, character: int | None) -> tuple[int, int]:
    if line is None or character is None:
        raise ValueError("This operation requires line and character.")
    if line < 1 or line > len(lines):
        raise ValueError("line is out of range.")
    line_text = lines[line - 1]
    if character < 1 or character > max(len(line_text), 1):
        raise ValueError("character is out of range.")
    return line, character


def _format_definition_result(names: list[jedi.api.classes.Name]) -> dict:
    if not names:
        return {
            "status": "success",
            "result_count": 0,
            "result": "No definition found.",
        }
    lines = [f"Found {len(names)} definition(s):"]
    for name in names[:_MAX_RESULTS]:
        lines.append(f"- {name.name} ({name.type}) at {_format_location(name)}")
    if len(names) > _MAX_RESULTS:
        lines.append(f"... omitted {len(names) - _MAX_RESULTS} definitions")
    return {
        "status": "success",
        "result_count": len(names),
        "result": "\n".join(lines),
    }


def _format_reference_result(names: list[jedi.api.classes.Name]) -> dict:
    if not names:
        return {
            "status": "success",
            "result_count": 0,
            "result": "No references found.",
        }
    lines = [f"Found {len(names)} reference(s):"]
    for name in names[:_MAX_RESULTS]:
        lines.append(f"- {name.name} at {_format_location(name)}")
    if len(names) > _MAX_RESULTS:
        lines.append(f"... omitted {len(names) - _MAX_RESULTS} references")
    return {
        "status": "success",
        "result_count": len(names),
        "result": "\n".join(lines),
    }


def _format_hover_result(names: list[jedi.api.classes.Name]) -> dict:
    if not names:
        return {
            "status": "success",
            "result_count": 0,
            "result": "No hover information available.",
        }
    primary = names[0]
    lines = [primary.description]
    try:
        doc = primary.docstring(raw=True).strip()
    except Exception:
        doc = ""
    if doc:
        lines.append("")
        lines.append(doc)
    return {
        "status": "success",
        "result_count": 1,
        "result": "\n".join(lines).strip(),
    }


def _format_document_symbols(names: list[jedi.api.classes.Name]) -> dict:
    if not names:
        return {
            "status": "success",
            "result_count": 0,
            "result": "No symbols found.",
        }
    lines = [f"Found {len(names)} symbol(s):"]
    for name in names[:_MAX_RESULTS]:
        line = getattr(name, "line", None)
        column = getattr(name, "column", None)
        if line is not None and column is not None:
            location = f"{line}:{column + 1}"
        elif line is not None:
            location = str(line)
        else:
            location = "?"
        lines.append(f"- {name.name} ({name.type}) at {location}")
    if len(names) > _MAX_RESULTS:
        lines.append(f"... omitted {len(names) - _MAX_RESULTS} symbols")
    return {
        "status": "success",
        "result_count": len(names),
        "result": "\n".join(lines),
    }


def _format_workspace_symbols(names: list[jedi.api.classes.Name]) -> dict:
    if not names:
        return {
            "status": "success",
            "result_count": 0,
            "result": "No workspace symbols found.",
        }
    lines = [f"Found {len(names)} workspace symbol(s):"]
    for name in names[:_MAX_RESULTS]:
        lines.append(f"- {name.name} ({name.type}) at {_format_location(name)}")
    if len(names) > _MAX_RESULTS:
        lines.append(f"... omitted {len(names) - _MAX_RESULTS} symbols")
    return {
        "status": "success",
        "result_count": len(names),
        "result": "\n".join(lines),
    }


def lsp_tool(
    operation: str,
    file_path: str,
    *,
    line: int | None = None,
    character: int | None = None,
    query: str = "",
) -> dict:
    """Run a Python-only semantic code query using Jedi.

    This is a lightweight Python-only substitute for IDE/LSP navigation. Use it
    when text search is not enough and you need symbol-aware results such as
    definitions, references, hover info, document symbols, or workspace symbol
    search.

    Supported operations:
      - goToDefinition
      - findReferences
      - hover
      - documentSymbol
      - workspaceSymbol

    Example input for goToDefinition:
      operation="goToDefinition", file_path="pkg/app.py", line=12, character=8

    Example output for goToDefinition:
      {"status": "success", "operation": "goToDefinition", "result_count": 1, "result": "Found 1 definition(s):\\n- foo (function) at pkg/lib.py:4:1"}

    Example input for findReferences:
      operation="findReferences", file_path="pkg/app.py", line=12, character=8

    Example output for findReferences:
      {"status": "success", "operation": "findReferences", "result_count": 3, "result": "Found 3 reference(s):\\n- foo at pkg/app.py:12:8\\n- foo at pkg/lib.py:20:5"}

    Example input for hover:
      operation="hover", file_path="pkg/app.py", line=12, character=8

    Example output for hover:
      {"status": "success", "operation": "hover", "result_count": 1, "result": "function foo\\n\\nfoo(x: int) -> str"}

    Example input for documentSymbol:
      operation="documentSymbol", file_path="pkg/app.py"

    Example output for documentSymbol:
      {"status": "success", "operation": "documentSymbol", "result_count": 5, "result": "Found 5 symbol(s):\\n- Foo (class) at 1:1\\n- bar (function) at 4:5"}

    Example input for workspaceSymbol:
      operation="workspaceSymbol", file_path=".", query="Foo"

    Example output for workspaceSymbol:
      {"status": "success", "operation": "workspaceSymbol", "query": "Foo", "result_count": 2, "result": "Found 2 workspace symbol(s):\\n- Foo (class) at pkg/app.py:1:1\\n- FooBuilder (class) at pkg/build.py:3:1"}
    """
    try:
        if operation not in _SUPPORTED_OPERATIONS:
            return {
                "status": "error",
                "error_message": (
                    f"Unsupported operation: {operation}. "
                    f"Supported operations: {', '.join(sorted(_SUPPORTED_OPERATIONS))}"
                ),
            }

        workspace_root = get_workspace_root()

        if operation == "workspaceSymbol":
            root_path = resolve_workspace_path(file_path or ".")
            project = _make_project(root_path)
            search_query = query.strip()
            if not search_query:
                return {
                    "status": "error",
                    "error_message": "workspaceSymbol requires a non-empty query.",
                }
            names = [
                name
                for name in project.search(search_query, all_scopes=True)
                if _workspace_path_filter(name, workspace_root)
            ]
            formatted = _format_workspace_symbols(names)
            return {
                **formatted,
                "operation": operation,
                "file_path": to_workspace_relative(root_path),
                "query": search_query,
            }

        resolved_path = resolve_workspace_path(file_path)
        script, lines = _load_script(resolved_path)

        if operation in {"goToDefinition", "findReferences", "hover"}:
            valid_line, valid_character = _validate_position(lines, line, character)
            zero_based_column = valid_character - 1

            if operation == "goToDefinition":
                names = [
                    name
                    for name in script.infer(valid_line, zero_based_column)
                    if _workspace_path_filter(name, workspace_root)
                ]
                formatted = _format_definition_result(names)
            elif operation == "findReferences":
                names = [
                    name
                    for name in script.get_references(valid_line, zero_based_column)
                    if _workspace_path_filter(name, workspace_root)
                ]
                formatted = _format_reference_result(names)
            else:
                names = script.infer(valid_line, zero_based_column)
                formatted = _format_hover_result(names)

            return {
                **formatted,
                "operation": operation,
                "file_path": to_workspace_relative(resolved_path),
                "line": valid_line,
                "character": valid_character,
            }

        if operation == "documentSymbol":
            names = [
                name
                for name in script.get_names(all_scopes=True, definitions=True, references=False)
                if getattr(name, "line", None) is not None
            ]
            formatted = _format_document_symbols(names)
            return {
                **formatted,
                "operation": operation,
                "file_path": to_workspace_relative(resolved_path),
            }

        return {"status": "error", "error_message": f"Unhandled operation: {operation}"}
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}
