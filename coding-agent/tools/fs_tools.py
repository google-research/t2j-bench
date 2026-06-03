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

import asyncio
import difflib
import fnmatch
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .workspace import (
    RestrictedWorkspacePathError,
    get_workspace_root,
    resolve_workspace_path,
    to_workspace_relative,
)

_OUTPUT_LIMIT = 12000
_SEARCH_MAX_FILE_BYTES = 1_000_000
_SEARCH_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "dist",
    "build",
}
_SEARCH_IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".o",
    ".a",
    ".class",
    ".jar",
    ".zip",
    ".tar",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".mp3",
    ".wav",
    ".mp4",
}

_FILE_READ_STATE_KEY = "_coding_agent_file_reads"


def _truncate(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    omitted = len(text) - _OUTPUT_LIMIT
    return f"{text[:_OUTPUT_LIMIT]}\n...<truncated {omitted} chars>"


def _get_file_read_state(tool_context: Any | None) -> dict[str, dict[str, Any]]:
    if tool_context is None:
        return {}
    state = getattr(tool_context, "state", None)
    if state is None:
        return {}
    current = state.get(_FILE_READ_STATE_KEY)
    if isinstance(current, dict):
        return current
    current = {}
    state[_FILE_READ_STATE_KEY] = current
    return current


def _record_file_read(
    tool_context: Any | None,
    *,
    file_path: Path,
    fully_read: bool,
) -> None:
    if tool_context is None:
        return
    read_state = _get_file_read_state(tool_context)
    try:
        mtime = int(file_path.stat().st_mtime_ns)
    except OSError:
        return
    content_hash = None
    if fully_read:
        try:
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError:
            content_hash = None
    read_state[to_workspace_relative(file_path)] = {
        "timestamp_ns": mtime,
        "is_partial_view": not fully_read,
        "content_hash": content_hash,
    }


def _validate_prior_read(
    tool_context: Any | None,
    *,
    file_path: Path,
    require_full_read: bool,
) -> dict[str, Any] | None:
    if tool_context is None:
        return None
    relative_path = to_workspace_relative(file_path)
    read_state = _get_file_read_state(tool_context)
    info = read_state.get(relative_path)
    if not isinstance(info, dict):
        return {
            "status": "error",
            "path": relative_path,
            "error_message": "File has not been read yet. Read it first before modifying it.",
        }
    if require_full_read and info.get("is_partial_view"):
        return {
            "status": "error",
            "path": relative_path,
            "error_message": (
                "File was only partially read. Read the full file before overwriting an existing file."
            ),
        }
    try:
        current_mtime = int(file_path.stat().st_mtime_ns)
    except OSError as exc:
        return {
            "status": "error",
            "path": relative_path,
            "error_message": str(exc),
        }
    read_mtime = info.get("timestamp_ns")
    if isinstance(read_mtime, int) and current_mtime > read_mtime:
        return {
            "status": "error",
            "path": relative_path,
            "error_message": (
                "File has been modified since it was read. Read it again before editing or overwriting it."
            ),
        }
    recorded_hash = info.get("content_hash")
    if isinstance(recorded_hash, str) and recorded_hash:
        try:
            current_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError as exc:
            return {
                "status": "error",
                "path": relative_path,
                "error_message": str(exc),
            }
        if current_hash != recorded_hash:
            return {
                "status": "error",
                "path": relative_path,
                "error_message": (
                    "File has been modified since it was read. Read it again before editing or overwriting it."
                ),
            }
    return None

def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _as_bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _as_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _parse_list_files_request(request: dict[str, object]) -> tuple[str, bool, int, str | None]:
    return (
        _as_str(request.get("path"), "."),
        _as_bool(request.get("recursive"), True),
        _as_int(request.get("max_entries"), 200),
        _as_str(request.get("exclude"), "") or None,
    )


def _parse_read_file_request(request: dict[str, object]) -> tuple[str, int, int]:
    path = request.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("Each read_files request must include a non-empty string 'path'.")
    return (
        path,
        _as_int(request.get("start_line"), 1),
        _as_int(request.get("end_line"), 200),
    )


def _parse_search_text_request(request: dict[str, object]) -> tuple[str, str, int]:
    pattern = request.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("Each search_text_parallel request must include a non-empty string 'pattern'.")
    return (
        pattern,
        _as_str(request.get("path"), "."),
        _as_int(request.get("max_matches"), 100),
    )


def _normalize_keywords(keywords: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        cleaned = keyword.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


def _is_probably_binary(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return True
    return b"\x00" in sample


def _iter_search_candidates(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    candidates: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in _SEARCH_IGNORED_DIRS and not name.startswith(".")
        ]
        current_path = Path(current_root)
        for filename in filenames:
            path = current_path / filename
            if filename.startswith("."):
                continue
            if path.suffix.lower() in _SEARCH_IGNORED_SUFFIXES:
                continue
            candidates.append(path)
    return candidates


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _scan_keyword_file(
    file_path: Path,
    root: Path,
    *,
    keywords: list[str],
    match_mode: str,
    case_sensitive: bool,
    max_snippets_per_file: int,
    context_lines: int,
) -> dict | None:
    try:
        if file_path.stat().st_size > _SEARCH_MAX_FILE_BYTES or _is_probably_binary(file_path):
            return None
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    haystack = text if case_sensitive else text.lower()
    relative_path = to_workspace_relative(file_path)
    path_haystack = relative_path if case_sensitive else relative_path.lower()
    prepared_keywords = keywords if case_sensitive else [keyword.lower() for keyword in keywords]
    lines = text.splitlines()

    matched_keywords: list[str] = []
    keyword_occurrences: dict[str, int] = {}
    path_keyword_hits: list[str] = []
    line_match_ranges: list[tuple[int, int]] = []
    snippets: list[dict] = []
    best_window_coverage = 0

    for original_keyword, prepared_keyword in zip(keywords, prepared_keywords):
        occurrences = haystack.count(prepared_keyword)
        path_hits = path_haystack.count(prepared_keyword)
        total_hits = occurrences + path_hits
        if total_hits > 0:
            matched_keywords.append(original_keyword)
            keyword_occurrences[original_keyword] = total_hits
        if path_hits > 0:
            path_keyword_hits.append(original_keyword)

    if match_mode == "all" and len(matched_keywords) < len(keywords):
        return None
    if not matched_keywords:
        return None

    for index, line in enumerate(lines, start=1):
        line_haystack = line if case_sensitive else line.lower()
        line_keywords = [
            keyword
            for keyword, prepared_keyword in zip(keywords, prepared_keywords)
            if prepared_keyword in line_haystack
        ]
        if not line_keywords:
            continue
        best_window_coverage = max(best_window_coverage, len(set(line_keywords)))
        line_match_ranges.append(
            (max(1, index - context_lines), min(len(lines), index + context_lines))
        )

    for start_line, end_line in _merge_ranges(line_match_ranges)[:max_snippets_per_file]:
        snippet_lines: list[str] = []
        snippet_keywords: set[str] = set()
        for line_index in range(start_line, end_line + 1):
            line_text = lines[line_index - 1]
            line_haystack = line_text if case_sensitive else line_text.lower()
            for keyword, prepared_keyword in zip(keywords, prepared_keywords):
                if prepared_keyword in line_haystack:
                    snippet_keywords.add(keyword)
            snippet_lines.append(f"{line_index}: {line_text}")
        snippets.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "matched_keywords": sorted(snippet_keywords),
                "content": "\n".join(snippet_lines),
            }
        )

    coverage = len(matched_keywords)
    total_occurrences = sum(keyword_occurrences.values())
    score = (
        coverage * 100
        + total_occurrences * 4
        + len(path_keyword_hits) * 20
        + best_window_coverage * 15
    )
    return {
        "path": relative_path,
        "score": score,
        "matched_keywords": matched_keywords,
        "keyword_coverage": f"{coverage}/{len(keywords)}",
        "keyword_occurrences": keyword_occurrences,
        "path_keyword_hits": path_keyword_hits,
        "snippets": snippets,
    }


def keyword_search(
    keywords: list[str],
    target_folder: str = ".",
    *,
    match_mode: str = "any",
    case_sensitive: bool = False,
    max_results: int = 20,
    max_snippets_per_file: int = 3,
    context_lines: int = 1,
) -> dict:
    """Run a ranked multi-keyword search inside a target folder.

    Use this to narrow a large codebase quickly when you have several related
    keywords and want the most relevant files first. The search skips common
    cache/build directories, ignores likely binary files, scans files in
    parallel, and ranks results by keyword coverage, hit density, path matches,
    and snippet proximity.

    Example input:
        keywords=["loader", "tokenizer", "config"]
        target_folder="src"

    Example output:
        {"status": "success", "target_folder": "src", "results": [{"path": "src/loader.py", "score": 220, "snippets": [...]}]}
    """
    try:
        normalized_keywords = _normalize_keywords(keywords)
        if not normalized_keywords:
            return {"status": "error", "error_message": "keywords must contain at least one non-empty string."}
        if match_mode not in {"any", "all"}:
            return {"status": "error", "error_message": "match_mode must be either 'any' or 'all'."}

        root = resolve_workspace_path(target_folder)
        if not root.exists():
            return {"status": "error", "error_message": f"Path does not exist: {target_folder}"}

        candidates = _iter_search_candidates(root)
        if not candidates:
            return {
                "status": "success",
                "target_folder": to_workspace_relative(root),
                "keywords": normalized_keywords,
                "match_mode": match_mode,
                "results": [],
                "searched_files": 0,
            }

        max_workers = min(32, max(4, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scanned = list(
                executor.map(
                    lambda path: _scan_keyword_file(
                        path,
                        root,
                        keywords=normalized_keywords,
                        match_mode=match_mode,
                        case_sensitive=case_sensitive,
                        max_snippets_per_file=max_snippets_per_file,
                        context_lines=context_lines,
                    ),
                    candidates,
                )
            )

        results = [item for item in scanned if item is not None]
        results.sort(key=lambda item: (-item["score"], item["path"]))
        truncated = len(results) > max_results
        return {
            "status": "success",
            "target_folder": to_workspace_relative(root),
            "keywords": normalized_keywords,
            "match_mode": match_mode,
            "case_sensitive": case_sensitive,
            "searched_files": len(candidates),
            "returned_results": min(len(results), max_results),
            "truncated": truncated,
            "results": results[:max_results],
        }
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


def _list_files_impl(
    path: str = ".",
    recursive: bool = True,
    max_entries: int = 200,
    exclude: str | None = None,
) -> dict:
    """List files or directories inside the workspace.

    Use this for lightweight file-system discovery before reaching for shell
    commands. Paths are resolved relative to the workspace root. The result is
    bounded by `max_entries` and reports whether truncation occurred.
    """
    try:
        workspace = get_workspace_root()
        root = resolve_workspace_path(path)
        if not root.exists():
            return {"status": "error", "error_message": f"Path does not exist: {path}"}

        exclude_regex = None
        if exclude:
            try:
                exclude_regex = re.compile(exclude)
            except re.error as exc:
                return {"status": "error", "error_message": f"Invalid exclude regex: {exc}"}

        if root.is_file():
            entries = [to_workspace_relative(root)]
        elif recursive:
            entries = sorted(
                to_workspace_relative(item)
                for item in root.rglob("*")
            )
        else:
            entries = sorted(
                to_workspace_relative(item)
                for item in root.iterdir()
            )

        if exclude_regex is not None:
            entries = [entry for entry in entries if not exclude_regex.search(entry)]

        truncated = False
        if len(entries) > max_entries:
            entries = entries[:max_entries]
            truncated = True

        return {
            "status": "success",
            "path": to_workspace_relative(root),
            "entries": entries,
            "truncated": truncated,
            "exclude": exclude or "",
            "working_directory": str(workspace),
        }
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


def _read_file_impl(path: str, start_line: int = 1, end_line: int = 200) -> dict:
    """Read a text file from the workspace, optionally by line range.

    Use this when you need exact file contents for analysis or editing. The
    returned content is line-numbered and preserves the full selected range.
    """
    try:
        file_path = resolve_workspace_path(path)
        if not file_path.exists():
            return {"status": "error", "error_message": f"File does not exist: {path}"}
        if not file_path.is_file():
            return {"status": "error", "error_message": f"Path is not a file: {path}"}

        lines = file_path.read_text(encoding="utf-8").splitlines()
        start = max(start_line, 1)
        end = max(end_line, start)
        selected = lines[start - 1:end]
        numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))

        return {
            "status": "success",
            "path": to_workspace_relative(file_path),
            "start_line": start,
            "end_line": min(end, len(lines)),
            "content": numbered,
        }
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


def write_file(path: str, content: str, tool_context=None) -> dict:
    """Write a file inside the workspace.

    Existing files are overwritten. Missing parent directories are created
    automatically. If the file already exists, read it first before writing.

    Example input:
        path="notes/output.txt", content="done\n"

    Example output:
        {"status": "success", "path": "notes/output.txt", "bytes_written": 5}
    """
    try:
        file_path = resolve_workspace_path(path)
        existed = file_path.exists()
        original_content = ""
        if file_path.exists():
            prior_read_error = _validate_prior_read(
                tool_context,
                file_path=file_path,
                require_full_read=True,
            )
            if prior_read_error is not None:
                return prior_read_error
            original_content = file_path.read_text(encoding="utf-8")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        relative_path = to_workspace_relative(file_path)
        result = {
            "status": "success",
            "type": "update" if existed else "create",
            "path": relative_path,
            "filePath": relative_path,
            "bytes_written": len(content.encode("utf-8")),
            "line_count": _line_count(content),
            "original_line_count": _line_count(original_content),
            "content_chars": len(content),
            "content_preview": content[:2000],
            "original_content_preview": original_content[:2000] if existed else None,
            "patch": _build_patch(relative_path, original_content, content),
        }
        _record_file_read(tool_context, file_path=file_path, fully_read=True)
        return result
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


def _restore_trailing_newline(original: str, updated: str) -> str:
    if original.endswith("\n") and not updated.endswith("\n"):
        return updated + "\n"
    if not original.endswith("\n") and updated.endswith("\n"):
        return updated[:-1]
    return updated


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _build_patch(path: str, original: str, updated: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=path,
            tofile=path,
            lineterm="",
        )
    )


def _calculate_exact_replacement(original: str, old_text: str, new_text: str) -> tuple[str, int] | None:
    occurrences = original.count(old_text)
    if occurrences == 0:
        return None
    return _restore_trailing_newline(original, original.replace(old_text, new_text)), occurrences


def _calculate_flexible_replacement(original: str, old_text: str, new_text: str) -> tuple[str, int] | None:
    source_lines = original.splitlines(keepends=True)
    search_lines = old_text.split("\n")
    replace_lines = new_text.split("\n")
    stripped_search = [line.strip() for line in search_lines]
    if not stripped_search:
        return None

    occurrences = 0
    i = 0
    while i <= len(source_lines) - len(stripped_search):
        window = source_lines[i:i + len(stripped_search)]
        window_stripped = [line.rstrip("\r\n").strip() for line in window]
        if window_stripped == stripped_search:
            occurrences += 1
            indent_match = re.match(r"^([ \t]*)", window[0])
            indent = indent_match.group(1) if indent_match else ""
            replacement = [f"{indent}{line}" for line in replace_lines]
            if window[-1].endswith("\r\n"):
                replacement_text = "\r\n".join(replacement)
            else:
                replacement_text = "\n".join(replacement)
            if window[-1].endswith("\n"):
                replacement_text += "\n"
            source_lines[i:i + len(stripped_search)] = [replacement_text]
            i += 1
        else:
            i += 1

    if occurrences == 0:
        return None
    updated = "".join(source_lines)
    return _restore_trailing_newline(original, updated), occurrences


def _calculate_regex_replacement(original: str, old_text: str, new_text: str) -> tuple[str, int] | None:
    normalized_old = old_text.replace("\r\n", "\n")
    normalized_new = new_text.replace("\r\n", "\n")
    tokens = re.findall(r"\S+", normalized_old)
    if not tokens:
        return None

    pattern = r"\s*".join(re.escape(token) for token in tokens)
    regex = re.compile(rf"^([ \t]*){pattern}", re.MULTILINE)
    match = regex.search(original)
    if not match:
        return None

    indent = match.group(1) or ""
    replacement = "\n".join(f"{indent}{line}" for line in normalized_new.split("\n"))
    updated = regex.sub(replacement, original, count=1)
    return _restore_trailing_newline(original, updated), 1


def _find_flexible_old_text_candidates(original: str, old_text: str) -> list[str]:
    source_lines = original.splitlines(keepends=True)
    search_lines = old_text.split("\n")
    stripped_search = [line.strip() for line in search_lines]
    if not stripped_search:
        return []

    candidates: list[str] = []
    for i in range(len(source_lines) - len(stripped_search) + 1):
        window = source_lines[i:i + len(stripped_search)]
        window_stripped = [line.rstrip("\r\n").strip() for line in window]
        if window_stripped == stripped_search:
            candidates.append("".join(window))
    return candidates


def _find_regex_old_text_candidate(original: str, old_text: str) -> str | None:
    normalized_old = old_text.replace("\r\n", "\n")
    tokens = re.findall(r"\S+", normalized_old)
    if not tokens:
        return None

    pattern = r"\s*".join(re.escape(token) for token in tokens)
    regex = re.compile(rf"^([ \t]*){pattern}", re.MULTILINE)
    match = regex.search(original)
    if not match:
        return None
    return match.group(0)


def _attempt_patch_correction(
    original: str,
    old_text: str,
    new_text: str,
    expected_replacements: int,
) -> tuple[str, int, str, str] | None:
    flexible_candidates = _find_flexible_old_text_candidates(original, old_text)
    if len(flexible_candidates) == expected_replacements:
        corrected_old_text = flexible_candidates[0]
        corrected_count = original.count(corrected_old_text)
        if corrected_count == expected_replacements:
            updated = _restore_trailing_newline(
                original,
                original.replace(corrected_old_text, new_text),
            )
            return updated, corrected_count, "corrected_flexible", corrected_old_text

    corrected_old_text = _find_regex_old_text_candidate(original, old_text)
    if corrected_old_text is not None:
        corrected_count = original.count(corrected_old_text)
        if corrected_count == expected_replacements:
            updated = _restore_trailing_newline(
                original,
                original.replace(corrected_old_text, new_text),
            )
            return updated, corrected_count, "corrected_regex", corrected_old_text

    return None


def patch_file(
    path: str,
    old_text: str,
    new_text: str,
    *,
    expected_replacements: int = 1,
    tool_context=None,
) -> dict:
    """Replace text inside a workspace file using a search-and-replace edit.

    Use this for targeted edits when you want to update only part of a file
    rather than rewriting the whole file. Prefer giving `old_text` with enough
    surrounding context to uniquely identify the target location. By default
    exactly one replacement is expected. Set `expected_replacements` when you
    intentionally want to replace multiple occurrences.

    Example input:
        path="src/app.py", old_text="return old_value", new_text="return new_value"

    Example output:
        {"status": "success", "path": "src/app.py", "match_strategy": "exact", "match_count": 1}
    """
    try:
        if not path:
            return {"status": "error", "error_message": "Path must be a non-empty string."}
        if not isinstance(old_text, str):
            return {"status": "error", "error_message": "old_text must be a string."}
        if not isinstance(new_text, str):
            return {"status": "error", "error_message": "new_text must be a string."}
        if not isinstance(expected_replacements, int) or expected_replacements < 1:
            return {"status": "error", "error_message": "expected_replacements must be an integer >= 1."}

        file_path = resolve_workspace_path(path)
        if old_text == "":
            if file_path.exists():
                return {
                    "status": "error",
                    "error_message": "File already exists. old_text must be non-empty when editing an existing file.",
                    "path": to_workspace_relative(file_path),
                }
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_text, encoding="utf-8")
            relative_path = to_workspace_relative(file_path)
            return {
                "status": "success",
                "type": "create",
                "path": relative_path,
                "filePath": relative_path,
                "match_strategy": "create",
                "match_count": 1,
                "bytes_written": len(new_text.encode("utf-8")),
                "line_count": _line_count(new_text),
                "original_line_count": 0,
                "content_chars": len(new_text),
                "replaceAll": False,
                "oldString": "",
                "newString": new_text,
                "content_preview": new_text[:2000],
                "original_content_preview": None,
                "patch": _build_patch(relative_path, "", new_text),
            }
        if not file_path.exists():
            return {
                "status": "error",
                "error_message": f"File not found: {path}. Use empty old_text only when creating a new file.",
            }
        if not file_path.is_file():
            return {"status": "error", "error_message": f"Path is not a file: {path}"}

        prior_read_error = _validate_prior_read(
            tool_context,
            file_path=file_path,
            require_full_read=False,
        )
        if prior_read_error is not None:
            return prior_read_error

        original = file_path.read_text(encoding="utf-8")
        correction_used = False
        corrected_old_text: str | None = None
        replacement = _calculate_exact_replacement(original, old_text, new_text)
        strategy_name = "exact" if replacement is not None else ""

        if replacement is None:
            corrected = _attempt_patch_correction(
                original,
                old_text,
                new_text,
                expected_replacements,
            )
            if corrected is not None:
                updated, match_count, strategy_name, corrected_old_text = corrected
                replacement = (updated, match_count)
                correction_used = True

        if replacement is None:
            for current_strategy_name, current_strategy in [
                ("flexible", _calculate_flexible_replacement),
                ("regex", _calculate_regex_replacement),
            ]:
                result = current_strategy(original, old_text, new_text)
                if result is not None:
                    replacement = result
                    strategy_name = current_strategy_name
                    break

        if replacement is None:
            return {
                "status": "error",
                "error_message": (
                    "Failed to edit, could not find the string to replace. "
                    "Re-read the file and provide a more exact old_text snippet with surrounding context."
                ),
                "path": to_workspace_relative(file_path),
            }

        updated, match_count = replacement
        if match_count != expected_replacements:
            return {
                "status": "error",
                "error_message": (
                    f"Failed to edit, expected {expected_replacements} replacement(s) "
                    f"but found {match_count}."
                ),
                "path": to_workspace_relative(file_path),
                "match_count": match_count,
            }
        if updated == original:
            return {
                "status": "error",
                "error_message": "No changes to apply. old_text and new_text produce identical file content.",
                "path": to_workspace_relative(file_path),
                "match_count": match_count,
            }
        file_path.write_text(updated, encoding="utf-8")
        relative_path = to_workspace_relative(file_path)
        result = {
            "status": "success",
            "type": "update",
            "path": relative_path,
            "filePath": relative_path,
            "match_strategy": strategy_name,
            "match_count": match_count,
            "correction_used": correction_used,
            "corrected_old_text": corrected_old_text,
            "bytes_written": len(updated.encode("utf-8")),
            "line_count": _line_count(updated),
            "original_line_count": _line_count(original),
            "content_chars": len(updated),
            "replaceAll": expected_replacements > 1,
            "oldString": old_text,
            "newString": new_text,
            "content_preview": updated[:2000],
            "original_content_preview": original[:2000],
            "patch": _build_patch(relative_path, original, updated),
        }
        _record_file_read(tool_context, file_path=file_path, fully_read=True)
        return result
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


def _search_text_impl(pattern: str, path: str = ".", max_matches: int = 100) -> dict:
    return search_text(pattern=pattern, path=path, max_matches=max_matches)


def list_files(
    path: str = ".",
    recursive: bool = True,
    max_entries: int = 200,
    exclude: str | None = None,
) -> dict:
    """List files or directories for a single path request.

    Use `exclude` to filter returned paths with a regular expression after the
    directory walk. This is useful for hiding cache folders, compiled files, or
    generated artifacts from the result set.

    Example input:
        path="src", exclude="__pycache__|\\.pyc$"

    Example output:
        {"status": "success", "path": "src", "exclude": "__pycache__|\\.pyc$", "entries": ["src/app.py", "src/utils.py"]}
    """
    return _list_files_impl(path=path, recursive=recursive, max_entries=max_entries, exclude=exclude)


def read_file(path: str, start_line: int = 1, end_line: int = 500, tool_context=None) -> dict:
    """Read one file or file slice for a single request.

    Example input:
        path="src/app.py", start_line=1, end_line=40

    Example output:
        {"status": "success", "path": "src/app.py", "start_line": 1, "end_line": 40, "content": "1: import ..."}
    """
    result = _read_file_impl(path=path, start_line=start_line, end_line=end_line)
    if result.get("status") == "success":
        try:
            file_path = resolve_workspace_path(path)
            returned_end = result.get("end_line")
            fully_read = start_line <= 1 and isinstance(returned_end, int) and end_line >= returned_end
            _record_file_read(tool_context, file_path=file_path, fully_read=fully_read)
        except Exception:
            pass
    return result


def search_text(
    pattern: str | None = None,
    path: str = ".",
    max_matches: int = 100,
    *,
    keywords: list[str] | None = None,
    target_folder: str | None = None,
    match_mode: str = "any",
    case_sensitive: bool = False,
    include_glob: str | None = None,
    context_lines: int = 0,
    use_regex: bool = False,
    regex_pattern: str | None = None,
    regex_flags: str = "",
) -> dict:
    """Search text within a target folder with grep-like controls.

    This is the default tool for searching code instead of shell patterns like
    `cat ... | grep ...`. It supports either a single `pattern` or multiple
    `keywords`, or a regular expression via `use_regex=True` and
    `regex_pattern=...`, optional folder scoping via `target_folder`
    relative to `path`,
    line-context snippets, and file filtering with `include_glob`. Each match
    includes the matched line plus the surrounding file-content slice
    requested by `context_lines`.

    Example input:
        keywords=["load_model", "AutoTokenizer"]
        target_folder="src"
        context_lines=1

    Example input (regex):
        use_regex=True
        regex_pattern="remat|checkpoint"
        target_folder="src"
        context_lines=2

    Example output:
        {"status": "success", "path": "src", "matched_files": 2, "matches": [{"path": "src/loader.py", "line": 18, "matched_keywords": ["load_model"], "matched_texts": [], "file_content": "17: ...\n18: load_model(...)\n19: ..."}]}
    """
    try:
        if use_regex:
            compiled_flags = 0
            normalized_flags = regex_flags.lower()
            flag_map = {
                "i": re.IGNORECASE,
                "m": re.MULTILINE,
                "s": re.DOTALL,
                "x": re.VERBOSE,
            }
            for flag_char in normalized_flags:
                if flag_char not in flag_map:
                    return {"status": "error", "error_message": f"Unsupported regex flag: {flag_char}"}
                compiled_flags |= flag_map[flag_char]
            if not case_sensitive and "i" not in normalized_flags:
                compiled_flags |= re.IGNORECASE

            active_regex_pattern = regex_pattern if isinstance(regex_pattern, str) and regex_pattern.strip() else pattern
            if not isinstance(active_regex_pattern, str) or not active_regex_pattern.strip():
                return {"status": "error", "error_message": "Provide a non-empty regex_pattern or pattern when use_regex=True."}
            try:
                compiled_regex = re.compile(active_regex_pattern, compiled_flags)
            except re.error as exc:
                return {"status": "error", "error_message": f"Invalid regex pattern: {exc}"}
            keywords_to_search: list[str] = []
        else:
            if keywords is None:
                if not isinstance(pattern, str) or not pattern.strip():
                    return {"status": "error", "error_message": "Provide a non-empty pattern or keywords."}
                keywords_to_search = [pattern.strip()]
            else:
                keywords_to_search = _normalize_keywords(keywords)
                if not keywords_to_search:
                    return {"status": "error", "error_message": "keywords must contain at least one non-empty string."}

        if not use_regex and match_mode not in {"any", "all"}:
            return {"status": "error", "error_message": "match_mode must be either 'any' or 'all'."}

        base_root = resolve_workspace_path(path)
        if target_folder is None:
            root = base_root
        else:
            target_root = Path(target_folder).expanduser()
            if target_root.is_absolute():
                root = resolve_workspace_path(str(target_root))
            else:
                root = resolve_workspace_path(str(base_root / target_root))
        if not root.exists():
            searched_path = target_folder if target_folder is not None else path
            return {"status": "error", "error_message": f"Path does not exist: {searched_path}"}

        prepared_keywords = keywords_to_search if case_sensitive else [keyword.lower() for keyword in keywords_to_search]
        candidates = _iter_search_candidates(root)
        matches: list[dict] = []
        matched_files = 0

        for file_path in candidates:
            if include_glob and not fnmatch.fnmatch(file_path.name, include_glob) and not fnmatch.fnmatch(
                to_workspace_relative(file_path), include_glob
            ):
                continue
            try:
                if file_path.stat().st_size > _SEARCH_MAX_FILE_BYTES or _is_probably_binary(file_path):
                    continue
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            file_had_match = False
            for line_no, line in enumerate(lines, start=1):
                if use_regex:
                    regex_matches = list(compiled_regex.finditer(line))
                    if not regex_matches:
                        continue
                    matched_keywords = []
                    matched_texts = [match.group(0) for match in regex_matches if match.group(0)]
                else:
                    line_haystack = line if case_sensitive else line.lower()
                    matched_keywords = [
                        keyword
                        for keyword, prepared_keyword in zip(keywords_to_search, prepared_keywords)
                        if prepared_keyword in line_haystack
                    ]
                    if not matched_keywords:
                        continue
                    if match_mode == "all" and len(matched_keywords) < len(keywords_to_search):
                        continue
                    matched_texts = []

                file_had_match = True
                start_line = max(1, line_no - context_lines)
                end_line = min(len(lines), line_no + context_lines)
                snippet = "\n".join(
                    f"{index}: {lines[index - 1]}" for index in range(start_line, end_line + 1)
                )
                matches.append(
                    {
                        "path": to_workspace_relative(file_path),
                        "line": line_no,
                        "matched_keywords": matched_keywords,
                        "matched_texts": matched_texts,
                        "text": line.strip(),
                        "context_start_line": start_line,
                        "context_end_line": end_line,
                        "context": snippet,
                        "file_content": snippet,
                    }
                )
                if len(matches) >= max_matches:
                    return {
                        "status": "success",
                        "path": to_workspace_relative(root),
                        "pattern": pattern,
                        "keywords": keywords_to_search,
                        "use_regex": use_regex,
                        "regex_pattern": active_regex_pattern if use_regex else None,
                        "regex_flags": regex_flags if use_regex else "",
                        "match_mode": match_mode,
                        "case_sensitive": case_sensitive,
                        "include_glob": include_glob,
                        "matched_files": matched_files + (1 if file_had_match else 0),
                        "matches": matches,
                        "truncated": True,
                    }
            if file_had_match:
                matched_files += 1

        return {
            "status": "success",
            "path": to_workspace_relative(root),
            "pattern": pattern,
            "keywords": keywords_to_search,
            "use_regex": use_regex,
            "regex_pattern": active_regex_pattern if use_regex else None,
            "regex_flags": regex_flags if use_regex else "",
            "match_mode": match_mode,
            "case_sensitive": case_sensitive,
            "include_glob": include_glob,
            "matched_files": matched_files,
            "matches": matches,
            "truncated": False,
        }
    except RestrictedWorkspacePathError as exc:
        return exc.to_warning_payload()
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


async def list_files_parallel(requests: list[dict[str, object]]) -> dict:
    """List multiple paths concurrently and return all results together.

    Use this when you already know the set of directories or files you want to
    inspect. Each request may include `path`, `recursive`, `max_entries`, and
    `exclude` for regex-based path exclusion.
    Parameter errors are returned as structured tool errors instead of raising.

    Example input:
        requests=[{"path": "src", "exclude": "__pycache__|\\.pyc$"}, {"path": "tests"}]

    Example output:
        {"status": "success", "results": [{"status": "success", "path": "src", "exclude": "__pycache__|\\.pyc$", "entries": ["src/app.py"]}]}
    """
    try:
        parsed_requests = [_parse_list_files_request(request) for request in requests]
        results = await asyncio.gather(
            *[
                asyncio.to_thread(_list_files_impl, path, recursive, max_entries, exclude)
                for path, recursive, max_entries, exclude in parsed_requests
            ],
            return_exceptions=True,
        )
        normalized_results = [
            result if not isinstance(result, Exception) else {"status": "error", "error_message": str(result)}
            for result in results
        ]
        return {"status": "success", "results": normalized_results}
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


async def read_files(requests: list[dict[str, object]]) -> dict:
    """Read multiple files or file slices concurrently and return all results together.

    Use this when you need several files at once. Each request must include a
    `path` and may include `start_line` and `end_line`. Invalid requests return
    structured errors instead of aborting the agent loop.
    """
    try:
        parsed_requests = [_parse_read_file_request(request) for request in requests]
        results = await asyncio.gather(
            *[
                asyncio.to_thread(_read_file_impl, path, start_line, end_line)
                for path, start_line, end_line in parsed_requests
            ],
            return_exceptions=True,
        )
        normalized_results = [
            result if not isinstance(result, Exception) else {"status": "error", "error_message": str(result)}
            for result in results
        ]
        return {"status": "success", "results": normalized_results}
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


async def search_text_parallel(requests: list[dict[str, object]]) -> dict:
    """Run multiple literal-text searches concurrently and return all results together.

    Use this when you want to search different paths or patterns in one tool
    call. Each request must include `pattern` and may include `path` and
    `max_matches`. Invalid requests return structured errors instead of raising.
    """
    try:
        parsed_requests = [_parse_search_text_request(request) for request in requests]
        results = await asyncio.gather(
            *[
                asyncio.to_thread(_search_text_impl, pattern, path, max_matches)
                for pattern, path, max_matches in parsed_requests
            ],
            return_exceptions=True,
        )
        normalized_results = [
            result if not isinstance(result, Exception) else {"status": "error", "error_message": str(result)}
            for result in results
        ]
        return {"status": "success", "results": normalized_results}
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}
