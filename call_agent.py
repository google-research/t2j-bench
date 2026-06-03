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
import argparse
import base64
from contextlib import aclosing
import inspect
import json
import os
import re
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
import textwrap
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
CODING_AGENT_ROOT = REPO_ROOT / "coding-agent"
AGENT_ENV_PATH = CODING_AGENT_ROOT / "coding_agent" / ".env"
_PROMPT_FILE_REF_RE = re.compile(r"(?<!\S)@([^\s]+)")
_OMITTED_LINE_RE = re.compile(r"^\.\.\.\s+omitted\s+(\d+)\s+(.+?)(?:\s+\.\.\.)?$")
_PENDING_TOOL_CALLS: dict[str, dict[str, Any]] = {}
_LAST_USAGE_TEXT_BY_AUTHOR: dict[str, str] = {}
_ROOT_WORKSPACE_STATE_KEY = "_coding_agent_root_workspace"
_DEFAULT_MEMORY_FILE_REL_PATH = ".coding_agent_memory/working_memory.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the coding_agent ADK agent against a specific workspace.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text, a .md file path, or @/path/to/file.md. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Workspace directory the agent is allowed to operate in.",
    )
    parser.add_argument(
        "--save-path",
        help="Optional path to save the full conversation history as JSON.",
    )
    parser.add_argument(
        "--load-history",
        help="Optional path to an existing saved conversation JSON to preload before sending the new prompt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed event and tool information.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Allow file tools to operate outside the workspace root.",
    )
    parser.add_argument(
        "--agent",
        default="client_agent",
        choices=("client_agent", "iterative_agent"),
        help="Which top-level agent to run.",
    )
    parser.add_argument("--user-id", default="cli_user")
    parser.add_argument("--session-id")
    parser.add_argument("--app-name", default="coding_agent")
    return parser.parse_args()


def _resolve_prompt_file_ref(ref: str, base_dir: Path | None) -> Path:
    candidate = Path(ref).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if base_dir is not None:
            candidates.append((base_dir / candidate).expanduser())
        candidates.append(candidate)

    for item in candidates:
        resolved = item.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    raise SystemExit(f"Prompt file reference not found: @{ref}")


def _append_prompt_file_refs(prompt: str, *, base_dir: Path | None) -> str:
    refs: list[Path] = []
    seen: set[Path] = set()

    for match in _PROMPT_FILE_REF_RE.finditer(prompt):
        path = _resolve_prompt_file_ref(match.group(1), base_dir)
        if path not in seen:
            seen.add(path)
            refs.append(path)

    if not refs:
        return prompt

    sections = [prompt.rstrip()]
    for path in refs:
        content = path.read_text(encoding="utf-8")
        sections.append(f"\n\n## {path.name}\n\n{content.rstrip()}")
    return "".join(sections).strip()


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        raw = args.prompt.strip()
        candidate = raw[1:] if raw.startswith("@") else raw
        candidate_path = Path(candidate).expanduser()
        if candidate_path.suffix.lower() == ".md" and candidate_path.exists() and candidate_path.is_file():
            resolved_prompt_path = candidate_path.resolve()
            prompt = resolved_prompt_path.read_text(encoding="utf-8")
            return _append_prompt_file_refs(prompt, base_dir=resolved_prompt_path.parent)
        return _append_prompt_file_refs(args.prompt, base_dir=None)
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return _append_prompt_file_refs(data, base_dir=None)
    raise SystemExit("A prompt is required, either as an argument or via stdin.")


def _inject_workspace_constraint(prompt: str, workspace: Path) -> str:
    constraint = (
        f"\n\nWorkspace boundary:\n"
        f"- Your workspace is strictly limited to `{workspace}`.\n"
        f"- Do not read, list, search, inspect, or modify files outside this workspace.\n"
    )
    return prompt.rstrip() + constraint


def _ensure_coding_agent_memory_file(workspace: Path) -> None:
    memory_path = workspace / _DEFAULT_MEMORY_FILE_REL_PATH
    if memory_path.exists():
        return
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "\n".join(
            [
                "# Working Memory",
                "",
                "## Directory Structure",
                "- Pending update.",
                "",
                "## Important File Snippets",
                "- Pending update.",
                "",
                "## Important Findings",
                "- Pending update.",
                "",
                "## Current Task Progress",
                "- Pending update.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _serialize_model(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return {
                "__type__": "bytes",
                "encoding": "utf-8",
                "data": value.decode("utf-8"),
            }
        except UnicodeDecodeError:
            return {
                "__type__": "bytes",
                "encoding": "base64",
                "data": base64.b64encode(value).decode("ascii"),
            }
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True, mode="json")
        except TypeError:
            dumped = value.model_dump(exclude_none=True)
        return _serialize_model(dumped)
    if hasattr(value, "dict") and callable(value.dict):
        return _serialize_model(value.dict())
    if isinstance(value, dict):
        return {str(key): _serialize_model(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_model(item) for item in value]
    return str(value)


def _event_to_record(index: int, event: Any) -> dict[str, Any]:
    return {
        "index": index,
        "author": getattr(event, "author", None),
        "timestamp": getattr(event, "timestamp", None),
        "is_final_response": event.is_final_response() if hasattr(event, "is_final_response") else False,
        "event": _serialize_model(event),
    }


def _extract_text_parts(event: Any) -> list[str]:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return texts


def _summarize_token_usage(events: list[Any]) -> dict[str, Any]:
    totals = {
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "total_token_count": 0,
    }
    counted_events = 0

    for event in events:
        usage = getattr(event, "usage_metadata", None)
        if usage is None:
            continue
        counted_events += 1
        for key in totals:
            value = getattr(usage, key, None)
            if isinstance(value, int):
                totals[key] += value

    totals["events_with_usage"] = counted_events
    return totals


def _format_token_usage_summary(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    if summary.get("prompt_token_count"):
        parts.append(f"prompt={summary['prompt_token_count']}")
    if summary.get("cached_content_token_count"):
        parts.append(f"cached={summary['cached_content_token_count']}")
    if summary.get("candidates_token_count"):
        parts.append(f"output={summary['candidates_token_count']}")
    if summary.get("thoughts_token_count"):
        parts.append(f"thoughts={summary['thoughts_token_count']}")
    if summary.get("total_token_count"):
        parts.append(f"total={summary['total_token_count']}")
    if summary.get("events_with_usage") is not None:
        parts.append(f"events={summary['events_with_usage']}")
    return "Token consumption: " + " ".join(parts) if parts else ""


def _format_verbose_event(event: Any) -> list[str]:
    lines: list[str] = []
    author = getattr(event, "author", "unknown")
    lines.extend(f"{author}: {text}" for text in _extract_text_parts(event))

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            lines.append(f"{author}: [tool call] {function_call.name} {function_call.args}")
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            response_payload = _serialize_model(function_response.response)
            lines.append(f"{author}: [tool response] {function_response.name} {json.dumps(response_payload, ensure_ascii=False)}")
    return lines


def _truncate_preview_lines(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"… +{omitted} lines"])


def _summarize_tool_response_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("output"), str) and payload["output"].strip():
            return _truncate_preview_lines(payload["output"])
        prioritized: list[str] = []
        for key in (
            "status",
            "error_message",
            "path",
            "match_strategy",
            "match_count",
            "bytes_written",
            "exit_code",
            "command_status",
            "session_id",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                prioritized.append(f"{key}: {value}")
        if prioritized:
            return _truncate_preview_lines("\n".join(prioritized))
    serialized = json.dumps(_serialize_model(payload), ensure_ascii=False, indent=2)
    return _truncate_preview_lines(serialized)


def _is_terminal_tool(tool_name: str) -> bool:
    return tool_name in {
        "bash_exec",
        "python_exec",
        "run_bash",
        "read_command_session",
        "wait",
        "terminate_command_session",
        "run_evaluator_command",
    }


def _tool_call_key(tool_name: str, identifier: str | None) -> str:
    return f"{tool_name}:{identifier or uuid.uuid4().hex}"


def _extract_terminal_command(tool_name: str, args: dict[str, Any], payload: Any) -> str:
    if tool_name in {"bash_exec", "run_bash", "run_evaluator_command"}:
        value = args.get("command")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if tool_name == "python_exec":
        code = args.get("code")
        if isinstance(code, str) and code.strip():
            return "python -c " + code.strip()
        return "python snippet"

    if isinstance(payload, dict):
        output = payload.get("output")
        if isinstance(output, str):
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.startswith("- command: "):
                    return stripped[len("- command: ") :].strip()
    return tool_name


def _extract_tool_invocation_text(tool_name: str, args: dict[str, Any], payload: Any) -> str:
    if _is_terminal_tool(tool_name):
        return _extract_terminal_command(tool_name, args, payload)

    prioritized_keys = (
        "path",
        "target_folder",
        "pattern",
        "keywords",
        "regex_pattern",
        "delivery_folder_path",
        "summary",
        "session_id",
        "request",
        "reason",
    )
    parts = [tool_name]
    for key in prioritized_keys:
        value = args.get(key)
        if value in (None, "", [], {}):
            continue
        rendered = _short_arg_value(value)
        if rendered:
            parts.append(f"{key}={rendered}")
    if len(parts) == 1 and args:
        serialized = json.dumps(_serialize_model(args), ensure_ascii=False, sort_keys=True)
        parts.append(serialized)
    return " ".join(parts)


def _short_arg_value(value: Any) -> str:
    serialized = _serialize_model(value)
    if isinstance(serialized, list):
        return "[" + ", ".join(str(item) for item in serialized[:4]) + (", ..." if len(serialized) > 4 else "") + "]"
    text = str(serialized)
    return text if len(text) <= 160 else text[:157] + "..."


def _extract_terminal_output_text(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("output"), str):
        text = payload["output"]
    else:
        text = _summarize_tool_response_payload(payload)

    lines = text.splitlines()
    for marker in ("New output:", "Output:", "Recent output:", "Stdout:", "Stderr:"):
        for index, line in enumerate(lines):
            if line.strip() == marker:
                return "\n".join(lines[index + 1 :]).strip()
    return text.strip()


def _normalize_display_line(line: str) -> str:
    stripped = line.strip()
    match = _OMITTED_LINE_RE.match(stripped)
    if match:
        return f"… +{match.group(1)} {match.group(2)}"
    return line


def _normalize_display_text(text: str) -> str:
    if not text:
        return text
    return "\n".join(_normalize_display_line(line) for line in text.splitlines()).strip()


def _extract_structured_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped in {"Key facts:", "Content:", "Error:"}:
            current = stripped[:-1].lower().replace(" ", "_")
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(raw_line)
    return {
        key: "\n".join(lines).strip()
        for key, lines in sections.items()
        if "\n".join(lines).strip()
    }


def _extract_structured_output_text(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("output"), str):
        text = payload["output"]
        sections = _extract_structured_sections(text)
        if sections.get("content"):
            return _normalize_display_text(sections["content"])
        if sections.get("error"):
            return _normalize_display_text(sections["error"])
        if sections.get("key_facts"):
            return _normalize_display_text(sections["key_facts"])
        return _normalize_display_text(text)
    return _normalize_display_text(_summarize_tool_response_payload(payload).strip())


def _extract_tool_output_text(tool_name: str, payload: Any) -> str:
    payload = _serialize_model(payload)
    if tool_name == "run_bash" and isinstance(payload, dict):
        output_context_lines = payload.get("output_context_lines")
        output_new_lines = payload.get("output_new_lines")
        output_lines = payload.get("output_lines")
        chunks: list[str] = []
        if isinstance(output_context_lines, list) and output_context_lines:
            chunks.extend(str(line) for line in output_context_lines)
        if isinstance(output_new_lines, list) and output_new_lines:
            chunks.extend(str(line) for line in output_new_lines)
        elif isinstance(output_lines, list) and output_lines:
            chunks.extend(str(line) for line in output_lines)
        if chunks:
            return "\n".join(chunks).strip()
        output = payload.get("output")
        if isinstance(output, str) and output.strip():
            return output.strip()
        return ""
    if _is_terminal_tool(tool_name) and _is_payload_with_output(payload):
        return _extract_terminal_output_text(payload)
    return _extract_structured_output_text(payload)


def _is_omission_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("… +") or bool(_OMITTED_LINE_RE.match(stripped))


def _is_payload_with_output(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("output"), str)


def _render_tool_block(prefix: str, command: str, output_text: str) -> str:
    wrapped_command = textwrap.wrap(command, width=118) or [command]
    lines = [f"• {prefix} {wrapped_command[0]}"]
    lines.extend(f"  │ {line}" for line in wrapped_command[1:])
    rendered_output = _render_tool_output_append(output_text)
    lines.extend(f"  {line}" for line in rendered_output.splitlines())
    return "\n".join(lines)


def _render_tool_call_block(prefix: str, command: str) -> str:
    wrapped_command = textwrap.wrap(command, width=118) or [command]
    lines = [f"• {prefix} {wrapped_command[0]}"]
    lines.extend(f"  │ {line}" for line in wrapped_command[1:])
    return "\n".join(lines)


def _render_tool_output_append(output_text: str) -> str:
    output_lines = output_text.splitlines() if output_text else []
    if not output_lines:
        return "└ <no output>"
    normalized_lines = [_normalize_display_line(line) for line in output_lines]
    head_count = 5
    tail_count = 5
    if len(normalized_lines) <= head_count + tail_count:
        lines = [f"└ {normalized_lines[0]}"]
        for line in normalized_lines[1:]:
            lines.append(f"  {line}")
        return "\n".join(lines)

    head_lines = normalized_lines[:head_count]
    tail_lines = normalized_lines[-tail_count:]
    middle_lines = normalized_lines[head_count:-tail_count]
    preserved_middle = [line for line in middle_lines if _is_omission_line(line)]
    lines = [f"└ {head_lines[0]}"]
    for line in head_lines[1:]:
        lines.append(f"  {line}")
    if preserved_middle:
        for line in preserved_middle:
            lines.append(f"  {line}")
    else:
        omitted = len(normalized_lines) - head_count - tail_count
        lines.append(f"  … +{omitted} lines")
    for line in tail_lines:
        lines.append(f"  {line}")
    return "\n".join(lines)


def _print_cli_event(event: Any) -> None:
    usage = getattr(event, "usage_metadata", None)
    author = getattr(event, "author", "unknown")
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []

    def usage_text() -> str:
        if usage is None:
            return ""
        usage_bits: list[str] = []
        for label, attr in (
            ("prompt", "prompt_token_count"),
            ("cached", "cached_content_token_count"),
            ("output", "candidates_token_count"),
            ("thoughts", "thoughts_token_count"),
            ("total", "total_token_count"),
        ):
            value = getattr(usage, attr, None)
            if value is not None:
                usage_bits.append(f"{label}={value}")
        return " ".join(usage_bits)

    current_usage_bits = usage_text()
    if current_usage_bits:
        _LAST_USAGE_TEXT_BY_AUTHOR[author] = current_usage_bits

    def print_header(*, tool_name: str | None = None) -> None:
        parts: list[str] = [author]
        if tool_name:
            parts.append(f"tool: {tool_name}")
        usage_bits = current_usage_bits or _LAST_USAGE_TEXT_BY_AUTHOR.get(author, "")
        if usage_bits:
            parts.append(f"token consumption: {usage_bits}")
        print(f"\n[{'|'.join(parts)}]")

    for part in parts:
        text = getattr(part, "text", None)
        if text:
            print_header()
            print("  " + _truncate_preview_lines(text).replace("\n", "\n  "))

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            args = function_call.args or {}
            tool_name = function_call.name
            serialized_args = _serialize_model(args)
            _PENDING_TOOL_CALLS[_tool_call_key(tool_name, getattr(function_call, "id", None))] = {
                "tool_name": tool_name,
                "args": serialized_args,
                "printed": False,
            }

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            tool_name = getattr(function_response, "name", "")
            key = _tool_call_key(tool_name, getattr(function_response, "id", None))
            pending = _PENDING_TOOL_CALLS.pop(key, {"tool_name": tool_name, "args": {}, "printed": False})
            payload = _serialize_model(function_response.response)
            invocation_text = _extract_tool_invocation_text(tool_name, pending.get("args", {}), payload)
            output_text = _extract_tool_output_text(tool_name, payload)
            print_header(tool_name=tool_name)
            prefix = "Ran" if _is_terminal_tool(tool_name) else "Used"
            print("  " + _render_tool_block(prefix, invocation_text, output_text).replace("\n", "\n  "))


def _resolve_maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        import asyncio

        return asyncio.run(value)
    return value


def _load_simple_env_file(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)
        loaded[key] = value
    return loaded


def _resolve_save_path(raw_save_path: str | None) -> Path | None:
    if not raw_save_path:
        return None

    candidate = Path(raw_save_path).expanduser().resolve()
    suffix = candidate.suffix.lower()
    if candidate.exists() and candidate.is_dir():
        filename = f"conversation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        return candidate / filename
    if suffix != ".json":
        return candidate / f"conversation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    return candidate


def _load_history_payload(raw_path: str | None) -> dict[str, Any] | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Conversation history file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Conversation history path is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse conversation history JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Conversation history JSON must be an object.")
    payload["_loaded_from_path"] = str(path)
    return payload


def _deserialize_event_records(payload: dict[str, Any]) -> list[Any]:
    from google.adk.events.event import Event

    records = payload.get("events")
    if not isinstance(records, list):
        return []

    events: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_event = record.get("event")
        if not isinstance(raw_event, dict):
            continue
        events.append(Event.model_validate(raw_event))
    return events


async def _run_agent_async(
    *,
    runner: Any,
    user_id: str,
    session_id: str,
    new_message: Any,
    verbose: bool,
) -> list[Any]:
    events: list[Any] = []
    async with aclosing(
        runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        )
    ) as agen:
        async for event in agen:
            if verbose:
                _print_cli_event(event)
                sys.stdout.flush()
            events.append(event)
    return events


def main() -> None:
    args = _parse_args()
    loaded_history_payload = _load_history_payload(args.load_history)
    prompt = _read_prompt(args)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise SystemExit(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise SystemExit(f"Workspace is not a directory: {workspace}")
    _ensure_coding_agent_memory_file(workspace)
    prompt = _inject_workspace_constraint(prompt, workspace)

    os.environ["CODING_AGENT_WORKSPACE"] = str(workspace)
    os.environ["CODING_AGENT_YOLO"] = "1" if args.yolo else "0"
    os.environ["CODING_AGENT_VERBOSE"] = "1" if args.verbose else "0"
    os.environ["CODING_AGENT_STREAM_SUBAGENT_LOGS"] = "1" if args.verbose else "0"
    loaded_env = _load_simple_env_file(AGENT_ENV_PATH)
    if str(CODING_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(CODING_AGENT_ROOT))

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    if args.agent == "iterative_agent":
        from iterative_agent.agent import root_agent
    else:
        from coding_agent.agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(
        app_name=args.app_name,
        agent=root_agent,
        session_service=session_service,
    )
    session = _resolve_maybe_await(
        session_service.create_session(
            app_name=args.app_name,
            user_id=args.user_id,
            session_id=args.session_id or f"session-{uuid.uuid4().hex[:12]}",
            state={_ROOT_WORKSPACE_STATE_KEY: str(workspace)},
        )
    )

    loaded_history_events: list[Any] = []
    if loaded_history_payload is not None:
        loaded_history_events = _deserialize_event_records(loaded_history_payload)
        for event in loaded_history_events:
            _resolve_maybe_await(session_service.append_event(session=session, event=event))

    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    events: list[Any] = []
    run_error: dict[str, str] | None = None
    interrupted = False
    try:
        events = asyncio.run(
            _run_agent_async(
                runner=runner,
                user_id=args.user_id,
                session_id=session.id,
                new_message=new_message,
                verbose=args.verbose,
            )
        )
    except KeyboardInterrupt:
        interrupted = True
        run_error = {
            "type": "KeyboardInterrupt",
            "message": "Interrupted by user with Ctrl+C.",
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        run_error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            _resolve_maybe_await(runner.close())
        except Exception:
            pass

    final_chunks: list[str] = []
    for event in events:
        if hasattr(event, "is_final_response") and event.is_final_response():
            final_chunks.extend(_extract_text_parts(event))

    final_response = "\n".join(chunk for chunk in final_chunks if chunk).strip()
    token_usage_summary = _summarize_token_usage(events)
    if final_response:
        if args.verbose:
            print("\n=== final response ===")
        print(final_response)
        formatted_token_usage = _format_token_usage_summary(token_usage_summary)
        if formatted_token_usage:
            print(f"\n{formatted_token_usage}")
    elif interrupted:
        print("Agent run interrupted by user.", file=sys.stderr)
    elif run_error:
        print(f"Agent run failed: {run_error['type']}: {run_error['message']}", file=sys.stderr)

    save_path = _resolve_save_path(args.save_path)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "app_name": args.app_name,
                "user_id": args.user_id,
                "session_id": session.id,
                "workspace": str(workspace),
                "prompt": prompt,
                "verbose": args.verbose,
                "yolo": args.yolo,
                "interrupted": interrupted,
                "agent_env_path": str(AGENT_ENV_PATH),
                "agent_name": args.agent,
                "loaded_env_keys": sorted(loaded_env.keys()),
                "load_history_path": loaded_history_payload.get("_loaded_from_path") if loaded_history_payload else None,
                "loaded_history_event_count": len(loaded_history_events),
            },
            "summary": {
                "num_events": len(events),
                "final_response": final_response,
                "token_usage": token_usage_summary,
                "error": run_error,
            },
            "events": [_event_to_record(index, event) for index, event in enumerate(events, start=1)],
        }
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved conversation history to {save_path}")

    if interrupted or run_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
