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

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import textwrap
from typing import Any
import uuid

from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
from google.adk.tools.agent_tool import _get_input_schema, _get_output_schema
from google.adk.utils._schema_utils import validate_schema
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from pydantic import BaseModel

from tools.workspace import (
    get_workspace_root,
    resolve_workspace_path,
    to_workspace_relative,
)


_VERBOSE_ENV_VAR = "CODING_AGENT_VERBOSE"
_STREAM_SUBAGENT_LOGS_ENV_VAR = "CODING_AGENT_STREAM_SUBAGENT_LOGS"
_PRINT_SUBAGENT_FINAL_ENV_VAR = "CODING_AGENT_PRINT_SUBAGENT_FINAL"
_FORMAT_CODEBASE_OUTPUT_TOOL = "format_codebase_output"
_MAX_EXCERPT_LINES = 200
_OMITTED_LINE_RE = re.compile(r"^\.\.\.\s+omitted\s+(\d+)\s+(.+?)(?:\s+\.\.\.)?$")
_PENDING_TOOL_CALLS: dict[str, dict[str, Any]] = {}
_LAST_USAGE_TEXT_BY_AUTHOR: dict[str, str] = {}
_DEFAULT_MEMORY_FILE_REL_PATH = ".coding_agent_memory/working_memory.md"
_SUBAGENT_MEMORY_DIR_REL_PATH = ".coding_agent_memory/subagents"
_ROOT_WORKSPACE_STATE_KEY = "_coding_agent_root_workspace"
_MEMORY_FILE_STATE_KEY = "_coding_agent_memory_file"
_LAST_COMPACTION_END_TS_STATE_KEY = "_coding_agent_last_compaction_end_timestamp"
_LAST_COMPACTION_EVENT_COUNT_STATE_KEY = "_coding_agent_last_compaction_event_count"


class WorkspaceScopedAgentRequest(BaseModel):
    request: str
    workspace: str
    restricted_mode: bool = False


def _serialize_model(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True, mode="json")
        except TypeError:
            dumped = value.model_dump(exclude_none=True)
        return _serialize_model(dumped)
    if isinstance(value, dict):
        return {str(key): _serialize_model(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_model(item) for item in value]
    return str(value)


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
        "run_bash",
        "read_command_session",
        "wait",
        "terminate_command_session",
        "run_evaluator_command",
    }


def _tool_call_key(tool_name: str, identifier: str | None) -> str:
    return f"{tool_name}:{identifier or 'no-id'}"


def _extract_terminal_command(tool_name: str, args: dict[str, Any], payload: Any) -> str:
    if tool_name in {"bash_exec", "run_bash", "run_evaluator_command"}:
        value = args.get("command")
        if isinstance(value, str) and value.strip():
            return value.strip()

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


def _subagent_display_name(parent_agent_name: str, child_agent_name: str) -> str:
    if child_agent_name == parent_agent_name:
        return f"{child_agent_name}>subagent"
    return f"{parent_agent_name}>{child_agent_name}"


def _print_streamed_event(event: Any, *, display_name: str | None = None) -> None:
    usage = getattr(event, "usage_metadata", None)
    author = display_name or getattr(event, "author", "unknown")
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
        header_parts: list[str] = [author]
        if tool_name:
            header_parts.append(f"tool: {tool_name}")
        usage_bits = current_usage_bits or _LAST_USAGE_TEXT_BY_AUTHOR.get(author, "")
        if usage_bits:
            header_parts.append(f"token consumption: {usage_bits}")
        print(f"\n[{'|'.join(header_parts)}]")

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


def _render_codebase_excerpt(path: str, start_line: int, end_line: int) -> str:
    try:
        file_path = resolve_workspace_path(path)
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return (
            f"[{path}:{start_line}-{end_line}]\n"
            f"<error reading file: {exc}>"
        )

    start = max(start_line, 1)
    end = max(end_line, start)
    if end - start + 1 > _MAX_EXCERPT_LINES:
        end = start + _MAX_EXCERPT_LINES - 1

    selected = lines[start - 1 : end]
    numbered = "\n".join(
        f"{idx}: {line}" for idx, line in enumerate(selected, start=start)
    )
    header = f"[{to_workspace_relative(file_path)}:{start}-{min(end, len(lines))}]"
    return f"{header}\n{numbered}" if numbered else f"{header}\n<no content>"


def _render_formatted_codebase_output(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None

    files = response.get("files")
    summary = response.get("summary")
    if not isinstance(files, list):
        return None

    sections: list[str] = []
    if isinstance(summary, str) and summary.strip():
        sections.append(f"Summary\n{summary.strip()}")

    excerpts: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        start_line = item.get("start_line", 1)
        end_line = item.get("end_line", start_line)
        if not isinstance(path, str) or not path:
            continue
        if not isinstance(start_line, int) or isinstance(start_line, bool):
            start_line = 1
        if not isinstance(end_line, int) or isinstance(end_line, bool):
            end_line = start_line
        excerpts.append(_render_codebase_excerpt(path, start_line, end_line))

    if excerpts:
        sections.append("Selected file excerpts\n\n" + "\n\n".join(excerpts))

    return "\n\n".join(sections).strip() or None


def _resolve_scoped_workspace(raw_workspace: str) -> Path:
    parent_workspace = get_workspace_root()
    candidate = Path(raw_workspace).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (parent_workspace / candidate).resolve()
    )
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Workspace does not exist or is not a directory: {raw_workspace}")
    return resolved


def _default_memory_template() -> str:
    return "\n".join(
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
    )


def _ensure_coding_agent_memory_file(workspace: Path, memory_rel_path: str) -> Path:
    memory_path = workspace / memory_rel_path
    if memory_path.exists():
        return memory_path
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(_default_memory_template(), encoding="utf-8")
    return memory_path


def _prepare_scoped_child_state(parent_state: dict[str, Any], scoped_workspace: Path) -> dict[str, Any]:
    child_state = dict(parent_state)
    subagent_memory_rel_path = (
        f"{_SUBAGENT_MEMORY_DIR_REL_PATH}/{uuid.uuid4().hex}/working_memory.md"
    )
    _ensure_coding_agent_memory_file(scoped_workspace, subagent_memory_rel_path)
    child_state[_ROOT_WORKSPACE_STATE_KEY] = str(scoped_workspace)
    child_state[_MEMORY_FILE_STATE_KEY] = subagent_memory_rel_path
    child_state.pop(_LAST_COMPACTION_END_TS_STATE_KEY, None)
    child_state.pop(_LAST_COMPACTION_EVENT_COUNT_STATE_KEY, None)
    return child_state


@contextmanager
def _temporary_workspace_override(workspace: Path):
    previous_workspace = os.environ.get("CODING_AGENT_WORKSPACE")
    os.environ["CODING_AGENT_WORKSPACE"] = str(workspace)
    try:
        yield
    finally:
        if previous_workspace is None:
            os.environ.pop("CODING_AGENT_WORKSPACE", None)
        else:
            os.environ["CODING_AGENT_WORKSPACE"] = previous_workspace


@contextmanager
def _temporary_restricted_mode_override(restricted_mode: bool):
    previous_value = os.environ.get("CODING_AGENT_RESTRICTED_MODE")
    value = "1" if restricted_mode else "0"
    os.environ["CODING_AGENT_RESTRICTED_MODE"] = value
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("CODING_AGENT_RESTRICTED_MODE", None)
        else:
            os.environ["CODING_AGENT_RESTRICTED_MODE"] = previous_value


class StreamingAgentTool(AgentTool):
    """AgentTool that can stream child agent events to stdout in verbose mode."""

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context,
    ) -> Any:
        return await self._run_async_with_state(
            args=args,
            tool_context=tool_context,
            state_dict=self._build_child_state_dict(tool_context),
        )

    async def _run_async_with_state(
        self,
        *,
        args: dict[str, Any],
        tool_context,
        state_dict: dict[str, Any],
    ) -> Any:
        if self.skip_summarization:
            tool_context.actions.skip_summarization = True

        input_schema = _get_input_schema(self.agent)
        if input_schema:
            input_value = input_schema.model_validate(args)
            content = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=input_value.model_dump_json(exclude_none=True)
                    )
                ],
            )
        else:
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=args["request"])],
            )

        invocation_context = tool_context._invocation_context
        parent_app_name = invocation_context.app_name if invocation_context else None
        child_app_name = parent_app_name or self.agent.name
        plugins = (
            tool_context._invocation_context.plugin_manager.plugins
            if self.include_plugins
            else None
        )
        runner = Runner(
            app_name=child_app_name,
            agent=self.agent,
            artifact_service=ForwardingArtifactService(tool_context),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
            credential_service=tool_context._invocation_context.credential_service,
            plugins=plugins,
        )

        session = await runner.session_service.create_session(
            app_name=child_app_name,
            user_id=tool_context._invocation_context.user_id,
            state=state_dict,
        )

        stream_logs = os.environ.get(_VERBOSE_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } and os.environ.get(_STREAM_SUBAGENT_LOGS_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        display_name = _subagent_display_name(tool_context.agent_name, self.agent.name)

        last_content = None
        formatted_codebase_output = None
        try:
            async with Aclosing(
                runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=content,
                )
            ) as agen:
                async for event in agen:
                    if event.actions.state_delta:
                        tool_context.state.update(event.actions.state_delta)
                    if stream_logs:
                        _print_streamed_event(event, display_name=display_name)
                        sys.stdout.flush()
                    for function_response in event.get_function_responses():
                        if getattr(function_response, "name", None) == _FORMAT_CODEBASE_OUTPUT_TOOL:
                            formatted_codebase_output = function_response.response
                    if event.content:
                        last_content = event.content
        finally:
            await runner.close()

        if formatted_codebase_output is not None:
            merged_text = _render_formatted_codebase_output(formatted_codebase_output) or ""
        elif last_content is None or last_content.parts is None:
            return ""
        else:
            merged_text = "\n".join(
                p.text for p in last_content.parts if p.text and not p.thought
            )
        print_final_value = os.environ.get(_PRINT_SUBAGENT_FINAL_ENV_VAR)
        print_final = True if print_final_value is None else print_final_value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if print_final and merged_text:
            print(f"\n[{display_name}]\n{merged_text}\n")
            sys.stdout.flush()
        output_schema = _get_output_schema(self.agent)
        if output_schema:
            return validate_schema(output_schema, merged_text)
        return merged_text

    def _build_child_state_dict(self, tool_context) -> dict[str, Any]:
        return {
            k: v
            for k, v in tool_context.state.to_dict().items()
            if not k.startswith("_adk")
        }


class WorkspaceScopedStreamingAgentTool(StreamingAgentTool):
    """Streaming sub-agent tool that runs the child agent inside a scoped workspace."""

    def __init__(
        self,
        agent,
        *,
        name: str | None = None,
        description: str | None = None,
        skip_summarization: bool = False,
        include_plugins: bool = True,
    ):
        super().__init__(
            agent=agent,
            skip_summarization=skip_summarization,
            include_plugins=include_plugins,
        )
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

    def _get_declaration(self) -> types.FunctionDeclaration:
        declaration = super()._get_declaration()
        declaration.name = self.name
        declaration.description = self.description
        declaration.parameters = None
        declaration.parameters_json_schema = {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "workspace": {"type": "string"},
                "restricted_mode": {"type": "boolean"},
            },
            "required": ["request", "workspace"],
        }
        return declaration

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context,
    ) -> Any:
        parsed = WorkspaceScopedAgentRequest.model_validate(args)
        scoped_workspace = _resolve_scoped_workspace(parsed.workspace)
        if not hasattr(tool_context.state, "to_dict") or not callable(tool_context.state.to_dict):
            raise ValueError("Tool context state does not support to_dict().")
        parent_state = {
            k: v
            for k, v in tool_context.state.to_dict().items()
            if not k.startswith("_adk")
        }
        child_state = _prepare_scoped_child_state(parent_state, scoped_workspace)
        with _temporary_workspace_override(scoped_workspace):
            with _temporary_restricted_mode_override(parsed.restricted_mode):
                return await self._run_async_with_state(
                    args={"request": parsed.request},
                    tool_context=tool_context,
                    state_dict=child_state,
                )
