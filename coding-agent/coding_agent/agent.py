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

import asyncio
import copy
import json
import logging
import os
import random
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from coding_agent.streaming_agent_tool import (
    StreamingAgentTool,
    WorkspaceScopedStreamingAgentTool
)
from tools.workspace import get_workspace_root
from tools import (
    bash_exec,
    check_running_sessions,
    format_codebase_output,
    keyword_search,
    list_files,
    lsp_tool,
    patch_file,
    read_command_session,
    read_file,
    read_todos,
    report_truncated_model_output,
    search_text,
    terminate_command_session,
    wait,
    write_file,
    write_todos,
)


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        return value
    return default


def _env_flag(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_value(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    value = _env_value(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_optional_float(name: str) -> float | None:
    value = _env_value(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_thinking_level(default: str = "low") -> types.ThinkingLevel | None:
    value = _env_value("CODING_AGENT_THINKING_LEVEL", default)
    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized or normalized in {"off", "none", "false", "0"}:
        return None

    mapping = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }
    return mapping.get(normalized, types.ThinkingLevel.LOW)


def _env_thinking_budget() -> int | None:
    """Thinking budget for the ThinkingConfig, Claude-only.

    adk-python's Anthropic integration requires ``thinking_budget`` to be set
    explicitly whenever a ThinkingConfig is provided (0 disables, -1 selects
    adaptive/model-chosen depth, positive ints are a manual token budget).
    Gemini instead uses ``thinking_level``, so only surface a budget for Claude
    models to leave the Gemini path untouched. Defaults to -1 (adaptive), which
    is the value adk recommends for Claude Opus 4.6 / 4.7+.
    """
    model = _env_value("CODING_AGENT_MODEL", "") or ""
    if "claude" not in model.lower():
        return None
    return _env_int("CODING_AGENT_THINKING_BUDGET", -1)


def _configure_retry_logging() -> None:
    if not _env_flag("CODING_AGENT_LOG_RETRIES", True):
        return

    logger = logging.getLogger("google.genai._api_client")
    logger.setLevel(logging.INFO)

    if any(getattr(handler, "name", "") == "coding_agent_retry_handler" for handler in logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.name = "coding_agent_retry_handler"
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False


_configure_retry_logging()

_DEFAULT_COMPACTION_CONTEXT_LIMIT = 1048576
_DEFAULT_COMPACTION_TRIGGER_RATIO = 0.85
_DEFAULT_COMPACTION_KEEP_RATIO = 0.10
_DEFAULT_COMPACTION_CURRENT_USAGE_KEEP_RATIO = 0.40
_DEFAULT_COMPACTION_MIN_RETAINED_EVENTS = 6
_DEFAULT_COMPACTION_MIN_EVENTS_TO_COMPACT = 4
_DEFAULT_COMPACTION_DEBOUNCE_TAIL_EVENTS = 4
_DEFAULT_MAX_OUTPUT_TOKENS = 32768
_DEFAULT_COMPACTION_ARCHIVE_DIR = ".coding_agent_memory/compactions"
_DEFAULT_MEMORY_FILE_REL_PATH = ".coding_agent_memory/working_memory.md"
_DEFAULT_MEMORY_HISTORY_ROUNDS = 5
_MODEL_NAME = _env_value("CODING_AGENT_MODEL")
_LITELLM_PROVIDER_PREFIXES = (
    "anthropic/",
    "azure/",
    "bedrock/",
    "deepinfra/",
    "groq/",
    "ollama/",
    "openai/",
    "openrouter/",
    "perplexity/",
    "together_ai/",
    "vertex_ai/",
)
_LITELLM_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_OPENAI_GOOGLE_AUTH_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_OPENAI_GOOGLE_AUTH_REFRESH_SKEW_SECONDS = 300
_openai_google_auth_credentials = None
_openai_google_auth_lock = threading.Lock()
_CURRENT_CONTEXT_LIMIT = _env_int("CODING_AGENT_COMPACT_CONTEXT_LIMIT", _DEFAULT_COMPACTION_CONTEXT_LIMIT)
_CURRENT_KEEP_RATIO = _env_float("CODING_AGENT_COMPACT_KEEP_RATIO", _DEFAULT_COMPACTION_KEEP_RATIO)
_MEMORY_FILE_STATE_KEY = "_coding_agent_memory_file"
_LAST_COMPACTION_END_TS_STATE_KEY = "_coding_agent_last_compaction_end_timestamp"
_LAST_COMPACTION_EVENT_COUNT_STATE_KEY = "_coding_agent_last_compaction_event_count"
_ROOT_WORKSPACE_STATE_KEY = "_coding_agent_root_workspace"


def _should_refresh_openai_google_auth() -> bool:
    explicit = _env_value("CODING_AGENT_REFRESH_OPENAI_GOOGLE_AUTH")
    if explicit is not None:
        return _env_flag("CODING_AGENT_REFRESH_OPENAI_GOOGLE_AUTH", False)
    base_url = _env_value("OPENAI_BASE_URL") or _env_value("OPENAI_API_BASE") or ""
    return bool(_MODEL_NAME) and _MODEL_NAME.startswith("openai/") and "googleapis.com" in base_url


def _seconds_until_expiry(expiry) -> float | None:
    if expiry is None:
        return None
    if getattr(expiry, "tzinfo", None) is not None:
        now = datetime.now(expiry.tzinfo)
    else:
        now = datetime.utcnow()
    return (expiry - now).total_seconds()


def _refresh_openai_google_auth_token_if_needed() -> None:
    if not _should_refresh_openai_google_auth():
        return

    global _openai_google_auth_credentials
    with _openai_google_auth_lock:
        if _openai_google_auth_credentials is None:
            import google.auth

            _openai_google_auth_credentials, _ = google.auth.default(
                scopes=list(_OPENAI_GOOGLE_AUTH_SCOPES)
            )

        credentials = _openai_google_auth_credentials
        expires_in = _seconds_until_expiry(getattr(credentials, "expiry", None))
        needs_refresh = (
            not getattr(credentials, "valid", False)
            or getattr(credentials, "expired", False)
            or expires_in is None
            or expires_in < _OPENAI_GOOGLE_AUTH_REFRESH_SKEW_SECONDS
        )
        if needs_refresh:
            from google.auth.transport.requests import Request

            credentials.refresh(Request())

        token = getattr(credentials, "token", None)
        if not token:
            raise RuntimeError(
                "Google ADC did not produce an access token for the "
                "OpenAI-compatible Vertex endpoint."
            )
        os.environ["OPENAI_API_KEY"] = token


def _uses_litellm_provider(model_name: str) -> bool:
    return model_name.startswith(_LITELLM_PROVIDER_PREFIXES)


def _litellm_completion_kwargs() -> dict:
    kwargs: dict = {}
    timeout = _env_optional_float("CODING_AGENT_LITELLM_TIMEOUT")
    if timeout is not None:
        kwargs["timeout"] = timeout
    max_retries = _env_optional_int("CODING_AGENT_LITELLM_MAX_RETRIES")
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return kwargs


def _exception_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_retryable_litellm_error(exc: BaseException) -> bool:
    retryable_names = (
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    )
    for item in _iter_exception_chain(exc):
        if _exception_status_code(item) in _LITELLM_RETRYABLE_STATUS_CODES:
            return True
        if item.__class__.__name__ in retryable_names:
            return True
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    for item in _iter_exception_chain(exc):
        headers = getattr(item, "headers", None)
        response = getattr(item, "response", None)
        if headers is None and response is not None:
            headers = getattr(response, "headers", None)
        if not headers:
            continue
        value = None
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            try:
                value = headers.get(key)
            except AttributeError:
                value = headers.get(key) if isinstance(headers, dict) else None
            if value:
                break
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def _litellm_retry_delay(exc: BaseException, failed_attempt_index: int) -> float:
    initial_delay = _env_float("CODING_AGENT_LITELLM_RETRY_INITIAL_DELAY", 30.0)
    max_delay = _env_float("CODING_AGENT_LITELLM_RETRY_MAX_DELAY", 300.0)
    multiplier = _env_float("CODING_AGENT_LITELLM_RETRY_MULTIPLIER", 2.0)
    jitter_ratio = _env_float("CODING_AGENT_LITELLM_RETRY_JITTER", 0.25)
    initial_delay = max(0.0, initial_delay)
    max_delay = max(initial_delay, max_delay)
    multiplier = max(1.0, multiplier)
    delay = min(
        max_delay,
        initial_delay * (multiplier ** max(0, failed_attempt_index - 1)),
    )
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        delay = min(max_delay, max(delay, retry_after))
    if jitter_ratio > 0 and delay > 0:
        delay += random.uniform(0.0, delay * jitter_ratio)
    return delay


def _agent_model():
    if not _MODEL_NAME:
        raise RuntimeError(
            "CODING_AGENT_MODEL environment variable is required. "
            "Set it to the model identifier you want the coding agent to use "
            "(e.g. a LiteLLM-style 'provider/model' string such as "
            "'openai/gpt-4o' or 'anthropic/claude-3-5-sonnet-latest')."
        )

    if not _uses_litellm_provider(_MODEL_NAME):
        return _MODEL_NAME

    from google.adk.models.lite_llm import LiteLlm

    class RetryingLiteLlm(LiteLlm):
        async def generate_content_async(self, llm_request, stream: bool = False):
            attempts = max(1, _env_int("CODING_AGENT_LITELLM_RETRY_ATTEMPTS", 8))
            for attempt in range(1, attempts + 1):
                try:
                    if hasattr(llm_request, "model_copy"):
                        request_for_attempt = llm_request.model_copy(deep=True)
                    else:
                        request_for_attempt = copy.deepcopy(llm_request)
                    async for response in super().generate_content_async(
                        request_for_attempt,
                        stream=stream,
                    ):
                        yield response
                    return
                except Exception as exc:
                    if attempt >= attempts or not _is_retryable_litellm_error(exc):
                        raise
                    delay = _litellm_retry_delay(exc, attempt)
                    print(
                        f"[litellm retry] attempt={attempt}/{attempts} "
                        f"model={self.model} error={exc.__class__.__name__} "
                        f"sleep_seconds={delay:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )
                    await asyncio.sleep(delay)

    return RetryingLiteLlm(model=_MODEL_NAME, **_litellm_completion_kwargs())


def _content_signature(content: types.Content | None) -> str:
    if content is None:
        return ""
    return content.model_dump_json(exclude_none=True)


def _serialize_part(part: types.Part) -> str:
    if part.text and not getattr(part, "thought", False):
        return f"text: {part.text}"
    if part.function_call:
        args = part.function_call.args or {}
        return f"function_call {part.function_call.name}: {json.dumps(args, ensure_ascii=False, sort_keys=True)}"
    if part.function_response:
        return (
            f"function_response {part.function_response.name}: "
            f"{json.dumps(part.function_response.response, ensure_ascii=False, sort_keys=True)}"
        )
    if part.inline_data:
        return "inline_data: <omitted>"
    if part.file_data:
        return "file_data: <omitted>"
    return ""


def _serialize_contents(contents: list[types.Content]) -> str:
    blocks: list[str] = []
    for index, content in enumerate(contents, start=1):
        lines: list[str] = [f"[{index}] role={content.role or 'unknown'}"]
        for part in content.parts or []:
            serialized = _serialize_part(part)
            if serialized:
                lines.append(serialized)
        if len(lines) > 1:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _count_content_text_chars(contents: list[types.Content]) -> int:
    total = 0
    for content in contents:
        for part in content.parts or []:
            if part.text:
                total += len(part.text)
    return total


def _count_serialized_content_chars(contents: list[types.Content]) -> int:
    return len(_serialize_contents(contents))


def _sanitize_compacted_history(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            continue
        lowered = line.lower()
        if lowered.startswith("call:"):
            continue
        if "function_call" in lowered:
            continue
        if lowered.startswith("```") or lowered.endswith("```"):
            continue
        if "<ctrl" in lowered:
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()


def _extract_user_instruction_text(user_content: types.Content | None) -> str:
    if user_content is None or not user_content.parts:
        return ""
    parts: list[str] = []
    for part in user_content.parts:
        if part.text:
            parts.append(part.text)
    return "\n".join(parts).strip()


def _get_memory_workspace(session_state: dict[str, object] | None) -> Path | None:
    root = None if not session_state else session_state.get(_ROOT_WORKSPACE_STATE_KEY)
    if isinstance(root, str) and root.strip():
        return Path(root)
    workspace = _env_value("CODING_AGENT_WORKSPACE")
    if workspace:
        return Path(workspace)
    return None


def _ensure_memory_file(
    session_state: dict[str, object] | None,
) -> tuple[Path | None, str | None]:
    workspace_path = _get_memory_workspace(session_state)
    if workspace_path is None:
        return None, None
    memory_rel_path = None if not session_state else session_state.get(_MEMORY_FILE_STATE_KEY)
    if not isinstance(memory_rel_path, str) or not memory_rel_path.strip():
        memory_rel_path = _DEFAULT_MEMORY_FILE_REL_PATH
    memory_path = workspace_path / memory_rel_path
    if not memory_path.exists():
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
    return memory_path, memory_rel_path


def _load_memory_file_text(session_state: dict[str, object] | None) -> str | None:
    memory_path, _ = _ensure_memory_file(session_state)
    if memory_path is None or not memory_path.exists():
        return None
    return memory_path.read_text(encoding="utf-8").strip()


@contextmanager
def _temporary_workspace_override(workspace: Path | None):
    if workspace is None:
        yield
        return
    previous_workspace = os.environ.get("CODING_AGENT_WORKSPACE")
    os.environ["CODING_AGENT_WORKSPACE"] = str(workspace)
    try:
        yield
    finally:
        if previous_workspace is None:
            os.environ.pop("CODING_AGENT_WORKSPACE", None)
        else:
            os.environ["CODING_AGENT_WORKSPACE"] = previous_workspace


_MEMORY_INSTRUCTION_MARKER = "## Working Memory Context"


def _inject_memory_after_user_instruction(llm_request, session_state: dict[str, object] | None) -> None:
    memory_text = _load_memory_file_text(session_state)
    if not memory_text:
        return

    existing_contents = list(getattr(llm_request, "contents", []) or [])
    if not existing_contents:
        return
    first_content = copy.deepcopy(existing_contents[0])
    if not first_content.parts:
        return
    for part in first_content.parts:
        if not part.text or getattr(part, "thought", False):
            continue
        base_text = part.text
        marker_index = base_text.find(_MEMORY_INSTRUCTION_MARKER)
        if marker_index != -1:
            base_text = base_text[:marker_index].rstrip()
        part.text = (
            f"{base_text}\n\n"
            f"{_MEMORY_INSTRUCTION_MARKER}\n"
            "Use this compressed memory as older context. If it conflicts with the recent raw conversation, prefer the recent raw conversation.\n\n"
            f"{memory_text}"
        ).strip()
        llm_request.contents = [first_content] + existing_contents[1:]
        return


def _is_textual_user_content(content: types.Content | None) -> bool:
    if content is None or content.role != "user":
        return False
    if any(part.function_response is not None for part in (content.parts or [])):
        return False
    for part in content.parts or []:
        if part.text and not getattr(part, "thought", False):
            return True
    return False


def _estimate_content_tokens(content: types.Content) -> int:
    return max(1, len(_serialize_contents([content])) // 4)


def _content_function_call_ids(content: types.Content) -> set[str]:
    ids: set[str] = set()
    for part in content.parts or []:
        function_call = getattr(part, "function_call", None)
        function_call_id = getattr(function_call, "id", None)
        if isinstance(function_call_id, str) and function_call_id:
            ids.add(function_call_id)
    return ids


def _content_function_response_ids(content: types.Content) -> set[str]:
    ids: set[str] = set()
    for part in content.parts or []:
        function_response = getattr(part, "function_response", None)
        function_response_id = getattr(function_response, "id", None)
        if isinstance(function_response_id, str) and function_response_id:
            ids.add(function_response_id)
    return ids


def _has_dangling_tool_results(contents: list[types.Content]) -> bool:
    seen_call_ids: set[str] = set()
    for content in contents:
        seen_call_ids.update(_content_function_call_ids(content))
        response_ids = _content_function_response_ids(content)
        if any(response_id not in seen_call_ids for response_id in response_ids):
            return True
    return False


def _trim_contents_to_recent_budget(
    contents: list[types.Content],
    *,
    context_limit: int,
    keep_ratio: float,
) -> list[types.Content]:
    if not contents:
        return contents

    latest_user_index = None
    for index in range(len(contents) - 1, -1, -1):
        if _is_textual_user_content(contents[index]):
            latest_user_index = index
            break

    if latest_user_index is None:
        return contents
    base_contents = [contents[latest_user_index]]
    tail_contents = contents[latest_user_index + 1 :]
    if not tail_contents:
        return base_contents

    token_budget = max(1, int(context_limit * keep_ratio))
    base_tokens = sum(_estimate_content_tokens(content) for content in base_contents)
    tail_budget = max(token_budget - base_tokens, 0)
    if tail_budget <= 0:
        return base_contents
    kept_tail_reversed: list[types.Content] = []
    used_tokens = 0
    for content in reversed(tail_contents):
        content_tokens = _estimate_content_tokens(content)
        if kept_tail_reversed and used_tokens + content_tokens > tail_budget:
            break
        if not kept_tail_reversed and content_tokens > tail_budget:
            kept_tail_reversed.append(content)
            break
        kept_tail_reversed.append(content)
        used_tokens += content_tokens

    retained_tail = list(reversed(kept_tail_reversed))
    if _has_dangling_tool_results(retained_tail):
        retained_tail = tail_contents
    return base_contents + retained_tail


async def _run_history_compactor(
    callback_context,
    history_contents: list[types.Content],
    focus_prompt: str | None = None,
    memory_file_path: str | None = None,
    current_task_progress: str | None = None,
) -> str | None:
    history_text = _serialize_contents(history_contents)
    if not history_text.strip():
        return None

    compactor_input = _build_history_compactor_input(
        history_text=history_text,
        focus_prompt=focus_prompt,
        memory_file_path=memory_file_path,
        current_task_progress=current_task_progress,
    )

    runner = Runner(
        app_name=callback_context.agent_name,
        agent=history_compactor,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=callback_context._invocation_context.credential_service,
    )
    session_state = getattr(callback_context.session, "state", None)
    compactor_state: dict[str, object] = {}
    if isinstance(session_state, dict):
        root_workspace = session_state.get(_ROOT_WORKSPACE_STATE_KEY)
        memory_file = session_state.get(_MEMORY_FILE_STATE_KEY)
        if isinstance(root_workspace, str) and root_workspace.strip():
            compactor_state[_ROOT_WORKSPACE_STATE_KEY] = root_workspace
        if isinstance(memory_file, str) and memory_file.strip():
            compactor_state[_MEMORY_FILE_STATE_KEY] = memory_file
    session = await runner.session_service.create_session(
        app_name=callback_context.agent_name,
        user_id=callback_context.user_id,
        state=compactor_state,
    )

    try:
        memory_workspace = _get_memory_workspace(session_state if isinstance(session_state, dict) else None)
        with _temporary_workspace_override(memory_workspace):
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=compactor_input)],
                ),
            ):
                del event
    finally:
        await runner.close()

    return _load_memory_file_text(callback_context.session.state)


def _build_history_compactor_input(
    *,
    history_text: str,
    focus_prompt: str | None,
    memory_file_path: str | None,
    current_task_progress: str | None,
) -> str:
    prompt_lines: list[str] = []
    normalized_focus_prompt = (focus_prompt or "").strip()
    if normalized_focus_prompt:
        prompt_lines.extend(
            [
                "Focus prompt for this compaction:",
                normalized_focus_prompt,
                "",
                "Conversation history to compress:",
                "",
            ]
        )
    if memory_file_path:
        prompt_lines.extend(
            [
                "Memory file to update:",
                memory_file_path,
                "",
                "Read that file fully, then update that same file with patch_file so it becomes the merged durable memory after this compaction.",
                "",
            ]
        )
    normalized_current_task_progress = (current_task_progress or "").strip()
    if normalized_current_task_progress:
        prompt_lines.extend(
            [
                "Current task progress from the client agent:",
                normalized_current_task_progress,
                "",
                "Use this to update the `Current Task Progress` section. Do not invent or rewrite that section beyond what this explicit progress update supports.",
                "",
            ]
        )
    else:
        prompt_lines.extend(
            [
                "Current task progress from the client agent:",
                "<none provided>",
                "",
                "Do not update the `Current Task Progress` section unless the client agent explicitly provides a progress update.",
                "",
            ]
        )
    if history_text.strip():
        prompt_lines.extend(
            [
                "Conversation history to compress:",
                "",
                history_text,
            ]
        )
    return "\n".join(prompt_lines).strip()


def _find_preserved_user_event_index(
    session_events: list[Event], user_content: types.Content | None
) -> int | None:
    if user_content is None:
        return None

    target_signature = _content_signature(user_content)
    for index in range(len(session_events) - 1, -1, -1):
        event = session_events[index]
        if event.author != "user" or not event.content:
            continue
        if _content_signature(event.content) == target_signature:
            return index
    return None


def _recent_compaction_exists(session_events: list[Event], session_state: dict[str, object] | None) -> bool:
    if not session_events or not session_state:
        return False
    last_event_count = session_state.get(_LAST_COMPACTION_EVENT_COUNT_STATE_KEY)
    if not isinstance(last_event_count, int):
        return False
    return len(session_events) - last_event_count <= _DEFAULT_COMPACTION_DEBOUNCE_TAIL_EVENTS


def _event_serialized_chars(event: Event) -> int:
    if event.content is None:
        return 0
    return len(_serialize_contents([event.content]))


def _select_events_to_compact(
    session_events: list[Event],
    *,
    session_state: dict[str, object] | None,
    user_content: types.Content | None,
    current_usage_tokens: int,
    context_limit: int,
    keep_ratio: float,
) -> list[Event]:
    from google.adk.apps.compaction import (
        _pending_function_call_ids,
        _safe_token_compaction_split_index,
        _truncate_events_before_pending_function_call,
    )

    if not session_events:
        return []

    preserved_index = _find_preserved_user_event_index(session_events, user_content)
    last_compaction_end = 0.0
    if session_state:
        last_compaction_end_value = session_state.get(_LAST_COMPACTION_END_TS_STATE_KEY)
        if isinstance(last_compaction_end_value, (int, float)):
            last_compaction_end = float(last_compaction_end_value)
    candidate_events = [
        event
        for index, event in enumerate(session_events)
        if event.content is not None
        and index != preserved_index
        and event.timestamp > last_compaction_end
    ]
    if len(candidate_events) <= _DEFAULT_COMPACTION_MIN_RETAINED_EVENTS:
        return []

    target_keep_tokens = min(
        max(int(context_limit * keep_ratio), 1),
        max(int(current_usage_tokens * _DEFAULT_COMPACTION_CURRENT_USAGE_KEEP_RATIO), 1),
    )
    if target_keep_tokens <= 0:
        return []

    avg_tokens_per_event = max(current_usage_tokens / max(len(candidate_events), 1), 1.0)
    event_retention_size = int(round(target_keep_tokens / avg_tokens_per_event))
    event_retention_size = max(_DEFAULT_COMPACTION_MIN_RETAINED_EVENTS, event_retention_size)
    event_retention_size = min(event_retention_size, len(candidate_events) - 1)
    if event_retention_size <= 0:
        return []

    split_index = _safe_token_compaction_split_index(
        candidate_events=candidate_events,
        event_retention_size=event_retention_size,
    )
    events_to_compact = candidate_events[:split_index]
    if len(events_to_compact) < _DEFAULT_COMPACTION_MIN_EVENTS_TO_COMPACT:
        return []

    pending_ids = _pending_function_call_ids(session_events)
    events_to_compact = _truncate_events_before_pending_function_call(events_to_compact, pending_ids)
    if len(events_to_compact) < _DEFAULT_COMPACTION_MIN_EVENTS_TO_COMPACT:
        return []
    return events_to_compact


def _build_compaction_summary_input(
    session_events: list[Event],
    session_state: dict[str, object] | None,
    events_to_compact: list[Event],
) -> tuple[list[types.Content], float, float]:
    summary_contents: list[types.Content] = []
    compaction_start = events_to_compact[0].timestamp
    last_compaction_end_value = None if session_state is None else session_state.get(
        _LAST_COMPACTION_END_TS_STATE_KEY
    )
    if isinstance(last_compaction_end_value, (int, float)):
        compaction_start = float(last_compaction_end_value)

    for event in events_to_compact:
        if event.content is not None:
            summary_contents.append(copy.deepcopy(event.content))

    return summary_contents, compaction_start, events_to_compact[-1].timestamp


def _write_compaction_archive(
    *,
    agent_name: str,
    reason: str,
    focus_prompt: str | None,
    memory_text: str,
    memory_file_path: str | None,
    compacted_event_count: int,
    retained_recent_event_count: int,
    current_usage_tokens: int,
    context_limit: int,
) -> str | None:
    workspace = _env_value("CODING_AGENT_WORKSPACE")
    if not workspace:
        return None

    workspace_path = Path(workspace)
    archive_dir = workspace_path / _DEFAULT_COMPACTION_ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"{stamp}-{agent_name}.md"
    normalized_focus_prompt = (focus_prompt or "").strip()
    archive_path.write_text(
        "\n".join(
            [
                "# Memory Compaction Snapshot",
                f"Agent: {agent_name}",
                f"Timestamp: {stamp}",
                f"Reason: {reason}",
                f"Focus prompt: {normalized_focus_prompt}" if normalized_focus_prompt else "Focus prompt: <none>",
                f"Memory file: {memory_file_path}" if memory_file_path else "Memory file: <none>",
                f"Current usage tokens: {current_usage_tokens}",
                f"Context limit: {context_limit}",
                f"Compacted event count: {compacted_event_count}",
                f"Retained recent event count: {retained_recent_event_count}",
                "",
                "## Memory File Snapshot",
                memory_text.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        return str(archive_path.relative_to(workspace_path))
    except ValueError:
        return str(archive_path)


async def _apply_history_compaction(
    callback_context,
    *,
    reason: str,
    focus_prompt: str | None,
    current_task_progress: str | None,
    current_usage_tokens: int,
    context_limit: int,
    keep_ratio: float,
) -> dict[str, object] | None:
    memory_path, memory_rel_path = _ensure_memory_file(callback_context.session.state)
    if memory_path is None or memory_rel_path is None:
        return None

    events_to_compact = _select_events_to_compact(
        callback_context.session.events,
        session_state=callback_context.session.state,
        user_content=callback_context.user_content,
        current_usage_tokens=current_usage_tokens,
        context_limit=context_limit,
        keep_ratio=keep_ratio,
    )
    if not events_to_compact:
        return None

    summary_contents, compaction_start, compaction_end = _build_compaction_summary_input(
        callback_context.session.events,
        callback_context.session.state,
        events_to_compact,
    )
    compacted_history = await _run_history_compactor(
        callback_context,
        summary_contents,
        focus_prompt=focus_prompt,
        memory_file_path=memory_rel_path,
        current_task_progress=current_task_progress,
    )
    if not compacted_history:
        return None
    compacted_history = _sanitize_compacted_history(compacted_history)
    if not compacted_history:
        return None

    retained_recent_event_count = len(
        [
            event
            for event in callback_context.session.events
            if event.content is not None
            and not (event.actions and event.actions.compaction)
            and event.timestamp > compaction_end
        ]
    )
    archive_path = _write_compaction_archive(
        agent_name=callback_context.agent_name,
        reason=reason,
        focus_prompt=focus_prompt,
        memory_text=compacted_history,
        memory_file_path=memory_rel_path,
        compacted_event_count=len(events_to_compact),
        retained_recent_event_count=retained_recent_event_count,
        current_usage_tokens=current_usage_tokens,
        context_limit=context_limit,
    )
    state_event = Event(
        invocation_id=callback_context.invocation_id,
        author=callback_context.agent_name,
        branch=callback_context.session.events[-1].branch if callback_context.session.events else None,
        actions=EventActions(
            state_delta={
                _MEMORY_FILE_STATE_KEY: memory_rel_path,
                _LAST_COMPACTION_END_TS_STATE_KEY: compaction_end,
                _LAST_COMPACTION_EVENT_COUNT_STATE_KEY: len(callback_context.session.events) + 1,
            },
        ),
    )
    await callback_context._invocation_context.session_service.append_event(
        session=callback_context.session,
        event=state_event,
    )
    return {
        "compacted_event_count": len(events_to_compact),
        "retained_recent_event_count": retained_recent_event_count,
        "archive_path": archive_path,
        "compaction_start": compaction_start,
        "compaction_end": compaction_end,
        "memory_file": memory_rel_path,
    }


async def compact_memory(
    reason: str,
    current_task_progress: str,
    prompt: str | None = None,
    tool_context=None,
) -> dict[str, object]:
    """Compact older conversation history while keeping a recent raw context window.

    Use this when you are at a clean task boundary, after extracting conclusions
    from a large amount of context, before reading a lot of new context, or
    before starting a long multi-step execution phase.

    This clears all prior raw conversation history except the top-level user
    instruction and the system instruction. Always provide a concise
    `current_task_progress` summary so the compactor can preserve the current
    execution state explicitly.

    Example input:
      reason="Finished repository investigation; about to start multi-file edits."
      current_task_progress="Repository structure is mapped. Next step: patch loader and trainer."

    Example input with focus prompt:
      current_task_progress="Attempt 2 failed in evaluator due to adapter import mismatch. Next step: patch adapter imports and rerun."
      prompt="Preserve evaluator assumptions, adapter contracts, and unresolved risks."

    Example output:
      {
        "status": "success",
        "compacted_event_count": 18,
        "retained_recent_event_count": 7,
        "archive_path": ".coding_agent_memory/compactions/20260329T120000Z-client_agent.md"
      }
    """
    from google.adk.apps.compaction import _latest_prompt_token_count

    context_limit = _env_int("CODING_AGENT_COMPACT_CONTEXT_LIMIT", _DEFAULT_COMPACTION_CONTEXT_LIMIT)
    keep_ratio = _env_float("CODING_AGENT_COMPACT_KEEP_RATIO", _DEFAULT_COMPACTION_KEEP_RATIO)
    current_usage_tokens = _latest_prompt_token_count(
        tool_context.session.events,
        current_branch=getattr(tool_context._invocation_context, "branch", None),
        agent_name=tool_context.agent_name,
    ) or 0
    result = await _apply_history_compaction(
        tool_context,
        reason=reason,
        focus_prompt=prompt,
        current_task_progress=current_task_progress,
        current_usage_tokens=current_usage_tokens,
        context_limit=context_limit,
        keep_ratio=keep_ratio,
    )
    if not result:
        return {
            "status": "noop",
            "reason": reason,
            "current_usage_tokens": current_usage_tokens,
            "output": "No eligible older history segment was compacted.",
        }
    return {
        "status": "success",
        "reason": reason,
        "current_usage_tokens": current_usage_tokens,
        **result,
        "output": (
            f"Compacted {result['compacted_event_count']} older events and retained "
            f"{result['retained_recent_event_count']} recent events."
        ),
    }


async def _maybe_compact_history(callback_context, llm_request):
    _refresh_openai_google_auth_token_if_needed()

    has_memory_file = bool(callback_context.session.state.get(_MEMORY_FILE_STATE_KEY))
    context_limit = _env_int("CODING_AGENT_COMPACT_CONTEXT_LIMIT", _DEFAULT_COMPACTION_CONTEXT_LIMIT)
    keep_ratio = _env_float("CODING_AGENT_COMPACT_KEEP_RATIO", _DEFAULT_COMPACTION_KEEP_RATIO)
    if context_limit <= 0:
        context_limit = _DEFAULT_COMPACTION_CONTEXT_LIMIT
    if keep_ratio <= 0:
        keep_ratio = _DEFAULT_COMPACTION_KEEP_RATIO
    keep_budget_tokens = max(1, int(context_limit * keep_ratio))

    from google.adk.apps.compaction import _latest_prompt_token_count

    current_usage_tokens = _latest_prompt_token_count(
        callback_context.session.events,
        current_branch=getattr(callback_context._invocation_context, "branch", None),
        agent_name=callback_context.agent_name,
    ) or 0
    current_usage_ratio = current_usage_tokens / context_limit

    if has_memory_file:
        _inject_memory_after_user_instruction(llm_request, callback_context.session.state)
        if current_usage_tokens > keep_budget_tokens:
            llm_request.contents = _trim_contents_to_recent_budget(
                list(getattr(llm_request, "contents", []) or []),
                context_limit=context_limit,
                keep_ratio=keep_ratio,
            )

    if _env_flag("CODING_AGENT_DEBUG_SYSTEM_INSTRUCTION", False):
        system_instruction = getattr(getattr(llm_request, "config", None), "system_instruction", None)
        if isinstance(system_instruction, str):
            preview = system_instruction[:400].replace("\n", "\\n")
            print(
                f"[{callback_context.agent_name} system_instruction] "
                f"len={len(system_instruction)} preview={preview}",
                flush=True,
            )
        else:
            print(
                f"[{callback_context.agent_name} system_instruction] "
                f"type={type(system_instruction).__name__}",
                flush=True,
            )

    trigger_ratio = _env_float("CODING_AGENT_COMPACT_TRIGGER_RATIO", _DEFAULT_COMPACTION_TRIGGER_RATIO)
    if trigger_ratio <= 0:
        trigger_ratio = _DEFAULT_COMPACTION_TRIGGER_RATIO

    if current_usage_ratio < trigger_ratio:
        return None
    if _recent_compaction_exists(
        callback_context.session.events,
        callback_context.session.state,
    ):
        return None

    before_contents = list(getattr(llm_request, "contents", []) or [])
    before_content_count = len(before_contents)
    before_text_chars = _count_content_text_chars(before_contents)
    before_serialized_chars = _count_serialized_content_chars(before_contents)

    result = await _apply_history_compaction(
        callback_context,
        reason=(
            f"Automatic safety compaction triggered at context usage "
            f"{current_usage_ratio:.1%} before the next model call."
        ),
        focus_prompt=None,
        current_task_progress=None,
        current_usage_tokens=current_usage_tokens,
        context_limit=context_limit,
        keep_ratio=keep_ratio,
    )
    if not result:
        return None

    _inject_memory_after_user_instruction(llm_request, callback_context.session.state)
    llm_request.contents = _trim_contents_to_recent_budget(
        list(getattr(llm_request, "contents", []) or []),
        context_limit=context_limit,
        keep_ratio=keep_ratio,
    )
    after_contents = list(llm_request.contents or [])
    after_content_count = len(after_contents)
    after_text_chars = _count_content_text_chars(after_contents)
    after_serialized_chars = _count_serialized_content_chars(after_contents)

    print(
        f"[{callback_context.agent_name} compact_history] "
        f"context_limit={context_limit} "
        f"current_usage_tokens={current_usage_tokens} "
        f"current_usage_ratio={current_usage_ratio:.3f} "
        f"compacted_event_count={result['compacted_event_count']} "
        f"retained_recent_event_count={result['retained_recent_event_count']} "
        f"memory_file={result['memory_file']} "
        f"keep_ratio={keep_ratio:.2f} "
        f"before_contents={before_content_count} "
        f"before_text_chars={before_text_chars} "
        f"before_serialized_chars={before_serialized_chars} "
        f"after_contents={after_content_count} "
        f"after_text_chars={after_text_chars} "
        f"after_serialized_chars={after_serialized_chars}",
        flush=True,
    )
    return None


async def _refresh_auth_before_model(callback_context, llm_request):
    _refresh_openai_google_auth_token_if_needed()
    return None


def _print_model_thoughts(callback_context, llm_response):
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        finish_reason = getattr(llm_response, "finish_reason", None) if llm_response else None
        error_message = getattr(llm_response, "error_message", None) or ""
        if finish_reason == types.FinishReason.MAX_TOKENS and (
            "truncated" in error_message.lower() or "max" in error_message.lower()
        ):
            max_output_tokens = None
            invocation_context = getattr(callback_context, "_invocation_context", None)
            agent = getattr(invocation_context, "agent", None)
            config = getattr(agent, "generate_content_config", None)
            if config is not None:
                max_output_tokens = getattr(config, "max_output_tokens", None)
            return llm_response.model_copy(
                update={
                    "content": types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(
                                name="report_truncated_model_output",
                                args={
                                    "error_message": error_message or "Model output exceeded max_output_tokens.",
                                    "max_output_tokens": max_output_tokens,
                                },
                            )
                        ],
                    ),
                    "error_code": None,
                    "error_message": None,
                    "finish_reason": types.FinishReason.STOP,
                }
            )
        return None

    function_calls = []
    for part in llm_response.content.parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            function_calls.append(function_call)

    if _env_flag("CODING_AGENT_PRINT_TOOL_CALL_COUNTS", True):
        tool_names = [fc.name or "<unknown>" for fc in function_calls]
        print(f"[{callback_context.agent_name} tool_calls] count={len(function_calls)} names={tool_names}")
        sys.stdout.flush()

    thought_parts = [part for part in llm_response.content.parts if getattr(part, "thought", False)]

    if not _env_flag("CODING_AGENT_PRINT_THOUGHTS", True):
        for part in thought_parts:
            if part.text:
                part.text = None
        return None

    thought_texts: list[str] = []
    thought_signatures = 0
    for part in thought_parts:
        if getattr(part, "thought_signature", None) is not None:
            thought_signatures += 1
        if part.text:
            thought_texts.append(part.text)

    if not thought_texts and not thought_signatures:
        return None

    print(f"\n[{callback_context.agent_name} thoughts]")
    for text in thought_texts:
        print(text)

    usage = getattr(llm_response, "usage_metadata", None)
    thoughts_token_count = (
        usage.thoughts_token_count if usage and usage.thoughts_token_count is not None else None
    )
    if thought_signatures or thoughts_token_count is not None:
        stats: list[str] = []
        if thoughts_token_count is not None:
            stats.append(f"thought_tokens={thoughts_token_count}")
        if thought_signatures:
            stats.append(f"thought_signatures={thought_signatures}")
        print(f"[{callback_context.agent_name} thoughts meta] {' '.join(stats)}")

    for part in thought_parts:
        if part.text:
            part.text = None

    sys.stdout.flush()
    return None


_COMMON_CONFIG = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            initial_delay=_env_float("CODING_AGENT_RETRY_INITIAL_DELAY", 0.1),
            max_delay=_env_float("CODING_AGENT_RETRY_MAX_DELAY", 0.5),
            attempts=_env_int("CODING_AGENT_RETRY_ATTEMPTS", 100),
        ),
    ),
    max_output_tokens=_env_int("CODING_AGENT_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_OUTPUT_TOKENS),
    thinking_config=types.ThinkingConfig(
        include_thoughts=_env_flag("CODING_AGENT_INCLUDE_THOUGHTS", True),
        thinking_level=_env_thinking_level(),
        thinking_budget=_env_thinking_budget(),
    ),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True,
    ),
)

_INVESTIGATOR_CONFIG = types.GenerateContentConfig(
    http_options=_COMMON_CONFIG.http_options,
    max_output_tokens=_COMMON_CONFIG.max_output_tokens,
    thinking_config=types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.LOW,
        thinking_budget=_env_thinking_budget(),
    ),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True,
    ),
)

_COMPACTOR_CONFIG = types.GenerateContentConfig(
    http_options=_COMMON_CONFIG.http_options,
    max_output_tokens=_COMMON_CONFIG.max_output_tokens,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True,
    ),
)


history_compactor = LlmAgent(
    name="history_compactor",
    model=_agent_model(),
    instruction="""Update one durable memory file for the coding task.

The input will provide:
- a memory file path that must be updated in place,
- an optional focus prompt,
- an optional current task progress update from the client agent,
- and the conversation history that should be compressed.

Workflow:
1. Read the memory file fully.
2. Merge the old durable memory with the new conversation history.
3. Update the same memory file with `patch_file`.
4. If the first patch is incomplete, read the file again and continue patching.

This is a multi-step editing agent, not a single-call transformer.
It may read and patch the same memory file multiple times before stopping.
Stop only when the memory file itself is in the desired final state.

What to preserve:
- durable context that will still matter after older raw history is removed,
- important repository structure,
- important file snippets or references,
- important findings and decisions,
- the current task progress and unresolved risks.

`Current Task Progress` is special:
- Do not infer or rewrite it on your own from the conversation history.
- Only update that section when the client agent explicitly provides a current task progress update.
- If no explicit progress update is provided, leave that section unchanged.

What to remove:
- duplicate facts,
- stale low-signal chatter,
- transient tool noise,
- text that does not matter after older raw history is removed.

If a focus prompt is present, prioritize preserving the information that best supports that focus.

Hard constraints:
- Write neutral durable memory, not an assistant continuation.
- Do not write in first person.
- Do not emit tool invocations or quoted assistant drafts into the memory file.
- Keep the file concise and merged forward; do not append a new block each time.
- Keep the memory as normal markdown. Reuse and update the existing file structure instead of rewriting it into a new rigid format unless the current structure is clearly broken.

After the file is satisfactory, return a brief confirmation.""",
    description="Compacts long-running execution history into a dense, high-fidelity summary.",
    tools=[read_file, patch_file],
    before_model_callback=_refresh_auth_before_model,
    generate_content_config=_COMPACTOR_CONFIG,
)


class VerifierInput(BaseModel):
    delivery_folder_path: str = Field(
        description="Workspace-relative folder path containing the deliverable to verify."
    )
    summary: str = Field(
        description="Short summary of what was changed or what should be verified."
    )
    user_instruction: str = Field(
        description=(
            "The original top-level acceptance target for the task, injected automatically "
            "from the current run after @file expansion. Use it as the verification target."
        )
    )


def _inject_verifier_user_instruction(tool, args, tool_context):
    tool_name = getattr(tool, "name", "")

    if tool_name in {"write_todos", "read_todos"}:
        if not args.get("agent_name"):
            args["agent_name"] = tool_context.agent_name
        if not args.get("workspace_root"):
            session_state = getattr(tool_context._invocation_context.session, "state", {}) or {}
            rooted_workspace = session_state.get(_ROOT_WORKSPACE_STATE_KEY)
            args["workspace_root"] = str(rooted_workspace or get_workspace_root())

    if isinstance(tool, StreamingAgentTool) and getattr(tool, "agent", None) is verifier:
        if not args.get("user_instruction"):
            args["user_instruction"] = _extract_user_instruction_text(tool_context.user_content)

    return None


codebase_investigator = LlmAgent(
    name="codebase_investigator",
    model=_agent_model(),
    instruction="""You are a read-only CLI investigation agent specializing in repository research.

# Core Mandates

- **Read-Only:** Do not edit files and do not run mutating shell commands.
- **Conventions:** Infer behavior from actual repository evidence, not from generic assumptions.
- **Scope Control:** Stay tightly scoped to the request. Do not drift into broad exploration.
- **High-Signal Handoff:** Return only the excerpts that materially support the answer.

# Primary Workflow

Operate using **Research -> Selection -> Handoff**.

1. **Research:** Map the relevant code paths and validate assumptions with the available read tools.
   - Use `keyword_search` for ranked multi-keyword discovery inside a target folder.
   - Use `list_files` to understand file structure and candidate modules.
   - Use `search_text` as the default grep-like search tool.
   - `search_text` already returns the matched line plus the surrounding file-content slice controlled by `context_lines`. Use it first for search with context before doing separate reads.
   - Use `read_file` only when you need exact file content or a larger, more precise range.
   - Search and read in parallel when the requests are independent.

2. **Selection:** Decide which evidence actually matters.
   - Prefer small, targeted excerpts from files you already read.
   - You may return multiple slices from the same file and slices from multiple different files.
   - Do not include long excerpts when a smaller slice is enough.

3. **Handoff:** Return a compact evidence package.
   - Call `format_codebase_output` exactly once.
   - Put the selected file slices in that call.
   - The summary must be especially short and should contain only the highest-signal conclusion.

# Output Contract

- Use normal assistant text only for minimal coordination.
- The real final handoff must come through `format_codebase_output`.
""",
    description="Read-only codebase investigator that searches, reads, and returns selected excerpts plus a short summary.",
    tools=[
        keyword_search,
        list_files,
        read_file,
        search_text,
        format_codebase_output,
    ],
    before_model_callback=_refresh_auth_before_model,
    generate_content_config=_INVESTIGATOR_CONFIG,
)


planner = LlmAgent(
    name="planner",
    model=_agent_model(),
    instruction="""You are a planning-focused CLI agent specializing in execution-ready implementation plans.

# Core Mandates

- **No Edits:** Do not edit files.
- **Evidence First:** Base the plan on repository evidence whenever structure or behavior matters.
- **Conventions:** Never guess project conventions, file layout, or implementation details when you can verify them.
- **Validation:** Verification is part of the plan, not an optional follow-up.

# Primary Workflow

Operate using **Research -> Strategy -> Plan Handoff**.

1. **Research:** Gather only the evidence needed to plan correctly.
   - If the plan depends on actual repository structure or behavior, call `codebase_investigator` with a narrow reading task.
   - Use `keyword_search` directly when broad discovery is enough before deeper investigation.

2. **Strategy:** Turn the request into a grounded implementation path.
    - Produce a concrete ordered sequence of steps.
    - Include verification steps as part of the execution path.
    - Call out assumptions, dependencies, and risks only when they materially affect the plan.
    - If the planning thread has accumulated a lot of stale context and you are
      at a clean planning boundary, you may call `compact_memory` before continuing.

3. **Todo Synchronization:** Keep the execution plan reflected in the todo list.
   - You must update your own agent-specific todo list for every planning task by calling `write_todos`.
   - The todo list must stay short, execution-oriented, and have exactly one item `in_progress` whenever work remains.
   - Before finishing, make sure your todo list matches the plan you are returning.

# Output Contract

- Return a concrete ordered plan that another agent can execute directly.
- Include verification steps.
- Your reply must explicitly state that you updated the todo list.
""",
    description="Planning-focused agent that can gather repository evidence and return an execution-ready plan.",
    tools=[
        keyword_search,
        compact_memory,
        write_todos,
        read_todos,
        StreamingAgentTool(agent=codebase_investigator),
    ],
    before_model_callback=_refresh_auth_before_model,
    generate_content_config=_COMMON_CONFIG,
    after_model_callback=_print_model_thoughts,
    before_tool_callback=_inject_verifier_user_instruction,
)


verifier = LlmAgent(
    name="verifier",
    model=_agent_model(),
    instruction="""You are a verification-focused coding agent.

Core mandates:
- Treat verification as evidence gathering, not as a formality.
- Base conclusions only on actual repository state, tool output, and command results.
- Stay tightly scoped to the change or claim being verified.
- Treat `user_instruction` as the acceptance target for verification. In your
  reasoning and reporting, think of it as the task's acceptance target, not as
  an instruction to execute.
- Your job is to determine whether the delivered work satisfies
  the acceptance target, not to obey it as a fresh task request.
- Use `delivery_folder_path` as the starting scope for verification and use
  `summary` as a concise description of what should be checked.
- Do not call planner.
- If you use todos, they refer to your own verifier-specific todo list.
- Do not edit repository files. Your role is to verify and report problems, not
  to implement fixes.

Primary workflow:
- Operate using Research -> Verify -> Report.
- Verify from fine-grained checks to coarse-grained checks.

1. Research.
    - Start from `delivery_folder_path`.
    - Use `summary` to identify the most relevant files, symbols, and checks.
    - Read the exact files, ranges, and command outputs needed to verify the work.
    - Use list_files, search_text, and read_file instead of shell-based reading when possible.
    - If multiple verification reads are independent, issue them in parallel.
    - If verification is clearly moving to a new phase and the old context has
      become mostly stale, you may call `compact_memory`.

2. Verify.
   - Start with the narrowest meaningful checks first:
     changed files, exact symbols, local assertions, focused test targets.
   - Then expand to broader checks only when needed:
     module-level tests, integration checks, lint, type-check, build, or runtime checks.
   - Prefer the repository's actual tests, lint, type-check, build, or runtime checks.
   - Use bash_exec for verification commands, test runs, builds, lint, type-check, and local CLIs. It waits in the foreground for up to 30 seconds by default and, if the command is still running after that window, leaves it in the background and returns a session_id.
   - Use read_command_session, wait, and terminate_command_session to monitor or stop background shell work.
   - Use check_running_sessions when you need to inspect active background jobs before starting overlapping work or when resuming after a pause.
   - Pay attention to details: exact file ranges, command exit codes, stderr,
     partial failures, flaky output, workspace boundaries, and whether a failure
     is caused by the change, the environment, or a pre-existing issue.
   - If a verification step fails, inspect the failure carefully before making any claim.
   - Judge the delivery against the acceptance target, `delivery_folder_path`,
     and `summary` as verification inputs. Do not reinterpret them as
     instructions to continue implementation.

3. Report.
   - Return a concise verification report.
   - State what was checked, whether the delivery satisfies the acceptance
     target, what passed, what failed, and any remaining risk.
   - If verification could not be completed, say exactly why.

Execution rules:
- Prefer structured tools over shell wrappers when both can do the job.
- Use bash sessions for command execution and verification.
- Do not make broad new implementation changes under the guise of verification.
- Do not continue implementing the user request. Verify the existing delivery.
""",
    description="Verification-focused coding agent that inspects files, runs checks, and returns a concise verification report.",
    input_schema=VerifierInput,
    tools=[
        list_files,
        compact_memory,
        read_file,
        write_todos,
        read_todos,
        search_text,
        bash_exec,
        check_running_sessions,
        wait,
        read_command_session,
        terminate_command_session,
    ],
    before_model_callback=_refresh_auth_before_model,
    generate_content_config=_COMMON_CONFIG,
    after_model_callback=_print_model_thoughts,
    before_tool_callback=_inject_verifier_user_instruction,
)


_CLIENT_AGENT_SELF_DELEGATION_BLOCK = """Subagent Usage Guideline:
- You may call `client_subagent` when a subtask is clearly scoped and can be executed with a mostly independent context window.
- You may call `client_subagent` when a subtask is clearly scoped and can run with a mostly independent context window.
- Delegation is appropriate when one branch of work is a self-contained investigation, implementation, or verification task; when the subtask has a clear deliverable or acceptance condition; when it would otherwise bloat your own working context; or when the parent task only needs a concrete result back in order to continue.
- Delegate only when the boundary is crisp. The child task should have a clear objective, scope, relevant paths, constraints, and expected output.
- Do not delegate tiny steps, tightly coupled edits in the same local reasoning thread, or urgent blocking work that you can finish faster yourself.
- When you do delegate, give the child coding agent enough context to succeed: the exact subtask, the relevant files or directories, the important constraints, and the form of result it should return.
"""


_CLIENT_AGENT_INSTRUCTION = f"""You are an AI coding assistant.
You should work like a strong software engineer. Reduce uncertainty quickly, keep the
active context small, make the smallest change that can work, and verify the
riskiest assumptions first.

Current context limit: {_CURRENT_CONTEXT_LIMIT} prompt tokens.
After compaction, only a small recent raw window is kept, currently about {round(_CURRENT_KEEP_RATIO * 100)}% of the max context budget, plus the top-level user instruction and working memory.

Operate using Intestigation -> Planning -> Execute -> Validate.

Intestigation:
Start with the cheapest high-signal evidence and build understanding
progressively. Prefer paths, symbols, tests, APIs, and concrete errors over
large raw blobs. Read only what you need for the next decision, and prefer
direct evidence over speculation.
For Python code, use `lsp_tool` when you need semantic navigation such as
definitions, references, hover information, or symbol search.

Planning:
For simple work, proceed directly. For multi-step, risky, or design-heavy work,
call planner. For substantial work, keep a short agent-specific todo list with
exactly one item in progress. Always aim for the smallest deliverable that
satisfies the task. If you call `compact_memory`, remember that it deletes all
older raw conversation history and keeps only the top-level user instruction
plus a small recent raw window, roughly 10% of the maximum context budget.
Before calling it, explicitly summarize the current task progress you will
still need after compaction and pass that summary as `current_task_progress`.

Execution:
Keep only the information needed for the next step. Distill large reads and
outputs into concise conclusions instead of carrying raw detail forward. Make
the smallest change that can plausibly solve the problem. Prefer changes that
are easy to verify, easy to revert, and consistent with the existing codebase.
Avoid unrelated cleanup, speculative abstraction, and broad rewrites. Fast
development means reading narrowly, changing narrowly, and keeping momentum.

Validation:
Validation is required for completion. Run the narrowest meaningful
verification first, then broaden only as needed. Prefer the project's real
tests, lint, type-check, build, or runtime checks over generic substitutes. For
non-trivial verification, call verifier before concluding. If verification
fails, inspect the failure carefully, fix what is feasible, and rerun the
relevant check. If verification cannot be completed, say exactly why and
describe the remaining risk. Fast debugging means reproducing concrete
failures, checking the highest-risk assumption first, and learning from actual
outputs instead of guessing. Fast testing means starting with the cheapest
high-signal check, then broadening only if needed.

Delivery:
Deliver quickly: produce a working result, prove the key behavior, avoid
unnecessary polish, and report the exact completion state and remaining risks.
Fast delivery means shipping a coherent result another engineer can trust:
the correct change, evidence it works, and a clear statement of remaining
risks or blockers. Finish only when the task is resolved or the remaining
blocker is explicit.

Available tools:
- `list_files`: List files in a directory when you need structure or file discovery.
- `lsp_tool`: Use Python semantic navigation for definitions, references, hover, and symbols.
- `report_truncated_model_output`: Compress older context, keep only the top-level user instruction plus about {round(_CURRENT_KEEP_RATIO * 100)}% of the max context as recent raw context, and therefore requires an explicit `current_task_progress` summary.
- `read_file`: Read file contents when you need direct evidence from code or text.
- `write_todos`: Update the current agent's todo list for multi-step work.
- `read_todos`: Read the current agent's todo list to recover or check progress.
- `search_text`: Search file contents by keyword or pattern across a scoped directory.
- `patch_file`: Make a focused edit after reading the file and replacing a specific existing snippet.
- `write_file`: Create a new file or fully rewrite an existing file after reading it end-to-end.
- `bash_exec`: Run terminal commands, wait in the foreground for up to 30 seconds by default, and continue in the background with a session_id if still running.
- `check_running_sessions`: See which background bash sessions are still running.
- `wait`: Wait on a running bash session and fetch fresh output.
- `read_command_session`: Read recent output from an existing bash session.
- `terminate_command_session`: Stop a running bash session when it is no longer needed.
- `planner`: Produce a concrete plan and update todos for complex work.
- `verifier`: Check whether the current delivery satisfies the acceptance target.
- `client_subagent`: Delegate a clearly scoped child coding task to a separate context window.

""" + _CLIENT_AGENT_SELF_DELEGATION_BLOCK


_CLIENT_SUBAGENT_INSTRUCTION = _CLIENT_AGENT_INSTRUCTION.replace(
    _CLIENT_AGENT_SELF_DELEGATION_BLOCK,
    "",
).replace(
    "- `client_subagent`: Delegate a clearly scoped child coding task to a separate context window.",
    "",
)

_PLANNER_TOOL = StreamingAgentTool(agent=planner)
_PLANNER_TOOL.description = (
    "Use planner for multi-step, risky, or design-heavy work when you need an "
    "ordered execution plan and an updated todo list."
)

_VERIFIER_TOOL = StreamingAgentTool(agent=verifier)
_VERIFIER_TOOL.description = (
    "Use verifier for non-trivial validation. It inspects the delivery, runs "
    "checks, and reports whether the current result satisfies the acceptance target."
)


_CLIENT_BASE_TOOLS = [
    list_files,
    lsp_tool,
    compact_memory,
    read_file,
    write_todos,
    read_todos,
    search_text,
    patch_file,
    write_file,
    bash_exec,
    check_running_sessions,
    wait,
    read_command_session,
    terminate_command_session,
    report_truncated_model_output,
    _PLANNER_TOOL,
    _VERIFIER_TOOL,
]


client_subagent = LlmAgent(
    name="client_subagent",
    model=_agent_model(),
    instruction=_CLIENT_SUBAGENT_INSTRUCTION,
    description="Delegated coding sub-agent for scoped investigation, implementation, and verification subtasks. Subagent is not able to access the file out of its workspace.",
    tools=list(_CLIENT_BASE_TOOLS),
    before_model_callback=_maybe_compact_history,
    before_tool_callback=_inject_verifier_user_instruction,
    generate_content_config=_COMMON_CONFIG,
    after_model_callback=_print_model_thoughts,
)


client_agent = LlmAgent(
    name="client_agent",
    model=_agent_model(),
    instruction=_CLIENT_AGENT_INSTRUCTION,
    description="Primary coding agent for reading the repository, editing files, running commands, verifying changes, and coordinating planning when needed.",
    tools=list(_CLIENT_BASE_TOOLS) + [
        WorkspaceScopedStreamingAgentTool(
            agent=client_subagent,
            description=(
                "Use client_subagent for a clearly scoped child coding task. "
                "Pass `request` for the delegated task, `workspace` for the child "
                "working root, and optional `restricted_mode=true` to ban "
                "out-of-workspace file and shell paths."
            ),
        )
    ],
    before_model_callback=_maybe_compact_history,
    before_tool_callback=_inject_verifier_user_instruction,
    generate_content_config=_COMMON_CONFIG,
    after_model_callback=_print_model_thoughts,
)
root_agent = client_agent
