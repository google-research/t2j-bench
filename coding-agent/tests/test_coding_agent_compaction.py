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

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from coding_agent import agent as gemini_agent


def _content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def _function_response_content(name: str, index: int) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part.from_function_response(
                name=name,
                response={"output": f"tool-result-{index}"},
            )
        ],
    )


def _function_call_content(name: str, call_id: str) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part.from_function_call(
                name=name,
                args={"x": 1},
            )
        ],
    ).model_copy(
        update={
            "parts": [
                types.Part(
                    function_call=types.FunctionCall(
                        name=name,
                        args={"x": 1},
                        id=call_id,
                    )
                )
            ]
        }
    )


def _function_response_with_id(name: str, call_id: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part.from_function_response(
                name=name,
                response={"output": "done"},
            )
        ],
    ).model_copy(
        update={
            "parts": [
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name,
                        response={"output": "done"},
                        id=call_id,
                    )
                )
            ]
        }
    )


def _event(author: str, role: str, text: str, timestamp: float) -> Event:
    return Event(
        author=author,
        invocation_id=f"inv-{timestamp}",
        timestamp=timestamp,
        content=_content(role, text),
    )


def test_select_events_to_compact_preserves_current_user_and_recent_suffix():
    user_content = _content("user", "Top-level task instruction")
    session_events = [_event("user", "user", "Top-level task instruction", 1.0)]
    for index in range(2, 18):
        session_events.append(
            _event(
                "model" if index % 2 == 0 else "user",
                "model" if index % 2 == 0 else "user",
                f"event-{index}",
                float(index),
            )
        )

    selected = gemini_agent._select_events_to_compact(
        session_events,
        session_state={},
        user_content=user_content,
        current_usage_tokens=220_000,
        context_limit=1_048_576,
        keep_ratio=0.10,
    )

    assert selected
    assert session_events[0] not in selected
    assert selected[0].timestamp == 2.0
    assert selected[-1].timestamp < session_events[-1].timestamp
    assert len(selected) >= gemini_agent._DEFAULT_COMPACTION_MIN_EVENTS_TO_COMPACT


def test_trim_contents_to_recent_budget_keeps_latest_user_and_recent_tail():
    contents = [_content("user", "Top-level task instruction")]
    for index in range(1, 9):
        contents.append(_content("model", f"model-{index}"))
        contents.append(_function_response_content("dummy_tool", index))

    trimmed = gemini_agent._trim_contents_to_recent_budget(
        contents,
        context_limit=400,
        keep_ratio=0.10,
    )

    assert trimmed[0].parts[0].text == "Top-level task instruction"
    assert len(trimmed) > 1
    assert any(content.role == "model" for content in trimmed[1:])
    model_texts = [content.parts[0].text for content in trimmed if content.role == "model"]
    assert model_texts[-1] == "model-8"


def test_trim_contents_to_recent_budget_keeps_tool_call_result_pairs():
    contents = [
        _content("user", "Top-level task instruction"),
        _content("model", "plain-model-1"),
        _function_response_content("dummy_tool", 1),
        _function_call_content("write_file", "tool-1"),
        _function_response_with_id("write_file", "tool-1"),
    ]

    trimmed = gemini_agent._trim_contents_to_recent_budget(
        contents,
        context_limit=200,
        keep_ratio=0.10,
    )

    assert trimmed[0].parts[0].text == "Top-level task instruction"
    retained_call_ids = set()
    retained_response_ids = set()
    for content in trimmed[1:]:
        retained_call_ids.update(gemini_agent._content_function_call_ids(content))
        retained_response_ids.update(gemini_agent._content_function_response_ids(content))
    assert retained_response_ids.issubset(retained_call_ids)
    assert "tool-1" in retained_call_ids
    assert "tool-1" in retained_response_ids


def test_build_history_compactor_input_separates_snapshots_and_history():
    text = gemini_agent._build_history_compactor_input(
        history_text="[1] role=model\ntext: recent history",
        focus_prompt="Preserve evaluator contracts.",
        memory_file_path=".coding_agent_memory/working_memory.md",
        current_task_progress="Attempt 2 failed in evaluator; next step is adapter import fix.",
    )

    assert "Focus prompt for this compaction:" in text
    assert "Preserve evaluator contracts." in text
    assert "Memory file to update:" in text
    assert ".coding_agent_memory/working_memory.md" in text
    assert "Current task progress from the client agent:" in text
    assert "Attempt 2 failed in evaluator; next step is adapter import fix." in text
    assert "Conversation history to compress:" in text
    assert "[1] role=model\ntext: recent history" in text


def test_apply_history_compaction_updates_memory_file_and_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("CODING_AGENT_WORKSPACE", str(tmp_path))
    captured_focus_prompt = {}

    async def _fake_run_history_compactor(
        callback_context,
        history_contents,
        focus_prompt=None,
        memory_file_path=None,
        current_task_progress=None,
    ):
        del history_contents
        captured_focus_prompt["value"] = focus_prompt
        captured_focus_prompt["progress"] = current_task_progress
        memory_path = tmp_path / str(memory_file_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            (
                "# Working Memory\n\n"
                "## Directory Structure\n- repo/\n\n"
                "## Important File Snippets\n- foo.py updated\n\n"
                "## Important Findings\n- Important progress.\n\n"
                "## Current Task Progress\n- Run verification\n"
            ),
            encoding="utf-8",
        )
        return memory_path.read_text(encoding="utf-8")

    monkeypatch.setattr(gemini_agent, "_run_history_compactor", _fake_run_history_compactor)

    user_content = _content("user", "Top-level task instruction")
    session_events = [_event("user", "user", "Top-level task instruction", 1.0)]
    for index in range(2, 18):
        session_events.append(
            _event(
                "model" if index % 2 == 0 else "user",
                "model" if index % 2 == 0 else "user",
                f"event-{index}",
                float(index),
            )
        )

    session = SimpleNamespace(events=session_events, state={})

    class _SessionService:
        async def append_event(self, session, event):
            session.events.append(event)
            if event.actions and event.actions.state_delta:
                session.state.update(event.actions.state_delta)

    invocation_context = SimpleNamespace(
        session_service=_SessionService(),
        branch=None,
    )
    callback_context = SimpleNamespace(
        agent_name="client_agent",
        session=session,
        user_content=user_content,
        invocation_id="test-invocation",
        _invocation_context=invocation_context,
    )

    result = asyncio.run(
        gemini_agent._apply_history_compaction(
            callback_context,
            reason="Finished research; about to start implementation.",
            focus_prompt=None,
            current_task_progress=None,
            current_usage_tokens=220_000,
            context_limit=1_048_576,
            keep_ratio=0.10,
        )
    )

    assert result is not None
    assert result["compacted_event_count"] >= gemini_agent._DEFAULT_COMPACTION_MIN_EVENTS_TO_COMPACT
    assert result["archive_path"]
    assert result["memory_file"] == gemini_agent._DEFAULT_MEMORY_FILE_REL_PATH
    assert session.state[gemini_agent._MEMORY_FILE_STATE_KEY] == gemini_agent._DEFAULT_MEMORY_FILE_REL_PATH
    assert session.state[gemini_agent._LAST_COMPACTION_END_TS_STATE_KEY] == result["compaction_end"]
    assert captured_focus_prompt["value"] is None
    assert captured_focus_prompt["progress"] is None

    memory_path = tmp_path / gemini_agent._DEFAULT_MEMORY_FILE_REL_PATH
    assert memory_path.exists()
    memory_text = memory_path.read_text(encoding="utf-8")
    assert "Important progress." in memory_text
    assert "## Important Findings" in memory_text
    archive_path = tmp_path / result["archive_path"]
    assert archive_path.exists()
    assert "Memory Compaction Snapshot" in archive_path.read_text(encoding="utf-8")
    archive_text = archive_path.read_text(encoding="utf-8")
    assert "Focus prompt: <none>" in archive_text
    assert "Memory file: .coding_agent_memory/working_memory.md" in archive_text


def test_maybe_compact_history_injects_memory_file_after_user_instruction(tmp_path: Path):
    memory_path = tmp_path / ".coding_agent_memory" / "working_memory.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        (
            "# Working Memory\n\n"
            "## Directory Structure\n- src/\n\n"
            "## Important File Snippets\n- later.py\n\n"
            "## Important Findings\n- Preserve evaluator contracts.\n\n"
            "## Current Task Progress\n- verify later.py\n"
        ),
        encoding="utf-8",
    )
    session = SimpleNamespace(
        events=[],
        state={
            gemini_agent._MEMORY_FILE_STATE_KEY: ".coding_agent_memory/working_memory.md",
            gemini_agent._ROOT_WORKSPACE_STATE_KEY: str(tmp_path),
        },
    )
    callback_context = SimpleNamespace(
        agent_name="client_agent",
        session=session,
        user_content=_content("user", "Top-level task instruction"),
        invocation_id="test-invocation",
        _invocation_context=SimpleNamespace(branch=None),
    )
    llm_request = LlmRequest(
        contents=[
            _content("user", "Top-level task instruction"),
            *[
                content
                for index in range(1, 9)
                for content in (_content("model", f"model-{index}"), _function_response_content("dummy_tool", index))
            ],
        ],
        config=types.GenerateContentConfig(system_instruction="Base system instruction."),
    )
    asyncio.run(gemini_agent._maybe_compact_history(callback_context, llm_request))

    system_instruction = llm_request.config.system_instruction
    assert isinstance(system_instruction, str)
    assert system_instruction == "Base system instruction."
    merged_user_text = llm_request.contents[0].parts[0].text
    assert merged_user_text is not None
    assert "## Working Memory Context" in merged_user_text
    assert "Preserve evaluator contracts." in merged_user_text
    assert "Top-level task instruction" in merged_user_text
    assert len(llm_request.contents) > 1
    assert any(content.role == "model" for content in llm_request.contents[1:])


def test_maybe_compact_history_does_not_trim_before_any_compaction():
    session = SimpleNamespace(events=[], state={})
    callback_context = SimpleNamespace(
        agent_name="client_agent",
        session=session,
        user_content=_content("user", "Top-level task instruction"),
        invocation_id="test-invocation",
        _invocation_context=SimpleNamespace(branch=None),
    )
    llm_request = LlmRequest(
        contents=[
            _content("user", "Top-level task instruction"),
            *[
                content
                for index in range(1, 9)
                for content in (_content("model", f"model-{index}"), _function_response_content("dummy_tool", index))
            ],
        ],
        config=types.GenerateContentConfig(system_instruction="Base system instruction."),
    )

    original_len = len(llm_request.contents)
    asyncio.run(gemini_agent._maybe_compact_history(callback_context, llm_request))

    assert llm_request.config.system_instruction == "Base system instruction."
    assert len(llm_request.contents) == original_len


def test_compact_memory_passes_focus_prompt_to_history_compactor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("CODING_AGENT_WORKSPACE", str(tmp_path))
    captured_focus_prompt = {}

    async def _fake_run_history_compactor(
        callback_context,
        history_contents,
        focus_prompt=None,
        memory_file_path=None,
        current_task_progress=None,
    ):
        del callback_context, history_contents
        captured_focus_prompt["value"] = focus_prompt
        captured_focus_prompt["progress"] = current_task_progress
        memory_path = tmp_path / str(memory_file_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            (
                "# Working Memory\n\n"
                "## Directory Structure\n- repo/\n\n"
                "## Important File Snippets\n- foo.py updated\n\n"
                "## Important Findings\n- Preserved evaluator assumptions.\n\n"
                "## Current Task Progress\n- Run verification\n"
            ),
            encoding="utf-8",
        )
        return memory_path.read_text(encoding="utf-8")

    monkeypatch.setattr(gemini_agent, "_run_history_compactor", _fake_run_history_compactor)

    user_content = _content("user", "Top-level task instruction")
    session_events = [_event("user", "user", "Top-level task instruction", 1.0)]
    for index in range(2, 18):
        session_events.append(
            _event(
                "model" if index % 2 == 0 else "user",
                "model" if index % 2 == 0 else "user",
                f"event-{index}",
                float(index),
            )
        )

    session = SimpleNamespace(events=session_events, state={})

    class _SessionService:
        async def append_event(self, session, event):
            session.events.append(event)
            if event.actions and event.actions.state_delta:
                session.state.update(event.actions.state_delta)

    tool_context = SimpleNamespace(
        agent_name="client_agent",
        session=session,
        user_content=user_content,
        invocation_id="test-invocation",
        _invocation_context=SimpleNamespace(session_service=_SessionService(), branch=None),
    )

    result = asyncio.run(
        gemini_agent.compact_memory(
            reason="Need to preserve evaluator contracts.",
            prompt="Preserve evaluator assumptions, adapter contracts, and unresolved risks.",
            current_task_progress="Attempt 2 failed in evaluator due to adapter import mismatch. Next step: patch adapter imports and rerun.",
            tool_context=tool_context,
        )
    )

    assert result["status"] == "success"
    assert "prompt" not in result
    assert captured_focus_prompt["value"] == "Preserve evaluator assumptions, adapter contracts, and unresolved risks."
    assert captured_focus_prompt["progress"] == (
        "Attempt 2 failed in evaluator due to adapter import mismatch. Next step: patch adapter imports and rerun."
    )
    assert session.state[gemini_agent._MEMORY_FILE_STATE_KEY] == gemini_agent._DEFAULT_MEMORY_FILE_REL_PATH
    archive_path = tmp_path / result["archive_path"]
    archive_text = archive_path.read_text(encoding="utf-8")
    assert "Focus prompt: Preserve evaluator assumptions, adapter contracts, and unresolved risks." in archive_text


def test_run_history_compactor_uses_session_memory_workspace_and_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    scoped_workspace = tmp_path / "subagent_workspace"
    memory_rel_path = ".coding_agent_memory/subagents/child-1/working_memory.md"
    memory_path = scoped_workspace / memory_rel_path
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text("# Child Memory\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class _FakeSessionService:
        async def create_session(self, app_name, user_id, state):
            captured["app_name"] = app_name
            captured["user_id"] = user_id
            captured["state"] = dict(state)
            return SimpleNamespace(user_id=user_id, id="fake-session")

    class _FakeRunner:
        def __init__(self, *, app_name, agent, session_service, memory_service, credential_service):
            del agent, session_service, memory_service, credential_service
            captured["runner_app_name"] = app_name
            self.session_service = _FakeSessionService()

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id, session_id, new_message
            captured["workspace_during_run"] = os.environ.get("CODING_AGENT_WORKSPACE")
            if False:
                yield None
            return

        async def close(self):
            return None

    monkeypatch.setattr(gemini_agent, "Runner", _FakeRunner)

    callback_context = SimpleNamespace(
        agent_name="client_subagent",
        user_id="test-user",
        session=SimpleNamespace(
            state={
                gemini_agent._ROOT_WORKSPACE_STATE_KEY: str(scoped_workspace),
                gemini_agent._MEMORY_FILE_STATE_KEY: memory_rel_path,
            }
        ),
        _invocation_context=SimpleNamespace(credential_service=None),
    )

    result = asyncio.run(
        gemini_agent._run_history_compactor(
            callback_context,
            history_contents=[_content("user", "compress this")],
            focus_prompt="Keep subagent-specific findings.",
            memory_file_path=memory_rel_path,
            current_task_progress="Subagent is investigating adapter imports.",
        )
    )

    assert result == "# Child Memory"
    assert captured["workspace_during_run"] == str(scoped_workspace)
    assert captured["state"][gemini_agent._ROOT_WORKSPACE_STATE_KEY] == str(scoped_workspace)
    assert captured["state"][gemini_agent._MEMORY_FILE_STATE_KEY] == memory_rel_path


def test_apply_history_compaction_still_updates_memory_when_already_below_keep_budget_after_prior_compaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("CODING_AGENT_WORKSPACE", str(tmp_path))
    memory_path = tmp_path / gemini_agent._DEFAULT_MEMORY_FILE_REL_PATH
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text("# Working Memory\n", encoding="utf-8")

    captured = {"called": False}

    async def _fake_run_history_compactor(
        callback_context,
        history_contents,
        focus_prompt=None,
        memory_file_path=None,
        current_task_progress=None,
    ):
        del callback_context, history_contents, focus_prompt, current_task_progress
        captured["called"] = True
        target = tmp_path / str(memory_file_path)
        target.write_text("# Updated Memory\n", encoding="utf-8")
        return target.read_text(encoding="utf-8")

    monkeypatch.setattr(gemini_agent, "_run_history_compactor", _fake_run_history_compactor)

    user_content = _content("user", "Top-level task instruction")
    session_events = [_event("user", "user", "Top-level task instruction", 1.0)]
    for index in range(2, 18):
        session_events.append(
            _event(
                "model" if index % 2 == 0 else "user",
                "model" if index % 2 == 0 else "user",
                f"event-{index}",
                float(index),
            )
        )

    class _SessionService:
        async def append_event(self, session, event):
            session.events.append(event)
            session.state.update(event.actions.state_delta or {})

    callback_context = SimpleNamespace(
        agent_name="client_agent",
        session=SimpleNamespace(
            events=session_events,
            state={
                gemini_agent._ROOT_WORKSPACE_STATE_KEY: str(tmp_path),
                gemini_agent._MEMORY_FILE_STATE_KEY: gemini_agent._DEFAULT_MEMORY_FILE_REL_PATH,
                gemini_agent._LAST_COMPACTION_END_TS_STATE_KEY: 5.0,
            },
        ),
        user_content=user_content,
        invocation_id="test-invocation",
        _invocation_context=SimpleNamespace(session_service=_SessionService(), branch=None),
    )

    result = asyncio.run(
        gemini_agent._apply_history_compaction(
            callback_context,
            reason="Second compaction should noop.",
            focus_prompt=None,
            current_task_progress="Still working.",
            current_usage_tokens=40_000,
            context_limit=1_048_576,
            keep_ratio=0.10,
        )
    )

    assert result is not None
    assert captured["called"] is True


def test_maybe_compact_history_does_not_trim_again_when_below_keep_budget(tmp_path: Path):
    memory_path = tmp_path / ".coding_agent_memory" / "working_memory.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "# Working Memory\n\n## Important Findings\n- Keep this memory.\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(
        events=[],
        state={
            gemini_agent._MEMORY_FILE_STATE_KEY: ".coding_agent_memory/working_memory.md",
            gemini_agent._ROOT_WORKSPACE_STATE_KEY: str(tmp_path),
        },
    )
    callback_context = SimpleNamespace(
        agent_name="client_agent",
        session=session,
        user_content=_content("user", "Top-level task instruction"),
        invocation_id="test-invocation",
        _invocation_context=SimpleNamespace(branch=None),
    )
    llm_request = LlmRequest(
        contents=[
            _content("user", "Top-level task instruction"),
            *[
                content
                for index in range(1, 5)
                for content in (_content("model", f"model-{index}"), _function_response_content("dummy_tool", index))
            ],
        ],
        config=types.GenerateContentConfig(system_instruction="Base system instruction."),
    )

    original_len = len(llm_request.contents)
    asyncio.run(gemini_agent._maybe_compact_history(callback_context, llm_request))

    assert len(llm_request.contents) == original_len
    merged_user_text = llm_request.contents[0].parts[0].text
    assert merged_user_text is not None
    assert "## Working Memory Context" in merged_user_text
