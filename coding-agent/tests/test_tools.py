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
import threading
from collections import deque
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import sys

import pytest

ADK_SRC = Path(__file__).resolve().parents[2] / "adk-python" / "src"
if ADK_SRC.exists():
    sys.path.insert(0, str(ADK_SRC))

from anthropic import types as anthropic_types
from google.adk.flows.llm_flows.functions import (
    _format_tool_result_for_model,
    _offload_large_tool_output,
)
from google.adk.models.anthropic_llm import AnthropicLlm, NOT_GIVEN
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from coding_agent.agent import _inject_verifier_user_instruction
from coding_agent import agent as coding_agent_module
from coding_agent import streaming_agent_tool as streaming_agent_tool_module
from tools import (
    activate_skill,
    bash_exec,
    check_running_sessions,
    format_codebase_output,
    keyword_search,
    list_files,
    list_skills,
    lsp_tool,
    patch_file,
    read_command_session,
    read_file,
    read_memory,
    read_todos,
    report_truncated_model_output,
    run_bash,
    save_memory,
    search_text,
    terminate_command_session,
    wait,
    write_file,
    write_todos,
)
from tools import command_sessions as command_sessions_module
from tools import memory_tools as memory_tools_module


@pytest.fixture(autouse=True)
def workspace_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CODING_AGENT_WORKSPACE", str(workspace))
    monkeypatch.setenv("CODING_AGENT_YOLO", "0")
    yield workspace
    command_sessions_module._cleanup_sessions()
    with command_sessions_module._SESSIONS_LOCK:
        command_sessions_module._SESSIONS.clear()


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "GEMINI.md"
    monkeypatch.setattr(memory_tools_module, "_MEMORY_DIR", memory_dir)
    monkeypatch.setattr(memory_tools_module, "_MEMORY_FILE", memory_file)
    yield


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _DummyToolContext:
    def __init__(self):
        self.state = {}


def test_anthropic_llm_uses_llm_request_max_output_tokens():
    captured: dict[str, object] = {}

    class _FakeMessage(SimpleNamespace):
        def model_dump_json(self, **kwargs):
            del kwargs
            return "{}"

    class _FakeClient:
        class _Messages:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return _FakeMessage(
                    content=[anthropic_types.TextBlock(text="ok", type="text")],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    stop_reason="end_turn",
                )

        @property
        def messages(self):
            return self._Messages()

    class _TestAnthropic(AnthropicLlm):
        @property
        def _anthropic_client(self):
            return _FakeClient()

    llm = _TestAnthropic(model="claude-opus-4-6", max_tokens=8192)
    request = LlmRequest(
        model="claude-opus-4-6",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
        config=types.GenerateContentConfig(max_output_tokens=32768),
    )

    async def _collect():
        return [response async for response in llm.generate_content_async(request, stream=False)]

    responses = asyncio.run(_collect())
    assert len(responses) == 1
    assert captured["max_tokens"] == 32768
    assert captured["thinking"] == {"type": "adaptive"}


def test_anthropic_llm_only_enables_adaptive_thinking_for_claude_46():
    captured: dict[str, object] = {}

    class _FakeMessage(SimpleNamespace):
        def model_dump_json(self, **kwargs):
            del kwargs
            return "{}"

    class _FakeClient:
        class _Messages:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return _FakeMessage(
                    content=[anthropic_types.TextBlock(text="ok", type="text")],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    stop_reason="end_turn",
                )

        @property
        def messages(self):
            return self._Messages()

    class _TestAnthropic(AnthropicLlm):
        @property
        def _anthropic_client(self):
            return _FakeClient()

    llm = _TestAnthropic(model="claude-3-7-sonnet-20250219", max_tokens=8192)
    request = LlmRequest(
        model="claude-3-7-sonnet-20250219",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
        config=types.GenerateContentConfig(max_output_tokens=8192),
    )

    async def _collect():
        return [response async for response in llm.generate_content_async(request, stream=False)]

    responses = asyncio.run(_collect())
    assert len(responses) == 1
    assert captured["thinking"] is NOT_GIVEN


def test_anthropic_llm_falls_back_to_streaming_for_long_request_error():
    class _FakeClient:
        class _Messages:
            async def create(self, **kwargs):
                del kwargs
                raise ValueError(
                    "Streaming is required for operations that may take longer than 10 minutes. "
                    "See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details"
                )

        @property
        def messages(self):
            return self._Messages()

    class _TestAnthropic(AnthropicLlm):
        @property
        def _anthropic_client(self):
            return _FakeClient()

    llm = _TestAnthropic(model="claude-opus-4-6", max_tokens=8192)
    request = LlmRequest(
        model="claude-opus-4-6",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
        config=types.GenerateContentConfig(max_output_tokens=32768),
    )

    final = LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text="ok")]),
        partial=False,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=2,
        ),
    )

    async def _fake_stream(*args, **kwargs):
        del args, kwargs
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part.from_text(text="o")]),
            partial=True,
        )
        yield final

    llm._generate_content_streaming = _fake_stream  # type: ignore[method-assign]

    async def _collect():
      return [response async for response in llm.generate_content_async(request, stream=False)]

    responses = asyncio.run(_collect())
    assert len(responses) == 1
    assert responses[0].partial is False
    assert responses[0].content.parts[0].text == "ok"


def test_anthropic_llm_nonstream_tool_request_uses_internal_streaming():
    dummy_tool = BaseTool(name="dummy", description="dummy tool")

    class _FakeClient:
        class _Messages:
            async def create(self, **kwargs):
                raise AssertionError("non-stream messages.create should not be used for tool-enabled requests")

        @property
        def messages(self):
            return self._Messages()

    class _TestAnthropic(AnthropicLlm):
        @property
        def _anthropic_client(self):
            return _FakeClient()

    llm = _TestAnthropic(model="claude-opus-4-6", max_tokens=8192)
    request = LlmRequest(
        model="claude-opus-4-6",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
        config=types.GenerateContentConfig(max_output_tokens=32768),
        tools_dict={"dummy": dummy_tool},
    )

    final = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="dummy", args={"x": 1})],
        ),
        partial=False,
    )

    async def _fake_collect(*args, **kwargs):
        del args, kwargs
        return final

    llm._collect_streaming_final_response = _fake_collect  # type: ignore[method-assign]

    async def _collect():
        return [response async for response in llm.generate_content_async(request, stream=False)]

    responses = asyncio.run(_collect())
    assert len(responses) == 1
    assert responses[0].partial is False
    assert responses[0].content.parts[0].function_call.name == "dummy"


def test_anthropic_streaming_emits_final_response_on_tool_use_stop_reason():
    class _FakeMessages:
        async def create(self, **kwargs):
            del kwargs

            async def _agen():
                yield SimpleNamespace(
                    type="message_start",
                    message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=1)),
                )
                yield SimpleNamespace(
                    type="content_block_start",
                    index=0,
                    content_block=anthropic_types.TextBlock(text="", type="text"),
                )
                yield SimpleNamespace(
                    type="content_block_delta",
                    index=0,
                    delta=anthropic_types.TextDelta(text="Let's test:", type="text_delta"),
                )
                yield SimpleNamespace(
                    type="content_block_start",
                    index=1,
                    content_block=anthropic_types.ToolUseBlock(
                        id="toolu_123",
                        name="bash_exec",
                        input={},
                        type="tool_use",
                    ),
                )
                yield SimpleNamespace(
                    type="content_block_delta",
                    index=1,
                    delta=anthropic_types.InputJSONDelta(
                        partial_json='{"command":"echo hi"}',
                        type="input_json_delta",
                    ),
                )
                yield SimpleNamespace(
                    type="message_delta",
                    usage=SimpleNamespace(output_tokens=20),
                    delta=SimpleNamespace(stop_reason="tool_use"),
                )
                yield SimpleNamespace(type="message_stop")

            return _agen()

    class _FakeClient:
        @property
        def messages(self):
            return _FakeMessages()

    class _TestAnthropic(AnthropicLlm):
        @property
        def _anthropic_client(self):
            return _FakeClient()

    llm = _TestAnthropic(model="claude-opus-4-6", max_tokens=8192)
    request = LlmRequest(
        model="claude-opus-4-6",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
        config=types.GenerateContentConfig(max_output_tokens=32768),
    )

    async def _collect():
        return [response async for response in llm._generate_content_streaming(request, [], NOT_GIVEN, NOT_GIVEN)]

    responses = asyncio.run(_collect())
    assert len(responses) == 2
    assert responses[0].partial is True
    assert responses[1].partial is False
    parts = responses[1].content.parts
    assert parts[0].text == "Let's test:"
    assert parts[1].function_call.name == "bash_exec"
    assert parts[1].function_call.args["command"] == "echo hi"


def test_after_model_callback_converts_max_tokens_truncation_into_tool_call(workspace_env: Path):
    llm_response = LlmResponse(
        finish_reason=coding_agent_module.types.FinishReason.MAX_TOKENS,
        error_code=str(coding_agent_module.types.FinishReason.MAX_TOKENS),
        error_message=(
            "Tool call arguments were truncated while streaming and could not be "
            "parsed as valid JSON. Increase `max_output_tokens` and retry."
        ),
    )
    callback_context = SimpleNamespace(
        _invocation_context=SimpleNamespace(
            agent=SimpleNamespace(
                generate_content_config=SimpleNamespace(max_output_tokens=32768)
            )
        )
    )

    altered = coding_agent_module._print_model_thoughts(callback_context, llm_response)

    assert altered is not None
    assert altered.error_code is None
    assert altered.error_message is None
    assert altered.finish_reason == coding_agent_module.types.FinishReason.STOP
    function_call = altered.content.parts[0].function_call
    assert function_call is not None
    assert function_call.name == "report_truncated_model_output"
    assert function_call.args["max_output_tokens"] == 32768
    assert "truncated" in function_call.args["error_message"].lower()


def test_write_file_and_read_file(workspace_env: Path):
    result = write_file("src/app.py", "line1\nline2\nline3\n")
    assert result["status"] == "success"
    assert result["path"] == "src/app.py"

    read_result = read_file("src/app.py", start_line=2, end_line=3)
    assert read_result["status"] == "success"
    assert read_result["start_line"] == 2
    assert read_result["end_line"] == 3
    assert read_result["content"] == "2: line2\n3: line3"


def test_write_file_requires_read_for_existing_file(workspace_env: Path):
    _write(workspace_env / "src" / "app.py", "old\n")
    tool_context = _DummyToolContext()

    result = write_file("src/app.py", "new\n", tool_context=tool_context)

    assert result["status"] == "error"
    assert "Read it first" in result["error_message"]


def test_write_file_allows_existing_file_after_full_read(workspace_env: Path):
    _write(workspace_env / "src" / "app.py", "old\n")
    tool_context = _DummyToolContext()

    read_result = read_file("src/app.py", start_line=1, end_line=500, tool_context=tool_context)
    assert read_result["status"] == "success"

    result = write_file("src/app.py", "new\n", tool_context=tool_context)

    assert result["status"] == "success"
    assert result["type"] == "update"
    assert result["filePath"] == "src/app.py"
    assert result["patch"].startswith("--- src/app.py")
    assert (workspace_env / "src" / "app.py").read_text(encoding="utf-8") == "new\n"


def test_write_file_rejects_partial_read_for_existing_file(workspace_env: Path):
    _write(workspace_env / "src" / "app.py", "1\n2\n3\n")
    tool_context = _DummyToolContext()

    read_result = read_file("src/app.py", start_line=2, end_line=2, tool_context=tool_context)
    assert read_result["status"] == "success"

    result = write_file("src/app.py", "new\n", tool_context=tool_context)

    assert result["status"] == "error"
    assert "partially read" in result["error_message"]


def test_patch_file_requires_read_for_existing_file(workspace_env: Path):
    _write(workspace_env / "src" / "app.py", "return old_value\n")
    tool_context = _DummyToolContext()

    result = patch_file(
        "src/app.py",
        "return old_value",
        "return new_value",
        tool_context=tool_context,
    )

    assert result["status"] == "error"
    assert "Read it first" in result["error_message"]


def test_patch_file_allows_edit_after_read(workspace_env: Path):
    _write(workspace_env / "src" / "app.py", "return old_value\n")
    tool_context = _DummyToolContext()

    read_result = read_file("src/app.py", tool_context=tool_context)
    assert read_result["status"] == "success"

    result = patch_file(
        "src/app.py",
        "return old_value",
        "return new_value",
        tool_context=tool_context,
    )

    assert result["status"] == "success"
    assert result["type"] == "update"
    assert result["filePath"] == "src/app.py"
    assert result["patch"].startswith("--- src/app.py")
    assert result["replaceAll"] is False
    assert "new_value" in (workspace_env / "src" / "app.py").read_text(encoding="utf-8")


def test_patch_file_rejects_if_file_changed_after_read(workspace_env: Path):
    file_path = workspace_env / "src" / "app.py"
    _write(file_path, "return old_value\n")
    tool_context = _DummyToolContext()

    read_result = read_file("src/app.py", tool_context=tool_context)
    assert read_result["status"] == "success"

    _write(file_path, "return changed_elsewhere\n")

    result = patch_file(
        "src/app.py",
        "return changed_elsewhere",
        "return new_value",
        tool_context=tool_context,
    )

    assert result["status"] == "error"
    assert "modified since it was read" in result["error_message"]


def test_list_files(workspace_env: Path):
    _write(workspace_env / "a.txt", "a")
    _write(workspace_env / "nested" / "b.txt", "b")

    result = list_files(".", recursive=True)
    assert result["status"] == "success"
    assert "a.txt" in result["entries"]
    assert "nested/b.txt" in result["entries"]
    assert result["truncated"] is False


def test_list_files_exclude_regex(workspace_env: Path):
    _write(workspace_env / "src" / "keep.py", "x")
    _write(workspace_env / "src" / "__pycache__" / "drop.pyc", "x")
    _write(workspace_env / "src" / "drop.pyc", "x")

    result = list_files("src", recursive=True, exclude=r"__pycache__|\.pyc$")
    assert result["status"] == "success"
    assert result["exclude"] == r"__pycache__|\.pyc$"
    assert "src/keep.py" in result["entries"]
    assert all("__pycache__" not in entry for entry in result["entries"])
    assert all(not entry.endswith(".pyc") for entry in result["entries"])


def test_list_files_invalid_exclude_regex(workspace_env: Path):
    result = list_files(".", exclude="(")
    assert result["status"] == "error"
    assert "Invalid exclude regex" in result["error_message"]


def test_search_text_keywords_and_context(workspace_env: Path):
    _write(
        workspace_env / "pkg" / "loader.py",
        "def load_model():\n    return 'ok'\n\nclass AutoTokenizer:\n    pass\n",
    )

    result = search_text(
        keywords=["load_model", "AutoTokenizer"],
        target_folder="pkg",
        context_lines=1,
    )

    assert result["status"] == "success"
    assert result["matched_files"] == 1
    assert result["matches"]
    assert any("load_model" in match["file_content"] for match in result["matches"])
    assert any("AutoTokenizer" in match["file_content"] for match in result["matches"])


def test_search_text_regex(workspace_env: Path):
    _write(
        workspace_env / "pkg" / "model.py",
        "checkpoint_fn = True\nother = 1\nremat_layer = False\n",
    )

    result = search_text(
        use_regex=True,
        regex_pattern=r"remat|checkpoint",
        target_folder="pkg",
        context_lines=0,
    )

    assert result["status"] == "success"
    assert result["use_regex"] is True
    assert result["matched_files"] == 1
    assert len(result["matches"]) == 2
    matched_texts = [text for match in result["matches"] for text in match["matched_texts"]]
    assert "checkpoint" in matched_texts
    assert "remat" in matched_texts


def test_search_text_target_folder_is_scoped_relative_to_path(workspace_env: Path):
    _write(
        workspace_env / "attempt_1" / "ppo" / "good.py",
        "import torch\n",
    )
    _write(
        workspace_env / "ppo" / "bad.py",
        "import torch\n",
    )

    result = search_text(
        keywords=["import torch"],
        path="attempt_1",
        target_folder="ppo",
    )

    assert result["status"] == "success"
    assert result["path"] == "attempt_1/ppo"
    assert result["matched_files"] == 1
    assert [match["path"] for match in result["matches"]] == ["attempt_1/ppo/good.py"]


def test_search_text_invalid_regex_flag(workspace_env: Path):
    result = search_text(
        use_regex=True,
        regex_pattern="foo",
        regex_flags="z",
    )
    assert result["status"] == "error"
    assert "Unsupported regex flag" in result["error_message"]


def test_keyword_search(workspace_env: Path):
    _write(workspace_env / "src" / "loader.py", "load tokenizer config\n")
    _write(workspace_env / "src" / "other.py", "config only\n")

    result = keyword_search(["loader", "config"], target_folder="src")
    assert result["status"] == "success"
    assert result["returned_results"] >= 1
    assert result["results"][0]["path"] == "src/loader.py"


def test_lsp_tool_go_to_definition_and_hover(workspace_env: Path):
    _write(
        workspace_env / "pkg" / "mod.py",
        "class Foo:\n"
        "    def bar(self):\n"
        "        \"\"\"Doc for bar.\"\"\"\n"
        "        return 1\n"
        "\n"
        "foo = Foo()\n"
        "foo.bar()\n",
    )

    definition = lsp_tool(
        "goToDefinition",
        "pkg/mod.py",
        line=7,
        character=5,
    )
    assert definition["status"] == "success"
    assert definition["operation"] == "goToDefinition"
    assert definition["result_count"] >= 1
    assert "bar" in definition["result"]

    hover = lsp_tool(
        "hover",
        "pkg/mod.py",
        line=7,
        character=5,
    )
    assert hover["status"] == "success"
    assert hover["operation"] == "hover"
    assert hover["result_count"] == 1


def test_lsp_tool_find_references_and_document_symbols(workspace_env: Path):
    _write(
        workspace_env / "pkg" / "refs.py",
        "class Foo:\n"
        "    pass\n"
        "\n"
        "foo = Foo()\n"
        "bar = Foo()\n",
    )

    references = lsp_tool(
        "findReferences",
        "pkg/refs.py",
        line=1,
        character=8,
    )
    assert references["status"] == "success"
    assert references["operation"] == "findReferences"
    assert references["result_count"] >= 3
    assert "pkg/refs.py" in references["result"]

    symbols = lsp_tool("documentSymbol", "pkg/refs.py")
    assert symbols["status"] == "success"
    assert symbols["operation"] == "documentSymbol"
    assert "Foo" in symbols["result"]


def test_lsp_tool_workspace_symbol(workspace_env: Path):
    _write(workspace_env / "pkg" / "a.py", "class Alpha:\n    pass\n")
    _write(workspace_env / "pkg" / "b.py", "def beta():\n    return Alpha\n")

    result = lsp_tool("workspaceSymbol", ".", query="Alpha")
    assert result["status"] == "success"
    assert result["operation"] == "workspaceSymbol"
    assert result["query"] == "Alpha"
    assert "Alpha" in result["result"]


def test_prepare_scoped_child_state_creates_isolated_subagent_memory(workspace_env: Path):
    main_memory_path = workspace_env / ".coding_agent_memory" / "working_memory.md"
    main_memory_path.parent.mkdir(parents=True, exist_ok=True)
    main_memory_path.write_text("# Main Memory\n", encoding="utf-8")

    scoped_workspace = workspace_env / "delegated"
    scoped_workspace.mkdir()

    parent_state = {
        "_coding_agent_root_workspace": str(workspace_env),
        "_coding_agent_memory_file": ".coding_agent_memory/working_memory.md",
        "_coding_agent_last_compaction_end_timestamp": "2026-04-01T00:00:00Z",
        "_coding_agent_last_compaction_event_count": 42,
    }

    child_state_one = streaming_agent_tool_module._prepare_scoped_child_state(
        parent_state,
        scoped_workspace,
    )
    child_state_two = streaming_agent_tool_module._prepare_scoped_child_state(
        parent_state,
        scoped_workspace,
    )

    assert child_state_one["_coding_agent_root_workspace"] == str(scoped_workspace)
    assert child_state_one["_coding_agent_memory_file"] != parent_state["_coding_agent_memory_file"]
    assert child_state_one["_coding_agent_memory_file"] != child_state_two["_coding_agent_memory_file"]
    assert "_coding_agent_last_compaction_end_timestamp" not in child_state_one
    assert "_coding_agent_last_compaction_event_count" not in child_state_one

    child_memory_path = scoped_workspace / child_state_one["_coding_agent_memory_file"]
    assert child_memory_path.exists()
    child_text = child_memory_path.read_text(encoding="utf-8")
    assert "# Working Memory" in child_text
    assert "## Current Task Progress" in child_text
    assert child_memory_path != main_memory_path


def test_patch_file_exact(workspace_env: Path):
    _write(workspace_env / "src" / "app.py", "value = 1\n")

    result = patch_file("src/app.py", "value = 1", "value = 2")
    assert result["status"] == "success"
    assert result["match_strategy"] == "exact"
    assert (workspace_env / "src" / "app.py").read_text(encoding="utf-8") == "value = 2\n"


def test_patch_file_create_new_file(workspace_env: Path):
    result = patch_file("new/file.txt", "", "created\n")
    assert result["status"] == "success"
    assert result["type"] == "create"
    assert result["filePath"] == "new/file.txt"
    assert result["match_strategy"] == "create"
    assert (workspace_env / "new" / "file.txt").read_text(encoding="utf-8") == "created\n"


def test_write_and_patch_formatting_matches_create_update_style(workspace_env: Path):
    write_result = {
        "status": "success",
        "type": "create",
        "filePath": "src/new.py",
        "bytes_written": 12,
        "line_count": 2,
        "patch": "--- src/new.py\n+++ src/new.py\n@@ -0,0 +1,2 @@\n+one\n+two",
    }
    write_formatted = _format_tool_result_for_model("write_file", write_result)["output"]
    assert "Created src/new.py" in write_formatted
    assert "Patch:" in write_formatted
    assert "--- src/new.py" in write_formatted

    patch_result = {
        "status": "success",
        "type": "update",
        "filePath": "src/app.py",
        "match_strategy": "exact",
        "match_count": 1,
        "bytes_written": 20,
        "line_count": 1,
        "patch": "--- src/app.py\n+++ src/app.py\n@@ -1 +1 @@\n-old\n+new",
    }
    patch_formatted = _format_tool_result_for_model("patch_file", patch_result)["output"]
    assert "Updated src/app.py" in patch_formatted
    assert "strategy=exact" in patch_formatted
    assert "Patch:" in patch_formatted


def test_report_truncated_model_output_formats_clear_retry_error(workspace_env: Path):
    result = report_truncated_model_output(
        "Tool call arguments were truncated while streaming and could not be parsed as valid JSON.",
        max_output_tokens=32768,
    )
    assert result["status"] == "error"
    formatted = _format_tool_result_for_model("report_truncated_model_output", result)["output"]
    assert "Tool: report_truncated_model_output" in formatted
    assert "max_output_tokens: 32768" in formatted
    assert "Retry guidance:" in formatted


def test_bash_exec(workspace_env: Path):
    result = bash_exec("printf 'hello\\n'")
    assert result["status"] == "success"
    assert result["exit_code"] == 0
    assert result["output"] == "hello"


def test_bash_exec_collapses_carriage_return_overwrites(workspace_env: Path):
    result = bash_exec(
        "python - <<'PY'\n"
        "import sys\n"
        "sys.stdout.write('step 1\\r')\n"
        "sys.stdout.write('step 2\\r')\n"
        "sys.stdout.write('done\\n')\n"
        "sys.stdout.flush()\n"
        "PY"
    )
    assert result["status"] == "success"
    assert result["output"] == "done"


def test_bash_exec_rejects_sudo(workspace_env: Path):
    result = bash_exec("sudo ls")
    assert result["status"] == "error"
    assert "sudo" in result["error_message"]


def test_bash_exec_honors_timeout_ms(workspace_env: Path):
    result = bash_exec("python - <<'PY'\nimport time\ntime.sleep(0.2)\nPY", timeout_ms=50)
    assert result["status"] == "success"
    assert result["timed_out"] is True
    assert result["backgrounded"] is True
    assert result["timeout_ms"] == 50
    assert "session_id" in result
    terminate_result = terminate_command_session(result["session_id"])
    assert terminate_result["status"] == "success"


def test_terminal_tools_return_stdout_and_stderr_to_model(workspace_env: Path):
    bash_result = bash_exec(
        "python - <<'PY'\n"
        "import sys\n"
        "print('bash out')\n"
        "sys.stderr.write('bash err\\n')\n"
        "raise SystemExit(1)\n"
        "PY"
    )
    assert bash_result["status"] == "error"
    assert "bash out" in bash_result["output"]
    assert "bash err" in bash_result["output"]
    bash_formatted = _format_tool_result_for_model("bash_exec", bash_result)["output"]
    assert "Output:" in bash_formatted and "bash out" in bash_formatted
    assert "bash err" in bash_formatted


def test_bash_exec_backgrounds_long_running_command(workspace_env: Path):
    result = bash_exec(
        "python -u - <<'PY'\n"
        "import time\n"
        "print('started')\n"
        "time.sleep(0.3)\n"
        "print('done')\n"
        "PY",
        timeout_ms=50,
    )
    assert result["status"] == "success"
    assert result["backgrounded"] is True
    assert result["timed_out"] is True
    assert "session_id" in result
    assert "background" in result["message"].lower()
    formatted = _format_tool_result_for_model("bash_exec", result)["output"]
    assert "session_id:" in formatted
    assert "backgrounded: true" in formatted
    terminate_result = terminate_command_session(result["session_id"])
    assert terminate_result["status"] == "success"


def test_bash_exec_backgrounds_long_running_command_with_tail_pipeline(workspace_env: Path):
    result = bash_exec(
        "python -u - <<'PY'\n"
        "import sys, time\n"
        "print('started')\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.3)\n"
        "print('done')\n"
        "PY\n"
        "2>&1 | tail -40",
        timeout_ms=50,
    )
    assert result["status"] == "success"
    assert result["backgrounded"] is True
    assert result["timed_out"] is True
    assert "session_id" in result
    assert "background" in result["message"].lower()
    assert "started" in result["output"]
    terminate_result = terminate_command_session(result["session_id"])
    assert terminate_result["status"] == "success"


def test_bash_exec_backgrounds_silent_long_running_command(workspace_env: Path):
    result = bash_exec(
        "python -u - <<'PY'\n"
        "import time\n"
        "time.sleep(0.3)\n"
        "print('done')\n"
        "PY",
        timeout_ms=50,
    )
    assert result["status"] == "success"
    assert result["backgrounded"] is True
    assert result["timed_out"] is True
    assert "session_id" in result
    terminate_result = terminate_command_session(result["session_id"])
    assert terminate_result["status"] == "success"


def test_bash_exec_reports_signal_termination_as_error(workspace_env: Path):
    result = bash_exec("kill -TERM $$")
    assert result["status"] == "error"
    assert result["exit_code"] == -15
    assert "signal 15" in result["error_message"].lower()


def test_run_bash_starts_new_session(workspace_env: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _FakeStdout:
        def read(self, _size=4096):
            return b""

        def close(self):
            return None

    class _FakeProcess:
        pid = 99999

        def __init__(self):
            self.stdout = _FakeStdout()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def _fake_popen(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(command_sessions_module.subprocess, "Popen", _fake_popen)
    result = run_bash("printf 'done\\n'", initial_wait_seconds=0.0)
    assert result["status"] == "success"
    assert captured["kwargs"]["start_new_session"] is True


def test_terminate_command_session_uses_process_group_signal(
    workspace_env: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[int, int]] = []

    class _FakeProcess:
        pid = 43210

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

        def terminate(self):
            raise AssertionError("terminate() should not be the primary path")

        def kill(self):
            raise AssertionError("kill() should not be the primary path")

    monkeypatch.setattr(
        command_sessions_module.os,
        "killpg",
        lambda pid, sig: calls.append((pid, sig)),
    )

    session = command_sessions_module._CommandSession.__new__(command_sessions_module._CommandSession)
    session.session_id = "session-123"
    session.process = _FakeProcess()
    session.command = "python train.py"
    session.kind = "bash"
    session.workdir = Path("/tmp")
    session.created_at = 0.0
    session._lock = threading.Lock()
    session._output_buffer = command_sessions_module.TerminalOutputBuffer(max_lines=2000)
    session._output_cursor_seq = 0
    session._last_output_at = 0.0
    session._last_read_at = 0.0
    session._termination_reason = None
    session._output_thread = threading.Thread()
    session._wait_for_reader_drain = lambda timeout_seconds=0.2: None

    termination = session.terminate(reason="manual")
    assert termination["command_status"] == "terminated"
    assert calls == [(43210, signal.SIGTERM)]


def test_large_tool_output_is_offloaded_to_workspace_tmp(workspace_env: Path):
    long_output = "A" * 600000
    result = _offload_large_tool_output(
        "bash_exec",
        {"output": long_output},
    )
    assert result["truncated"] is True
    assert result["original_output_chars"] == len(long_output)
    assert result["output_char_limit"] == 500000
    assert "full_output_path" in result
    assert "tool output truncated" in result["output"]
    saved_path = workspace_env / result["full_output_path"]
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == long_output
    assert len(result["output"]) <= 500000


def test_todo_tools(workspace_env: Path):
    write_result = write_todos(
        [
            {"description": "Inspect loader", "status": "completed"},
            {"description": "Patch config", "status": "in_progress"},
        ],
        agent_name="client_agent",
        workspace_root=str(workspace_env),
    )
    assert write_result["status"] == "success"
    assert write_result["agent_name"] == "client_agent"
    assert write_result["todo_file"] == str(workspace_env / ".adk" / "coding_agent_todos.json")

    planner_write_result = write_todos(
        [
            {"description": "Plan rollout", "status": "in_progress"},
        ],
        agent_name="planner",
        workspace_root=str(workspace_env),
    )
    assert planner_write_result["status"] == "success"
    assert planner_write_result["agent_name"] == "planner"

    read_result = read_todos(agent_name="client_agent", workspace_root=str(workspace_env))
    assert read_result["status"] == "success"
    assert read_result["agent_name"] == "client_agent"
    assert read_result["todos"][1]["description"] == "Patch config"
    assert read_result["todo_file"] == str(workspace_env / ".adk" / "coding_agent_todos.json")

    planner_read_result = read_todos(agent_name="planner", workspace_root=str(workspace_env))
    assert planner_read_result["status"] == "success"
    assert planner_read_result["agent_name"] == "planner"
    assert planner_read_result["todos"] == [
        {"description": "Plan rollout", "status": "in_progress"}
    ]


def test_todo_tools_use_explicit_workspace_root_even_if_env_changes(tmp_path: Path, monkeypatch):
    root_workspace = tmp_path / "root"
    other_workspace = tmp_path / "other"
    root_workspace.mkdir()
    other_workspace.mkdir()

    monkeypatch.setenv("CODING_AGENT_WORKSPACE", str(other_workspace))

    session = SimpleNamespace(state={"_coding_agent_root_workspace": str(root_workspace)})
    tool_context = SimpleNamespace(
        agent_name="client_agent",
        _invocation_context=SimpleNamespace(session=session),
    )

    args = {"todos": [{"description": "Task", "status": "in_progress"}]}
    patched = _inject_verifier_user_instruction(
        SimpleNamespace(name="write_todos"),
        args,
        tool_context,
    )

    assert patched is None
    result = write_todos(**args)
    assert result["status"] == "success"
    assert result["todo_file"] == str(root_workspace / ".adk" / "coding_agent_todos.json")
    assert Path(result["todo_file"]).exists()


def test_memory_tools():
    save_result = save_memory("Prefer concise answers.")
    assert save_result["status"] == "success"

    read_result = read_memory()
    assert read_result["status"] == "success"
    assert "Prefer concise answers." in read_result["content"]


def test_skills_tools(workspace_env: Path):
    skill_dir = workspace_env / ".gemini" / "skills" / "demo-skill"
    _write(
        skill_dir / "SKILL.md",
        "---\nname: demo-skill\ndescription: Demo description\n---\nUse this carefully.\n",
    )
    _write(skill_dir / "notes.txt", "resource\n")

    list_result = list_skills()
    assert list_result["status"] == "success"
    assert any(skill["name"] == "demo-skill" for skill in list_result["skills"])

    activate_result = activate_skill("demo-skill")
    assert activate_result["status"] == "success"
    assert activate_result["activated_skill"]["name"] == "demo-skill"
    assert "Use this carefully." in activate_result["activated_skill"]["body"]
    assert any(path.endswith("notes.txt") for path in activate_result["activated_skill"]["resource_files"])


def test_format_codebase_output():
    result = format_codebase_output(
        files=[
            {"path": "src/app.py", "start_line": 10, "end_line": 20},
            {"path": "src/app.py", "start_line": 30, "end_line": 30},
        ],
        summary="Short summary.",
    )
    assert result["status"] == "success"
    assert result["files"][0]["path"] == "src/app.py"
    assert result["summary"] == "Short summary."


def test_command_session_tools(workspace_env: Path):
    command = (
        "python -u - <<'PY'\n"
        "import sys, time\n"
        "print('out1')\n"
        "sys.stderr.write('err1\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(0.2)\n"
        "print('out2')\n"
        "sys.stderr.write('err2\\n')\n"
        "sys.stderr.flush()\n"
        "PY"
    )
    start_result = run_bash(command, initial_wait_seconds=0.05)
    assert start_result["status"] == "success"
    assert start_result["backgrounded"] is True
    assert start_result["command_status"] == "running"
    assert "background" in start_result["message"].lower()
    session_id = start_result["session_id"]

    collected_output = list(start_result["output_lines"])
    wait_result = {"status": "success", "output_lines": []}
    for _ in range(6):
        wait_result = wait(session_id, wait_seconds=0.2)
        assert wait_result["status"] == "success"
        collected_output.extend(wait_result["output_lines"])
        if collected_output:
            break

    merged_output = "\n".join(collected_output)
    assert "out1" in merged_output or "out2" in merged_output
    assert "err1" in merged_output or "err2" in merged_output

    read_result = read_command_session(session_id, wait_seconds=0.0)
    assert read_result["status"] == "success"
    assert read_result["command_status"] in {"running", "completed"}

    terminate_result = terminate_command_session(session_id)
    assert terminate_result["status"] == "success"
    assert terminate_result["termination"]["command_status"] in {"terminated", "completed"}


def test_run_bash_rejects_sudo(workspace_env: Path):
    result = run_bash("sudo whoami")
    assert result["status"] == "error"
    assert "sudo" in result["error_message"]


def test_run_bash_completes_within_initial_window(workspace_env: Path):
    result = run_bash("printf 'done\\n'", initial_wait_seconds=0.2)
    assert result["status"] == "success"
    assert result["backgrounded"] is False
    assert result["command_status"] == "completed"
    assert "finished within the initial" in result["message"]
    assert "done" in "\n".join(result["output_lines"])


def test_run_bash_collapses_carriage_return_progress(workspace_env: Path):
    result = run_bash(
        "python - <<'PY'\n"
        "import sys, time\n"
        "for index in range(3):\n"
        "    sys.stdout.write(f'progress {index}\\r')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.02)\n"
        "print('done')\n"
        "PY",
        initial_wait_seconds=0.5,
    )
    assert result["status"] == "success"
    assert result["backgrounded"] is False
    assert result["output_lines"] == ["done"]


def test_check_running_sessions(workspace_env: Path):
    command = (
        "python -u - <<'PY'\n"
        "import time\n"
        "print('started')\n"
        "time.sleep(1.0)\n"
        "print('done')\n"
        "PY"
    )
    start_result = run_bash(command, initial_wait_seconds=0.05)
    assert start_result["status"] == "success"
    session_id = start_result["session_id"]

    running_result = check_running_sessions()
    assert running_result["status"] == "success"
    assert running_result["running_count"] >= 1
    matching = [session for session in running_result["sessions"] if session["session_id"] == session_id]
    assert matching
    assert "running_seconds" in matching[0]

    terminate_result = terminate_command_session(session_id)
    assert terminate_result["status"] == "success"


def test_command_session_auto_terminates_after_idle():
    class _FakeProcess:
        pid = 123

        def __init__(self):
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    session = command_sessions_module._CommandSession.__new__(command_sessions_module._CommandSession)
    session.session_id = "session-123"
    session.process = _FakeProcess()
    session.command = "python train.py"
    session.kind = "bash"
    session.workdir = Path("/tmp")
    session.created_at = 0.0
    session._lock = threading.Lock()
    session._output_lines = deque(maxlen=2000)
    session._output_next_seq = 1
    session._output_cursor_seq = 0
    session._last_output_at = 0.0
    session._last_read_at = 0.0
    session._termination_reason = None
    session._output_thread = threading.Thread()
    session._wait_for_reader_drain = lambda timeout_seconds=0.2: None

    assert session.should_auto_terminate(now=301.0, idle_timeout_seconds=300.0) is True
    termination = session.terminate(reason="idle_timeout")
    assert termination["command_status"] == "terminated"
    assert termination["termination_reason"] == "idle_timeout"
    assert session.process.terminated is True


def test_session_tool_formatting_includes_output():
    payload = {
        "status": "success",
        "session_id": "session-123",
        "command": "python train.py",
        "command_status": "running",
        "output_context_lines": ["old out", "old err"],
        "output_new_lines": ["session out", "session err"],
    }
    formatted = _format_tool_result_for_model("wait", payload)["output"]
    assert "Recent output:" in formatted
    assert "New output:" in formatted
    assert "session out" in formatted
    assert "session err" in formatted
    assert "old out" in formatted
    assert "old err" in formatted


def test_command_session_read_returns_recent_context_plus_new_output():
    class _FakeProcess:
        pid = 123

        def poll(self):
            return None

    session = command_sessions_module._CommandSession.__new__(command_sessions_module._CommandSession)
    session.session_id = "session-123"
    session.process = _FakeProcess()
    session.command = "python train.py"
    session.kind = "bash"
    session.workdir = Path("/tmp")
    session.created_at = 0.0
    session._lock = threading.Lock()
    session._output_buffer = command_sessions_module.TerminalOutputBuffer(max_lines=2000)
    session._output_buffer.feed("out 1\nerr 1\nout 2\nerr 2\nout 3\nerr 3\n")
    session._output_cursor_seq = 4
    session._last_output_at = 0.0
    session._last_read_at = 0.0
    session._termination_reason = None
    session._wait_for_reader_drain = lambda timeout_seconds=0.2: None
    session.snapshot = lambda: {
        "session_id": session.session_id,
        "kind": session.kind,
        "command": session.command,
        "working_directory": str(session.workdir),
        "pid": session.process.pid,
        "command_status": "running",
        "is_running": True,
        "exit_code": None,
        "running_seconds": 0.0,
    }

    payload = session.read(max_output_lines=2)
    assert payload["output_context_lines"] == [
        "out 1",
        "err 1",
        "out 2",
        "err 2",
    ]
    assert payload["output_new_lines"] == ["out 3", "err 3"]
    assert payload["output_lines"] == [
        "out 1",
        "err 1",
        "out 2",
        "err 2",
        "out 3",
        "err 3",
    ]


def test_all_exported_tools_have_basic_coverage():
    expected_exports = {
        "activate_skill",
        "bash_exec",
        "check_running_sessions",
        "format_codebase_output",
        "keyword_search",
        "list_files",
        "list_skills",
        "lsp_tool",
        "patch_file",
        "read_command_session",
        "read_file",
        "read_memory",
        "read_todos",
        "report_truncated_model_output",
        "save_memory",
        "search_text",
        "run_bash",
        "terminate_command_session",
        "wait",
        "write_file",
        "write_todos",
    }
    from tools import __all__ as exported_tools

    assert set(exported_tools) == expected_exports
