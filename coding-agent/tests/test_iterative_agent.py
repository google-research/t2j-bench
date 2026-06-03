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
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ADK_SRC = REPO_ROOT / "adk-python" / "src"

if ADK_SRC.exists():
    sys.path.insert(0, str(ADK_SRC))
sys.path.insert(0, str(REPO_ROOT / "coding-agent"))

from iterative_agent.scoped_client_tool import ScopedCodingAgentTool
from iterative_agent.tools import prepare_repository_attempt, run_evaluator_command


@pytest.fixture(autouse=True)
def workspace_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CODING_AGENT_WORKSPACE", str(workspace))
    monkeypatch.setenv("CODING_AGENT_YOLO", "0")
    monkeypatch.setenv("CODING_AGENT_VERBOSE", "0")
    monkeypatch.setenv("CODING_AGENT_STREAM_SUBAGENT_LOGS", "0")
    yield workspace


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_repository_attempt_copies_source_repository(workspace_env: Path):
    source_repo = workspace_env / "source_repo"
    _write(source_repo / "README.md", "# source\n")
    _write(source_repo / "pkg" / "module.py", "value = 1\n")
    _write(source_repo / "__pycache__" / "ignored.pyc", "x")

    result = prepare_repository_attempt("source_repo")

    assert result["status"] == "success"
    candidate_repo = workspace_env / result["candidate_repository_path"]
    assert candidate_repo.is_dir()
    assert (candidate_repo / "README.md").read_text(encoding="utf-8") == "# source\n"
    assert (candidate_repo / "pkg" / "module.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not (candidate_repo / "__pycache__").exists()


def test_run_evaluator_command_returns_merged_output_and_artifacts(workspace_env: Path):
    evaluator_dir = workspace_env / "evaluator"
    original_repo = workspace_env / "original"
    candidate_repo = workspace_env / "candidate"
    evaluator_dir.mkdir()
    original_repo.mkdir()
    candidate_repo.mkdir()

    result = run_evaluator_command(
        evaluator_path="evaluator",
        original_repository_path="original",
        candidate_repository_path="candidate",
        command=(
            "printf 'out\\n'; "
            "python - <<'PY'\n"
            "import sys, pathlib\n"
            "sys.stdout.write('more\\n')\n"
            "sys.stderr.write('err\\n')\n"
            "pathlib.Path('reports').mkdir(exist_ok=True)\n"
            "pathlib.Path('reports/new_eval_summary.json').write_text('{\"pass\": true}')\n"
            "PY"
        ),
        timeout_seconds=30,
    )

    assert result["status"] == "success"
    assert result["exit_code"] == 0
    assert "out" in result["output"]
    assert "more" in result["output"]
    assert "err" in result["output"]
    assert result["evaluator_log_path"].endswith("evaluator_command.log")
    log_path = workspace_env / result["evaluator_log_path"]
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "out" in log_text
    assert "more" in log_text
    assert "err" in log_text
    assert any(path.endswith("reports/new_eval_summary.json") for path in result["recent_artifacts"])


def test_run_evaluator_command_rejects_sudo(workspace_env: Path):
    evaluator_dir = workspace_env / "evaluator"
    original_repo = workspace_env / "original"
    candidate_repo = workspace_env / "candidate"
    evaluator_dir.mkdir()
    original_repo.mkdir()
    candidate_repo.mkdir()

    result = run_evaluator_command(
        evaluator_path="evaluator",
        original_repository_path="original",
        candidate_repository_path="candidate",
        command="sudo python -m lib.run_eval config.yaml",
        timeout_seconds=30,
    )

    assert result["status"] == "error"
    assert "sudo" in result["error_message"]


class _DummyState:
    def __init__(self):
        self._state = {}

    def to_dict(self):
        return dict(self._state)

    def update(self, delta):
        if delta:
            self._state.update(delta)


class _DummyToolContext:
    def __init__(self):
        self.state = _DummyState()
        self._invocation_context = type(
            "InvocationContext",
            (),
            {
                "app_name": "test_app",
                "user_id": "test_user",
                "credential_service": None,
                "plugin_manager": type("PluginManager", (), {"plugins": []})(),
            },
        )()


def test_scoped_coding_agent_tool_runs_nested_root_agent(
    workspace_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    candidate_repo = workspace_env / "candidate_repo"
    candidate_repo.mkdir()
    _write(candidate_repo / "README.md", "# candidate\n")

    import iterative_agent.scoped_client_tool as scoped_module

    recorded = {}

    class DummySession:
        id = "nested-session"

    class DummySessionService:
        async def create_session(self, *, app_name, user_id, session_id):
            recorded["session"] = {
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
            }
            return DummySession()

    class DummyRunner:
        def __init__(self, *, app_name, agent, session_service):
            recorded["runner_init"] = {
                "app_name": app_name,
                "agent": agent,
                "session_service_type": type(session_service).__name__,
            }

        async def run_async(self, *, user_id, session_id, new_message):
            recorded["run_async"] = {
                "user_id": user_id,
                "session_id": session_id,
                "prompt": new_message.parts[0].text,
            }
            yield type(
                "Event",
                (),
                {
                    "is_final_response": lambda self: True,
                    "content": type(
                        "Content",
                        (),
                        {
                            "parts": [type("Part", (), {"text": "conversion finished"})()],
                        },
                    )(),
                    "model_dump": lambda self, exclude_none=True, mode="json": {
                        "author": "client_agent",
                        "content": {"parts": [{"text": "conversion finished"}]},
                    },
                },
            )()

        async def close(self):
            recorded["closed"] = True

    monkeypatch.setattr(scoped_module, "InMemorySessionService", DummySessionService)
    monkeypatch.setattr(scoped_module, "Runner", DummyRunner)
    monkeypatch.setattr(scoped_module, "coding_agent_root_agent", object())

    tool = ScopedCodingAgentTool()
    result = asyncio.run(
        tool.run_async(
            args={
                "candidate_repository_path": "candidate_repo",
                "instruction": "Preserve the public API and satisfy the required contracts.",
            },
            tool_context=_DummyToolContext(),
        )
    )

    assert result["status"] == "success"
    assert result["candidate_repository_path"] == "candidate_repo"
    assert result["final_response"] == "conversion finished"
    assert result["output"] == "conversion finished"
    assert recorded["runner_init"]["app_name"] == "test_app"
    assert recorded["session"]["user_id"] == "test_user"
    assert str(candidate_repo) in recorded["run_async"]["prompt"]
    assert "Workspace boundary:" in recorded["run_async"]["prompt"]
    assert recorded["closed"] is True
