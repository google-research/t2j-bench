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

from .command_sessions import run_bash

_DEFAULT_TIMEOUT_MS = 30_000


def bash_exec(command: str, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> dict:
    """Run a Bash command inside the workspace root.

    Use this for terminal work: tests, builds, linters, package managers,
    git, and local CLIs. Do not use it for ordinary file reading, writing,
    editing, or searching when a structured file tool can do the job better.
    The command runs in a Bash session. By default it waits in the foreground
    for 30 seconds (`timeout_ms=30000`). If the command is still running after
    that foreground window, it is left running in the background and the tool
    returns a `session_id` for later polling.

    Example input:
        command="pytest tests/test_loader.py -q", timeout_ms=30000

    Example output:
        {"status": "success", "command": "pytest tests/test_loader.py -q", "output": "..."}
    """
    normalized_timeout_ms = max(1, int(timeout_ms))
    result = run_bash(command, initial_wait_seconds=normalized_timeout_ms / 1000.0)
    result = dict(result)
    result["command"] = command
    result["timeout_ms"] = normalized_timeout_ms

    output_lines = result.get("output_lines")
    if isinstance(output_lines, list):
        result["output"] = "\n".join(str(line) for line in output_lines)

    if result.get("status") != "success":
        return result

    if result.get("backgrounded"):
        result["timed_out"] = True
        return result

    result["timed_out"] = False
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        result["status"] = "error"
        if exit_code < 0:
            signal_number = -exit_code
            result["error_message"] = (
                f"Command terminated by signal {signal_number}."
            )
        else:
            result["error_message"] = f"Command exited with code {exit_code}."
    result.pop("session_id", None)
    result.pop("backgrounded", None)
    result.pop("message", None)
    result.pop("next_action_hint", None)
    result.pop("initial_wait_seconds", None)
    return result
