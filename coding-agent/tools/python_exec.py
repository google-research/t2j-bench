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

import subprocess
import sys

from .workspace import get_workspace_root

def python_exec(code: str, timeout: int = 30) -> dict:
    """Run a Python snippet inside the workspace root.

    Use this for quick data inspection, small prototypes, or project-local
    Python logic when a shell command would be awkward. The snippet runs with
    the current interpreter from the workspace root.

    Args:
        code: The complete Python code snippet to execute.
        timeout: The timeout of the execution in seconds. Default in 30s.


    Returns:
        A structured result containing `status`, `exit_code`, `stdout`,
        `stderr`, and `working_directory`. Non-zero exit codes are returned as
        tool errors, but stdout and stderr are preserved for diagnosis.

    Example input:
        code="import pathlib; print(pathlib.Path('.').resolve())"

    Example output:
        {"status": "success", "exit_code": 0, "stdout": "/workspace/..."}
    """
    workdir = get_workspace_root()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        return {
            "status": "error",
            "error_message": f"Python code timed out after {timeout} seconds.",
            "stdout": stdout,
            "stderr": stderr,
            "working_directory": str(workdir),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Failed to execute python code: {exc}",
            "working_directory": str(workdir),
        }

    status = "success" if completed.returncode == 0 else "error"
    return {
        "status": status,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "working_directory": str(workdir),
    }
