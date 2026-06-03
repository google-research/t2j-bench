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

import atexit
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .shell_safety import (
    find_forbidden_shell_token,
    find_restricted_workspace_command_path,
    restricted_workspace_command_warning,
)
from .shell_output import TerminalOutputBuffer
from .workspace import get_workspace_root, is_restricted_mode

_MAX_BUFFER_LINES = 2000
_DEFAULT_INITIAL_WAIT_SECONDS = 30.0
_DEFAULT_POLL_WAIT_SECONDS = 1.0
_READ_POLL_INTERVAL_SECONDS = 0.05
_RECENT_CONTEXT_LINES = 10
_IDLE_TIMEOUT_SECONDS = 300.0
_WATCHDOG_POLL_INTERVAL_SECONDS = 5.0


class _CommandSession:
    def __init__(self, *, process: subprocess.Popen[bytes], command: str, kind: str, workdir: Path) -> None:
        self.session_id = uuid.uuid4().hex
        self.process = process
        self.command = command
        self.kind = kind
        self.workdir = workdir
        self.created_at = time.time()
        self._lock = threading.Lock()
        self._output_buffer = TerminalOutputBuffer(max_lines=_MAX_BUFFER_LINES)
        self._output_cursor_seq = 0
        self._last_output_at = self.created_at
        self._last_read_at = self.created_at
        self._termination_reason: str | None = None
        self._output_thread = threading.Thread(
            target=self._stream_reader,
            args=(self.process.stdout,),
            daemon=True,
        )
        self._output_thread.start()

    def _stream_reader(self, stream) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                with self._lock:
                    text = chunk.decode(errors="replace")
                    self._output_buffer.feed(text)
                    self._last_output_at = time.time()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _wait_for_reader_drain(self, timeout_seconds: float = 0.2) -> None:
        if self.process.poll() is None:
            return
        self._output_thread.join(timeout=timeout_seconds)

    def _status(self) -> str:
        if self._termination_reason is not None:
            return "terminated"
        return "running" if self.process.poll() is None else "completed"

    def snapshot(self) -> dict:
        command_status = self._status()
        payload = {
            "session_id": self.session_id,
            "kind": self.kind,
            "command": self.command,
            "working_directory": str(self.workdir),
            "pid": self.process.pid,
            "command_status": command_status,
            "is_running": command_status == "running",
            "exit_code": self.process.poll(),
            "running_seconds": round(time.time() - self.created_at, 3),
            "seconds_since_last_output": round(time.time() - self._last_output_at, 3),
            "seconds_since_last_read": round(time.time() - self._last_read_at, 3),
        }
        if self._termination_reason is not None:
            payload["termination_reason"] = self._termination_reason
        return payload

    def read(self, *, max_output_lines: int) -> dict:
        self._wait_for_reader_drain()
        with self._lock:
            self._last_read_at = time.time()
            output_items = self._output_buffer.items()
            unread_output = [(seq, line) for seq, line in output_items if seq > self._output_cursor_seq]
            prior_output_context = [
                (seq, line) for seq, line in output_items if seq <= self._output_cursor_seq
            ][-_RECENT_CONTEXT_LINES :]

            output_selected = unread_output[:max_output_lines]
            output_lines = [line for _, line in prior_output_context + output_selected]
            output_context_lines = [line for _, line in prior_output_context]
            output_new_lines = [line for _, line in output_selected]

            if output_selected:
                self._output_cursor_seq = output_selected[-1][0]

            oldest_output_seq = output_items[0][0] if output_items else self._output_cursor_seq
            output_buffer_truncated = bool(output_items) and self._output_cursor_seq < oldest_output_seq - 1

        payload = self.snapshot()
        payload.update(
            {
                "output_lines": output_lines,
                "output_context_lines": output_context_lines,
                "output_new_lines": output_new_lines,
                "output_context_lines_returned": len(output_context_lines),
                "output_new_lines_returned": len(output_new_lines),
                "output_lines_returned": len(output_lines),
                "output_lines_remaining": max(len(unread_output) - len(output_selected), 0),
                "output_buffer_truncated": output_buffer_truncated,
            }
        )
        return payload

    def terminate(self, *, reason: str = "manual") -> dict:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except Exception:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    self.process.kill()
                self.process.wait(timeout=5)
        self._termination_reason = reason
        payload = self.snapshot()
        payload["command_status"] = "terminated"
        payload["is_running"] = False
        return payload

    def should_auto_terminate(self, *, now: float | None = None, idle_timeout_seconds: float = _IDLE_TIMEOUT_SECONDS) -> bool:
        if self.process.poll() is not None or self._termination_reason is not None:
            return False
        current_time = time.time() if now is None else now
        return (
            current_time - self._last_output_at > idle_timeout_seconds
            and current_time - self._last_read_at > idle_timeout_seconds
        )


_SESSIONS: dict[str, _CommandSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_session(session_id: str) -> _CommandSession:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise ValueError(f"Unknown command session: {session_id}")
    return session


def _register_session(session: _CommandSession) -> _CommandSession:
    with _SESSIONS_LOCK:
        _SESSIONS[session.session_id] = session
    return session


def _unregister_session(session_id: str) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(session_id, None)


def _cleanup_sessions() -> None:
    with _SESSIONS_LOCK:
        sessions = list(_SESSIONS.values())
    for session in sessions:
        if session.process.poll() is None:
            try:
                session.process.kill()
            except Exception:
                pass


def _session_watchdog_loop() -> None:
    while True:
        time.sleep(_WATCHDOG_POLL_INTERVAL_SECONDS)
        with _SESSIONS_LOCK:
            sessions = list(_SESSIONS.values())
        for session in sessions:
            try:
                if session.should_auto_terminate():
                    session.terminate(reason="idle_timeout")
            except Exception:
                continue


atexit.register(_cleanup_sessions)
_WATCHDOG_THREAD = threading.Thread(target=_session_watchdog_loop, daemon=True)
_WATCHDOG_THREAD.start()


def check_running_sessions(limit: int = 20) -> dict:
    """List currently running bash sessions.

    Use this before starting another shell job, when resuming work after a pause,
    or when you need to decide which existing session to poll next.

    Example input:
        limit=10

    Example output:
        {"status": "success", "running_count": 1, "sessions": [{"session_id": "...", "command_status": "running", "running_seconds": 12.3}]}
    """
    try:
        normalized_limit = max(1, limit)
        with _SESSIONS_LOCK:
            sessions = list(_SESSIONS.values())
        running_sessions = [
            session.snapshot()
            for session in sessions
            if session.process.poll() is None
        ]
        running_sessions.sort(key=lambda item: item.get("running_seconds", 0.0), reverse=True)
        truncated = len(running_sessions) > normalized_limit
        selected = running_sessions[:normalized_limit]
        return {
            "status": "success",
            "running_count": len(running_sessions),
            "returned_sessions": len(selected),
            "truncated": truncated,
            "sessions": selected,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_message": str(exc),
        }


def run_bash(command: str, initial_wait_seconds: float = _DEFAULT_INITIAL_WAIT_SECONDS) -> dict:
    """Start a new Bash session and return a session handle.

    Use this when a command may stream output over time or run longer than a
    normal one-shot shell call. The tool waits for an initial foreground window
    first. The default foreground window is 30 seconds. If the command finishes
    within that window, the tool returns the captured output and closes the
    session. Otherwise it returns the output seen so far, explicitly states
    that the command continues in the background, and returns a session handle
    for later polling.

    Example input:
        command="python train.py --steps 10000"

    Example output:
        {"status": "success", "session_id": "...", "backgrounded": true, "command_status": "running", "output_lines": ["..."]}
    """
    workdir = get_workspace_root()
    forbidden = find_forbidden_shell_token(command)
    if forbidden:
        return {
            "status": "error",
            "error_message": f"Forbidden shell token detected: {forbidden}. `sudo` commands are not allowed.",
            "working_directory": str(workdir),
        }
    offending_path = find_restricted_workspace_command_path(
        command,
        workdir,
        restricted_mode=is_restricted_mode(),
    )
    if offending_path:
        return restricted_workspace_command_warning(command, offending_path, workdir)
    try:
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            start_new_session=True,
        )
        session = _register_session(
            _CommandSession(process=process, command=command, kind="bash", workdir=workdir)
        )
        if initial_wait_seconds > 0:
            deadline = time.time() + initial_wait_seconds
            while session.process.poll() is None and time.time() < deadline:
                time.sleep(min(_READ_POLL_INTERVAL_SECONDS, max(deadline - time.time(), 0.0)))
        payload = session.read(max_output_lines=80)
        payload["status"] = "success"
        payload["initial_wait_seconds"] = initial_wait_seconds
        wait_seconds_text = f"{initial_wait_seconds:g}"
        if payload["command_status"] == "running":
            payload["backgrounded"] = True
            payload["message"] = (
                f"The command did not finish within the initial {wait_seconds_text}-second window "
                "and continues running in the background."
            )
            payload["next_action_hint"] = (
                "The command is still running in the background. Use wait or read_command_session "
                "with this session_id to fetch more output. Use terminate_command_session to stop it."
            )
        else:
            payload["backgrounded"] = False
            payload["message"] = (
                f"The command finished within the initial {wait_seconds_text}-second window."
            )
            payload["next_action_hint"] = (
                "The command already finished; no background session remains."
            )
            _unregister_session(session.session_id)
        return payload
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Failed to start bash session: {exc}",
            "working_directory": str(workdir),
        }


start_bash_session = run_bash


def read_command_session(
    session_id: str,
    max_output_lines: int = 80,
    wait_seconds: float = _DEFAULT_POLL_WAIT_SECONDS,
) -> dict:
    """Return recent session output and the current status of a running session.

    Use this to poll a running session that was previously returned by
    `bash_exec` or `run_bash`. The returned output lists contain the unread
    increment plus up to the previous 10 already-read lines as context.

    Example input:
        session_id="abc123"

    Example output:
        {"status": "success", "session_id": "abc123", "command_status": "running", "output_lines": ["step 20"]}
    """
    try:
        session = _get_session(session_id)
        deadline = time.time() + max(wait_seconds, 0.0)
        payload = None
        while True:
            payload = session.read(max_output_lines=max_output_lines)
            has_new_output = bool(payload["output_new_lines"])
            if has_new_output or payload["command_status"] != "running":
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(_READ_POLL_INTERVAL_SECONDS, remaining))
        payload["status"] = "success"
        payload["next_action_hint"] = (
            "Use the returned lines and the run status to decide whether to keep polling or terminate the session."
        )
        return payload
    except Exception as exc:
        return {
            "status": "error",
            "error_message": str(exc),
        }


def wait(
    session_id: str,
    wait_seconds: float = _DEFAULT_POLL_WAIT_SECONDS,
    max_output_lines: int = 80,
) -> dict:
    """Wait briefly for a running command session, then return fresh output.

    Use this when a foreground shell command was pushed into the background and
    you want to pause briefly before checking the session again without
    manually passing `wait_seconds` to `read_command_session`.

    Example input:
        session_id="abc123", wait_seconds=2.0

    Example output:
        {"status": "success", "session_id": "abc123", "command_status": "running", "output_lines": ["step 200"]}
    """
    return read_command_session(
        session_id=session_id,
        max_output_lines=max_output_lines,
        wait_seconds=wait_seconds,
    )


def terminate_command_session(
    session_id: str,
    max_output_lines: int = 160,
) -> dict:
    """Stop a background command session and return the final unread output.

    Example input:
        session_id="abc123"

    Example output:
        {"status": "success", "session_id": "abc123", "command_status": "terminated", "output_lines": ["done"]}
    """
    try:
        session = _get_session(session_id)
        termination = session.terminate()
        payload = session.read(max_output_lines=max_output_lines)
        payload["status"] = "success"
        payload["termination"] = termination
        return payload
    except Exception as exc:
        return {
            "status": "error",
            "error_message": str(exc),
        }
