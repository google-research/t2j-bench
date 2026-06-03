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

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_AGENT_PATH = REPO_ROOT / "call_agent.py"
DEFAULT_CONTAINER_REPO_ROOT = Path("/opt/t2j_bench")
DEFAULT_CONTAINER_WORKSPACE = Path("/attempt_workspace")
DEFAULT_CONTAINER_META = Path("/attempt_meta")
DEFAULT_HISTORY_DIRNAME = "agent_history"
DEFAULT_CODEX_HOME_DIRNAME = ".codex"
DEFAULT_CODEX_LAST_MESSAGE = "codex_last_message.md"
DEFAULT_GEMINI_HOME_DIRNAME = ".gemini"
DEFAULT_GEMINI_LAST_MESSAGE = "gemini_last_message.md"
DEFAULT_GEMINI_TRUSTED_FOLDERS = "gemini_trusted_folders.json"
DEFAULT_CLAUDE_HOME_DIRNAME = ".claude"
DEFAULT_CLAUDE_LAST_MESSAGE = "claude_last_message.md"
DEFAULT_CONTAINER_PYTHON = Path("/opt/conda/envs/lf/bin/python")
DEFAULT_CONTAINER_HOME = Path("/home/coding_agent")
DEFAULT_CONTAINER_CODEX_HOME = DEFAULT_CONTAINER_HOME / DEFAULT_CODEX_HOME_DIRNAME
DEFAULT_CONTAINER_GEMINI_HOME = DEFAULT_CONTAINER_HOME / DEFAULT_GEMINI_HOME_DIRNAME
DEFAULT_CONTAINER_CLAUDE_HOME = DEFAULT_CONTAINER_HOME / DEFAULT_CLAUDE_HOME_DIRNAME
_WORKSPACE_STATE_DIRS = (".coding_agent_memory", ".adk", ".attempt", ".tmp")
_TORCH_ORIG_MOUNT_RE = re.compile(r"^(dpo|ppo|sft)/([^/:]+)/torch_orig(?:$|:)")
_HOST_CODEX_FILES = ("auth.json", "config.toml", "managed-config.toml")
_HOST_GEMINI_FILES = ("oauth_creds.json", "google_accounts.json")
_HOST_GEMINI_WRITABLE_FILES = {"oauth_creds.json", "google_accounts.json"}
_HOST_CLAUDE_FILES = (".credentials.json",)
_CODEX_ENV_ALLOWLIST = {"CODEX_API_KEY", "CODEX_CA_CERTIFICATE"}
_GEMINI_ENV_DENY_PREFIXES = ("GEMINI_CLI_IDE_",)
_CLAUDE_ENV_DENY_PREFIXES = ("CLAUDE_CODE_IDE_", "CLAUDE_CODE_SSE_")
_SENSITIVE_ENV_NAME_FRAGMENTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_ENV_SKIP_KEYS = {
    "PATH",
    "PYTHONPATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
    "PWD",
    "OLDPWD",
    "SHLVL",
    "_",
    "PS1",
    "CODEX_HOME",
    "GEMINI_CLI_HOME",
    "GEMINI_CLI_TRUSTED_FOLDERS_PATH",
    "CLAUDE_CONFIG_DIR",
    "LOGNAME",
    "SUDO_USER",
    "USER",
    "USERNAME",
}
_ENV_SKIP_PREFIXES = ("CONDA_PREFIX",)


class _DockerRunInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signum)


def _resolve_docker_cmd() -> list[str]:
    docker_cmd = ["docker"]
    try:
        result = subprocess.run(
            docker_cmd + ["ps"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0:
        return docker_cmd

    sudo_docker_cmd = ["sudo", "-n", "docker"]
    result = subprocess.run(
        sudo_docker_cmd + ["ps"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode == 0:
        return sudo_docker_cmd

    raise SystemExit(
        "Unable to access Docker daemon with either 'docker' or 'sudo -n docker'."
    )


def _normalize_gpus_arg(gpus: str) -> str:
    value = gpus.strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered in {"all", "none"}:
        return value
    if value.isdigit():
        return value
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value
    return f'"{value}"'


def _resolved_path_text(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _host_path_fragments() -> tuple[str, ...]:
    fragments = []
    for path in (REPO_ROOT, Path.home()):
        fragment = _resolved_path_text(path)
        if fragment and fragment != "/":
            fragments.append(fragment)
    return tuple(sorted(fragments, key=len, reverse=True))


def _contains_host_path(value: str) -> bool:
    return any(fragment in value for fragment in _host_path_fragments())


def _passthrough_env_pairs() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key, value in os.environ.items():
        if key in _ENV_SKIP_KEYS:
            continue
        if key.startswith(_ENV_SKIP_PREFIXES):
            continue
        if _contains_host_path(value):
            continue
        if key.startswith("CODEX_") and key not in _CODEX_ENV_ALLOWLIST:
            continue
        if any(key.startswith(prefix) for prefix in _GEMINI_ENV_DENY_PREFIXES):
            continue
        if any(key.startswith(prefix) for prefix in _CLAUDE_ENV_DENY_PREFIXES):
            continue
        items.append((key, value))
    return sorted(items)


def _redact_env_assignment(value: str) -> str:
    key, separator, _ = value.partition("=")
    if not separator:
        return value
    if any(fragment in key.upper() for fragment in _SENSITIVE_ENV_NAME_FRAGMENTS):
        return f"{key}=<redacted>"
    return value


def _host_path_preview_replacements() -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for path, label in (
        (REPO_ROOT, "<repo-root>"),
        (Path.home(), "<home>"),
    ):
        source = _resolved_path_text(path)
        if source and source != "/":
            replacements.append((source, label))
    return sorted(replacements, key=lambda item: len(item[0]), reverse=True)


def _redact_host_paths_for_preview(value: str) -> str:
    redacted = value
    for source, label in _host_path_preview_replacements():
        redacted = redacted.replace(source, label)
    return redacted


def _docker_command_preview(docker_cmd: list[str]) -> str:
    parts: list[str] = []
    redact_next_env = False
    for part in docker_cmd:
        if redact_next_env:
            part = _redact_env_assignment(part)
            redact_next_env = False
        elif part in {"-e", "--env"}:
            redact_next_env = True
        part = _redact_host_paths_for_preview(part)
        parts.append(shlex.quote(part))
    return " ".join(parts)


def _host_user_spec() -> str:
    uid = os.getuid()
    gid = os.getgid()
    if uid == 0 and os.environ.get("SUDO_UID"):
        uid = int(os.environ["SUDO_UID"])
        gid = int(os.environ.get("SUDO_GID", gid))
    return f"{uid}:{gid}"


def _sanitize_docker_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not name:
        name = "attempt"
    if not name[0].isalnum():
        name = f"attempt-{name}"
    return name[:120]


def _default_container_name(attempt_root: Path) -> str:
    stem = _sanitize_docker_name(attempt_root.name)
    return _sanitize_docker_name(f"t2j-bench-attempt-{stem}-{os.getpid()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected coding agent inside a Docker container that "
            "already has a configured conda env named 'lf'."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Docker image that contains the configured 'lf' conda environment.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Host workspace root. Selected files/directories are mounted from here.",
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help=(
            "Workspace-relative path to mount into the container workspace. "
            "Repeatable. Supports RELPATH or RELPATH:CONTAINER_RELPATH. "
            "If omitted, the entire workspace is mounted."
        ),
    )
    parser.add_argument(
        "--prompt",
        help="Inline prompt text. Mutually exclusive with --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Host path to a prompt file. Mutually exclusive with --prompt.",
    )
    parser.add_argument(
        "--agent",
        default="client_agent",
        choices=("client_agent", "iterative_agent", "codex", "gemini", "claude"),
        help="Top-level agent to run inside the container.",
    )
    parser.add_argument(
        "--attempt-root",
        help=(
            "Host directory where attempt artifacts are written. Defaults to "
            "<workspace>/.attempt_docker/<timestamp>."
        ),
    )
    parser.add_argument(
        "--name",
        help="Optional Docker container name.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable verbose agent output. For Claude this switches to realtime "
            "stream-json output so long-running attempts show progress."
        ),
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help=(
            "Pass --yolo to call_agent.py/Gemini CLI, Codex dangerous mode, "
            "or Claude --dangerously-skip-permissions."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the docker command and exit without running it.",
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help=(
            "Value passed to Docker --gpus. Defaults to 'all'. "
            "Use 'none' to disable GPU access."
        ),
    )
    return parser.parse_args()


def _require_prompt(args: argparse.Namespace) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise SystemExit("Provide exactly one of --prompt or --prompt-file.")

    if args.prompt_file:
        path = Path(args.prompt_file).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise SystemExit(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    return args.prompt


def _parse_mount_spec(spec: str, workspace_root: Path) -> tuple[Path, Path]:
    raw_src, raw_dst = (spec.split(":", 1) + [None])[:2]
    host_path = (workspace_root / raw_src).resolve()
    if not host_path.exists():
        raise SystemExit(f"Mount source does not exist: {host_path}")

    try:
        host_path.relative_to(workspace_root)
    except ValueError as exc:
        raise SystemExit(
            f"Mount source escapes workspace root: {host_path} (workspace={workspace_root})"
        ) from exc

    container_rel = Path(raw_dst) if raw_dst else Path(raw_src)
    if container_rel.is_absolute():
        raise SystemExit(f"Container mount target must be relative: {container_rel}")

    return host_path, container_rel


def _build_attempt_root(args: argparse.Namespace, workspace_root: Path) -> Path:
    if args.attempt_root:
        attempt_root = Path(args.attempt_root).expanduser().resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt_name = ts
        for spec in getattr(args, "mount", []) or []:
            match = _TORCH_ORIG_MOUNT_RE.match(spec)
            if match:
                training_method, model_name = match.groups()
                attempt_name = f"{model_name}_{training_method}_{ts}"
                break
        attempt_root = workspace_root / ".attempt_docker" / attempt_name
    attempt_root.mkdir(parents=True, exist_ok=True)
    return attempt_root


def _ensure_workspace_state_dirs(root: Path) -> None:
    for name in _WORKSPACE_STATE_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def _write_attempt_files(attempt_root: Path, prompt_text: str) -> tuple[Path, Path]:
    prompt_path = attempt_root / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    history_dir = attempt_root / DEFAULT_HISTORY_DIRNAME
    history_dir.mkdir(parents=True, exist_ok=True)
    return prompt_path, history_dir


def _write_gemini_trusted_folders(attempt_root: Path) -> Path:
    trusted_folders_path = attempt_root / DEFAULT_GEMINI_TRUSTED_FOLDERS
    trusted_folders_path.write_text(
        json.dumps(
            {str(DEFAULT_CONTAINER_WORKSPACE): "TRUST_FOLDER"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return trusted_folders_path


def _read_host_gemini_auth_type() -> str:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "true":
        return "vertex-ai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini-api-key"

    source_settings = Path.home() / DEFAULT_GEMINI_HOME_DIRNAME / "settings.json"
    try:
        settings = json.loads(source_settings.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "vertex-ai"

    auth_type = (
        settings.get("security", {})
        .get("auth", {})
        .get("selectedType")
    )
    if isinstance(auth_type, str) and auth_type:
        return auth_type
    return "vertex-ai"


def _write_gemini_settings(attempt_root: Path) -> Path:
    gemini_home = _ensure_gemini_home_dir(attempt_root)
    settings_path = gemini_home / "settings.json"
    settings = {
        "security": {
            "auth": {
                "selectedType": _read_host_gemini_auth_type(),
            },
        },
        "ide": {
            "enabled": False,
        },
        "hasSeenIdeIntegrationNudge": True,
        "general": {
            "sessionRetention": {
                "enabled": False,
            },
        },
        "tools": {
            "autoAccept": True,
        },
        "ui": {
            "hideWindowTitle": True,
        },
    }
    settings_path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return settings_path


def _read_host_claude_env_settings() -> dict[str, str]:
    source_settings = Path.home() / DEFAULT_CLAUDE_HOME_DIRNAME / "settings.json"
    try:
        settings = json.loads(source_settings.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    env_settings = settings.get("env")
    if not isinstance(env_settings, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in env_settings.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }


def _write_claude_settings(attempt_root: Path) -> Path:
    claude_home = _ensure_claude_home_dir(attempt_root)
    settings_path = claude_home / "settings.json"
    settings = {
        "autoMemoryEnabled": False,
        "env": _read_host_claude_env_settings(),
        "skipDangerousModePermissionPrompt": True,
    }
    settings_path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return settings_path


def _container_python_command(python_args: str) -> str:
    return (
        f"if [ -x {shlex.quote(str(DEFAULT_CONTAINER_PYTHON))} ]; then "
        f"{shlex.quote(str(DEFAULT_CONTAINER_PYTHON))} {python_args}; "
        "elif command -v conda >/dev/null 2>&1; then "
        f"conda run -n lf --no-capture-output python {python_args}; "
        "elif [ -x /opt/conda/bin/conda ]; then "
        f"/opt/conda/bin/conda run -n lf --no-capture-output python {python_args}; "
        "else "
        "echo 'lf python not found in container' >&2; exit 127; "
        "fi"
    )


def _build_call_agent_command(
    *,
    agent: str,
    verbose: bool,
    yolo: bool,
    prompt_path: Path,
    history_dir: Path,
) -> str:
    args: list[str] = [
        str(DEFAULT_CONTAINER_REPO_ROOT / "call_agent.py"),
        "--workspace",
        str(DEFAULT_CONTAINER_WORKSPACE),
        "--save-path",
        str(DEFAULT_CONTAINER_META / DEFAULT_HISTORY_DIRNAME),
        "--agent",
        agent,
    ]
    if verbose:
        args.append("--verbose")
    if yolo:
        args.append("--yolo")
    args.append(str(DEFAULT_CONTAINER_META / prompt_path.name))
    return " ".join(shlex.quote(part) for part in args)


def _build_codex_exec_command(
    *,
    prompt_path: Path,
    yolo: bool,
) -> str:
    prompt_file = DEFAULT_CONTAINER_META / prompt_path.name
    output_file = DEFAULT_CONTAINER_META / DEFAULT_CODEX_LAST_MESSAGE
    if yolo:
        safety_flags = ["--dangerously-bypass-approvals-and-sandbox"]
    else:
        safety_flags = [
            "--sandbox",
            "workspace-write",
            "--ask-for-approval",
            "never",
        ]

    codex_args = [
        "codex",
        "exec",
        "--cd",
        str(DEFAULT_CONTAINER_WORKSPACE),
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(output_file),
        *safety_flags,
    ]
    quoted_args = " ".join(shlex.quote(part) for part in codex_args)
    return (
        'model_args=(); '
        'if [ -n "${CODING_AGENT_MODEL:-}" ]; then '
        'model_args+=(--model "${CODING_AGENT_MODEL}"); '
        "fi; "
        'reasoning_effort="${CODING_AGENT_REASONING_EFFORT:-high}"; '
        'config_args=(-c "model_reasoning_effort=${reasoning_effort}"); '
        f"{quoted_args} "
        '"${model_args[@]}" "${config_args[@]}" '
        f"- < {shlex.quote(str(prompt_file))}"
    )


def _build_gemini_exec_command(
    *,
    prompt_path: Path,
    yolo: bool,
) -> str:
    prompt_file = DEFAULT_CONTAINER_META / prompt_path.name
    output_file = DEFAULT_CONTAINER_META / DEFAULT_GEMINI_LAST_MESSAGE
    gemini_args = [
        "gemini",
        "--output-format",
        "text",
        "--prompt",
        "",
    ]
    if yolo:
        gemini_args.append("--yolo")

    quoted_args = " ".join(shlex.quote(part) for part in gemini_args)
    return (
        'model_name="${CODING_AGENT_MODEL:-gemini-3.1-pro-preview}"; '
        f"{quoted_args} "
        '--model "${model_name}" '
        f"< {shlex.quote(str(prompt_file))} | tee {shlex.quote(str(output_file))}"
    )


def _build_claude_exec_command(
    *,
    prompt_path: Path,
    verbose: bool,
    yolo: bool,
) -> str:
    prompt_file = DEFAULT_CONTAINER_META / prompt_path.name
    output_file = DEFAULT_CONTAINER_META / DEFAULT_CLAUDE_LAST_MESSAGE
    claude_args = [
        "claude",
        "--print",
        "--no-session-persistence",
        "--setting-sources",
        "user",
    ]
    if verbose:
        claude_args += [
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--include-hook-events",
        ]
    else:
        claude_args += [
            "--output-format",
            "text",
        ]
    if yolo:
        claude_args.append("--dangerously-skip-permissions")

    quoted_args = " ".join(shlex.quote(part) for part in claude_args)
    return (
        'model_name="${CODING_AGENT_MODEL:-opus}"; '
        f"{quoted_args} "
        '--model "${model_name}" '
        f"< {shlex.quote(str(prompt_file))} | tee {shlex.quote(str(output_file))}"
    )


def _ensure_codex_home_dir(root: Path) -> Path:
    codex_home_dir = root / DEFAULT_CODEX_HOME_DIRNAME
    codex_home_dir.mkdir(parents=True, exist_ok=True)
    return codex_home_dir


def _ensure_gemini_home_dir(root: Path) -> Path:
    gemini_home_dir = root / DEFAULT_GEMINI_HOME_DIRNAME
    gemini_home_dir.mkdir(parents=True, exist_ok=True)
    return gemini_home_dir


def _ensure_claude_home_dir(root: Path) -> Path:
    claude_home_dir = root / DEFAULT_CLAUDE_HOME_DIRNAME
    claude_home_dir.mkdir(parents=True, exist_ok=True)
    return claude_home_dir


def _add_codex_mounts(docker_cmd: list[str], attempt_root: Path) -> None:
    host_codex_home = _ensure_codex_home_dir(attempt_root)
    docker_cmd += [
        "--mount",
        f"type=bind,src={host_codex_home},dst={DEFAULT_CONTAINER_CODEX_HOME}",
    ]

    source_codex_home = Path.home() / DEFAULT_CODEX_HOME_DIRNAME
    for filename in _HOST_CODEX_FILES:
        source_file = source_codex_home / filename
        if not source_file.exists() or not source_file.is_file():
            continue
        mount_spec = (
            f"type=bind,src={source_file},dst={DEFAULT_CONTAINER_CODEX_HOME / filename}"
        )
        if filename != "auth.json":
            mount_spec += ",readonly"
        docker_cmd += [
            "--mount",
            mount_spec,
        ]


def _add_gemini_mounts(docker_cmd: list[str], attempt_root: Path) -> None:
    host_gemini_home = _ensure_gemini_home_dir(attempt_root)
    docker_cmd += [
        "--mount",
        f"type=bind,src={host_gemini_home},dst={DEFAULT_CONTAINER_GEMINI_HOME}",
    ]

    source_gemini_home = Path.home() / DEFAULT_GEMINI_HOME_DIRNAME
    for filename in _HOST_GEMINI_FILES:
        source_file = source_gemini_home / filename
        if not source_file.exists() or not source_file.is_file():
            continue
        mount_spec = (
            f"type=bind,src={source_file},dst={DEFAULT_CONTAINER_GEMINI_HOME / filename}"
        )
        if filename not in _HOST_GEMINI_WRITABLE_FILES:
            mount_spec += ",readonly"
        docker_cmd += [
            "--mount",
            mount_spec,
        ]


def _add_claude_mounts(docker_cmd: list[str], attempt_root: Path) -> None:
    host_claude_home = _ensure_claude_home_dir(attempt_root)
    docker_cmd += [
        "--mount",
        f"type=bind,src={host_claude_home},dst={DEFAULT_CONTAINER_CLAUDE_HOME}",
    ]

    source_claude_home = Path.home() / DEFAULT_CLAUDE_HOME_DIRNAME
    for filename in _HOST_CLAUDE_FILES:
        source_file = source_claude_home / filename
        if not source_file.exists() or not source_file.is_file():
            continue
        docker_cmd += [
            "--mount",
            f"type=bind,src={source_file},dst={DEFAULT_CONTAINER_CLAUDE_HOME / filename}",
        ]


def _build_docker_command(
    *,
    image: str,
    workspace_root: Path,
    mount_specs: list[str],
    attempt_root: Path,
    prompt_path: Path,
    history_dir: Path,
    agent: str,
    verbose: bool,
    yolo: bool,
    container_name: str | None,
    gpus: str,
) -> list[str]:
    docker_cmd: list[str] = _resolve_docker_cmd() + ["run", "--rm"]
    if agent not in {"codex", "gemini", "claude"} and sys.stdin.isatty() and sys.stdout.isatty():
        docker_cmd.append("-it")
    if container_name is None:
        container_name = _default_container_name(attempt_root)
    if container_name:
        docker_cmd += ["--name", container_name]
    if agent == "claude":
        docker_cmd += ["--user", _host_user_spec()]
    if gpus and gpus.lower() != "none":
        docker_cmd += ["--gpus", _normalize_gpus_arg(gpus)]

    docker_cmd += [
        "--mount",
        f"type=bind,src={REPO_ROOT},dst={DEFAULT_CONTAINER_REPO_ROOT},readonly",
        "--mount",
        f"type=bind,src={attempt_root},dst={DEFAULT_CONTAINER_META}",
    ]
    if agent == "codex":
        _add_codex_mounts(docker_cmd, attempt_root)
    elif agent == "gemini":
        _add_gemini_mounts(docker_cmd, attempt_root)
    elif agent == "claude":
        _add_claude_mounts(docker_cmd, attempt_root)

    if mount_specs:
        _ensure_workspace_state_dirs(attempt_root)
        for name in _WORKSPACE_STATE_DIRS:
            host_state_dir = attempt_root / name
            docker_cmd += [
                "--mount",
                (
                    f"type=bind,src={host_state_dir},"
                    f"dst={DEFAULT_CONTAINER_WORKSPACE / name}"
                ),
            ]
        for spec in mount_specs:
            host_path, container_rel = _parse_mount_spec(spec, workspace_root)
            docker_cmd += [
                "--mount",
                (
                    f"type=bind,src={host_path},"
                    f"dst={DEFAULT_CONTAINER_WORKSPACE / container_rel}"
                ),
            ]
    else:
        docker_cmd += [
            "--mount",
            f"type=bind,src={workspace_root},dst={DEFAULT_CONTAINER_WORKSPACE}",
        ]

    gcloud_config = Path.home() / ".config" / "gcloud"
    if gcloud_config.exists():
        docker_cmd += [
            "--mount",
            f"type=bind,src={gcloud_config},dst=/root/.config/gcloud,readonly",
            "--mount",
            f"type=bind,src={gcloud_config},dst={DEFAULT_CONTAINER_HOME / '.config' / 'gcloud'},readonly",
        ]

    huggingface_cache = Path.home() / ".cache" / "huggingface"
    if huggingface_cache.exists():
        docker_cmd += [
            "--mount",
            f"type=bind,src={huggingface_cache},dst={DEFAULT_CONTAINER_HOME / '.cache' / 'huggingface'}",
        ]
    huggingface_config = Path.home() / ".huggingface"
    if huggingface_config.exists():
        docker_cmd += [
            "--mount",
            f"type=bind,src={huggingface_config},dst={DEFAULT_CONTAINER_HOME / '.huggingface'}",
        ]

    for key, value in _passthrough_env_pairs():
        docker_cmd += ["-e", f"{key}={value}"]

    docker_cmd += [
        "-e",
        f"HOME={DEFAULT_CONTAINER_HOME}",
        "-e",
        f"CODEX_HOME={DEFAULT_CONTAINER_CODEX_HOME}",
        "-e",
        f"GEMINI_CLI_HOME={DEFAULT_CONTAINER_HOME}",
        "-e",
        f"GEMINI_CLI_TRUSTED_FOLDERS_PATH={DEFAULT_CONTAINER_META / DEFAULT_GEMINI_TRUSTED_FOLDERS}",
        "-e",
        f"XDG_CACHE_HOME={DEFAULT_CONTAINER_HOME / '.cache'}",
        "-e",
        f"HF_HOME={DEFAULT_CONTAINER_HOME / '.cache' / 'huggingface'}",
        "-e",
        f"HUGGINGFACE_HUB_CACHE={DEFAULT_CONTAINER_HOME / '.cache' / 'huggingface' / 'hub'}",
        "-e",
        # Prepend coding-agent to the editable adk-python/LlamaFactory source
        # trees the image exposes on PYTHONPATH. These packages are installed
        # editable and excluded from the packed conda env (conda-pack
        # --ignore-editable-packages), so they are importable only via these
        # paths; overriding PYTHONPATH with just coding-agent would hide them.
        "PYTHONPATH="
        + ":".join(
            str(p)
            for p in (
                DEFAULT_CONTAINER_REPO_ROOT / "coding-agent",
                DEFAULT_CONTAINER_HOME / "adk-python" / "src",
                DEFAULT_CONTAINER_HOME / "LlamaFactory" / "src",
            )
        ),
    ]
    docker_cmd.append(image)

    setup_dirs = [
        DEFAULT_CONTAINER_WORKSPACE,
        DEFAULT_CONTAINER_META / DEFAULT_HISTORY_DIRNAME,
    ]
    if agent == "codex":
        setup_dirs.append(DEFAULT_CONTAINER_CODEX_HOME)
        agent_exec = _build_codex_exec_command(
            prompt_path=prompt_path,
            yolo=yolo,
        )
        working_dir = DEFAULT_CONTAINER_WORKSPACE
    elif agent == "gemini":
        setup_dirs.append(DEFAULT_CONTAINER_GEMINI_HOME)
        agent_exec = _build_gemini_exec_command(
            prompt_path=prompt_path,
            yolo=yolo,
        )
        working_dir = DEFAULT_CONTAINER_WORKSPACE
    elif agent == "claude":
        setup_dirs.append(DEFAULT_CONTAINER_CLAUDE_HOME)
        agent_exec = _build_claude_exec_command(
            prompt_path=prompt_path,
            verbose=verbose,
            yolo=yolo,
        )
        working_dir = DEFAULT_CONTAINER_WORKSPACE
    else:
        call_agent_cmd = _build_call_agent_command(
            agent=agent,
            verbose=verbose,
            yolo=yolo,
            prompt_path=prompt_path,
            history_dir=history_dir,
        )
        agent_exec = _container_python_command(call_agent_cmd)
        working_dir = DEFAULT_CONTAINER_REPO_ROOT

    mkdir_cmd = "mkdir -p " + " ".join(
        shlex.quote(str(path)) for path in setup_dirs
    )
    shell_cmd = (
        "set -euo pipefail; "
        f"{mkdir_cmd}; "
        f"cd {shlex.quote(str(working_dir))}; "
        f"{agent_exec}"
    )
    docker_cmd += ["bash", "-lc", shell_cmd]
    return docker_cmd


def _docker_client_cmd(docker_cmd: list[str]) -> list[str]:
    try:
        run_index = docker_cmd.index("run")
    except ValueError:
        return _resolve_docker_cmd()
    return docker_cmd[:run_index]


def _extract_container_name(docker_cmd: list[str]) -> str | None:
    for index, part in enumerate(docker_cmd[:-1]):
        if part == "--name":
            return docker_cmd[index + 1]
    return None


def _stop_docker_container(docker_client_cmd: list[str], container_name: str | None) -> None:
    if not container_name:
        return
    print(f"Stopping Docker container {container_name}...", file=sys.stderr, flush=True)
    subprocess.run(
        docker_client_cmd + ["rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _interrupt_returncode(signum: int) -> int:
    return 128 + signum


def _run_docker_command(docker_cmd: list[str]) -> int:
    docker_client_cmd = _docker_client_cmd(docker_cmd)
    container_name = _extract_container_name(docker_cmd)
    process: subprocess.Popen[str] | None = None

    def raise_interrupted(signum: int, _frame: object) -> None:
        raise _DockerRunInterrupted(signum)

    previous_sigint = signal.signal(signal.SIGINT, raise_interrupted)
    previous_sigterm = signal.signal(signal.SIGTERM, raise_interrupted)
    try:
        process = subprocess.Popen(docker_cmd, text=True)
        returncode = process.wait()
        if returncode in {
            _interrupt_returncode(signal.SIGINT),
            _interrupt_returncode(signal.SIGTERM),
        }:
            _stop_docker_container(docker_client_cmd, container_name)
        return returncode
    except KeyboardInterrupt:
        signum = signal.SIGINT
    except _DockerRunInterrupted as exc:
        signum = exc.signum
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    _stop_docker_container(docker_client_cmd, container_name)
    if process is not None and process.poll() is None:
        try:
            process.send_signal(signum)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
                process.wait()
    return _interrupt_returncode(signum)


def main() -> int:
    args = _parse_args()
    workspace_root = Path(args.workspace).expanduser().resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise SystemExit(f"Workspace directory not found: {workspace_root}")

    prompt_text = _require_prompt(args)
    attempt_root = _build_attempt_root(args, workspace_root)
    prompt_path, history_dir = _write_attempt_files(attempt_root, prompt_text)
    if args.agent == "gemini":
        _write_gemini_settings(attempt_root)
        _write_gemini_trusted_folders(attempt_root)
    elif args.agent == "claude":
        _write_claude_settings(attempt_root)

    docker_cmd = _build_docker_command(
        image=args.image,
        workspace_root=workspace_root,
        mount_specs=args.mount,
        attempt_root=attempt_root,
        prompt_path=prompt_path,
        history_dir=history_dir,
        agent=args.agent,
        verbose=args.verbose,
        yolo=args.yolo,
        container_name=args.name,
        gpus=args.gpus,
    )

    command_preview = _docker_command_preview(docker_cmd)
    (attempt_root / "docker_command.sh").write_text(
        command_preview + "\n",
        encoding="utf-8",
    )

    print(f"Attempt root: {_redact_host_paths_for_preview(str(attempt_root))}")
    print(f"Prompt file: {_redact_host_paths_for_preview(str(prompt_path))}")
    print(
        "Conversation history dir: "
        f"{_redact_host_paths_for_preview(str(history_dir))}"
    )
    if args.agent == "codex":
        print(
            "Codex last message file: "
            f"{_redact_host_paths_for_preview(str(attempt_root / DEFAULT_CODEX_LAST_MESSAGE))}"
        )
    elif args.agent == "gemini":
        print(
            "Gemini last message file: "
            f"{_redact_host_paths_for_preview(str(attempt_root / DEFAULT_GEMINI_LAST_MESSAGE))}"
        )
    elif args.agent == "claude":
        print(
            "Claude last message file: "
            f"{_redact_host_paths_for_preview(str(attempt_root / DEFAULT_CLAUDE_LAST_MESSAGE))}"
        )
    print("Docker command:")
    print(command_preview)

    if args.dry_run:
        return 0

    return _run_docker_command(docker_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
