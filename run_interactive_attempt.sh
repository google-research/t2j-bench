#!/usr/bin/env bash
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

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_interactive_attempt.sh

Interactive launcher for coding-agent Docker attempts.

It lets you choose:
- task/model repo under dataset/{sft,ppo,dpo}/<model>
- top-level agent
- agent env profile and CODING_AGENT_MODEL override
- docker image
- attempt GPU mapping
- whether to run pytest-based evaluation afterward
- whether to resume an existing batch timestamp and skip completed targets
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
CODING_AGENT_DIR="${REPO_ROOT}/coding-agent"
WORKSPACE_ROOT="${REPO_ROOT}/dataset"
ATTEMPT_SCRIPTS_DIR="${CODING_AGENT_DIR}/docker/coding_agent_lf/attempt_scripts"
INSTRUCTIONS_DIR="${WORKSPACE_ROOT}/instructions"
RUNNER="${CODING_AGENT_DIR}/scripts/run_attempt_in_docker.py"
EVAL_ROOT="${WORKSPACE_ROOT}/eval"
EVAL_CONFIG_DIR="${EVAL_ROOT}/configs"
EVAL_TEMPLATE_DIR="${EVAL_CONFIG_DIR}/templates"
EVAL_MODEL_REGISTRY="${EVAL_CONFIG_DIR}/model_registry.yaml"
BATCH_LOG_ROOT="${WORKSPACE_ROOT}/.attempt_docker/batch_runs"
DEFAULT_ENV_FILE="${CODING_AGENT_DIR}/coding_agent/.env"
EXAMPLE_ENV_FILE="${CODING_AGENT_DIR}/coding_agent/.env.example"
USER_HOME="${HOME:-$(cd "${REPO_ROOT}/.." && pwd)}"
DEFAULT_PYTHON_BIN="${USER_HOME}/miniconda3/envs/lf/bin/python"
MISSING_CANDIDATE_EXIT_CODE=20
ATTEMPT_STATUS_FILE=".interactive_attempt_status"
ATTEMPT_TIMEOUT_SECONDS="${RUN_INTERACTIVE_ATTEMPT_TIMEOUT_SECONDS:-}"
AUTO_EVAL_AFTER_ATTEMPT_FAILURE="${RUN_INTERACTIVE_AUTO_EVAL_AFTER_FAILURE:-0}"

# Resolve the interpreter used for evaluator/helper code. It needs the `lf`
# environment (torch, jax, pytest, yaml, ...), so prefer that env over whatever
# happens to be active (e.g. base). Order: explicit override -> lf env -> active
# env -> historical default.
PYTHON_BIN="${CODING_AGENT_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in \
    "${HOME}/.conda/envs/lf/bin/python" \
    "${HOME}/miniconda3/envs/lf/bin/python" \
    "/opt/conda/envs/lf/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  lf_prefix="$(conda env list | awk '$1=="lf" {print $NF}')"
  if [[ -n "${lf_prefix}" && -x "${lf_prefix}/bin/python" ]]; then
    PYTHON_BIN="${lf_prefix}/bin/python"
  fi
fi
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set CODING_AGENT_PYTHON_BIN to the lf env python, e.g. \$HOME/.conda/envs/lf/bin/python" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi

ACTIVE_CHILD_PIDS=()
ACTIVE_CONTAINER_NAMES=()
INTERRUPT_CLEANUP_RUNNING=0

sanitize_docker_name() {
  local raw="$1"
  local name
  name="$(printf '%s' "${raw}" | tr -c '[:alnum:]_.-' '-')"
  name="${name##[-.]}"
  name="${name%%[-.]}"
  if [[ -z "${name}" ]]; then
    name="attempt"
  fi
  if [[ ! "${name:0:1}" =~ [[:alnum:]] ]]; then
    name="attempt-${name}"
  fi
  printf '%s' "${name:0:120}"
}

attempt_container_name() {
  local task="$1"
  local model="$2"
  local attempt_root="$3"
  sanitize_docker_name "t2j-bench-attempt-${task}-${model}-$(basename "${attempt_root}")"
}

docker_rm_container() {
  local container_name="$1"
  if [[ -z "${container_name}" ]]; then
    return 0
  fi
  if docker ps >/dev/null 2>&1; then
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
  elif sudo -n docker ps >/dev/null 2>&1; then
    sudo -n docker rm -f "${container_name}" >/dev/null 2>&1 || true
  fi
}

terminate_pid_tree() {
  local pid="$1"
  local child
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  while IFS= read -r child; do
    [[ -n "${child}" ]] || continue
    terminate_pid_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -TERM "${pid}" >/dev/null 2>&1 || true
}

kill_pid_tree() {
  local pid="$1"
  local child
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  while IFS= read -r child; do
    [[ -n "${child}" ]] || continue
    kill_pid_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -KILL "${pid}" >/dev/null 2>&1 || true
}

remove_active_child_pid() {
  local target="$1"
  local pid
  local remaining=()
  for pid in "${ACTIVE_CHILD_PIDS[@]:-}"; do
    if [[ "${pid}" != "${target}" ]]; then
      remaining+=("${pid}")
    fi
  done
  ACTIVE_CHILD_PIDS=("${remaining[@]}")
}

cleanup_on_interrupt() {
  local pid container_name
  if (( INTERRUPT_CLEANUP_RUNNING )); then
    exit 130
  fi
  INTERRUPT_CLEANUP_RUNNING=1
  trap - INT TERM

  echo >&2
  echo "Interrupt received; stopping active attempts and Docker containers..." >&2

  for container_name in "${ACTIVE_CONTAINER_NAMES[@]:-}"; do
    docker_rm_container "${container_name}"
  done

  for pid in "${ACTIVE_CHILD_PIDS[@]:-}"; do
    terminate_pid_tree "${pid}"
  done

  sleep 2

  for pid in "${ACTIVE_CHILD_PIDS[@]:-}"; do
    kill_pid_tree "${pid}"
  done

  exit 130
}

trap cleanup_on_interrupt INT TERM

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

resolve_path_input() {
  local raw_path="$1"
  RAW_PATH="${raw_path}" BASE_DIR="${REPO_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

raw_path = os.environ["RAW_PATH"]
base_dir = Path(os.environ["BASE_DIR"])
candidate = Path(raw_path).expanduser()
if not candidate.is_absolute():
    candidate = base_dir / candidate
print(candidate.resolve(strict=False))
PY
}

read_env_value() {
  local env_file="$1"
  local target_key="$2"
  local raw_line line key value first_char last_char

  [[ -f "${env_file}" ]] || return 1

  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="$(trim_whitespace "${raw_line}")"
    [[ -z "${line}" || "${line:0:1}" == "#" || "${line}" != *"="* ]] && continue
    key="$(trim_whitespace "${line%%=*}")"
    value="$(trim_whitespace "${line#*=}")"
    if [[ "${key}" == "${target_key}" ]]; then
      if [[ ${#value} -ge 2 ]]; then
        first_char="${value:0:1}"
        last_char="${value: -1}"
        if [[ ( "${first_char}" == "'" && "${last_char}" == "'" ) || ( "${first_char}" == '"' && "${last_char}" == '"' ) ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi
      printf '%s' "${value}"
      return 0
    fi
  done < "${env_file}"

  return 1
}

clear_agent_env_profile() {
  unset GOOGLE_GENAI_USE_VERTEXAI || true
  unset GOOGLE_CLOUD_PROJECT || true
  unset GOOGLE_CLOUD_LOCATION || true
  unset OPENAI_BASE_URL || true
  unset OPENAI_API_BASE || true
  unset OPENAI_API_KEY || true
  unset CODING_AGENT_MODEL || true
  unset CODING_AGENT_REASONING_EFFORT || true
  unset CODING_AGENT_REFRESH_OPENAI_GOOGLE_AUTH || true
  unset CODING_AGENT_LITELLM_TIMEOUT || true
  unset CODING_AGENT_LITELLM_MAX_RETRIES || true
  unset CODING_AGENT_LITELLM_RETRY_ATTEMPTS || true
  unset CODING_AGENT_LITELLM_RETRY_INITIAL_DELAY || true
  unset CODING_AGENT_LITELLM_RETRY_MAX_DELAY || true
  unset CODING_AGENT_LITELLM_RETRY_MULTIPLIER || true
  unset CODING_AGENT_LITELLM_RETRY_JITTER || true
  unset CODING_AGENT_MAX_OUTPUT_TOKENS || true
  unset CODING_AGENT_THINKING_LEVEL || true
  unset CODING_AGENT_INCLUDE_THOUGHTS || true
  unset CODING_AGENT_PRINT_THOUGHTS || true
  unset CODING_AGENT_PRINT_TOOL_CALL_COUNTS || true
  unset CODING_AGENT_COMPACT_CONTEXT_LIMIT || true
  unset CODING_AGENT_COMPACT_TRIGGER_RATIO || true
  unset CODING_AGENT_COMPACT_KEEP_RATIO || true
  unset ADK_PRESERVE_THOUGHT_SIGNATURES || true
}

load_simple_env_file() {
  local env_file="$1"
  local raw_line line key value first_char last_char

  if [[ ! -f "${env_file}" ]]; then
    echo "Env file not found: ${env_file}" >&2
    return 1
  fi

  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="$(trim_whitespace "${raw_line}")"
    [[ -z "${line}" || "${line:0:1}" == "#" || "${line}" != *"="* ]] && continue
    key="$(trim_whitespace "${line%%=*}")"
    value="$(trim_whitespace "${line#*=}")"
    if [[ ${#value} -ge 2 ]]; then
      first_char="${value:0:1}"
      last_char="${value: -1}"
      if [[ ( "${first_char}" == "'" && "${last_char}" == "'" ) || ( "${first_char}" == '"' && "${last_char}" == '"' ) ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    export "${key}=${value}"
  done < "${env_file}"
}

prompt_text() {
  local prompt="$1"
  local default_value="${2-}"
  local reply
  if [[ -n "${default_value}" ]]; then
    read -r -p "${prompt} [${default_value}]: " reply
    if [[ -z "${reply}" ]]; then
      reply="${default_value}"
    fi
  else
    read -r -p "${prompt}: " reply
  fi
  printf '%s' "${reply}"
}

prompt_yes_no() {
  local prompt="$1"
  local default_value="${2:-y}"
  local reply normalized

  while true; do
    if [[ "${default_value}" == "y" ]]; then
      read -r -p "${prompt} [Y/n]: " reply
      reply="${reply:-y}"
    else
      read -r -p "${prompt} [y/N]: " reply
      reply="${reply:-n}"
    fi
    normalized="$(printf '%s' "${reply}" | tr '[:upper:]' '[:lower:]')"
    case "${normalized}" in
      y|yes)
        return 0
        ;;
      n|no)
        return 1
        ;;
    esac
    echo "Please answer y or n." >&2
  done
}

prompt_choice() {
  local prompt="$1"
  local default_index="$2"
  shift 2
  local options=("$@")
  local count="${#options[@]}"
  local reply index

  if (( count == 0 )); then
    echo "prompt_choice called with no options" >&2
    exit 1
  fi

  while true; do
    echo >&2
    echo "${prompt}" >&2
    for index in "${!options[@]}"; do
      printf '  %2d. %s' "$((index + 1))" "${options[$index]}" >&2
      if (( index + 1 == default_index )); then
        printf ' [default]' >&2
      fi
      printf '\n' >&2
    done
    read -r -p "Enter choice [${default_index}]: " reply
    reply="${reply:-${default_index}}"
    if [[ "${reply}" =~ ^[0-9]+$ ]] && (( reply >= 1 && reply <= count )); then
      printf '%s' "${options[$((reply - 1))]}"
      return 0
    fi
    echo "Invalid choice: ${reply}" >&2
  done
}

discover_models() {
  local task="$1"
  local models=()
  local supported_models=()
  local model_dir model_name

  mapfile -t supported_models < <(
    REGISTRY_PATH="${EVAL_MODEL_REGISTRY}" TASK_NAME="${task}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import yaml

registry_path = Path(os.environ["REGISTRY_PATH"]).resolve()
task = os.environ["TASK_NAME"]

if not registry_path.is_file():
    raise SystemExit(f"Evaluation model registry not found: {registry_path}")

registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
for model_name in sorted((registry.get(task) or {}).keys()):
    print(model_name)
PY
  )

  if (( ${#supported_models[@]} == 0 )); then
    echo "No supported models registered for task ${task} in ${EVAL_MODEL_REGISTRY}" >&2
    exit 1
  fi

  while IFS= read -r model_dir; do
    [[ -d "${model_dir}/torch_orig" ]] || continue
    model_name="$(basename "${model_dir}")"
    for supported_model in "${supported_models[@]}"; do
      if [[ "${model_name}" == "${supported_model}" ]]; then
        models+=("${model_name}")
        break
      fi
    done
  done < <(find "${WORKSPACE_ROOT}/${task}" -mindepth 1 -maxdepth 1 -type d | sort)

  if (( ${#models[@]} == 0 )); then
    echo "No supported models found under ${WORKSPACE_ROOT}/${task}" >&2
    exit 1
  fi

  printf '%s\n' "${models[@]}"
}

discover_all_targets() {
  local task
  local models=()
  local model_name

  ALL_TARGET_TASKS=()
  ALL_TARGET_MODELS=()

  for task in dpo ppo sft; do
    mapfile -t models < <(discover_models "${task}")
    for model_name in "${models[@]}"; do
      ALL_TARGET_TASKS+=("${task}")
      ALL_TARGET_MODELS+=("${model_name}")
    done
  done

  if (( ${#ALL_TARGET_TASKS[@]} == 0 )); then
    echo "No supported targets found for task=all" >&2
    exit 1
  fi
}

resolve_instruction_template() {
  local task="$1"
  local template_path="${INSTRUCTIONS_DIR}/${task}.md"
  if [[ ! -f "${template_path}" ]]; then
    echo "Instruction template not found: ${template_path}" >&2
    exit 1
  fi
  printf '%s' "${template_path}"
}

render_instruction_template() {
  local template_path="$1"
  local model_name="$2"
  local output_path="$3"

  TEMPLATE_PATH="${template_path}" MODEL_NAME="${model_name}" OUTPUT_PATH="${output_path}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

template_path = Path(os.environ["TEMPLATE_PATH"]).resolve()
model_name = os.environ["MODEL_NAME"]
output_path = Path(os.environ["OUTPUT_PATH"]).resolve()

text = template_path.read_text(encoding="utf-8")
text = text.replace("{{MODEL_NAME}}", model_name)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(text, encoding="utf-8")
PY
}

resolve_eval_template() {
  local task="$1"
  local template_path="${EVAL_TEMPLATE_DIR}/${task}.yaml"
  if [[ ! -f "${template_path}" ]]; then
    echo "Evaluation template not found: ${template_path}" >&2
    exit 1
  fi
  printf '%s' "${template_path}"
}

ensure_eval_supported() {
  local task="$1"
  local model="$2"

  REGISTRY_PATH="${EVAL_MODEL_REGISTRY}" TASK_NAME="${task}" MODEL_NAME="${model}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import sys
import yaml

registry_path = Path(os.environ["REGISTRY_PATH"]).resolve()
task = os.environ["TASK_NAME"]
model = os.environ["MODEL_NAME"]

if not registry_path.is_file():
    raise SystemExit(f"Evaluation model registry not found: {registry_path}")

registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
if task not in registry or model not in registry[task]:
    available = ", ".join(sorted((registry.get(task) or {}).keys()))
    if not available:
        available = "<none>"
    raise SystemExit(
        f"Automatic evaluation config is not defined for {task}/{model}. "
        f"Supported models for {task}: {available}"
    )
PY
}

render_eval_config() {
  local template_path="$1"
  local registry_path="$2"
  local task="$3"
  local model="$4"
  local reference_repo="$5"
  local candidate_repo="$6"
  local output_path="$7"
  local generated_output_dir="$8"
  local visible_devices="$9"

  TEMPLATE_PATH="${template_path}" \
  REGISTRY_PATH="${registry_path}" \
  TASK_NAME="${task}" \
  MODEL_NAME="${model}" \
  REFERENCE_REPO="${reference_repo}" \
  CANDIDATE_REPO="${candidate_repo}" \
  OUTPUT_PATH="${output_path}" \
  GENERATED_OUTPUT_DIR="${generated_output_dir}" \
  VISIBLE_DEVICES="${visible_devices}" \
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import copy
import os
import yaml

template_path = Path(os.environ["TEMPLATE_PATH"]).resolve()
registry_path = Path(os.environ["REGISTRY_PATH"]).resolve()
task = os.environ["TASK_NAME"]
model = os.environ["MODEL_NAME"]
reference_repo = str(Path(os.environ["REFERENCE_REPO"]).resolve())
candidate_repo = str(Path(os.environ["CANDIDATE_REPO"]).resolve())
output_path = Path(os.environ["OUTPUT_PATH"]).resolve()
generated_output_dir = str(Path(os.environ["GENERATED_OUTPUT_DIR"]).resolve())
visible_devices = [item.strip() for item in os.environ["VISIBLE_DEVICES"].split(",") if item.strip()]

template = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
model_meta = copy.deepcopy((registry.get(task) or {}).get(model))
if model_meta is None:
    supported = ", ".join(sorted((registry.get(task) or {}).keys())) or "<none>"
    raise SystemExit(
        f"Automatic evaluation config is not defined for {task}/{model}. "
        f"Supported models for {task}: {supported}"
    )

precision_profile = model_meta.pop("precision_profile", template.get("precision_profile", "bf16_compare"))
shared_config = model_meta.pop("shared_config", {})
if model_meta:
    raise SystemExit(f"Unexpected eval registry fields for {task}/{model}: {sorted(model_meta)}")

if visible_devices:
    reference_device = "cuda:0"
    candidate_device = "cuda:0"
    backend_device = "cuda:1" if len(visible_devices) >= 2 else "cuda:0"
else:
    reference_device = "cpu"
    candidate_device = "cpu"
    backend_device = "cpu"

template["precision_profile"] = precision_profile
template["output_dir"] = generated_output_dir
template["shared_config"] = shared_config
template["implementations"] = [
    {
        "name": template.get("reference_impl", "torch_orig"),
        "framework": "pytorch",
        "repo_dir": reference_repo,
        "device": reference_device,
    },
    {
        "name": template.get("candidate_impl", "jax_repo"),
        "framework": "flax",
        "repo_dir": candidate_repo,
        "device": candidate_device,
        "backend_device": backend_device,
    },
]

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
PY
}

resolve_env_profile() {
  local profile_choice custom_env
  local choices=(
    "default profile (.env)"
    "custom env file"
  )

  profile_choice="$(prompt_choice "Choose agent env profile" 1 "${choices[@]}")"
  case "${profile_choice}" in
    "default profile (.env)")
      if [[ ! -f "${DEFAULT_ENV_FILE}" ]]; then
        echo "Reminder: ${DEFAULT_ENV_FILE} does not exist." >&2
        echo "Create it before running, e.g. cp ${EXAMPLE_ENV_FILE} ${DEFAULT_ENV_FILE}" >&2
      fi
      printf '%s' "${DEFAULT_ENV_FILE}"
      ;;
    "custom env file")
      while true; do
        custom_env="$(prompt_text "Custom env file path")"
        custom_env="$(resolve_path_input "${custom_env}")"
        if [[ -f "${custom_env}" ]]; then
          printf '%s' "${custom_env}"
          return 0
        fi
        echo "Env file not found: ${custom_env}" >&2
      done
      ;;
  esac
}

resolve_codex_env_profile() {
  local profile_choice custom_env
  local choices=(
    "shell env only"
    "custom env file"
  )

  profile_choice="$(prompt_choice "Choose Codex env source" 1 "${choices[@]}")"
  case "${profile_choice}" in
    "shell env only")
      printf '%s' ""
      ;;
    "custom env file")
      while true; do
        custom_env="$(prompt_text "Custom env file path")"
        custom_env="$(resolve_path_input "${custom_env}")"
        if [[ -f "${custom_env}" ]]; then
          printf '%s' "${custom_env}"
          return 0
        fi
        echo "Env file not found: ${custom_env}" >&2
      done
      ;;
  esac
}

resolve_gemini_env_profile() {
  local profile_choice custom_env
  local choices=(
    "shell env only"
    "custom env file"
  )

  profile_choice="$(prompt_choice "Choose Gemini env source" 1 "${choices[@]}")"
  case "${profile_choice}" in
    "shell env only")
      printf '%s' ""
      ;;
    "custom env file")
      while true; do
        custom_env="$(prompt_text "Custom env file path")"
        custom_env="$(resolve_path_input "${custom_env}")"
        if [[ -f "${custom_env}" ]]; then
          printf '%s' "${custom_env}"
          return 0
        fi
        echo "Env file not found: ${custom_env}" >&2
      done
      ;;
  esac
}

resolve_claude_env_profile() {
  local profile_choice custom_env
  local choices=(
    "shell env only"
    "custom env file"
  )

  profile_choice="$(prompt_choice "Choose Claude env source" 1 "${choices[@]}")"
  case "${profile_choice}" in
    "shell env only")
      printf '%s' ""
      ;;
    "custom env file")
      while true; do
        custom_env="$(prompt_text "Custom env file path")"
        custom_env="$(resolve_path_input "${custom_env}")"
        if [[ -f "${custom_env}" ]]; then
          printf '%s' "${custom_env}"
          return 0
        fi
        echo "Env file not found: ${custom_env}" >&2
      done
      ;;
  esac
}

agent_uses_staged_jax_repo() {
  local agent_name="${1:-}"
  [[ "${agent_name}" == "client_agent" || "${agent_name}" == "codex" || "${agent_name}" == "gemini" || "${agent_name}" == "claude" ]]
}

derive_cuda_visible_devices() {
  local gpu_spec="$1"
  if [[ "${gpu_spec}" =~ ^device=([0-9,]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "${gpu_spec}" =~ ^\"device=([0-9,]+)\"$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "${gpu_spec}" =~ ^\'device=([0-9,]+)\'$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '%s' "0,1"
  fi
}

find_candidate_repo() {
  local attempt_root="$1"
  shift
  local candidate_hint

  for candidate_hint in "$@"; do
    if is_candidate_repo_dir "${candidate_hint}"; then
      printf '%s\n' "${candidate_hint}"
      return 0
    fi
  done

  find "${attempt_root}/.attempt" -maxdepth 2 -type d -path '*/attempt_*/jax_repo' | sort | tail -n 1
}

workspace_relative_path() {
  local abs_path="$1"

  ABS_PATH="${abs_path}" WORKSPACE_PATH="${WORKSPACE_ROOT}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

abs_path = Path(os.environ["ABS_PATH"]).resolve()
workspace = Path(os.environ["WORKSPACE_PATH"]).resolve()
print(abs_path.relative_to(workspace).as_posix())
PY
}

prepare_client_agent_stage() {
  local repo_dir="$1"
  local stage_parent="$2"

  rm -rf "${stage_parent}"
  mkdir -p "${stage_parent}"
  cp -a "${repo_dir}" "${stage_parent}/"
}

is_candidate_repo_dir() {
  local candidate_dir="${1:-}"

  [[ -n "${candidate_dir}" ]] || return 1
  [[ -d "${candidate_dir}" ]] || return 1
  [[ -f "${candidate_dir}/model/loader.py" ]] || return 1
  return 0
}

prepare_client_agent_root_jax_repo() {
  local root_jax_repo_dir="$1"

  rm -rf "${root_jax_repo_dir}"
  mkdir -p "${root_jax_repo_dir}"
}

finalize_client_agent_candidate() {
  local source_repo="$1"
  local attempt_root="$2"
  local archived_repo="${attempt_root}/jax_repo"

  if ! is_candidate_repo_dir "${source_repo}"; then
    return 1
  fi

  rm -rf "${archived_repo}"
  if cp -R "${source_repo}" "${archived_repo}"; then
    printf '%s\n' "${archived_repo}"
    return 0
  fi

  echo "Warning: failed to archive ${source_repo} into ${archived_repo}; using source path for evaluation." >&2
  printf '%s\n' "${source_repo}"
}

find_eval_summary_path() {
  local attempt_root="$1"

  ATTEMPT_ROOT="${attempt_root}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

attempt_root = Path(os.environ["ATTEMPT_ROOT"]).resolve()
preferred = sorted(
    attempt_root.glob("pytest_eval_outputs/**/new_eval_summary.json"),
    key=lambda path: path.stat().st_mtime,
)
if preferred:
    print(preferred[-1])
    raise SystemExit(0)

fallback = sorted(
    attempt_root.glob("**/new_eval_summary.json"),
    key=lambda path: path.stat().st_mtime,
)
if fallback:
    print(fallback[-1])
PY
}

write_attempt_status() {
  local attempt_root="$1"
  local task="$2"
  local model="$3"
  local status="$4"
  local exit_code="$5"
  local message="$6"
  local status_path="${attempt_root}/${ATTEMPT_STATUS_FILE}"

  mkdir -p "${attempt_root}"
  {
    printf 'task=%s\n' "${task}"
    printf 'model=%s\n' "${model}"
    printf 'status=%s\n' "${status}"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'message=%s\n' "${message}"
  } > "${status_path}"
}

read_attempt_status_value() {
  local status_path="$1"
  local key="$2"
  local line

  [[ -f "${status_path}" ]] || return 1
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" == "${key}="* ]]; then
      printf '%s' "${line#*=}"
      return 0
    fi
  done < "${status_path}"
  return 1
}

batch_target_completion_reason() {
  local task="$1"
  local model="$2"
  local attempt_root="$3"
  local log_file="$4"
  local summary_path status_path status

  summary_path="$(find_eval_summary_path "${attempt_root}")"
  if [[ -n "${summary_path}" ]]; then
    printf 'evaluation summary exists: %s' "${summary_path}"
    return 0
  fi

  status_path="${attempt_root}/${ATTEMPT_STATUS_FILE}"
  status="$(read_attempt_status_value "${status_path}" status || true)"
  case "${status}" in
    evaluation_completed)
      printf 'evaluation already completed'
      return 0
      ;;
    missing_candidate)
      printf 'previous attempt completed without a candidate jax_repo'
      return 0
      ;;
    attempt_completed)
      if (( ! RUN_EVAL )); then
        printf 'attempt already completed'
        return 0
      fi
      ;;
  esac

  if [[ -f "${log_file}" ]] && grep -F "No candidate jax_repo" "${log_file}" >/dev/null 2>&1 && grep -F "${task}/${model}" "${log_file}" >/dev/null 2>&1; then
    printf 'previous log shows no candidate jax_repo'
    return 0
  fi

  if (( ! RUN_EVAL )) && [[ -f "${log_file}" ]] && grep -F "Attempt exit code for ${task}/${model}: 0" "${log_file}" >/dev/null 2>&1; then
    printf 'previous log shows successful attempt'
    return 0
  fi

  return 1
}

latest_batch_timestamp() {
  local task="$1"
  local prefix latest_dir basename_value

  if [[ "${task}" == "all" ]]; then
    prefix="all"
  else
    prefix="${task}"
  fi

  [[ -d "${BATCH_LOG_ROOT}" ]] || return 1
  latest_dir="$(find "${BATCH_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null | sort | tail -n 1)"
  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi

  basename_value="$(basename "${latest_dir}")"
  printf '%s' "${basename_value#${prefix}_}"
}

validate_batch_timestamp() {
  local batch_timestamp="$1"

  if [[ ! "${batch_timestamp}" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
    echo "Invalid batch timestamp: ${batch_timestamp}" >&2
    echo "Expected format: YYYYMMDDTHHMMSSZ" >&2
    exit 1
  fi
}

print_eval_pass_rate_summary() {
  if (( $# == 0 )); then
    return 0
  fi

  "${PYTHON_BIN}" - "$@" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def find_summary(attempt_root: Path) -> Path | None:
    preferred = sorted(
        attempt_root.glob("pytest_eval_outputs/**/new_eval_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if preferred:
        return preferred[-1]

    fallback = sorted(
        attempt_root.glob("**/new_eval_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if fallback:
        return fallback[-1]
    return None


raw = sys.argv[1:]
if len(raw) % 2 != 0:
    raise SystemExit("Expected label/attempt_root pairs.")

rows = []
for idx in range(0, len(raw), 2):
    label = raw[idx]
    attempt_root = Path(raw[idx + 1]).resolve()
    summary_path = find_summary(attempt_root)
    row = {
        "label": label,
        "attempt_root": str(attempt_root),
        "summary_path": None if summary_path is None else str(summary_path),
        "overall": False,
        "spec": False,
        "numeric": False,
        "behavioral": False,
        "missing": summary_path is None,
    }
    if summary_path is not None:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        row["overall"] = data.get("overall") == "PASS"
        row["spec"] = bool((data.get("spec") or {}).get("passed"))
        row["numeric"] = bool((data.get("numeric") or {}).get("passed"))
        row["behavioral"] = bool((data.get("behavioral") or {}).get("passed"))
        row["missing"] = False
    rows.append(row)

total = len(rows)
with_summary = sum(0 if row["missing"] else 1 for row in rows)

def fmt_ratio(passed: int) -> str:
    pct = 0.0 if total == 0 else passed * 100.0 / total
    return f"{passed}/{total} ({pct:.1f}%)"

overall_pass = sum(1 for row in rows if row["overall"])
spec_pass = sum(1 for row in rows if row["spec"])
numeric_pass = sum(1 for row in rows if row["numeric"])
behavioral_pass = sum(1 for row in rows if row["behavioral"])

print()
print("Evaluation pass rates for this run")
print(f"  summaries found: {with_summary}/{total}")
print(f"  overall: {fmt_ratio(overall_pass)}")
print(f"  spec: {fmt_ratio(spec_pass)}")
print(f"  numeric: {fmt_ratio(numeric_pass)}")
print(f"  behavioral: {fmt_ratio(behavioral_pass)}")

print("  per target:")
for row in rows:
    if row["missing"]:
        print(f"    - {row['label']}: NO_SUMMARY ({row['attempt_root']})")
        continue
    overall = "PASS" if row["overall"] else "FAIL"
    spec = "PASS" if row["spec"] else "FAIL"
    numeric = "PASS" if row["numeric"] else "FAIL"
    behavioral = "PASS" if row["behavioral"] else "FAIL"
    print(
        f"    - {row['label']}: overall={overall} spec={spec} "
        f"numeric={numeric} behavioral={behavioral}"
    )
PY
}

parse_gpu_pairs_spec() {
  local spec="$1"
  local idx pair

  PARSED_GPU_PAIRS=()
  IFS=';' read -r -a raw_pairs <<< "${spec}"
  if (( ${#raw_pairs[@]} == 0 )); then
    echo "No GPU pairs configured" >&2
    exit 1
  fi

  for idx in "${!raw_pairs[@]}"; do
    pair="$(echo "${raw_pairs[$idx]}" | tr -d '[:space:]')"
    if [[ ! "${pair}" =~ ^[0-9]+,[0-9]+$ ]]; then
      echo "Invalid GPU pair: ${raw_pairs[$idx]}" >&2
      exit 1
    fi
    PARSED_GPU_PAIRS+=("${pair}")
  done
}

run_model_attempt() {
  local task="$1"
  local model="$2"
  local gpu_spec="$3"
  local eval_visible_devices="$4"
  local attempt_root="$5"
  local allow_eval_after_failure="${6:-0}"

  local repo_rel repo_dir mount_rel rendered_prompt_file runner_status candidate_repo
  local client_agent_stage_parent="" client_agent_model_jax_repo="" client_agent_root_jax_repo_dir=""
  local candidate_hint="" candidate_hint_fallback="" attempt_name="" docker_container_name=""
  local generated_eval_config generated_output_dir pytest_output_root eval_summary_path=""
  local prompt_template eval_template runner_args=() runner_cmd=() pytest_cmd=()

  repo_rel="${task}/${model}/torch_orig"
  repo_dir="${WORKSPACE_ROOT}/${repo_rel}"
  mount_rel="${repo_rel}"
  docker_container_name="${RUN_MODEL_CONTAINER_NAME:-$(attempt_container_name "${task}" "${model}" "${attempt_root}")}"
  ACTIVE_CONTAINER_NAMES+=("${docker_container_name}")
  if [[ ! -d "${repo_dir}" ]]; then
    echo "Selected repo not found: ${repo_dir}" >&2
    return 1
  fi

  if agent_uses_staged_jax_repo "${AGENT_NAME}"; then
    attempt_name="$(basename "${attempt_root}")"
    client_agent_stage_parent="${WORKSPACE_ROOT}/.attempt_docker/client_agent_stage/${attempt_name}/${task}/${model}"
    client_agent_model_jax_repo="${client_agent_stage_parent}/jax_repo"
    client_agent_root_jax_repo_dir="${WORKSPACE_ROOT}/.attempt_docker/client_agent_stage/${attempt_name}/workspace_root_jax_repo"
    prepare_client_agent_stage "${repo_dir}" "${client_agent_stage_parent}"
    prepare_client_agent_root_jax_repo "${client_agent_root_jax_repo_dir}"
    mount_rel="$(workspace_relative_path "${client_agent_stage_parent}"):${task}/${model}"
    candidate_hint="${client_agent_model_jax_repo}"
    candidate_hint_fallback="${client_agent_root_jax_repo_dir}"
  fi

  prompt_template="$(resolve_instruction_template "${task}")"
  rendered_prompt_file="${attempt_root}/rendered_prompt.md"

  echo
  echo "Starting attempt for ${task}/${model}"
  echo "  repo: ${repo_rel}"
  echo "  prompt template: ${prompt_template}"
  echo "  rendered prompt: ${rendered_prompt_file}"
  echo "  attempt gpus: ${gpu_spec}"
  echo "  attempt root: ${attempt_root}"
  echo "  docker container: ${docker_container_name}"
  if [[ -n "${ATTEMPT_TIMEOUT_SECONDS}" ]]; then
    echo "  attempt timeout: ${ATTEMPT_TIMEOUT_SECONDS}s"
  fi
  if agent_uses_staged_jax_repo "${AGENT_NAME}"; then
    echo "  isolated staged repo: ${client_agent_stage_parent}"
    echo "  persistent workspace jax_repo dir: ${client_agent_root_jax_repo_dir}"
  fi

  mkdir -p "${attempt_root}"
  render_instruction_template "${prompt_template}" "${model}" "${rendered_prompt_file}"

  if [[ -n "${ENV_FILE}" ]]; then
    clear_agent_env_profile
    load_simple_env_file "${ENV_FILE}"
  fi
  export CODING_AGENT_MODEL="${BASE_MODEL}"
  if [[ "${AGENT_NAME}" == "codex" ]]; then
    export CODING_AGENT_REASONING_EFFORT="${AGENT_REASONING_EFFORT}"
  fi

  runner_args=(
    --image "${DOCKER_IMAGE}"
    --workspace "${WORKSPACE_ROOT}"
    --attempt-root "${attempt_root}"
    --mount "${mount_rel}"
    --prompt-file "${rendered_prompt_file}"
    --agent "${AGENT_NAME}"
    --gpus "${gpu_spec}"
    --name "${docker_container_name}"
  )

  if ! agent_uses_staged_jax_repo "${AGENT_NAME}"; then
    runner_args+=(--mount "eval")
  else
    runner_args+=(--mount "$(workspace_relative_path "${client_agent_root_jax_repo_dir}"):jax_repo")
  fi

  if (( VERBOSE_FLAG )); then
    runner_args+=(--verbose)
  fi
  if (( YOLO_FLAG )); then
    runner_args+=(--yolo)
  fi

  runner_cmd=("${PYTHON_BIN}" "${RUNNER}" "${runner_args[@]}")
  if [[ -n "${ATTEMPT_TIMEOUT_SECONDS}" ]]; then
    runner_cmd=(timeout --foreground --kill-after=60s "${ATTEMPT_TIMEOUT_SECONDS}" "${runner_cmd[@]}")
  fi

  set +e
  "${runner_cmd[@]}"
  runner_status=$?
  set -e

  echo "Attempt exit code for ${task}/${model}: ${runner_status}"

  if agent_uses_staged_jax_repo "${AGENT_NAME}"; then
    if candidate_repo="$(finalize_client_agent_candidate "${client_agent_model_jax_repo}" "${attempt_root}")"; then
      candidate_hint="${candidate_repo}"
      echo "Archived staged jax_repo to ${candidate_repo}"
    elif candidate_repo="$(finalize_client_agent_candidate "${client_agent_root_jax_repo_dir}" "${attempt_root}")"; then
      candidate_hint="${candidate_repo}"
      echo "Archived staged jax_repo to ${candidate_repo}"
    else
      candidate_hint=""
      candidate_hint_fallback=""
    fi
  fi

  if agent_uses_staged_jax_repo "${AGENT_NAME}" && (( runner_status == 0 )); then
    candidate_repo="$(find_candidate_repo "${attempt_root}" "${candidate_hint}" "${candidate_hint_fallback}")"
    if [[ -z "${candidate_repo}" ]]; then
      echo "No candidate jax_repo found for ${task}/${model} after a successful attempt." >&2
      write_attempt_status "${attempt_root}" "${task}" "${model}" "missing_candidate" "${MISSING_CANDIDATE_EXIT_CODE}" "successful attempt did not produce a candidate jax_repo"
      return "${MISSING_CANDIDATE_EXIT_CODE}"
    fi
    candidate_hint="${candidate_repo}"
    candidate_hint_fallback=""
  fi

  if (( ! RUN_EVAL )); then
    if (( runner_status == 0 )); then
      write_attempt_status "${attempt_root}" "${task}" "${model}" "attempt_completed" "${runner_status}" "attempt completed without pytest evaluation"
    else
      write_attempt_status "${attempt_root}" "${task}" "${model}" "attempt_failed" "${runner_status}" "attempt failed before pytest evaluation"
    fi
    return "${runner_status}"
  fi

  eval_template="$(resolve_eval_template "${task}")"

  if (( runner_status != 0 )); then
    if (( allow_eval_after_failure )); then
      if (( AUTO_EVAL_AFTER_ATTEMPT_FAILURE )); then
        echo "Attempt failed for ${task}/${model}; trying pytest evaluation because RUN_INTERACTIVE_AUTO_EVAL_AFTER_FAILURE is enabled."
      elif ! prompt_yes_no "Attempt failed for ${task}/${model}. Try pytest evaluation anyway if a candidate repo exists?" "n"; then
        write_attempt_status "${attempt_root}" "${task}" "${model}" "attempt_failed" "${runner_status}" "attempt failed and pytest evaluation was not requested"
        return "${runner_status}"
      fi
    else
      write_attempt_status "${attempt_root}" "${task}" "${model}" "attempt_failed" "${runner_status}" "attempt failed before pytest evaluation"
      return "${runner_status}"
    fi
  fi

  candidate_repo="$(find_candidate_repo "${attempt_root}" "${candidate_hint}" "${candidate_hint_fallback}")"
  if [[ -z "${candidate_repo}" ]]; then
    echo "No candidate jax_repo found for ${task}/${model}; skipping evaluation for this target." >&2
    write_attempt_status "${attempt_root}" "${task}" "${model}" "missing_candidate" "${MISSING_CANDIDATE_EXIT_CODE}" "pytest evaluation skipped because no candidate jax_repo was found"
    return "${MISSING_CANDIDATE_EXIT_CODE}"
  fi

  generated_eval_config="${attempt_root}/pytest_eval_config.yaml"
  generated_output_dir="${attempt_root}/evaluation_output"
  pytest_output_root="${attempt_root}/pytest_eval_outputs"

  echo "Preparing evaluation config for ${task}/${model}"
  echo "  template: ${eval_template}"
  echo "  reference repo: ${repo_dir}"
  echo "  candidate repo: ${candidate_repo}"
  echo "  generated config: ${generated_eval_config}"

  render_eval_config \
    "${eval_template}" \
    "${EVAL_MODEL_REGISTRY}" \
    "${task}" \
    "${model}" \
    "${repo_dir}" \
    "${candidate_repo}" \
    "${generated_eval_config}" \
    "${generated_output_dir}" \
    "${eval_visible_devices}"

  echo "Running pytest evaluation for ${task}/${model}"
  pytest_cmd=(
    "${PYTHON_BIN}" -m pytest -q tests/test_eval_pipeline.py
    --eval-config "${generated_eval_config}"
    --eval-output-root "${pytest_output_root}"
  )

  if [[ -n "${PYTEST_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${PYTEST_EXTRA_ARGS} )
    pytest_cmd+=("${EXTRA_ARGS[@]}")
  fi

  set +e
  (
    cd "${EVAL_ROOT}"
    export PYTHONPATH=".:.."
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES="${eval_visible_devices}"
    "${pytest_cmd[@]}"
  )
  runner_status=$?
  set -e

  echo "Pytest exit code for ${task}/${model}: ${runner_status}"
  eval_summary_path="$(find_eval_summary_path "${attempt_root}")"
  if [[ -n "${eval_summary_path}" ]]; then
    write_attempt_status "${attempt_root}" "${task}" "${model}" "evaluation_completed" "${runner_status}" "evaluation summary produced at ${eval_summary_path}"
  elif (( runner_status == 0 )); then
    write_attempt_status "${attempt_root}" "${task}" "${model}" "evaluation_completed" "${runner_status}" "pytest evaluation completed without a discoverable summary"
  else
    write_attempt_status "${attempt_root}" "${task}" "${model}" "evaluation_failed" "${runner_status}" "pytest evaluation failed before producing a summary"
  fi

  if (( runner_status != 0 )) && [[ -n "${eval_summary_path}" ]]; then
    echo "Evaluation produced summary despite failing checks; treating ${task}/${model} as a completed pipeline run."
    echo "  summary: ${eval_summary_path}"
    return 0
  fi

  return "${runner_status}"
}

run_all_models() {
  local task="$1"
  local batch_size="$2"
  local batch_timestamp="$3"
  shift 3
  local models=("$@")

  local total_models="${#models[@]}"
  local batch_log_dir="${BATCH_LOG_ROOT}/${task}_${batch_timestamp}"
  local batch_start batch_end i slot gpu_pair gpu_spec log_file status container_name completion_reason
  local pids=() batch_models=() batch_logs=() batch_gpu_specs=() batch_failed=0
  local summary_args=() attempt_root=""

  mkdir -p "${batch_log_dir}"

  echo "Batch logs: ${batch_log_dir}"
  if (( RESUME_BATCH )); then
    echo "Resume mode: completed models for timestamp ${batch_timestamp} will be skipped."
  fi
  echo "Models:"
  for model_name in "${models[@]}"; do
    echo "  - ${model_name}"
  done

  for ((batch_start = 0; batch_start < total_models; batch_start += batch_size)); do
    batch_end=$((batch_start + batch_size))
    if (( batch_end > total_models )); then
      batch_end="${total_models}"
    fi

    echo
    echo "Starting batch $((batch_start / batch_size + 1)): models $((batch_start + 1))-${batch_end}"

    pids=()
    batch_models=()
    batch_logs=()
    batch_gpu_specs=()

    for ((i = batch_start; i < batch_end; ++i)); do
      slot=$((i - batch_start))
      model_name="${models[$i]}"
      gpu_pair="${PARSED_GPU_PAIRS[$slot]}"
      gpu_spec="device=${gpu_pair}"
      log_file="${batch_log_dir}/$(printf '%02d' "$((i + 1))")_${task}_${model_name}.log"
      attempt_root="${WORKSPACE_ROOT}/.attempt_docker/${model_name}_${task}_${batch_timestamp}"
      summary_args+=("${task}/${model_name}" "${attempt_root}")

      if (( RESUME_BATCH )) && completion_reason="$(batch_target_completion_reason "${task}" "${model_name}" "${attempt_root}" "${log_file}")"; then
        echo "Skipping completed ${task}/${model_name}: ${completion_reason}"
        continue
      fi

      container_name="$(attempt_container_name "${task}" "${model_name}" "${attempt_root}")"
      ACTIVE_CONTAINER_NAMES+=("${container_name}")

      echo "Launching ${task}/${model_name} on GPUs ${gpu_pair}"
      (
        RUN_MODEL_CONTAINER_NAME="${container_name}"
        run_model_attempt \
          "${task}" \
          "${model_name}" \
          "${gpu_spec}" \
          "$(derive_cuda_visible_devices "${gpu_spec}")" \
          "${attempt_root}" \
          "${AUTO_EVAL_AFTER_ATTEMPT_FAILURE}"
      ) > "${log_file}" 2>&1 &

      pids+=("$!")
      ACTIVE_CHILD_PIDS+=("$!")
      batch_models+=("${model_name}")
      batch_logs+=("${log_file}")
      batch_gpu_specs+=("${gpu_spec}")

      echo "  pid=$! log=${log_file}"
    done

    batch_failed=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        remove_active_child_pid "${pids[$i]}"
        echo "Finished ${task}/${batch_models[$i]} (${batch_gpu_specs[$i]})"
      else
        status=$?
        remove_active_child_pid "${pids[$i]}"
        if (( status == MISSING_CANDIDATE_EXIT_CODE )); then
          echo "No candidate jax_repo for ${task}/${batch_models[$i]} (${batch_gpu_specs[$i]}). Log: ${batch_logs[$i]}" >&2
        else
          echo "Failed ${task}/${batch_models[$i]} (${batch_gpu_specs[$i]}) with exit code ${status}. Log: ${batch_logs[$i]}" >&2
          batch_failed=1
        fi
      fi
    done

    if (( batch_failed )); then
      if (( RUN_EVAL )); then
        print_eval_pass_rate_summary "${summary_args[@]}"
      fi
      echo "Batch failed; stopping before the next batch." >&2
      return 1
    fi
  done

  if (( RUN_EVAL )); then
    print_eval_pass_rate_summary "${summary_args[@]}"
  fi

  echo
  echo "All ${task} batches completed successfully."
}

run_all_targets() {
  local batch_size="$1"
  local batch_timestamp="$2"

  local total_targets="${#ALL_TARGET_TASKS[@]}"
  local batch_log_dir="${BATCH_LOG_ROOT}/all_${batch_timestamp}"
  local batch_start batch_end i slot gpu_pair gpu_spec log_file task model status container_name completion_reason
  local pids=() batch_labels=() batch_logs=() batch_gpu_specs=() batch_failed=0
  local summary_args=() attempt_root=""

  mkdir -p "${batch_log_dir}"

  echo "Batch logs: ${batch_log_dir}"
  if (( RESUME_BATCH )); then
    echo "Resume mode: completed targets for timestamp ${batch_timestamp} will be skipped."
  fi
  echo "Targets:"
  for i in "${!ALL_TARGET_TASKS[@]}"; do
    echo "  - ${ALL_TARGET_TASKS[$i]}/${ALL_TARGET_MODELS[$i]}"
  done

  for ((batch_start = 0; batch_start < total_targets; batch_start += batch_size)); do
    batch_end=$((batch_start + batch_size))
    if (( batch_end > total_targets )); then
      batch_end="${total_targets}"
    fi

    echo
    echo "Starting batch $((batch_start / batch_size + 1)): targets $((batch_start + 1))-${batch_end}"

    pids=()
    batch_labels=()
    batch_logs=()
    batch_gpu_specs=()

    for ((i = batch_start; i < batch_end; ++i)); do
      slot=$((i - batch_start))
      task="${ALL_TARGET_TASKS[$i]}"
      model="${ALL_TARGET_MODELS[$i]}"
      gpu_pair="${PARSED_GPU_PAIRS[$slot]}"
      gpu_spec="device=${gpu_pair}"
      log_file="${batch_log_dir}/$(printf '%02d' "$((i + 1))")_${task}_${model}.log"
      attempt_root="${WORKSPACE_ROOT}/.attempt_docker/${model}_${task}_${batch_timestamp}"
      summary_args+=("${task}/${model}" "${attempt_root}")

      if (( RESUME_BATCH )) && completion_reason="$(batch_target_completion_reason "${task}" "${model}" "${attempt_root}" "${log_file}")"; then
        echo "Skipping completed ${task}/${model}: ${completion_reason}"
        continue
      fi

      container_name="$(attempt_container_name "${task}" "${model}" "${attempt_root}")"
      ACTIVE_CONTAINER_NAMES+=("${container_name}")

      echo "Launching ${task}/${model} on GPUs ${gpu_pair}"
      (
        RUN_MODEL_CONTAINER_NAME="${container_name}"
        run_model_attempt \
          "${task}" \
          "${model}" \
          "${gpu_spec}" \
          "$(derive_cuda_visible_devices "${gpu_spec}")" \
          "${attempt_root}" \
          "${AUTO_EVAL_AFTER_ATTEMPT_FAILURE}"
      ) > "${log_file}" 2>&1 &

      pids+=("$!")
      ACTIVE_CHILD_PIDS+=("$!")
      batch_labels+=("${task}/${model}")
      batch_logs+=("${log_file}")
      batch_gpu_specs+=("${gpu_spec}")

      echo "  pid=$! log=${log_file}"
    done

    batch_failed=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        remove_active_child_pid "${pids[$i]}"
        echo "Finished ${batch_labels[$i]} (${batch_gpu_specs[$i]})"
      else
        status=$?
        remove_active_child_pid "${pids[$i]}"
        if (( status == MISSING_CANDIDATE_EXIT_CODE )); then
          echo "No candidate jax_repo for ${batch_labels[$i]} (${batch_gpu_specs[$i]}). Log: ${batch_logs[$i]}" >&2
        else
          echo "Failed ${batch_labels[$i]} (${batch_gpu_specs[$i]}) with exit code ${status}. Log: ${batch_logs[$i]}" >&2
          batch_failed=1
        fi
      fi
    done

    if (( batch_failed )); then
      if (( RUN_EVAL )); then
        print_eval_pass_rate_summary "${summary_args[@]}"
      fi
      echo "Batch failed; stopping before the next batch." >&2
      return 1
    fi
  done

  if (( RUN_EVAL )); then
    print_eval_pass_rate_summary "${summary_args[@]}"
  fi

  echo
  echo "All task/model batches completed successfully."
}

TASK="$(prompt_choice "Choose training method" 1 sft ppo dpo all)"
MODELS=()
MODEL="all"
if [[ "${TASK}" == "all" ]]; then
  discover_all_targets
else
  mapfile -t MODELS < <(discover_models "${TASK}")
  MODEL_OPTIONS=("${MODELS[@]}" "all")
  MODEL="$(prompt_choice "Choose model for ${TASK}" 1 "${MODEL_OPTIONS[@]}")"
fi

AGENT_NAME="$(prompt_choice "Choose top-level agent" 1 client_agent iterative_agent codex gemini claude)"
case "${AGENT_NAME}" in
  codex)
    ENV_FILE="$(resolve_codex_env_profile)"
    DEFAULT_BASE_MODEL="gpt-5.4"
    AGENT_REASONING_EFFORT="high"
    ;;
  gemini)
    ENV_FILE="$(resolve_gemini_env_profile)"
    DEFAULT_BASE_MODEL="gemini-3.1-pro-preview"
    AGENT_REASONING_EFFORT=""
    ;;
  claude)
    ENV_FILE="$(resolve_claude_env_profile)"
    DEFAULT_BASE_MODEL="opus"
    AGENT_REASONING_EFFORT=""
    ;;
  *)
    ENV_FILE="$(resolve_env_profile)"
    DEFAULT_BASE_MODEL="$(read_env_value "${ENV_FILE}" CODING_AGENT_MODEL || true)"
    AGENT_REASONING_EFFORT=""
    ;;
esac

BASE_MODEL="$(prompt_text "Choose basemodel (CODING_AGENT_MODEL)" "${DEFAULT_BASE_MODEL}")"
if [[ -z "${BASE_MODEL}" ]]; then
  echo "Basemodel cannot be empty." >&2
  exit 1
fi
DOCKER_IMAGE="$(prompt_text "Docker image tag" "coding-agent-lf:latest")"
VERBOSE_FLAG=0
YOLO_FLAG=0
BATCH_SIZE=""
GPU_PAIRS_SPEC=""
ATTEMPT_GPU_SPEC=""
BATCH_TS=""
RESUME_BATCH=0

if [[ "${TASK}" == "all" || "${MODEL}" == "all" ]]; then
  DEFAULT_RESUME_TS="$(latest_batch_timestamp "${TASK}" || true)"
  default_batch_size=4
  if [[ "${TASK}" == "all" ]]; then
    if (( ${#ALL_TARGET_TASKS[@]} < default_batch_size )); then
      default_batch_size="${#ALL_TARGET_TASKS[@]}"
    fi
  elif (( ${#MODELS[@]} < default_batch_size )); then
    default_batch_size="${#MODELS[@]}"
  fi
  BATCH_SIZE="$(prompt_text "Batch size for concurrent runs" "${default_batch_size}")"
  if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid batch size: ${BATCH_SIZE}" >&2
    exit 1
  fi
  GPU_PAIRS_SPEC="$(prompt_text "GPU pairs for concurrent slots" "0,1;2,3;4,5;6,7")"
  parse_gpu_pairs_spec "${GPU_PAIRS_SPEC}"
  if (( BATCH_SIZE > ${#PARSED_GPU_PAIRS[@]} )); then
    echo "Batch size (${BATCH_SIZE}) exceeds available GPU pairs (${#PARSED_GPU_PAIRS[@]})" >&2
    exit 1
  fi
  if [[ -n "${DEFAULT_RESUME_TS}" ]]; then
    if prompt_yes_no "Resume existing batch timestamp ${DEFAULT_RESUME_TS} and skip completed targets?" "n"; then
      RESUME_BATCH=1
      BATCH_TS="$(prompt_text "Batch timestamp to resume" "${DEFAULT_RESUME_TS}")"
      validate_batch_timestamp "${BATCH_TS}"
    fi
  elif prompt_yes_no "Resume existing batch timestamp and skip completed targets?" "n"; then
    RESUME_BATCH=1
    BATCH_TS="$(prompt_text "Batch timestamp to resume")"
    validate_batch_timestamp "${BATCH_TS}"
  fi
  if [[ -z "${BATCH_TS}" ]]; then
    BATCH_TS="$(date -u +%Y%m%dT%H%M%SZ)"
  fi
else
  ATTEMPT_GPU_SPEC="$(prompt_text "Docker --gpus spec for attempt" "device=0,1")"
fi

if prompt_yes_no "Enable verbose agent output?" "y"; then
  VERBOSE_FLAG=1
fi
YOLO_DEFAULT="n"
if [[ "${AGENT_NAME}" == "codex" || "${AGENT_NAME}" == "gemini" || "${AGENT_NAME}" == "claude" ]]; then
  YOLO_DEFAULT="y"
fi
if prompt_yes_no "Enable permission bypass/yolo mode for the agent?" "${YOLO_DEFAULT}"; then
  YOLO_FLAG=1
fi

RUN_EVAL=0
if prompt_yes_no "Run pytest evaluation after attempt?" "y"; then
  RUN_EVAL=1
  if [[ "${TASK}" != "all" && "${MODEL}" != "all" ]]; then
    ensure_eval_supported "${TASK}" "${MODEL}"
  fi
fi

EVAL_TEMPLATE=""
EVAL_CUDA_VISIBLE_DEVICES=""
PYTEST_EXTRA_ARGS=""
if (( RUN_EVAL )); then
  if [[ "${TASK}" != "all" ]]; then
    EVAL_TEMPLATE="$(resolve_eval_template "${TASK}")"
  fi
  if [[ "${TASK}" != "all" && "${MODEL}" != "all" ]]; then
    DEFAULT_VISIBLE_DEVICES="$(derive_cuda_visible_devices "${ATTEMPT_GPU_SPEC}")"
    EVAL_CUDA_VISIBLE_DEVICES="$(prompt_text "CUDA_VISIBLE_DEVICES for pytest evaluation" "${DEFAULT_VISIBLE_DEVICES}")"
  fi
  PYTEST_EXTRA_ARGS="$(prompt_text "Extra pytest args (optional)" "")"
fi

echo
echo "Execution summary"
if [[ "${TASK}" == "all" ]]; then
  echo "  target set: all supported dpo/ppo/sft experiments"
  echo "  targets:"
  for i in "${!ALL_TARGET_TASKS[@]}"; do
    echo "    - ${ALL_TARGET_TASKS[$i]}/${ALL_TARGET_MODELS[$i]}"
  done
  echo "  prompt template: auto by task"
  echo "  batch size: ${BATCH_SIZE}"
  echo "  gpu pairs: ${GPU_PAIRS_SPEC}"
  echo "  batch timestamp: ${BATCH_TS}"
  echo "  resume completed targets: ${RESUME_BATCH}"
  echo "  batch log root: ${BATCH_LOG_ROOT}"
elif [[ "${MODEL}" == "all" ]]; then
  echo "  repo set: ${TASK}/{${MODELS[*]}}/torch_orig"
  echo "  prompt template: $(resolve_instruction_template "${TASK}")"
  echo "  models: ${MODELS[*]}"
  echo "  batch size: ${BATCH_SIZE}"
  echo "  gpu pairs: ${GPU_PAIRS_SPEC}"
  echo "  batch timestamp: ${BATCH_TS}"
  echo "  resume completed targets: ${RESUME_BATCH}"
  echo "  batch log root: ${BATCH_LOG_ROOT}"
else
  ATTEMPT_TS="$(date -u +%Y%m%dT%H%M%SZ)"
  ATTEMPT_ROOT="${WORKSPACE_ROOT}/.attempt_docker/${MODEL}_${TASK}_${ATTEMPT_TS}"
  RENDERED_PROMPT_FILE="${ATTEMPT_ROOT}/rendered_prompt.md"
  echo "  repo: ${TASK}/${MODEL}/torch_orig"
  echo "  prompt template: $(resolve_instruction_template "${TASK}")"
  echo "  rendered prompt: ${RENDERED_PROMPT_FILE}"
  echo "  attempt gpus: ${ATTEMPT_GPU_SPEC}"
  echo "  attempt root: ${ATTEMPT_ROOT}"
fi
echo "  agent: ${AGENT_NAME}"
if [[ -n "${ENV_FILE}" ]]; then
  echo "  env profile: ${ENV_FILE}"
else
  echo "  env profile: shell env only"
fi
echo "  basemodel: ${BASE_MODEL}"
if [[ "${AGENT_NAME}" == "codex" ]]; then
  echo "  reasoning effort: ${AGENT_REASONING_EFFORT}"
fi
if [[ -n "${GOOGLE_GENAI_USE_VERTEXAI:-}" ]]; then
  echo "  vertex ai: ${GOOGLE_GENAI_USE_VERTEXAI}"
fi
if [[ -n "${GOOGLE_CLOUD_LOCATION:-}" ]]; then
  echo "  effective cloud location: ${GOOGLE_CLOUD_LOCATION}"
fi
echo "  docker image: ${DOCKER_IMAGE}"
if (( RUN_EVAL )); then
  if [[ "${TASK}" == "all" ]]; then
    echo "  eval template: auto by task"
  else
    echo "  eval template: ${EVAL_TEMPLATE}"
  fi
  if [[ "${TASK}" != "all" && "${MODEL}" != "all" ]]; then
    echo "  eval CUDA_VISIBLE_DEVICES: ${EVAL_CUDA_VISIBLE_DEVICES}"
  else
    echo "  eval CUDA_VISIBLE_DEVICES: auto-derived from each GPU pair"
  fi
fi

if ! prompt_yes_no "Proceed with attempt?" "y"; then
  echo "Aborted."
  exit 0
fi

if [[ "${TASK}" == "all" ]]; then
  run_all_targets "${BATCH_SIZE}" "${BATCH_TS}"
elif [[ "${MODEL}" == "all" ]]; then
  run_all_models "${TASK}" "${BATCH_SIZE}" "${BATCH_TS}" "${MODELS[@]}"
else
  set +e
  run_model_attempt \
    "${TASK}" \
    "${MODEL}" \
    "${ATTEMPT_GPU_SPEC}" \
    "${EVAL_CUDA_VISIBLE_DEVICES}" \
    "${ATTEMPT_ROOT}" \
    1
  RUN_STATUS=$?
  set -e
  if (( RUN_EVAL )); then
    print_eval_pass_rate_summary "${TASK}/${MODEL}" "${ATTEMPT_ROOT}"
  fi
  exit "${RUN_STATUS}"
fi
