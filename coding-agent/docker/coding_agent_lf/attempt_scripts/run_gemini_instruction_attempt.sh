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
  echo "Usage: $0 <task> <model_dir> <prompt_name> [--gpus <spec>]" >&2
  echo "Example: $0 dpo llava_next llava_next_dpo --gpus device=2,3" >&2
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

TASK="$1"
MODEL_DIR="$2"
PROMPT_NAME="$3"
shift 3

GPU_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo "--gpus requires a value" >&2
        exit 1
      fi
      GPU_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODING_AGENT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${CODING_AGENT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${REPO_ROOT}/dataset"
RUNNER="${CODING_AGENT_DIR}/scripts/run_attempt_in_docker.py"
USER_HOME="${HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
DEFAULT_PYTHON_BIN="${USER_HOME}/miniconda3/envs/lf/bin/python"
PYTHON_BIN="${CODING_AGENT_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi
IMAGE_TAG="${CODING_AGENT_DOCKER_IMAGE:-coding-agent-lf:latest}"
GPU_SPEC="${CODING_AGENT_ATTEMPT_GPUS:-device=0,1}"
if [[ -n "${GPU_OVERRIDE}" ]]; then
  GPU_SPEC="${GPU_OVERRIDE}"
fi

PROMPT_FILE="${WORKSPACE_ROOT}/instructions/${PROMPT_NAME}.md"
MODEL_ROOT_REL="${TASK}/${MODEL_DIR}"
TORCH_ORIG_REL="${MODEL_ROOT_REL}/torch_orig"
TORCH_SRC_ALIAS_REL="${MODEL_ROOT_REL}/torch_orig:${MODEL_ROOT_REL}/torch_src"
TORCH_ROOT_ALIAS_REL="${MODEL_ROOT_REL}/torch_orig:torch_orig"
JAX_ATTEMPT_REL="${MODEL_ROOT_REL}/jax_attempt_gemini"
JAX_REPO_ROOT_ALIAS_REL="${MODEL_ROOT_REL}/jax_attempt_gemini:jax_repo"
EVAL_MOUNT="eval"
AGENT_ENV_FILE="${CODING_AGENT_DIR}/coding_agent/.env_gemini"
ATTEMPT_TS="$(date -u +%Y%m%dT%H%M%SZ)"
ATTEMPT_ROOT="${WORKSPACE_ROOT}/.attempt_docker/test_${MODEL_DIR}_${TASK}_${ATTEMPT_TS}"

if [[ ! -d "${WORKSPACE_ROOT}/${TORCH_ORIG_REL}" ]]; then
  echo "Model torch_orig not found: ${WORKSPACE_ROOT}/${TORCH_ORIG_REL}" >&2
  exit 1
fi

if [[ ! -d "${WORKSPACE_ROOT}/${EVAL_MOUNT}" ]]; then
  echo "Evaluator workspace not found: ${WORKSPACE_ROOT}/${EVAL_MOUNT}" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

if [[ ! -f "${AGENT_ENV_FILE}" ]]; then
  echo "Agent env file not found: ${AGENT_ENV_FILE}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -e "${WORKSPACE_ROOT}/${JAX_ATTEMPT_REL}" ]]; then
  rm -rf "${WORKSPACE_ROOT}/${JAX_ATTEMPT_REL}" 2>/dev/null || sudo -n rm -rf "${WORKSPACE_ROOT}/${JAX_ATTEMPT_REL}"
fi
mkdir -p "${WORKSPACE_ROOT}/${JAX_ATTEMPT_REL}"

exec "${PYTHON_BIN}" "${RUNNER}" \
  --image "${IMAGE_TAG}" \
  --workspace "${WORKSPACE_ROOT}" \
  --attempt-root "${ATTEMPT_ROOT}" \
  --mount "${TORCH_ORIG_REL}" \
  --mount "${TORCH_SRC_ALIAS_REL}" \
  --mount "${TORCH_ROOT_ALIAS_REL}" \
  --mount "${JAX_ATTEMPT_REL}" \
  --mount "${JAX_REPO_ROOT_ALIAS_REL}" \
  --mount "${EVAL_MOUNT}" \
  --prompt-file "${PROMPT_FILE}" \
  --agent-env-file "${AGENT_ENV_FILE}" \
  --agent client_agent \
  --gpus "${GPU_SPEC}" \
  --verbose
