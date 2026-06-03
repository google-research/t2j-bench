#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <task> <model> [--gpus <spec>]" >&2
  echo "Example: $0 ppo gemma3 --gpus device=2,3" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

TASK="$1"
MODEL="$2"
shift 2

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
DEFAULT_PROMPT_FILE="${SCRIPT_DIR}/../prompts/${TASK}_${MODEL}.md"
PROMPT_FILE="${CODING_AGENT_ATTEMPT_PROMPT:-${DEFAULT_PROMPT_FILE}}"
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
MODEL_MOUNT="${TASK}/${MODEL}/torch_orig"
EVAL_MOUNT="eval"

if [[ ! -d "${WORKSPACE_ROOT}/${MODEL_MOUNT}" ]]; then
  echo "Model workspace not found: ${WORKSPACE_ROOT}/${MODEL_MOUNT}" >&2
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${RUNNER}" \
  --image "${IMAGE_TAG}" \
  --workspace "${WORKSPACE_ROOT}" \
  --mount "${MODEL_MOUNT}" \
  --mount "${EVAL_MOUNT}" \
  --prompt-file "${PROMPT_FILE}" \
  --agent client_agent \
  --gpus "${GPU_SPEC}" \
  --verbose
