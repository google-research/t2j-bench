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
  echo "Usage: $0 <task> <model> [--gpus <spec>] [--log-file <path>]" >&2
  echo "Example: $0 dpo gemma3 --gpus device=2,3" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

TASK="$1"
MODEL="$2"
shift 2

GPU_SPEC=""
USER_LOG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo "--gpus requires a value" >&2
        exit 1
      fi
      GPU_SPEC="$2"
      shift 2
      ;;
    --log-file)
      if [[ $# -lt 2 ]]; then
        echo "--log-file requires a value" >&2
        exit 1
      fi
      USER_LOG_FILE="$2"
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
RUN_SCRIPT="${SCRIPT_DIR}/run_${TASK}_${MODEL}.sh"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Attempt wrapper not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

LOG_ROOT="${WORKSPACE_ROOT}/.attempt_docker"
mkdir -p "${LOG_ROOT}"

if [[ -n "${USER_LOG_FILE}" ]]; then
  LOG_FILE="${USER_LOG_FILE}"
  mkdir -p "$(dirname "${LOG_FILE}")"
else
  LOG_FILE="${LOG_ROOT}/run_${TASK}_${MODEL}.log"
fi

RUN_ARGS=()
if [[ -n "${GPU_SPEC}" ]]; then
  RUN_ARGS+=(--gpus "${GPU_SPEC}")
fi

nohup "${RUN_SCRIPT}" "${RUN_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started ${TASK}/${MODEL} attempt in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
