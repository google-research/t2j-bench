#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_attempt_and_verify.sh <task> <model> [--gpus <spec>]

Example:
  run_attempt_and_verify.sh ppo gemma3 --gpus device=0,1
EOF
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
ATTEMPT_SCRIPT="${SCRIPT_DIR}/run_${TASK}_${MODEL}.sh"
USER_HOME="${HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
DEFAULT_PYTHON_BIN="${USER_HOME}/miniconda3/envs/lf/bin/python"
PYTHON_BIN="${CODING_AGENT_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi
EVAL_CONFIG="${WORKSPACE_ROOT}/eval/configs/${TASK}_${MODEL}.yaml"
ATTEMPT_PREFIX="${MODEL}_${TASK}_"

if [[ ! -x "${ATTEMPT_SCRIPT}" ]]; then
  echo "Attempt script not found or not executable: ${ATTEMPT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "Evaluator config not found: ${EVAL_CONFIG}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

LATEST_BEFORE="$(
  ls -1dt "${WORKSPACE_ROOT}/.attempt_docker/${ATTEMPT_PREFIX}"* 2>/dev/null | head -n 1 || true
)"

echo "Running attempt via ${ATTEMPT_SCRIPT}"
if [[ -n "${GPU_OVERRIDE}" ]]; then
  "${ATTEMPT_SCRIPT}" --gpus "${GPU_OVERRIDE}"
else
  "${ATTEMPT_SCRIPT}"
fi

LATEST_AFTER="$(
  ls -1dt "${WORKSPACE_ROOT}/.attempt_docker/${ATTEMPT_PREFIX}"* 2>/dev/null | head -n 1 || true
)"

if [[ -z "${LATEST_AFTER}" ]]; then
  echo "No attempt directory found under ${WORKSPACE_ROOT}/.attempt_docker" >&2
  exit 1
fi

ATTEMPT_ROOT="${LATEST_AFTER}"
if [[ -n "${LATEST_BEFORE}" && "${LATEST_AFTER}" == "${LATEST_BEFORE}" ]]; then
  echo "Warning: newest attempt directory matches the previous latest one; reusing ${ATTEMPT_ROOT}" >&2
fi

CANDIDATE_REPO="$(
  find "${ATTEMPT_ROOT}/.attempt" -maxdepth 2 -type d -path '*/attempt_*/jax_repo' | sort | tail -n 1
)"

if [[ -z "${CANDIDATE_REPO}" ]]; then
  echo "No candidate jax_repo found under ${ATTEMPT_ROOT}/.attempt" >&2
  exit 1
fi

REFERENCE_REPO="${WORKSPACE_ROOT}/${TASK}/${MODEL}/torch_orig"
GENERATED_EVAL_CONFIG="${ATTEMPT_ROOT}/equivalence_eval_config.yaml"
GENERATED_OUTPUT_DIR="${ATTEMPT_ROOT}/equivalence_verification"

echo "Attempt root: ${ATTEMPT_ROOT}"
echo "Reference repo: ${REFERENCE_REPO}"
echo "Candidate repo: ${CANDIDATE_REPO}"
echo "Generated eval config: ${GENERATED_EVAL_CONFIG}"
echo "Verification output dir: ${GENERATED_OUTPUT_DIR}"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml

config_path = Path(${EVAL_CONFIG@Q})
reference_repo = Path(${REFERENCE_REPO@Q}).resolve()
candidate_repo = Path(${CANDIDATE_REPO@Q}).resolve()
output_dir = Path(${GENERATED_OUTPUT_DIR@Q}).resolve()
generated_path = Path(${GENERATED_EVAL_CONFIG@Q}).resolve()

with config_path.open("r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

data["output_dir"] = str(output_dir)
for impl in data.get("implementations", []):
    name = impl.get("name")
    if name == "torch_orig":
        impl["repo_dir"] = str(reference_repo)
    elif name == "jax_repo":
        impl["repo_dir"] = str(candidate_repo)

generated_path.parent.mkdir(parents=True, exist_ok=True)
with generated_path.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(data, fh, sort_keys=False)
PY

echo "Starting torch vs jax equivalence verification"
(
  cd "${WORKSPACE_ROOT}/eval"
  PYTHONPATH="${WORKSPACE_ROOT}/eval:${WORKSPACE_ROOT}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${PYTHON_BIN}" -m lib.run_eval "${GENERATED_EVAL_CONFIG}"
)
