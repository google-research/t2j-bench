#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_all_dpo_ppo_sft_once.sh [--batch-size <n>] [--gpu-pairs <spec>] [--log-root <path>] [--dry-run]

Options:
  --batch-size <n>   Number of scripts to run concurrently in each group. Default: 4
  --gpu-pairs <spec> Semicolon-separated GPU pairs, one pair per concurrent slot.
                     Default: 0,1;2,3;4,5;6,7
  --log-root <path>  Directory for batch log folders.
  --dry-run          Print the planned execution order without launching anything.

Example:
  run_all_dpo_ppo_sft_once.sh
  run_all_dpo_ppo_sft_once.sh --gpu-pairs '0,1;2,3;4,5;6,7'
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODING_AGENT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${CODING_AGENT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${REPO_ROOT}/dataset"

BATCH_SIZE=4
GPU_PAIRS_SPEC="0,1;2,3;4,5;6,7"
LOG_ROOT="${WORKSPACE_ROOT}/.attempt_docker/batch_runs"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size)
      if [[ $# -lt 2 ]]; then
        echo "--batch-size requires a value" >&2
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gpu-pairs)
      if [[ $# -lt 2 ]]; then
        echo "--gpu-pairs requires a value" >&2
        exit 1
      fi
      GPU_PAIRS_SPEC="$2"
      shift 2
      ;;
    --log-root)
      if [[ $# -lt 2 ]]; then
        echo "--log-root requires a value" >&2
        exit 1
      fi
      LOG_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --batch-size: ${BATCH_SIZE}" >&2
  exit 1
fi

IFS=';' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_SPEC}"
if [[ "${#GPU_PAIRS[@]}" -eq 0 ]]; then
  echo "No GPU pairs configured" >&2
  exit 1
fi

for idx in "${!GPU_PAIRS[@]}"; do
  pair="$(echo "${GPU_PAIRS[$idx]}" | tr -d '[:space:]')"
  if [[ ! "${pair}" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "Invalid GPU pair: ${GPU_PAIRS[$idx]}" >&2
    exit 1
  fi
  GPU_PAIRS[$idx]="${pair}"
done

if (( BATCH_SIZE > ${#GPU_PAIRS[@]} )); then
  echo "--batch-size (${BATCH_SIZE}) exceeds available GPU pairs (${#GPU_PAIRS[@]})" >&2
  exit 1
fi

mapfile -t TARGET_SCRIPTS < <(
  find "${SCRIPT_DIR}" -maxdepth 1 -type f \
    \( -name 'run_dpo*.sh' -o -name 'run_ppo*.sh' -o -name 'run_sft*.sh' \) \
    | sort
)

if [[ "${#TARGET_SCRIPTS[@]}" -eq 0 ]]; then
  echo "No target scripts found under ${SCRIPT_DIR}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG_DIR="${LOG_ROOT}/${TIMESTAMP}"

echo "Discovered ${#TARGET_SCRIPTS[@]} scripts."
echo "Batch size: ${BATCH_SIZE}"
echo "GPU pairs: ${GPU_PAIRS[*]}"
echo "Order:"
for idx in "${!TARGET_SCRIPTS[@]}"; do
  printf '  %2d. %s\n' "$((idx + 1))" "$(basename "${TARGET_SCRIPTS[$idx]}")"
done

if (( DRY_RUN )); then
  echo "Dry run only; nothing launched."
  exit 0
fi

mkdir -p "${RUN_LOG_DIR}"
echo "Logs: ${RUN_LOG_DIR}"

total_scripts="${#TARGET_SCRIPTS[@]}"
for ((batch_start = 0; batch_start < total_scripts; batch_start += BATCH_SIZE)); do
  batch_end=$((batch_start + BATCH_SIZE))
  if (( batch_end > total_scripts )); then
    batch_end="${total_scripts}"
  fi

  echo
  echo "Starting batch $((batch_start / BATCH_SIZE + 1)): scripts $((batch_start + 1))-${batch_end}"

  pids=()
  batch_scripts=()
  batch_gpu_specs=()
  batch_logs=()

  for ((i = batch_start; i < batch_end; ++i)); do
    slot=$((i - batch_start))
    script_path="${TARGET_SCRIPTS[$i]}"
    script_name="$(basename "${script_path}")"
    gpu_pair="${GPU_PAIRS[$slot]}"
    gpu_spec="device=${gpu_pair}"
    log_file="${RUN_LOG_DIR}/$(printf '%02d' "$((i + 1))")_${script_name%.sh}.log"

    echo "Launching ${script_name} on GPUs ${gpu_pair}"
    "${script_path}" --gpus "${gpu_spec}" > "${log_file}" 2>&1 &
    pid=$!

    pids+=("${pid}")
    batch_scripts+=("${script_name}")
    batch_gpu_specs+=("${gpu_spec}")
    batch_logs+=("${log_file}")

    echo "  pid=${pid} log=${log_file}"
  done

  batch_failed=0
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    script_name="${batch_scripts[$idx]}"
    gpu_spec="${batch_gpu_specs[$idx]}"
    log_file="${batch_logs[$idx]}"

    if wait "${pid}"; then
      echo "Finished ${script_name} (${gpu_spec})"
    else
      status=$?
      echo "Failed ${script_name} (${gpu_spec}) with exit code ${status}. Log: ${log_file}" >&2
      batch_failed=1
    fi
  done

  if (( batch_failed )); then
    echo "Batch failed; stopping before the next batch." >&2
    exit 1
  fi
done

echo
echo "All batches completed successfully."
