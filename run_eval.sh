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

# run_eval.sh — run one Harbor attempt of a t2j task with a coding agent, then grade it.
# Works for both task classes under tasks/ (training and inference), dispatching on
# [metadata].task_type in task.toml.
#
# Usage:
#   ./run_eval.sh [TASK] [-- extra harbor flags...]
#
# Examples:
#   AGENT_ENV=ANTHROPIC_API_KEY ./run_eval.sh                    # dpo_bloom (training)
#   AGENT_ENV=ANTHROPIC_API_KEY ./run_eval.sh core_llama3        # an inference task
#   AGENT_ENV=ANTHROPIC_API_KEY MODEL=claude-sonnet-5 ./run_eval.sh sft_qwen2
#   ./run_eval.sh dpo_bloom -- --debug                           # extra flags to `harbor run`
#
# Overridable via env: AGENT, MODEL, AGENT_ENV, MAX_RETRIES, N_CONCURRENT, JOBS_DIR,
#                      LOCKDIR, T2J_GPU_DEVICE_IDS, T2J_ALLOW_CONCURRENT.
#
# Prerequisites, in this order:
#   1. harbor CLI:      uv tool install harbor==0.20.0 (the harbor patches below assume 0.20.0)
#   2. Frozen goldens:  git lfs pull (into tasks/<TASK>/tests/golden/)
#   3. Per-task images: ./build_harbor_images.sh <TASK>
# The verifier image bakes the goldens at build time, so fetch them before building.
#
# Credentials: bring your own. AGENT_ENV is a comma- or space-separated list of host env
# var NAMES; each one that is set is forwarded to the agent container as --ae NAME=VALUE.
#   * claude-code runs inside the container, so its credentials must be forwarded
#     (e.g. ANTHROPIC_API_KEY, or ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN).
#   * terminus-2 and other host-side LiteLLM agents read this shell's env directly.
# Forwarded values appear in the harbor process's argv and in
# jobs/<run>/<trial>/config.json; harbor redacts the latter only for names matching
# KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH, so name secrets accordingly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------- configuration
usage() {
  cat <<'USAGE'
Usage: ./run_eval.sh [TASK] [-- extra Harbor flags...]

Default task: dpo_bloom
Environment: AGENT, MODEL, AGENT_ENV, N_CONCURRENT, MAX_RETRIES, JOBS_DIR,
             GOLDEN_PYTHON, LOCKDIR, T2J_GPU_DEVICE_IDS, T2J_ALLOW_CONCURRENT.

Examples:
  AGENT_ENV=ANTHROPIC_API_KEY ./run_eval.sh dpo_bloom
  AGENT_ENV=ANTHROPIC_API_KEY ./run_eval.sh core_llama3 -- --debug
USAGE
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
TASK=dpo_bloom
if [[ $# -gt 0 && "$1" != -- ]]; then TASK="$1"; shift; fi
EXTRA=()
if [[ "${1:-}" == -- ]]; then shift; EXTRA=("$@");
elif [[ $# -gt 0 ]]; then
  echo "ERROR: extra Harbor flags must follow '--'." >&2
  usage >&2
  exit 2
fi

AGENT="${AGENT:-claude-code}"
MODEL="${MODEL:-claude-opus-5}"
# Host env var names to forward into the agent container (see Credentials above).
AGENT_ENV="${AGENT_ENV:-}"
N_CONCURRENT="${N_CONCURRENT:-1}"
JOBS_DIR="${JOBS_DIR:-jobs}"

# Retries cover transient provider stalls that leave no jax_repo for harbor to collect.
MAX_RETRIES="${MAX_RETRIES:-1}"

TASK_PATH="tasks/${TASK}"
[[ "$TASK" =~ ^[a-z0-9]+_[a-z0-9_]+$ && -f "$TASK_PATH/task.toml" ]] || {
  echo "ERROR: task not found: $TASK (expected tasks/<task>/task.toml)." >&2
  exit 2
}

# The classes differ only in image naming; dispatch on [metadata].task_type.
TASK_TYPE="$(awk -F'"' '/^task_type[[:space:]]*=/ { print $2; exit }' "$TASK_PATH/task.toml")"
case "$TASK_TYPE" in
  training|inference) ;;
  *) echo "ERROR: unsupported task_type in $TASK_PATH/task.toml: $TASK_TYPE" >&2; exit 2 ;;
esac

if [[ "$TASK_TYPE" == inference ]]; then
  # core_gpt_oss -> core-gpt-oss: every underscore, as build_harbor_images.sh does.
  IMG_TAG="${TASK//_/-}"
  AGENT_IMG="t2j-inference-agent:${IMG_TAG}"
  VERIFIER_IMG="t2j-inference-verifier:${IMG_TAG}"
else
  # sft_llava_next -> sft-llava_next: first underscore only, as build_harbor_images.sh does.
  IMG_TAG="${TASK/_/-}"
  AGENT_IMG="t2j-agent:${IMG_TAG}"
  VERIFIER_IMG="t2j-verifier:${IMG_TAG}"
fi
# ---------------------------------------------------------------------- checks
command -v harbor >/dev/null 2>&1 || { echo "ERROR: harbor not on PATH. Install: uv tool install harbor==0.20.0"; exit 1; }
# Every task needs one H100; fail early with a clear message on other hardware.
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi not on PATH. This eval requires an H100 GPU (every task declares gpus=1)."
  echo "       Run on an H100 VM with the NVIDIA driver installed."
  exit 1; }
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
[[ -n "$GPU_NAMES" ]] || {
  echo "ERROR: nvidia-smi reports no GPUs. This eval requires an H100 GPU (every task declares gpus=1)."
  exit 1; }
if ! grep -qi 'H100' <<<"$GPU_NAMES"; then
  echo "ERROR: no H100 GPU detected. nvidia-smi reports:"
  sed 's/^/         - /' <<<"$GPU_NAMES"
  echo "       These tasks require an H100; other cards (e.g. an A100) are not supported."
  exit 1
fi

# Check both images up front: a missing verifier image would only fail after the agent run.
for img in "$AGENT_IMG" "$VERIFIER_IMG"; do
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    echo "ERROR: image ${img} not built."
    echo "       Build it with: ./build_harbor_images.sh ${TASK}"
    exit 1
  fi
done

# Check the host goldens are complete, using the task's own golden_manifest.py (the same
# list the in-container precheck uses). Exit codes: 0 complete, 1 incomplete (missing
# names on stdout), 2 could not check.
GOLDEN_DIR="${TASK_PATH}/tests/golden"
# The manifest needs `yaml`; prefer the authoring venv when present.
GOLDEN_PYTHON="${GOLDEN_PYTHON:-$( [[ -x .venv-inference/bin/python ]] && echo .venv-inference/bin/python || echo python3 )}"
GOLDEN_RC=0
GOLDEN_MISSING="$("$GOLDEN_PYTHON" "$ROOT/tools/check_goldens.py" "$TASK_PATH" 2>/dev/null)" || GOLDEN_RC=$?
# Only a non-empty exit 1 is a verdict; anything else means the check could not run.
if [[ "$GOLDEN_RC" != 0 ]] && { [[ "$GOLDEN_RC" != 1 ]] || [[ -z "$GOLDEN_MISSING" ]]; }; then
  GOLDEN_MISSING="__CHECK_UNAVAILABLE__"
fi
if [[ "$GOLDEN_MISSING" == "__CHECK_UNAVAILABLE__" ]]; then
  # Don't block on a missing host dependency; fall back to an emptiness check.
  echo ">> WARNING: could not read golden_manifest.py (missing yaml?); falling back to"
  echo "   an emptiness check. A stale-but-non-empty golden/ will NOT be caught here."
  if [[ ! -d "$GOLDEN_DIR" ]] || [[ -z "$(ls -A "$GOLDEN_DIR" 2>/dev/null)" ]]; then
    GOLDEN_MISSING="(golden/ is absent or empty)"
  else
    GOLDEN_MISSING=""
  fi
fi
if [[ -n "$GOLDEN_MISSING" ]]; then
  echo "ERROR: the frozen goldens at ${GOLDEN_DIR} are missing or incomplete."
  echo "       Missing: ${GOLDEN_MISSING}"
  echo
  echo "       They ship via Git-LFS under tasks/${TASK}/tests/golden/; run 'git lfs pull'"
  echo "       (they are not regenerated here) before building the verifier image or grading."
  exit 1
fi

# -------------------------------------------- goldens baked into the verifier image
# Harbor's separate verifier grades the goldens inside the image and never uploads the
# host tree, so the image must match the host goldens byte for byte.
HOST_GOLDEN_LIST="$(mktemp)"; IMG_GOLDEN_LIST="$(mktemp)"

# The only EXIT trap in this script (bash keeps just the last one registered).
_t2j_cleanup() {
  local rc=$?
  rm -f "$HOST_GOLDEN_LIST" "$IMG_GOLDEN_LIST"
  return "$rc"
}
trap _t2j_cleanup EXIT
( cd "$GOLDEN_DIR" && find . -type f | sort | xargs -r sha256sum ) > "$HOST_GOLDEN_LIST"
# `|| true`: an image without /tests/golden should reach the diff report below.
docker run --rm --entrypoint sh "$VERIFIER_IMG" \
  -c 'cd /tests/golden && find . -type f | sort | xargs -r sha256sum' \
  > "$IMG_GOLDEN_LIST" 2>/dev/null || true
if ! cmp -s "$HOST_GOLDEN_LIST" "$IMG_GOLDEN_LIST"; then
  echo "ERROR: the goldens baked into ${VERIFIER_IMG} do not match ${GOLDEN_DIR}."
  echo "       The verifier image is STALE -- rebuild it:"
  echo "         ./build_harbor_images.sh ${TASK}"
  echo
  echo "       This is not cosmetic. Harbor's separate verifier grades the goldens"
  echo "       INSIDE the image and never uploads the host tree over them"
  echo "       (trial.py:588 skip_tests_upload=True -> verifier.py:97 returns no test"
  echo "       source dirs -> verifier.py:147 skips the upload). The host tree matters"
  echo "       only because it is what the image is built FROM."
  echo
  awk 'NR==FNR { host[$2] = $1; next }
       { img[$2] = $1 }
       END {
         for (f in host) if (!(f in img))                     print "       only on the host:   " f
         for (f in img)  if (!(f in host))                    print "       only in the image:  " f
         for (f in host) if ((f in img) && host[f] != img[f]) print "       different bytes:    " f
       }' "$HOST_GOLDEN_LIST" "$IMG_GOLDEN_LIST" | sort
  exit 1
fi

# -------------------------------------------------------------- harbor patches
# Two edits to the installed harbor 0.20.0, re-applied idempotently on every run (they
# are lost on `uv tool upgrade harbor`). Each anchor must match exactly once, or this
# aborts rather than guessing.
#   1. GPU support: enable GPU allocation and pin to T2J_GPU_DEVICE_IDS when set.
#   2. Artifact scrubber: only redact secrets of >= 12 chars, so a short value (e.g. `1`)
#      cannot corrupt reward.json.
if ! python3 "$ROOT/tools/patch_harbor.py"; then
  echo "ERROR: could not patch the installed harbor for GPU allocation + artifact scrubbing."
  echo "       Both edits are required: the first lets a GPU task run at all (task.toml:"
  echo "       gpus = 1 for both phases); the second stops harbor turning every digit 1 --"
  echo "       reward.json included -- into [REDACTED]. The message above names what failed."
  echo "       Usually harbor is not installed (uv tool install harbor==0.20.0) or its source moved"
  echo "       and the patch in tools/patch_harbor.py needs updating."
  exit 1
fi

# ------------------------------------------------------------- one GPU trial per host
# Harbor maps a 1-GPU request to device 0 every time, so concurrent trials would share
# one card. Serialize trials with a host lock unless T2J_GPU_DEVICE_IDS pins a device
# (or T2J_ALLOW_CONCURRENT=1).
LOCKDIR="${LOCKDIR:-/tmp/t2j-jaxtrain-gpu.locks}"
mkdir -p "$LOCKDIR"
if [[ -n "${T2J_GPU_DEVICE_IDS:-}" ]]; then
  echo ">> pinned to GPU ${T2J_GPU_DEVICE_IDS} (host mutex not needed)"
elif [[ "${T2J_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  exec {gpufd}>"$LOCKDIR/harbor-gpu.lock"
  if ! flock -n "$gpufd"; then
    echo ">> another harbor GPU trial holds the host lock; waiting for it to finish..."
    echo "   (harbor pins every trial to GPU 0, so trials cannot safely overlap)"
    flock "$gpufd"
  fi
  echo ">> host GPU lock acquired ($LOCKDIR/harbor-gpu.lock)"
fi

# ------------------------------------------------ credentials & agent env (host)
# Forward the host env vars named in AGENT_ENV as --ae NAME=VALUE.
AE=()
FORWARDED=()
for _name in ${AGENT_ENV//,/ }; do
  [[ -n "$_name" ]] || continue
  [[ "$_name" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || {
    echo "ERROR: AGENT_ENV contains an invalid environment variable name: $_name" >&2
    exit 2
  }
  if [[ -z "${!_name:-}" ]]; then
    echo ">> WARNING: AGENT_ENV lists ${_name}, but it is unset or empty in this shell --"
    echo "   skipping it. The agent may fail to authenticate without it."
    continue
  fi
  AE+=( --ae "${_name}=${!_name}" )
  FORWARDED+=("$_name")
done

if [[ ${#FORWARDED[@]} -eq 0 ]]; then
  echo ">> NOTE: no agent env forwarded (AGENT_ENV is empty). An in-container agent such as"
  echo "   claude-code will need credentials forwarded -- e.g. AGENT_ENV=ANTHROPIC_API_KEY."
  echo "   A host-side agent such as terminus-2 reads this shell's environment directly."
  echo
  AUTH_DESC="none forwarded (AGENT_ENV empty)"
else
  AUTH_DESC="forwarded: ${FORWARDED[*]}"
fi

echo ">> task=${TASK}  path=${TASK_PATH}  agent=${AGENT}  model=${MODEL}"
echo ">> images: ${AGENT_IMG} + ${VERIFIER_IMG}"
echo ">> agent env: ${AUTH_DESC}"
echo ">> jobs_dir=${JOBS_DIR}  n_concurrent=${N_CONCURRENT}  max_retries=${MAX_RETRIES}"
echo ">> reward lands in ${JOBS_DIR}/<run>/<trial>/verifier/reward.json"
echo "   reward = spec_pass_rate * mean(numeric_score, behavior_score)"
echo

# ------------------------------------------------------------------------- run
# No `set -x`: forwarded values may be credentials. Echo the command with them redacted.
AE_SHOWN=()
for _ae_arg in "${AE[@]}"; do
  case "$_ae_arg" in
    --ae) AE_SHOWN+=("$_ae_arg") ;;
    *=*)  AE_SHOWN+=("${_ae_arg%%=*}=<redacted>") ;;
    *)    AE_SHOWN+=("$_ae_arg") ;;
  esac
done
echo "+ harbor run -p ${TASK_PATH} -a ${AGENT} -m ${MODEL} -n ${N_CONCURRENT} -o ${JOBS_DIR} -r ${MAX_RETRIES} ${AE_SHOWN[*]} --yes ${EXTRA[*]}"
harbor run \
  -p "$TASK_PATH" \
  -a "$AGENT" \
  -m "$MODEL" \
  -n "$N_CONCURRENT" \
  -o "$JOBS_DIR" \
  -r "$MAX_RETRIES" \
  "${AE[@]}" \
  --yes \
  "${EXTRA[@]}"
