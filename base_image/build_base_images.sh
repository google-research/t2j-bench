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

# Build the shared base images the per-task Harbor images stack on:
#
#   t2j-agent-base:latest           Dockerfile.agent           jax + torch dev env
#     -> t2j-model:{model}          Dockerfile.model           + weights at /data/{repo_id}
#   t2j-verifier-base:latest        Dockerfile.grade           torch-free grade env
#     -> t2j-verifier-model:{model} Dockerfile.verifier-model  + the same /data, copied
#                                                                from t2j-model:{model}
#
# Models and their HF repos are listed in models.txt.
#
# Usage (from anywhere):
#   base_image/build_base_images.sh                # both bases + every model in models.txt
#   base_image/build_base_images.sh bloom qwen2    # both bases + the named models
#   base_image/build_base_images.sh --bases-only   # just the two bases
#   base_image/build_base_images.sh --host-seed    # no docker: populate /data on this host
#
# Gated models (gemma3, llama3, paligemma) need a Hugging Face token, taken from $HF_TOKEN,
# $HF_TOKEN_FILE, or ~/.cache/huggingface/token and passed as a BuildKit secret.
#
# --host-seed fills /data/{repo_id} from the local HF cache (hardlinks), for running task
# code on the host. One-time setup:
#   sudo mkdir -p /data && sudo chown "$USER" /data
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER="${DOCKER:-docker}"
PYTHON="${PYTHON:-python3}"
MODELS_TXT="$HERE/models.txt"
HOST_DATA="${HOST_DATA:-/data}"

# `model  repo_id` pairs, comments and blank lines skipped.
repo_for() {
  awk -v m="$1" '!/^[[:space:]]*#/ && $1 == m {print $2; found=1; exit}
                 END {if (!found) exit 1}' "$MODELS_TXT" \
    || { echo "unknown model '$1' -- not in $MODELS_TXT" >&2; exit 2; }
}

all_models() {
  awk '!/^[[:space:]]*#/ && NF {print $1}' "$MODELS_TXT" | sort
}

bases_only=0
host_seed=0
models=()
for arg in "$@"; do
  case "$arg" in
    --bases-only) bases_only=1 ;;
    --host-seed)  host_seed=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) models+=("$arg") ;;
  esac
done
if [ ${#models[@]} -eq 0 ]; then
  mapfile -t models < <(all_models)
fi

# Resolve every model name up front so a typo fails before any build.
repos=()
for model in "${models[@]}"; do
  repo="$(repo_for "$model")" || exit $?
  repos+=("$repo")
done

if [ "$host_seed" -eq 1 ]; then
  if [ ! -w "$HOST_DATA" ]; then
    echo "!! $HOST_DATA is not writable. Run: sudo mkdir -p $HOST_DATA && sudo chown \"\$USER\" $HOST_DATA" >&2
    exit 1
  fi
  for i in "${!models[@]}"; do
    echo ">> seeding $HOST_DATA for ${models[$i]}  (${repos[$i]})"
    $PYTHON "$HERE/prepare_model.py" --repo "${repos[$i]}" --out "$HOST_DATA" --from-cache
  done
  echo ">> host seed done (${#models[@]} model(s) under $HOST_DATA)"
  exit 0
fi

# HF token for gated models. $HF_TOKEN is written to a 0600 temp file, since
# `--secret env=` needs a newer buildx than older docker CLIs ship.
GATED_MODELS="gemma3 llama3 paligemma"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
tmp_token=""
cleanup_token() { if [ -n "$tmp_token" ]; then rm -f "$tmp_token"; fi; }
trap cleanup_token EXIT

token_src=""
if [ -n "${HF_TOKEN:-}" ]; then
  tmp_token="$(mktemp)"
  chmod 600 "$tmp_token"
  printf '%s' "$HF_TOKEN" > "$tmp_token"
  token_src="$tmp_token"
elif [ -r "$HF_TOKEN_FILE" ]; then
  token_src="$HF_TOKEN_FILE"
fi

secret_args=()
if [ -n "$token_src" ]; then
  secret_args=(--secret "id=hf_token,src=$token_src")
elif [ "$bases_only" -eq 0 ]; then
  for model in "${models[@]}"; do
    case " $GATED_MODELS " in
      *" $model "*)
        echo "!! $model is gated on the hub and no token was found." >&2
        echo "   Set HF_TOKEN, or run: huggingface-cli login  (writes $HF_TOKEN_FILE)" >&2
        exit 1 ;;
    esac
  done
fi

# Dockerfile.model mounts the token with `RUN --mount=type=secret` (BuildKit only).
export DOCKER_BUILDKIT=1

# Build context is base_image/ only.
echo ">> building t2j-agent-base:latest"
$DOCKER build -f "$HERE/Dockerfile.agent" -t t2j-agent-base:latest "$HERE"
echo ">> building t2j-verifier-base:latest"
$DOCKER build -f "$HERE/Dockerfile.grade" -t t2j-verifier-base:latest "$HERE"

if [ "$bases_only" -eq 1 ]; then
  echo ">> done (bases only)"
  exit 0
fi

for i in "${!models[@]}"; do
  echo ">> building t2j-model:${models[$i]}  (${repos[$i]})"
  $DOCKER build -f "$HERE/Dockerfile.model" "${secret_args[@]}" \
    --build-arg "MODEL_REPO=${repos[$i]}" -t "t2j-model:${models[$i]}" "$HERE"
  # The verifier tier copies /data from the image just built, so both sides see the same bytes.
  echo ">> building t2j-verifier-model:${models[$i]}  (${repos[$i]})"
  $DOCKER build -f "$HERE/Dockerfile.verifier-model" \
    --build-arg "MODEL=${models[$i]}" --build-arg "MODEL_REPO=${repos[$i]}" \
    -t "t2j-verifier-model:${models[$i]}" "$HERE"
done
echo ">> done (2 bases + ${#models[@]} model(s) x 2 tiers)"
