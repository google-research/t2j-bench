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
#
# build_inference_bases.sh -- build the two base images for the inference task class
# ({core,ragged,chunked}_* under tasks/):
#
#   t2j-inference-verifier-base:latest  Dockerfile.inference-grade  jax 0.11 / flax 0.12.8, torch-free
#   t2j-inference-agent-base:latest     Dockerfile.inference-agent  the above + torch 2.10
#
# The weight images (t2j-model:{model}, t2j-verifier-model:{model}) are shared with the
# training class and built by build_base_images.sh. Per-task images pull the weights in
# with `COPY --from`, because those images are based on the older training jax stack.
#
# Usage (from anywhere):
#   base_image/build_inference_bases.sh                # both bases
#   base_image/build_inference_bases.sh --prove-guard  # also check the torch-free assert
#                                                      #   fails on an image with torch
#   DOCKER=podman base_image/build_inference_bases.sh  # override the CLI
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER="${DOCKER:-docker}"

VERIFIER_IMAGE="t2j-inference-verifier-base:latest"
AGENT_IMAGE="t2j-inference-agent-base:latest"

prove_guard=0
for arg in "$@"; do
  case "$arg" in
    --prove-guard) prove_guard=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

echo ">> building $VERIFIER_IMAGE"
$DOCKER build -f "$HERE/Dockerfile.inference-grade" -t "$VERIFIER_IMAGE" "$HERE"

echo ">> building $AGENT_IMAGE"
$DOCKER build -f "$HERE/Dockerfile.inference-agent" -t "$AGENT_IMAGE" "$HERE"

# --prove-guard: rerun the verifier's torch-free assert on the agent base (which has
# torch). The build must fail; if it succeeds, the assert isn't enforcing anything.
if [ "$prove_guard" -eq 1 ]; then
  echo ">> proving the torch-free assert fails closed (expect a FAILED build below)"
  guard_ctx="$(mktemp -d)"
  trap 'rm -rf "$guard_ctx"' EXIT
  cat > "$guard_ctx/Dockerfile" <<EOF
FROM $AGENT_IMAGE
RUN ! python -c "import torch" 2>/dev/null && echo "confirmed torch-free"
EOF
  if $DOCKER build --no-cache -f "$guard_ctx/Dockerfile" -t t2j-inference-guard-probe "$guard_ctx"; then
    $DOCKER rmi -f t2j-inference-guard-probe >/dev/null 2>&1 || true
    echo "!! FAIL: the torch-free assert PASSED in an image that has torch." >&2
    echo "   The assert in Dockerfile.inference-grade is not enforcing anything." >&2
    exit 1
  fi
  echo ">> guard proven: the assert aborts a build in which torch is importable"
fi

echo ">> done (2 bases: $VERIFIER_IMAGE, $AGENT_IMAGE)"
