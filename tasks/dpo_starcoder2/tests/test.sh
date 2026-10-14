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

# Harbor verifier entrypoint: grade the candidate against frozen goldens and write
# a reward file to /logs/verifier/. Torch-free.
set -uo pipefail
# Locate the agent's migrated repo. Harbor collects the top-level `artifacts`
# (/workspace/jax_repo) and re-materializes it in this separate verifier under
# /logs/artifacts/workspace/jax_repo. Prefer that; fall back to a search, then to
# the in-container /workspace path (shared-verifier mode).
if [ -z "${T2J_JAX_REPO:-}" ] || [ ! -f "${T2J_JAX_REPO}/config.py" ]; then
  if [ -f /logs/artifacts/workspace/jax_repo/config.py ]; then
    export T2J_JAX_REPO=/logs/artifacts/workspace/jax_repo
  elif [ -f /workspace/jax_repo/config.py ]; then
    export T2J_JAX_REPO=/workspace/jax_repo
  else
    _cand="$(find /logs/artifacts /workspace -maxdepth 5 -name config.py -path '*jax_repo*' 2>/dev/null | head -1)"
    [ -n "${_cand}" ] && export T2J_JAX_REPO="$(dirname "${_cand}")"
  fi
fi
echo ">> grading candidate repo at: ${T2J_JAX_REPO:-<NONE FOUND>}"
export T2J_SUMMARY_DIR=/logs/verifier                        # conftest writes eval_summary.json here
export PYTHONDONTWRITEBYTECODE=1                             # never leave __pycache__ in /tests
export TOKENIZERS_PARALLELISM=false                     # HF fork-safety; set in-container (removed the --ve: harbor scrubbed TOKEN* -> [REDACTED] and broke result.json)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# PREALLOCATE=false still caps the BFC pool at MEM_FRACTION, which defaults to 0.75 --
# 59 GiB of an 80 GiB card, and the 7B instances (mistral, llava_next) plus starcoder2
# need more than that to backpropagate a float32 upcast of a bf16 checkpoint. Raise the
# ceiling rather than preallocate it: the pool still grows on demand, so a smaller card
# behaves exactly as before and only hits the driver limit it would have hit anyway.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export JAX_DEFAULT_MATMUL_PRECISION=highest                  # GPU fp32 (A100 defaults fp32->TF32, breaks fp32 goldens)
mkdir -p /logs/verifier
python -m pytest -q --no-header -o addopts="" -p no:cacheprovider /tests || true
python /tests/summary_to_reward.py \
    /logs/verifier/eval_summary.json /logs/verifier/reward.json
