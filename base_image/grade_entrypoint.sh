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

# Grade one training task (torch-free). Grades solution/ref_jax_repo by default; set T2J_JAX_REPO
# to grade another repo (Harbor's tests/test.sh points it at /workspace/jax_repo).
#
# Usage: grade <repo-relative-task>  e.g. grade tasks/sft_qwen2
#
# Needs tests/golden/ and the checkpoint at /data/{repo_id}. The repo is only read:
# grading runs on a temp copy of tests/ and writes the summary to T2J_SUMMARY_DIR.
set -euo pipefail

INSTANCE="${1:?usage: grade tasks/<method>_<model>}"
REPO_ROOT="${T2J_REPO_ROOT:-$PWD}"
# Default outside the repo so grading never writes into a bind-mounted tasks/ tree.
SUMMARY_DIR="${T2J_SUMMARY_DIR:-/tmp/t2j-grade/${INSTANCE}}"

test_dir="${REPO_ROOT}/${INSTANCE}/tests"
graded_repo="${T2J_JAX_REPO:-${REPO_ROOT}/${INSTANCE}/solution/ref_jax_repo}"

# Require both metadata and parameter norms before grading.
for f in golden/meta.json golden/param_norms.json; do
  [[ -f "${test_dir}/${f}" ]] && continue
  echo "::error:: ${INSTANCE}: missing tests/${f} (bake into the verifier image, or fetch with 'git lfs pull')" >&2
  exit 3
done
model_dir="${T2J_DATA_ROOT:-/data}/$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_config"]["model_name_or_path"])' "${test_dir}/golden/meta.json")"
if [[ ! -d "${model_dir}" ]]; then
  echo "::error:: ${INSTANCE}: no checkpoint at ${model_dir} — the candidate's convert_checkpoint reads it (base_image/build_base_images.sh)" >&2
  exit 3
fi
if [[ ! -d "${graded_repo}" ]]; then
  echo "::error:: ${INSTANCE}: no repo to grade at ${graded_repo}" >&2
  exit 4
fi

# Grade a temp copy of tests/ so pytest's __pycache__ never lands in the repo.
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
grade_dir="${scratch}/tests"
cp -a "${test_dir}" "${grade_dir}"

mkdir -p "${SUMMARY_DIR}"
echo ">> grading ${INSTANCE} (torch-free): ${graded_repo}"
T2J_JAX_REPO="${graded_repo}" \
T2J_SUMMARY_DIR="${SUMMARY_DIR}" \
PYTHONDONTWRITEBYTECODE=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}" \
  python -m pytest -q --no-header -o addopts="" -p no:cacheprovider "${grade_dir}"
