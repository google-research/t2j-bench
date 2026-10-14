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

# One-time, per-clone setup: route git-lfs transfers for this repo to the Hugging Face
# Storage Bucket via tools/lfs_hf_bucket.py instead of GitHub's LFS server.
#
# git-lfs refuses to read transfer-agent settings from a committed .lfsconfig (it would let
# a repo run arbitrary programs), so this writes them to the clone's local .git/config.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
python="${PYTHON:-python3}"

command -v git-lfs >/dev/null || { echo "git-lfs is not installed (https://git-lfs.com)" >&2; exit 1; }
"$python" -c "import huggingface_hub" 2>/dev/null \
  || { echo "huggingface_hub is missing: $python -m pip install -U huggingface_hub" >&2; exit 1; }

git lfs install --local
git config lfs.standalonetransferagent hf-bucket
git config lfs.customtransfer.hf-bucket.path "$python"
git config lfs.customtransfer.hf-bucket.args "$root/tools/lfs_hf_bucket.py"
git config lfs.customtransfer.hf-bucket.concurrent true

echo "git-lfs -> hf://buckets/${T2J_LFS_BUCKET:-jiefengcgg/t2j-bench} configured."
echo "Fetch goldens with: git lfs pull   (or: git lfs pull --include='tasks/<task>/tests/golden/**')"
