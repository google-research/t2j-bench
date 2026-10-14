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

# Oracle solution: install the golden reference as the candidate to prove the task
# is solvable (Harbor copies solution/ to /solution and runs this from the workdir).
set -euo pipefail
mkdir -p /workspace/jax_repo
cp -r /solution/ref_jax_repo/. /workspace/jax_repo/
echo "installed golden solution -> /workspace/jax_repo"
