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
# build_harbor_images.sh -- build the per-task Harbor agent + verifier images, for both
# task classes (dispatching on [metadata].task_type in each task.toml).
#
#   training   {sft,dpo,ppo}_{model}  -> t2j-agent:{method}-{model}
#                                         t2j-verifier:{method}-{model}
#   inference  {core,ragged,chunked}_{family}
#                                      -> t2j-inference-agent:{task//_/-}
#                                         t2j-inference-verifier:{task//_/-}
#
# The per-task images stack on base and weight images (t2j-model:{model},
# t2j-verifier-model:{model}, plus the two inference bases). These are not built by
# default -- several wrap gated HF repos and need HF_TOKEN -- so a missing one fails the
# preflight with the command to run. Pass --with-bases to build them first.
#
# Builds use the repo root as context. For each task, .dockerignore is temporarily
# narrowed to that task (always excluding tasks/*/solution, the answer key) and restored
# afterwards; a lock (/tmp/t2j-dockerignore.lock) serializes concurrent builds.
#
# Usage (from anywhere):
#   ./build_harbor_images.sh                         # every task under tasks/
#   ./build_harbor_images.sh sft_qwen2 core_llama3   # just the named task(s)
#   ./build_harbor_images.sh --with-bases ...        # build the needed base + weight images first
#   ./build_harbor_images.sh --prove-guard           # check the answer-key exclusion works,
#                                                    #   build nothing else
#   ./build_harbor_images.sh --help
#
# Overridable via env: DOCKER, DOCKERIGNORE_LOCK, LOCK_WAIT, GOLDEN_PYTHON.
#
# If docker needs the docker group:
#   sg docker -c "PATH=/usr/bin:\$PATH ./build_harbor_images.sh sft_qwen2"
#
# Exit status: 0 if every build succeeded, 1 if any failed, 2 on a preflight error.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

DOCKER="${DOCKER:-docker}"
DOCKERIGNORE_LOCK="${DOCKERIGNORE_LOCK:-/tmp/t2j-dockerignore.lock}"
LOCK_WAIT="${LOCK_WAIT:-7200}"
# The golden manifest check needs `yaml`; prefer the authoring venv when present.
GOLDEN_PYTHON="${GOLDEN_PYTHON:-$( [ -x .venv-inference/bin/python ] && echo .venv-inference/bin/python || echo python3 )}"

export DOCKER_BUILDKIT=1

INF_AGENT_BASE="t2j-inference-agent-base:latest"
INF_VERIFIER_BASE="t2j-inference-verifier-base:latest"

# --help prints the header comment above (everything after the license).
usage() {
  awk '/^# limitations under the License\.$/ { start = 1; next }
       !start { next }
       /^#/ { sub(/^# ?/, ""); if (!body && $0 == "") next; body = 1; print; next }
       { exit }' "$0"
}

# [metadata].task_type of one task.
task_type_of() {
  awk -F'"' '/^task_type[[:space:]]*=/ { print $2; exit }' "tasks/$1/task.toml"
}

with_bases=0
prove_guard=0
tasks=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --with-bases) with_bases=1 ;;
    --prove-guard) prove_guard=1 ;;
    -*) echo "ERROR: unknown flag '$arg'." >&2; echo "Try: $0 --help" >&2; exit 2 ;;
    *) tasks+=("$arg") ;;
  esac
done

if [ ${#tasks[@]} -eq 0 ]; then
  # Every directory under tasks/ is a task.
  mapfile -t tasks < <(find tasks -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
fi
if [ ${#tasks[@]} -eq 0 ]; then
  echo "ERROR: no tasks under $ROOT/tasks." >&2
  exit 2
fi

# ------------------------------------------------------------------------- preflight
if ! $DOCKER info > /dev/null 2>&1; then
  echo "ERROR: '$DOCKER' cannot reach a docker daemon." >&2
  echo "       On this host:  sg docker -c \"PATH=/usr/bin:\\\$PATH $0 ${tasks[*]}\"" >&2
  exit 2
fi

# Distinct models behind the requested tasks (they share weight images).
mapfile -t models < <(printf '%s\n' "${tasks[@]}" | sed 's/^[^_]*_//' | sort -u)
any_inference=0
for task in "${tasks[@]}"; do
  [ "$(task_type_of "$task")" = inference ] && { any_inference=1; break; }
done

if [ "$with_bases" -eq 1 ]; then
  echo ">> building the training bases + weight tiers for: ${models[*]}"
  DOCKER="$DOCKER" base_image/build_base_images.sh "${models[@]}" || {
    echo "ERROR: weight/base tier build failed; nothing else was built." >&2; exit 2; }
  if [ "$any_inference" -eq 1 ]; then
    echo ">> building the two inference base tiers"
    DOCKER="$DOCKER" base_image/build_inference_bases.sh || {
      echo "ERROR: inference base tier build failed; nothing else was built." >&2; exit 2; }
  fi
fi

mapfile -t have_tags < <($DOCKER images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null)
# Exact whole-tag match (a grep -q pipeline can fail under pipefail via SIGPIPE).
have_tag() { case " ${have_tags[*]} " in *" $1 "*) return 0 ;; esac; return 1; }

problems=()
[ "$any_inference" -eq 1 ] && {
  have_tag "$INF_AGENT_BASE" || problems+=(
    "base image $INF_AGENT_BASE missing -- base_image/build_inference_bases.sh (or --with-bases)")
  have_tag "$INF_VERIFIER_BASE" || problems+=(
    "base image $INF_VERIFIER_BASE missing -- base_image/build_inference_bases.sh (or --with-bases)")
}
for task in "${tasks[@]}"; do
  tt="$(task_type_of "$task")"
  model="${task#*_}"
  dir="tasks/${task}"
  case "$tt" in
    training|inference) ;;
    *) problems+=("$task: task.toml has no [metadata].task_type -- cannot dispatch"); continue ;;
  esac
  for f in "environment/Dockerfile" "tests/Dockerfile" "tests/test.sh" "task.toml"; do
    [ -f "$dir/$f" ] || problems+=("$task: $dir/$f missing")
  done
  # Every local path the Dockerfiles COPY must exist (`COPY --from=` lines are skipped).
  while read -r src; do
    [ -n "$src" ] && [ ! -e "$src" ] && problems+=(
      "$task: $src is COPYed by a Dockerfile but is not on disk")
  done < <(awk '$1 == "COPY" && NF == 3 && $2 !~ /^--/ { print $2 }' \
             "$dir/environment/Dockerfile" "$dir/tests/Dockerfile" 2>/dev/null | sort -u)
  # The goldens get baked into the verifier image, so check them against the task's
  # golden_manifest.py. Exit codes: 0 complete, 1 incomplete (missing names on stdout),
  # 2 could not check (reason on stderr).
  golden_missing="$("$GOLDEN_PYTHON" "$ROOT/tools/check_goldens.py" "$dir" 2>&1)"; golden_rc=$?
  # An exit 1 that names nothing means GOLDEN_PYTHON didn't run; treat as cannot-check.
  if [ "$golden_rc" = 1 ] && [ -z "$golden_missing" ]; then
    golden_rc=9; golden_missing="$GOLDEN_PYTHON produced no answer"
  fi
  case "$golden_rc" in
    0) ;;
    1) problems+=(
       "$task: goldens INCOMPLETE -- missing: ${golden_missing} (they ship via Git-LFS; run: git lfs pull)") ;;
    *) # Can't run the check (e.g. no yaml): warn and fall back to a meta.json test.
       echo ">> WARNING: $task: could not run the golden manifest check:"
       echo "   ${golden_missing}"
       echo "   Falling back to a meta.json presence test. An INCOMPLETE golden set will"
       echo "   be baked into the verifier image and will NOT be caught here."
       [ -f "$dir/tests/golden/meta.json" ] || problems+=(
         "$task: $dir/tests/golden/meta.json missing -- fetch the Git-LFS goldens before building") ;;
  esac
  # Weight images are needed by both classes (training via FROM, inference via COPY --from).
  have_tag "t2j-model:${model}" || problems+=(
    "$task: weight tier t2j-model:${model} missing -- base_image/build_base_images.sh ${model} (gated repos need HF_TOKEN)")
  have_tag "t2j-verifier-model:${model}" || problems+=(
    "$task: weight tier t2j-verifier-model:${model} missing -- base_image/build_base_images.sh ${model} (gated repos need HF_TOKEN)")
done

if [ ${#problems[@]} -gt 0 ]; then
  echo "ERROR: preflight failed for ${#problems[@]} item(s); nothing was built." >&2
  printf '  %s\n' "${problems[@]}" >&2
  exit 2
fi

# ------------------------------------------------------- .dockerignore, under the lock
exec {lockfd}>"$DOCKERIGNORE_LOCK" || {
  echo "ERROR: cannot open the lock file $DOCKERIGNORE_LOCK." >&2; exit 2; }
if ! flock -n "$lockfd"; then
  echo ">> another build holds $DOCKERIGNORE_LOCK; waiting up to ${LOCK_WAIT}s"
  if ! flock -w "$LOCK_WAIT" "$lockfd"; then
    echo "ERROR: $DOCKERIGNORE_LOCK was held for more than ${LOCK_WAIT}s. Either a build" >&2
    echo "       is genuinely still running, or a crashed one left .dockerignore narrowed;" >&2
    echo "       check it against git before forcing anything." >&2
    exit 2
  fi
fi

# With the lock held, a narrowed file means a previous run died before restoring it.
if grep -qE '^!(tasks|jaxopt_tasks)/' .dockerignore; then
  echo "ERROR: $ROOT/.dockerignore is already narrowed to a single task:" >&2
  grep -nE '^!?(tasks|jaxopt_tasks)/' .dockerignore | sed 's/^/         /' >&2
  echo "       A previous build died without restoring it. Put the committed copy back" >&2
  echo "       first:  git show HEAD:.dockerignore > .dockerignore" >&2
  exit 2
fi

BACKUP="$(mktemp "${TMPDIR:-/tmp}/t2j-dockerignore.XXXXXX")"
cp -- .dockerignore "$BACKUP" || { echo "ERROR: cannot back up .dockerignore." >&2; exit 2; }

# shellcheck disable=SC2329  # invoked from the EXIT/INT/TERM traps installed below
restore_dockerignore() {
  # Idempotent: the signal trap calls this, then EXIT fires it again.
  [ -n "${BACKUP:-}" ] && [ -f "$BACKUP" ] || return 0
  cp -- "$BACKUP" .dockerignore
  # Verify the restore actually happened.
  if cmp -s "$BACKUP" .dockerignore; then
    rm -f -- "$BACKUP"
    return 0
  fi
  echo "FATAL: .dockerignore was NOT restored. The committed copy is preserved at:" >&2
  echo "         $BACKUP" >&2
  echo "       Put it back before building anything else:" >&2
  echo "         cp $BACKUP $ROOT/.dockerignore" >&2
  exit 1
}
trap restore_dockerignore EXIT
trap 'restore_dockerignore; exit 130' INT TERM

# Narrow .dockerignore to one task, always rendered from the backup (never the live file).
narrow_dockerignore() {
  local task="$1"
  { grep -vE '^!?tasks/' -- "$BACKUP"
    # Exclude the other tasks' trees (their goldens are several GB).
    echo "tasks/*"
    echo "!tasks/${task}"
    # The answer key: no Dockerfile COPYs it, and this keeps it out of every context.
    echo "tasks/*/solution"
    echo "tasks/${task}/solution"
    # Re-exclude bytecode: the `!tasks/<task>` line above re-includes it.
    echo "**/__pycache__"
    echo "**/*.pyc"
  } > .dockerignore
  # Sanity-check the narrowed file.
  grep -qxF "!tasks/${task}" .dockerignore \
    && grep -qxF "tasks/${task}/solution" .dockerignore \
    && grep -qxF "tasks/*/solution" .dockerignore
}

# ------------------------------------------------------------------------ prove-guard
#
# Check the answer-key exclusion in both directions against the real build context:
#   [1] without the exclusion, `COPY .../solution` succeeds (so the test can detect a leak);
#   [2] with it, the same COPY fails;
#   [3] with it, `COPY .../tests/golden` still succeeds.
# Uses `FROM scratch` with local output, so no images are created.
prove_guard() {
  local task="$1" rc=0
  local probe; probe="$(mktemp -d)"
  echo ">> proving the answer-key exclusion on tasks/${task}"

  printf 'FROM scratch\nCOPY tasks/%s/solution /solution\n' "$task" \
    > "$probe/answer_key.Dockerfile"
  printf 'FROM scratch\nCOPY tasks/%s/tests/golden /golden\n' "$task" \
    > "$probe/golden.Dockerfile"

  # [1] negative control: without the exclusion the answer key reaches the context
  grep -v '/solution$' -- "$BACKUP" > .dockerignore
  if $DOCKER build --no-cache -f "$probe/answer_key.Dockerfile" \
       -o "type=local,dest=$probe/out-nokey" . > "$probe/log-neg" 2>&1 \
     && [ -n "$(ls -A "$probe/out-nokey/solution/ref_jax_repo" 2>/dev/null)" ]; then
    echo "   [1/3] negative control OK: with the exclusion removed, the answer key reaches the context"
  else
    echo "!! [1/3] NEGATIVE CONTROL FAILED: with the exclusion REMOVED the answer key still" >&2
    echo "         did not arrive, so the guard below proves nothing. Log: $probe/log-neg" >&2
    rc=1
  fi

  # [2] the guard: with the exclusion the COPY must fail
  narrow_dockerignore "$task" || { echo "!! narrowing failed" >&2; rc=1; }
  if $DOCKER build --no-cache -f "$probe/answer_key.Dockerfile" \
       -o "type=local,dest=$probe/out-key" . > "$probe/log-guard" 2>&1; then
    echo "!! [2/3] GUARD FAILED: tasks/${task}/solution reached the build context" >&2
    echo "         with the exclusion in place. The answer key can enter an image." >&2
    rc=1
  else
    echo "   [2/3] guard proven: the COPY aborts the build -- $(grep -oiE 'no source files[^"]*|not found' "$probe/log-guard" | head -1)"
  fi

  # [3] positive control: the goldens must still reach the context
  if $DOCKER build --no-cache -f "$probe/golden.Dockerfile" \
       -o "type=local,dest=$probe/out-golden" . > "$probe/log-pos" 2>&1 \
     && [ -n "$(ls -A "$probe/out-golden/golden" 2>/dev/null)" ]; then
    echo "   [3/3] positive control OK: tests/golden/ still reaches the context ($(ls "$probe/out-golden/golden" | wc -l) artifacts)"
  else
    echo "!! [3/3] POSITIVE CONTROL FAILED: the narrowing also excluded tests/golden/," >&2
    echo "         so the verifier image would grade against nothing. Log: $probe/log-pos" >&2
    rc=1
  fi

  rm -rf -- "$probe"
  return "$rc"
}

if [ "$prove_guard" -eq 1 ]; then
  rc=0
  for task in "${tasks[@]}"; do prove_guard "$task" || rc=1; done
  exit "$rc"
fi

# ------------------------------------------------------------------------------ build
# Image tags. Training dashes the first underscore (sft_llava_next -> sft-llava_next);
# inference dashes every underscore (core_gpt_oss -> core-gpt-oss).
tag_for() {  # $1 = task -> prints "AGENT_TAG VERIFIER_TAG"
  local task="$1" name
  if [ "$(task_type_of "$task")" = inference ]; then
    name="${task//_/-}"
    printf 't2j-inference-agent:%s t2j-inference-verifier:%s' "$name" "$name"
  else
    name="${task%%_*}-${task#*_}"
    printf 't2j-agent:%s t2j-verifier:%s' "$name" "$name"
  fi
}

echo "$(date -u +%H:%M:%S) building ${#tasks[@]} task(s) x 2 images | docker=$DOCKER | lock=$DOCKERIGNORE_LOCK"
built=()
failed=()
for task in "${tasks[@]}"; do
  read -r agent_tag verifier_tag < <(tag_for "$task")
  if ! narrow_dockerignore "$task"; then
    echo "!! $task: narrowed .dockerignore failed its own sanity check; skipping"
    failed+=("$task (dockerignore)")
    continue
  fi
  echo ">> building ${agent_tag} and ${verifier_tag}"
  # shellcheck disable=SC2086  # $DOCKER may carry flags, e.g. DOCKER="docker --context h100"
  if ! $DOCKER build -f "tasks/${task}/environment/Dockerfile" -t "$agent_tag" .; then
    failed+=("$task (agent image)"); continue
  fi
  # shellcheck disable=SC2086
  if ! $DOCKER build -f "tasks/${task}/tests/Dockerfile" -t "$verifier_tag" .; then
    failed+=("$task (verifier image)"); continue
  fi
  built+=("$task")
done

# ---------------------------------------------------------------------------- summary
echo
echo "==== SUMMARY ===="
for task in "${tasks[@]}"; do
  read -r agent_tag verifier_tag < <(tag_for "$task")
  hit=""
  if [ ${#failed[@]} -gt 0 ]; then
    for f in "${failed[@]}"; do case "$f" in "$task "*) hit="$f";; esac; done
  fi
  if [ -n "$hit" ]; then
    printf "%-24s FAIL  %s\n" "$task" "${hit#"$task "}"
  else
    printf "%-24s OK    %s , %s\n" "$task" "$agent_tag" "$verifier_tag"
  fi
done
echo
echo "ALL DONE -- ${#built[@]} ok, ${#failed[@]} failed"
exit $(( ${#failed[@]} > 0 ))
