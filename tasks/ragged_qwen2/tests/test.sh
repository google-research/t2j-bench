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
# Harbor verifier entrypoint: grade the candidate against frozen goldens and write
# a reward file to /logs/verifier/. Torch-free.
set -uo pipefail

# A candidate must have config.py for the resolver and final grade guard.
_gradable() { [ -n "${T2J_JAX_REPO:-}" ] && [ -f "${T2J_JAX_REPO}/config.py" ]; }

# Prefer the conventional artifact path, then the original workspace path.
# The reference solution is excluded from the image, so an absent artifact cannot
# silently fall back to grading the reference implementation.
if ! _gradable; then
  if [ -f /logs/artifacts/workspace/jax_repo/config.py ]; then
    export T2J_JAX_REPO=/logs/artifacts/workspace/jax_repo
  elif [ -f /workspace/jax_repo/config.py ]; then
    export T2J_JAX_REPO=/workspace/jax_repo
  else
    _cand="$(find /logs/artifacts /workspace -maxdepth 5 -name config.py -path '*jax_repo*' 2>/dev/null | head -1)"
    [ -n "${_cand}" ] && export T2J_JAX_REPO="$(dirname "${_cand}")"
  fi
fi
if _gradable; then
  echo ">> grading candidate repo at: ${T2J_JAX_REPO}"
else
  echo ">> NO GRADABLE CANDIDATE at: ${T2J_JAX_REPO:-<no path resolved>}"
fi
# Run pytest as t2jgrade, keeping /tests, the reducer, and published rewards
# root-owned. Validate the effective permissions before executing candidate code.
# This prevents direct rewrites of those files, but does not authenticate the
# summary produced under the grading uid; forged measurements remain a limitation.
_GRADE_USER=t2jgrade
_SUMMARY_SRC_DIR=/run/t2j-grade      # the ONLY path the graded uid may write
_REDUCER=/opt/t2j-grade/summary_to_reward.py
if ! command -v setpriv >/dev/null 2>&1 || ! id -u "${_GRADE_USER}" >/dev/null 2>&1; then
  echo "!! aborting: this image cannot drop privileges for the graded phase --" >&2
  echo "   setpriv and the '${_GRADE_USER}' account are created by tests/Dockerfile." >&2
  echo "   Grading as root would let candidate code rewrite the reward reduction," >&2
  echo "   so NOTHING is graded and NO reward.json is written. Rebuild the verifier" >&2
  echo "   image from this task's tests/Dockerfile." >&2
  exit 3
fi
_GRADE_UID="$(id -u "${_GRADE_USER}")"
_GRADE_GID="$(id -g "${_GRADE_USER}")"
# Set HOME only on the unprivileged command. Exporting it globally would
# expose later root Python processes to the grading user's site-packages.
# setpriv preserves the remaining T2J_* and XLA_* environment settings.
_GRADE_HOME=/home/t2jgrade
# =========================================================================== #

# Root helpers ignore PYTHON* settings and user site-packages.
# Use -I -B for standalone reducers and stdlib-only heredocs; use -E -s -B
# for the precheck, which must import trusted sibling modules from /tests.
# These flags are not applied to candidate pytest, which uses its own imports.
_PY_ISOLATED="python -I -B"
_PY_TESTS="python -E -s -B"

export T2J_SUMMARY_DIR="${_SUMMARY_SRC_DIR}"                 # conftest writes eval_summary.json here
export PYTHONDONTWRITEBYTECODE=1                             # never leave __pycache__ in /tests
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# PREALLOCATE=false still caps the BFC pool at MEM_FRACTION, which defaults to 0.75 --
# 59 GiB of an 80 GiB card. Raise the ceiling rather than preallocate it: the pool
# still grows on demand, so a smaller card behaves exactly as before and only hits the
# driver limit it would have hit anyway.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export JAX_DEFAULT_MATMUL_PRECISION=highest                  # GPU fp32 (H100 defaults fp32->TF32, breaks fp32 goldens)
# XLA GEMM AUTOTUNING OFF, unconditionally. The autotuner times candidate kernels at
# compile time and picks the winner, so under card contention the same graph can lower
# to different kernels in two runs -- which is one of the drivers of this instance's
# measured bf16 cross-process spread (up to 1.9e-01 on the prefill logits; see
# instance.yaml). The grading device is a single pinned SKU, so autotuning buys nothing
# here and costs reproducibility. APPENDED, never assigned: XLA_FLAGS is a single flat
# string and an assignment would silently drop any flag the caller set.
#
# conftest sets the same flag for every entry point that does NOT come through this
# script (a developer's bare `pytest tasks/*/tests`), append-safely and
# idempotently, so the two cannot fight and neither can be forgotten.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"
# THE AUTHORING ESCAPE HATCH IS OFF IN A GRADED RUN. `T2J_INFERENCE_DATA_ROOT` rebinds
# the candidate loader's DATA_ROOT so the suite can run on a host with no /data; here
# the checkpoint IS at /data (both images bake it) and `test_loader_spec` grades that
# the loader resolved there. Unset rather than trusted-absent: an inherited value from
# the surrounding environment would redirect the loader and quietly weaken that check.
unset T2J_INFERENCE_DATA_ROOT
mkdir -p /logs/verifier
# NEITHER ARTIFACT SURVIVES INTO A SECOND TRIAL. Both are written at the END of a
# run -- `eval_summary.json` by conftest's `pytest_sessionfinish`, `reward.json` by
# `summary_to_reward.py` -- and BOTH are deliberately withheld on several paths
# (conftest cases 2/5/6; the exit-4 and no-summary guards below). Harbor gives each
# trial its own /logs, but a re-run into the SAME `--results` directory -- the
# documented way to re-grade one trial -- reuses it. Without this the second run's
# crash is reported at
# the FIRST run's score, which is the exact "a plausible number beats a missing
# file" failure the withholding exists to prevent, one invocation over.
rm -f /logs/verifier/eval_summary.json /logs/verifier/reward.json

# /logs/verifier IS ROOT'S, AND STAYS ROOT'S. It holds `reward.json`. Chowning it
# to the graded account so conftest could write its summary here would move the
# hole rather than close it: a detached watcher outliving pytest (this script is
# still PID 1) could rewrite the reward file after it is written. The graded uid
# gets a scratch directory of its own instead, under a root-owned parent so the
# DIRECTORY cannot be swapped, and root publishes the summary out of it below.
chmod 0755 /logs/verifier
rm -rf "${_SUMMARY_SRC_DIR}"
mkdir -p "${_SUMMARY_SRC_DIR}"
chown "${_GRADE_UID}:${_GRADE_GID}" "${_SUMMARY_SRC_DIR}"
chmod 0700 "${_SUMMARY_SRC_DIR}"

# THE SECOND GATE: THE DROP IS MEASURED, FROM INSIDE THE DROP, BEFORE A GPU-MINUTE
# IS SPENT.
#
# The gate at the top of this file checks that setpriv and the account EXIST. That
# is not the property this script depends on, and the difference is not academic:
# the hardening ships inside the image (`tests/Dockerfile` RUN steps) and no repo
# gate builds an image, so a verifier tagged before it landed has the account and a
# world-writable /tests. A `/logs` bind mount owned on the host by uid
# ${_GRADE_UID} is the same shape from the other direction. Either restores the
# measured hole -- candidate code rewriting the reward reduction, or the reward
# file itself -- and neither is visible to any static check in this repo.
#
# So ask the graded uid, in one setpriv call, what it can actually reach. Both
# directions are asserted: the four things it must NOT be able to do, and the one
# it MUST (write its own summary -- a scratch directory it cannot write would
# otherwise fail the grade far downstream with a confusing traceback).
#
# The reducer's /tests path is BUILT from `_REDUCER`'s basename rather than
# written out, because the harness refuses that literal anywhere in this file
# -- the whole point of the relocation is that
# nothing here reads the reducer from a candidate-writable path.
#
# A setpriv that fails outright lands here too: its error text is not "EFFECTIVE",
# so it aborts with that text quoted rather than proceeding.
_drop_check="$(setpriv --reuid="${_GRADE_UID}" --regid="${_GRADE_GID}" --clear-groups --no-new-privs -- \
  sh -c '
    bad=""
    [ -w /tests ]        && bad="${bad} /tests-is-writable"
    [ -e "$1" ]           && bad="${bad} the-reducer-is-still-in-/tests"
    [ -w "$2" ]           && bad="${bad} the-reducer-directory-is-writable"
    [ -w /logs/verifier ] && bad="${bad} /logs/verifier-is-writable"
    [ -w "$3" ]           || bad="${bad} the-scratch-summary-dir-is-NOT-writable"
    [ -n "${bad}" ] && { printf "%s" "${bad}"; exit 0; }
    printf "EFFECTIVE"
  ' _probe "/tests/${_REDUCER##*/}" "${_REDUCER%/*}" "${_SUMMARY_SRC_DIR}" 2>&1)"
if [ "${_drop_check}" != "EFFECTIVE" ]; then
  echo "!! aborting: the privilege drop is INEFFECTIVE in this container, so the" >&2
  echo "   graded phase could still write what the grade is read from. Reached as" >&2
  echo "   uid ${_GRADE_UID}:${_drop_check}" >&2
  echo "   The usual cause is a verifier image built before the hardening landed:" >&2
  echo "   rebuild it with ./build_harbor_images.sh <task>. A /logs mount owned" >&2
  echo "   by uid ${_GRADE_UID} on the host does this too. NOTHING is graded and NO" >&2
  echo "   reward.json is written." >&2
  exit 3
fi
# ...and the reduction has to exist where this script will look for it. Same
# ruling, checked as root because the graded uid deliberately cannot see it.
if [ ! -f "${_REDUCER}" ]; then
  echo "!! aborting: no reward reduction at ${_REDUCER}. This image was not built" >&2
  echo "   from this task's tests/Dockerfile. NOTHING is graded." >&2
  exit 3
fi

# THE CANDIDATE'S OWN TREE MUST BE READABLE BY THE GRADED UID. Harbor uploads it
# as root and a mode the agent chose; a 0600 file there would fail the whole grade
# on a permission error and report it as a port that scored nothing. Only r and X
# are added, never w, and `chmod -R` walks physically: with a `/cand/evil ->
# /secretdir` symlink planted in the tree, `chmod -R a+rX /cand` leaves both
# `/secretdir` (drw-------) and `/secretdir/secret` (-rw-------) untouched while
# widening the real file inside the tree. The ancestors get the traverse bit
# alone, one level at a time. Both are best-effort: the
# oracle mounts its repo `:ro`, where chmod fails with EROFS and the host modes
# already allow the read.
if _gradable; then
  chmod -R a+rX "${T2J_JAX_REPO}" 2>/dev/null
  _anc="$(dirname "${T2J_JAX_REPO}")"
  while [ "${_anc}" != "/" ] && [ -n "${_anc}" ]; do
    chmod a+rX "${_anc}" 2>/dev/null
    _anc="$(dirname "${_anc}")"
  done
fi

# HOST PRECONDITIONS, before a GPU-minute is spent inside pytest.
# Fatal on the wrong SKU, a loud warning on cuda_driver / jax_xla_version drift; see
# _grade_precheck.py's header for why those two severities differ and for why a wrong
# card deliberately leaves NO reward.json behind.
if ! ${_PY_TESTS} /tests/_grade_precheck.py --golden /tests/golden --out /logs/verifier; then
  echo "!! aborting: host preconditions not met, no grade was performed" >&2
  exit 3
fi

# THE ONE LINE THAT RUNS CANDIDATE CODE, and the only one that does not run as
# root. See the privilege-drop block at the top for what that closes and what it
# does not. `--clear-groups` drops root's supplementary groups (otherwise every
# group-writable root-owned path stays reachable); `--no-new-privs` forbids
# regaining them through a setuid binary. setpriv execs, so `$?` is pytest's own.
HOME="${_GRADE_HOME}" \
setpriv --reuid="${_GRADE_UID}" --regid="${_GRADE_GID}" --clear-groups --no-new-privs -- \
  python -m pytest -q --no-header -o addopts="" -p no:cacheprovider /tests
_pytest_rc=$?

# PUBLISH THE SUMMARY OUT OF THE GRADED UID'S REACH, IMMEDIATELY.
#
# Every guard below reads /logs/verifier/eval_summary.json, exactly as it did when
# conftest wrote there directly -- so this copy keeps the artifact at its
# documented path and keeps all four rulings that follow reading a file no
# unprivileged process can touch. What it buys: the reduction's input is frozen at
# the instant pytest returned. A watcher that survives pytest (Variant C: a
# double-forked setsid process rewriting the tally in a loop) now rewrites a file
# nothing will ever read again.
#
# O_NOFOLLOW IS THE POINT OF DOING THIS IN PYTHON. Root is reading out of a
# directory the graded uid owns, so a plain `cp` would happily follow a symlink
# planted there and copy an arbitrary file into an artifact that uid can read
# back. The refusal is atomic with the open; a `[ -L ]` test would race.
#
# ...AND NEITHER IS A FIFO, A HARDLINK, OR ANYTHING ELSE THAT IS NOT THE ARTIFACT.
# O_NOFOLLOW closes the symlink; it closes nothing else, and root is opening a name
# in a directory the graded uid owns, so the other file TYPES it can create there
# are part of the same question. `mkfifo` needs no privilege: with
# `/run/t2j-grade/eval_summary.json` a fifo planted by the graded uid,
# root's `os.open(..., O_RDONLY|O_NOFOLLOW)` does not return -- open(2) on a fifo
# with no writer blocks. The
# verifier hangs with no reward.json and the trial dies on Harbor's timeout: a denial
# of the grade rather than a forgery, and just as much a defect.
# O_NONBLOCK makes that open return immediately (and is a no-op on a regular file);
# the fstat then refuses it -- the publish step prints "refusing to
# publish ...: not a single-linked regular file" and exits 1, and the no-summary
# guard below rules on it exactly as it rules on a
# withheld artifact.
#
# `st_nlink != 1` is the hardlink case -- the graded uid naming a root-only file on
# the same filesystem and having root copy it into a world-readable artifact -- and
# it is DEFENCE IN DEPTH, not a demonstrated hole: the graded uid's
# `ln /etc/shadow /run/t2j-grade/eval_summary.json` is refused by the kernel with
# EPERM (fs.protected_hardlinks, which is not a namespaced sysctl and so is the
# grading HOST's setting, not this image's). It is one `os.fstat` field on a host
# that has it turned off. The fstat is on the FD the copy will read, so nothing here
# can be raced after the open.
#
# A MISSING SOURCE IS NOT AN ERROR HERE. Conftest withholds the summary on purpose
# in three documented cases; this exits nonzero, writes nothing, and the no-summary
# guard below -- which is where that ruling lives -- sees exactly what it saw before.
${_PY_ISOLATED} - "${_SUMMARY_SRC_DIR}/eval_summary.json" /logs/verifier/eval_summary.json <<'PYPUBLISH'
import os
import stat
import sys

src, dst = sys.argv[1], sys.argv[2]
try:
    fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
except OSError as exc:
    print(f"[t2j] no eval_summary.json published from {src}: {exc}", file=sys.stderr)
    raise SystemExit(1)
st = os.fstat(fd)
if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
    os.close(fd)
    print(f"[t2j] refusing to publish {src}: not a single-linked regular file "
          f"(mode {st.st_mode:#o}, nlink {st.st_nlink})", file=sys.stderr)
    raise SystemExit(1)
with os.fdopen(fd, "rb") as handle:
    data = handle.read(4 << 20)   # the honest artifact is ~10 KB; cap a hostile one
with open(dst, "wb") as out:
    out.write(data)
PYPUBLISH

# EXIT 4 IS PYTEST'S "YOU INVOKED THIS WRONG", AND HERE IT MEANS NO GRADE HAPPENED.
# All three of conftest's end-of-collection gates raise `pytest.UsageError`, which is
# exit 4: the golden-health gate, the geometry gate and the pinned-device gate. Every
# one of them is a statement about the HARNESS or the HOST, so it gets the same
# treatment the precheck above gives its own failures -- abort with NO reward.json,
# because `reward: 0.0` would say the port scored nothing and that is not what
# happened. Any other nonzero code MAY be an ordinary failing grade and must still
# be scored, which is why this is `-eq 4` and not `-ne 0` -- pytest exits 1 for
# "tests failed", which is precisely the case that has to reach a reward. The
# codes that are NOT ordinary failing grades (2, 3, and any future addition) are
# separated by the artifact they left behind rather than by their number: see the
# no-summary guard below, which is the general net under this specific one.
#
# Belt and braces with the precheck rather than a duplicate of it: the precheck runs
# the same benchmark self-checks first and would normally have caught the golden-health
# case already. This closes the window where the two disagree, and it is the only
# path by which the geometry and device gates can reach a reward file at all.
if [ "$_pytest_rc" -eq 4 ]; then
  echo "!! pytest exited 4 (usage error): one of the grader's preconditions refused" >&2
  echo "   this session. Nothing was graded and NO reward.json is written -- the" >&2
  echo "   message above says which precondition and what to repair." >&2
  exit 3
fi

# NO SUMMARY IS NOT A ZERO. THE SAME RULING AS EXIT 4, ONE LAYER DOWN.
#
# The guard above covers the codes conftest's four end-of-collection gates raise.
# It does NOT cover the OTHER way a session ends without an artifact: conftest
# withholds `eval_summary.json` on purpose in three documented cases -- items
# collected but none graded (case 2), a narrowed selection that graded something
# (case 5), and a PARTIAL tally (case 6, `0 < graded_outcomes < graded_items`) --
# and case 6's stated trigger includes "any exception escaping a hook once
# outcomes had begun accumulating".
#
# On the pinned pytest 9.1.0, such an exception is an INTERNALERROR with
# rc=3, `pytest_sessionfinish` still runs, case 6 withholds the artifact, and
# without this guard execution falls straight through to `summary_to_reward.py` -- whose
# `except (OSError, ValueError): s = {}` turns an absent file into an all-zero
# tally and whose `_rate` returns `0.0` for `ran == 0`. Exit status 0, a
# reward.json reading `{"reward": 0.0, "n_passed": 0}`, and a HARNESS fault filed
# as a port that scored nothing. `conftest._require_healthy_harness` documents
# that exact sequence as having already happened once ("a harness fault scored as
# a failed port, which is the one thing the gates in this module exist to
# prevent"); that fix removed one CAUSE, and this removes the CONVERSION, which
# is what makes the next unforeseen cause harmless.
#
# WHY THE RULING IS MADE HERE AND NOT IN THE REDUCTION. Not because the reduction
# is immovable: `summary_to_reward.py` is this class's own and editable -- it is
# authored once for this class, and is not among the pinned byte
# copies (`_t2j/sandbox.py`, `_t2j/purity.py`, `_t2j/_arrays.py`). A maintainer who
# assumed it was a frozen shared copy might conclude the
# opposite and build a third workaround somewhere else.
#
# The real reason is that the decision is about whether a reward file should EXIST,
# and only the caller can decline to produce one. A reduction taught to refuse
# would exit nonzero and this script would still have to choose what that means --
# which is the choice made below, once, at the one call site this class owns.
#
# WHY NOT `reward: 0.0`: Harbor copies reward.json's keys into the trial metrics,
# and a 0.0 there is indistinguishable from a candidate that failed every check.
# An ABSENT reward is distinguishable -- it is what the wrong-SKU precheck and the
# exit-4 gate already produce, and it is the state a reader of the logs has to
# investigate rather than average.
if [ ! -f /logs/verifier/eval_summary.json ]; then
  echo "!! pytest exited ${_pytest_rc} and wrote NO eval_summary.json, so this" >&2
  echo "   session graded nothing gradable. Either the grader withheld the" >&2
  echo "   artifact on purpose (see the [t2j] line above: a partial tally, a" >&2
  echo "   narrowed selection, or a collection with no graded items) or it died" >&2
  echo "   before writing it. Both are statements about the HARNESS, so this" >&2
  echo "   aborts with NO reward.json rather than converting an absent summary" >&2
  echo "   into reward 0.0 -- which would report a harness fault as a port that" >&2
  echo "   scored nothing." >&2
  exit 3
fi

# A SUMMARY THAT SAYS "NOTHING WAS GRADED" IS NOT A SCORE EITHER, and this is
# the sibling of the guard above rather than a new idea. `pytest_sessionfinish`
# case 1 DELIBERATELY writes a summary when the collection is empty, so that the
# reader learns which way the session got there -- that diagnosis is worth
# keeping. But the guard above only fires on an ABSENT summary, so a PRESENT one
# carrying `empty_collection` fell straight through to `summary_to_reward.py`
# and became `{"reward": 0.0, "n_passed": 0}`.
#
# `import <missing module>` in all three graded modules gives
# `Interrupted: 3 errors during collection`, pytest rc 2 -- which the exit-4 trap
# does not match -- and a reward.json of 0.0: a broken verifier IMAGE filed
# as a port that scored nothing, which is the exact sentence the block above says
# this script exists to prevent. `_grade_precheck.py` cannot catch it either: it
# reads the graded modules by AST and so never executes an import.
#
# Only the realistic one-module shape was already safe -- breaking a single
# module leaves the others collected, `_require_complete_collection` sees the
# gap and raises `pytest.UsageError`, and rc 4 aborts. This closes the all-three
# case, which is what a broken shared dependency produces.
_not_graded="$(${_PY_ISOLATED} - <<'PYEMPTY'
import json, sys
try:
    d = json.load(open("/logs/verifier/eval_summary.json"))
except Exception:
    sys.exit(0)
for key in ("empty_collection", "no_graded_outcomes"):
    if d.get(key):
        print(f"{key}: {d[key]}")
        break
PYEMPTY
)"
if [ -n "${_not_graded}" ]; then
  echo "!! the grader wrote a summary saying it graded NOTHING:" >&2
  echo "   ${_not_graded}" >&2
  echo "   That is a statement about the HARNESS -- a broken image, a narrowed" >&2
  echo "   invocation, a collection error -- not about the port. Aborting with NO" >&2
  echo "   reward.json rather than converting it into reward 0.0. The summary is" >&2
  echo "   left in place; read it to see which way the session got there." >&2
  exit 3
fi

# ...AND THE CONVERSION'S OWN EXIT STATUS IS CHECKED.
#
# This script runs `set -uo pipefail` with NO `-e` (deliberately -- the pytest
# call above must be allowed to fail so its code can be classified), so a
# `summary_to_reward.py` that dies leaves the script running and, with nothing
# after it in the not-gradable case, EXITING 0 WITH NO reward.json. An
# `eval_summary.json` holding a JSON *list* rather than an object survives the
# `except (OSError, ValueError)` -- `json.loads` succeeds -- and raises
# `AttributeError: 'list' object has no attribute 'get'`, outside
# the try: rc=1 and no reward.json, and without this guard the script would report success.
#
# Same ruling as above and for the same reason: a reduction that could not run is
# a harness fault, so it exits 3 with no reward file rather than leaving a silent
# green. A partial write cannot survive either -- `json.dump` is the last
# statement in `main()`, but a crash mid-write would leave truncated JSON, so the
# file is removed before exiting.
if ! ${_PY_ISOLATED} "${_REDUCER}" \
       /logs/verifier/eval_summary.json /logs/verifier/reward.json; then
  echo "!! summary_to_reward.py failed on an eval_summary.json that EXISTS, so" >&2
  echo "   the grade could not be reduced to a reward. That is a fault in the" >&2
  echo "   grader or in the artifact it wrote, not in the candidate: aborting" >&2
  echo "   with NO reward.json. The traceback above names the line." >&2
  rm -f /logs/verifier/reward.json
  exit 3
fi

# NOTHING GRADABLE EXITS NONZERO -- after reward.json is written, never instead of it.
#
# WHAT THIS EXIT STATUS IS. A LOG SIGNAL, and nothing more: harbor 0.20.0 discards the
# verifier's exit code entirely (environments/docker/docker.py::_compose_exec is called
# with check=False and verifier/verifier.py::verify() drops the ExecResult), so neither
# value can move a score. It is worth getting right anyway, because a wrong one points
# an investigator at the wrong subsystem -- which is exactly what the old predicate did.
#
# The reward file is still written FIRST and is still LOW (0.0). Harbor must
# read a score rather than an absence: a verifier that dies before writing one is
# recorded the same way as one that never ran, so "the agent produced nothing" would be
# indistinguishable from "the grading host fell over".
#
# Scoped to the NOT-GRADABLE case only. A candidate that imports and scores badly exits
# 0: this benchmark pays continuous partial credit, and turning every imperfect port
# into a nonzero exit would make the status meaningless.
#
# The message reports WHICH roots were looked at and what was found at each, because
# "the agent never wrote a repo" and "the agent wrote a repo with no config.py" are
# different faults in different subsystems and the old wording asserted the first.
if ! _gradable; then
  echo "!! NO GRADABLE CANDIDATE. Nothing was graded against a real port; the reward" >&2
  echo "   above is what this suite scores for one. A candidate is gradable when" >&2
  echo "   <root>/config.py exists. Roots examined:" >&2
  for _root in "${T2J_JAX_REPO:-}" /logs/artifacts/workspace/jax_repo /workspace/jax_repo; do
    [ -z "${_root}" ] && continue
    if [ ! -d "${_root}" ]; then
      echo "     ${_root}: no such directory" >&2
    elif [ ! -f "${_root}/config.py" ]; then
      echo "     ${_root}: EXISTS but has no config.py -- the agent produced a tree" >&2
      echo "       that does not mirror torch_orig; this is a candidate fault, not a" >&2
      echo "       collection fault" >&2
    else
      echo "     ${_root}: gradable (and should have been selected -- report this)" >&2
    fi
  done
  exit 4
fi
