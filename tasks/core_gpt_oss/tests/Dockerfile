# Verifier image for core/gpt_oss -- torch-free.
# Build with the REPO ROOT as context:
#   docker build -f tasks/core_gpt_oss/tests/Dockerfile \
#                -t t2j-inference-verifier:core-gpt-oss .
# or just: ./build_harbor_images.sh core_gpt_oss
#
# The base is the torch-free grade env (jax 0.11.0 / flax 0.12.8, build-asserted to have
# no torch). A hidden `import torch` in the candidate therefore fails hard here, which is
# the class's strongest independence guarantee.
FROM t2j-inference-verifier-base

# THE WEIGHTS ARRIVE BY `COPY --from`, NOT BY `FROM` -- same trap as the agent side.
# `t2j-verifier-model:gpt_oss` is reused unchanged from the training class and is built
# `FROM t2j-verifier-base` (jax 0.6.2 / flax 0.10.7); basing on it would replace this
# image's whole grade environment with that one.
#
# Grading needs both halves of /data/openai/gpt-oss-20b: the candidate's
# own model/loader.py opens tokenizer.json with the `tokenizers` pin, and its
# convert_checkpoint reads model.safetensors. The bytes came from the same source as the
# agent's copy, so the two sides provably see one checkpoint. No hub call, no token, and
# the verifier still runs `network_mode = "no-network"`.
COPY --from=t2j-verifier-model:gpt_oss /data /data

# Bake the whole grader tree: test_*.py, conftest, fixtures, _t2j/, golden/, plus
# test.sh, _grade_precheck.py and summary_to_reward.py. Harbor's separate verifier runs
# /tests/test.sh.
#
# golden/ IS GIT-IGNORED and must still land here. build_harbor_images.sh narrows
# .dockerignore to this task so that happens; the proof that it worked is the oracle
# score, not a reading of the build log (a missing artifact deselects or fails a check
# rather than breaking the COPY).
COPY tasks/core_gpt_oss/tests /tests

# THE GRADED PYTEST DOES NOT RUN AS ROOT, AND THE REWARD REDUCTION IS NOT ON A PATH IT
# CAN NAME.
#
# `test.sh` runs `python -m pytest /tests`, which imports and EXECUTES the candidate's
# repo in this container; it then reads the reward reducer off disk and reduces an
# eval_summary.json. Without the hardening below, the graded phase would run as root:
# `os.getuid()` 0, `/tests/summary_to_reward.py` W_OK True, open-for-append SUCCEEDS,
# `/tests` writable. Candidate module-level code could therefore replace the reduction
# with a script that writes a perfect tally -- reward 1.0 for a repo that ports nothing.
#
# Two changes, and `test.sh` depends on BOTH (it aborts with no reward.json if the
# account or setpriv is missing, rather than grading as root):
#
#   1. `t2jgrade` owns nothing. /tests is root-owned with every group/world write bit
#      stripped -- COPY preserves the source modes, so this is not assumed -- which
#      makes the reducer, conftest.py, the graded modules and golden/ read-only to the
#      graded phase. /logs/verifier, which holds reward.json, is root's too (test.sh).
#   2. The reducer moves out of /tests to a 0700 root-only directory, so the graded uid
#      cannot read it, let alone write it. Belt and braces with (1) rather than a
#      substitute for it: (1) is what actually closes the write class.
#
# NOT CLOSED: the
# candidate runs in the same PROCESS as the hook that writes eval_summary.json, so an
# in-session forgery of the tally is still possible in principle. Only reducing in a
# separate container or under a second uid closes that, and that is Harbor's to change.
#
# NO `USER` LINE, deliberately. test.sh must start as root: it writes reward.json into a
# directory the graded uid is not allowed to touch, and it drops privileges itself, for
# the pytest phase only, with setpriv (which -- unlike su/runuser -- preserves the
# environment the grade is pinned by).
RUN set -eux; \
    groupadd --gid 1101 t2jgrade; \
    useradd --uid 1101 --gid 1101 --no-log-init --create-home --shell /usr/sbin/nologin t2jgrade; \
    command -v setpriv; \
    mkdir -p /opt/t2j-grade; \
    mv /tests/summary_to_reward.py /opt/t2j-grade/summary_to_reward.py; \
    chown -R root:root /opt/t2j-grade /tests; \
    chmod 0700 /opt/t2j-grade; \
    chmod 0600 /opt/t2j-grade/summary_to_reward.py; \
    chmod -R go-w /tests; \
    test ! -e /tests/summary_to_reward.py

WORKDIR /tests
