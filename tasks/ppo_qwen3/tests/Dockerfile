# Verifier image for ppo/qwen3 — torch-free (EVAL_FRAMEWORK_DESIGN.md §4.2).
# Build with the REPO ROOT as context:
#   docker build -f tasks/ppo_qwen3/tests/Dockerfile -t t2j-verifier:ppo-qwen3 .
#
# Base = the torch-free grade env plus this model's weights
# (base_image/Dockerfile.verifier-model -> t2j-verifier-model:qwen3, built by
# base_image/build_base_images.sh).
#
# /data/Qwen/Qwen3-0.6B arrives with the base and nothing is fetched here — no hub call, no
# token, and the verifier still runs `network_mode = "no-network"`. Grading needs both
# halves of that directory: jax_repo/model/loader.py opens its tokenizer.json with the
# `tokenizers` pin, and since §4.3 the candidate's own convert_checkpoint reads its
# .safetensors. Those weights were COPYed from the torch-ful t2j-model:qwen3, so the
# grading side provably sees the same checkpoint the agent trained against.
FROM t2j-verifier-model:qwen3

# Bake the whole grader tree: test_*.py, conftest, fixtures, _t2j/, golden/, plus
# test.sh + summary_to_reward.py. Harbor's separate verifier runs /tests/test.sh.
# Bake the training dataset the in-band L6/T3 gates need (§7.4/§7.7): the grader runs the
# candidate's REAL training entrypoint, whose data/loader.py reads /data/<dataset>.jsonl +
# /data/dataset_info.json (dataset_dir defaults to /data in config.py). The verifier base
# is torch-free with no huggingface_hub, so it cannot re-run the prepare script; instead
# copy the already-materialized dataset artifacts (torch-free -- just data) straight out of
# this task's agent image, which build_harbor_images.sh builds immediately before this one.
# This is the SAME dataset the agent trained against, so the run stays hermetic/offline.
COPY --from=t2j-agent:ppo-qwen3 /data/dataset_info.json /data/
COPY --from=t2j-agent:ppo-qwen3 /data/*.jsonl /data/

COPY tasks/ppo_qwen3/tests /tests
WORKDIR /tests
