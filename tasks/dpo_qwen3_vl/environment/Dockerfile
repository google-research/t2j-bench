# Agent image for dpo/qwen3_vl. Build from the repo root with:
#   ./build_harbor_images.sh dpo_qwen3_vl
FROM t2j-model:qwen3_vl

# Bake the training data into /data at build time: the container has no network.
COPY tasks/dpo_qwen3_vl/environment/prepare_sightation_pref.py /tmp/prepare_sightation_pref.py
RUN python /tmp/prepare_sightation_pref.py --out /data \
    && rm -rf /tmp/prepare_sightation_pref.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/dpo_qwen3_vl/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
