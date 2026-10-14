# Agent image for ppo/qwen2_vl. Build from the repo root with:
#   ./build_harbor_images.sh ppo_qwen2_vl
FROM t2j-model:qwen2_vl

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_qwen2_vl/environment/prepare_geometry3k.py     /tmp/prepare_geometry3k.py
RUN python /tmp/prepare_geometry3k.py --out /data \
    && rm -rf /tmp/prepare_geometry3k.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_qwen2_vl/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
