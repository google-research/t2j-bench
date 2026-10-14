# Agent image for ppo/qwen2_omni. Build from the repo root with:
#   ./build_harbor_images.sh ppo_qwen2_omni
FROM t2j-model:qwen2_omni

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_qwen2_omni/environment/prepare_voicebench.py     /tmp/prepare_voicebench.py
RUN python /tmp/prepare_voicebench.py --out /data \
    && rm -rf /tmp/prepare_voicebench.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_qwen2_omni/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
