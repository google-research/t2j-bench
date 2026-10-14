# Agent image for ppo/bloom. Build from the repo root with:
#   ./build_harbor_images.sh ppo_bloom
FROM t2j-model:bloom

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_bloom/environment/prepare_aya.py            /tmp/prepare_aya.py
RUN python /tmp/prepare_aya.py --out /data \
    && rm -rf /tmp/prepare_aya.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_bloom/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
