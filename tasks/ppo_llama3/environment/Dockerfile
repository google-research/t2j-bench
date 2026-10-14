# Agent image for ppo/llama3. Build from the repo root with:
#   ./build_harbor_images.sh ppo_llama3
FROM t2j-model:llama3

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_llama3/environment/prepare_ifeval.py         /tmp/prepare_ifeval.py
RUN python /tmp/prepare_ifeval.py --out /data \
    && rm -rf /tmp/prepare_ifeval.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_llama3/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
