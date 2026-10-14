# Agent image for ppo/falcon. Build from the repo root with:
#   ./build_harbor_images.sh ppo_falcon
FROM t2j-model:falcon

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_falcon/environment/prepare_ifeval.py         /tmp/prepare_ifeval.py
RUN python /tmp/prepare_ifeval.py --out /data \
    && rm -rf /tmp/prepare_ifeval.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_falcon/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
