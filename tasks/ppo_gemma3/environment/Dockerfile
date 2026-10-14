# Agent image for ppo/gemma3. Build from the repo root with:
#   ./build_harbor_images.sh ppo_gemma3
FROM t2j-model:gemma3

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_gemma3/environment/prepare_geometry3k.py     /tmp/prepare_geometry3k.py
RUN python /tmp/prepare_geometry3k.py --out /data \
    && rm -rf /tmp/prepare_geometry3k.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_gemma3/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
