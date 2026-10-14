# Agent image for ppo/internvl. Build from the repo root with:
#   ./build_harbor_images.sh ppo_internvl
FROM t2j-model:internvl

# Bake the training data into /data at build time: the container has no network.
COPY tasks/ppo_internvl/environment/prepare_geometry3k.py     /tmp/prepare_geometry3k.py
RUN python /tmp/prepare_geometry3k.py --out /data \
    && rm -rf /tmp/prepare_geometry3k.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/ppo_internvl/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
