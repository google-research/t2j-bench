# Agent image for dpo/phi4. Build from the repo root with:
#   ./build_harbor_images.sh dpo_phi4
FROM t2j-model:phi4

# Bake the training data into /data at build time: the container has no network.
COPY tasks/dpo_phi4/environment/prepare_hh_rlhf.py        /tmp/prepare_hh_rlhf.py
RUN python /tmp/prepare_hh_rlhf.py --out /data \
    && rm -rf /tmp/prepare_hh_rlhf.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/dpo_phi4/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
