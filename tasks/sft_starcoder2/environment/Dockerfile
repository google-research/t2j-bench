# Agent image for sft/starcoder2. Build from the repo root with:
#   ./build_harbor_images.sh sft_starcoder2
FROM t2j-model:starcoder2

# Bake the training data into /data at build time: the container has no network.
COPY tasks/sft_starcoder2/environment/prepare_commitpackft.py   /tmp/prepare_commitpackft.py
RUN python /tmp/prepare_commitpackft.py --out /data \
    && rm -rf /tmp/prepare_commitpackft.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/sft_starcoder2/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
