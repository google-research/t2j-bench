# Agent image for sft/llava_next. Build from the repo root with:
#   ./build_harbor_images.sh sft_llava_next
FROM t2j-model:llava_next

# Bake the training data into /data at build time: the container has no network.
COPY tasks/sft_llava_next/environment/prepare_geometry3k.py     /tmp/prepare_geometry3k.py
RUN python /tmp/prepare_geometry3k.py --out /data \
    && rm -rf /tmp/prepare_geometry3k.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/sft_llava_next/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
