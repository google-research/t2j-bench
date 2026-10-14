# Agent image for sft/qwen2_omni. Build from the repo root with:
#   ./build_harbor_images.sh sft_qwen2_omni
FROM t2j-model:qwen2_omni

# Bake the training data into /data at build time: the container has no network.
COPY tasks/sft_qwen2_omni/environment/prepare_voice_assistant.py /tmp/prepare_voice_assistant.py
RUN python /tmp/prepare_voice_assistant.py --out /data \
    && rm -rf /tmp/prepare_voice_assistant.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/sft_qwen2_omni/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
