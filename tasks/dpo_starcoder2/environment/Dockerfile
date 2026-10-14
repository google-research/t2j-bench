# Agent image for dpo/starcoder2. Build from the repo root with:
#   ./build_harbor_images.sh dpo_starcoder2
FROM t2j-model:starcoder2

# Bake the training data into /data at build time: the container has no network.
COPY tasks/dpo_starcoder2/environment/prepare_code_vuln_dpo.py  /tmp/prepare_code_vuln_dpo.py
RUN python /tmp/prepare_code_vuln_dpo.py --out /data \
    && rm -rf /tmp/prepare_code_vuln_dpo.py /root/.cache/huggingface

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/dpo_starcoder2/environment/torch_orig                /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
