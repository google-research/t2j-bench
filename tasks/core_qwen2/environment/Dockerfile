# Agent image for core/qwen2. Build from the repo root with:
#   ./build_harbor_images.sh core_qwen2
FROM t2j-inference-agent-base

# Only the model weights are taken from the training-class image; basing on it would
# bring in its older jax/flax stack.
COPY --from=t2j-model:qwen2 /data /data

# The PyTorch reference repo the agent migrates (read-only).
COPY tasks/core_qwen2/environment/torch_orig  /workspace/torch_orig
ENV PYTHONPATH=/workspace/torch_orig

# Fail the build if the base image lost the numeric settings the grader uses.
RUN [ "$JAX_DEFAULT_MATMUL_PRECISION" = "highest" ] \
    && case "$XLA_FLAGS" in *--xla_gpu_autotune_level=0*) ;; \
       *) echo "agent base lost the graded numeric pins" >&2; exit 1 ;; esac \
    && echo "confirmed numeric pins: $JAX_DEFAULT_MATMUL_PRECISION / $XLA_FLAGS"

WORKDIR /workspace
# The agent writes its candidate to /workspace/jax_repo.
