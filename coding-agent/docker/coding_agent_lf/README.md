# coding_agent_lf Docker Image

All shell commands below assume you are running from the repo root.

This directory builds a Docker image for running `coding_agent` inside a
container while preserving the same Python environment layout as the current
host `lf` conda environment.

The main design goals are:
- container Python should match the active host `lf` environment
- GPU should be available inside the container
- editable local sources used by the agent should still resolve correctly

## What This Image Contains

The image is built from:
- the current host `lf` conda environment, packed with `conda-pack`
- the current editable `google_adk` source tree
- the current editable `LlamaFactory` source tree

The exact host source locations are resolved by `build_image.sh`. By default it
uses paths under `HOME`, and you can override them with:
- `CODING_AGENT_ADK_SRC`
- `CODING_AGENT_LLAMAFACTORY_SRC`
- `CODING_AGENT_CONDA_PACK_BIN`

## Files

- [Dockerfile](./Dockerfile)
- [build_image.sh](./build_image.sh)
- [run_attempt_in_docker.py](../../scripts/run_attempt_in_docker.py)

## Requirements

Host requirements:
- Linux x86_64
- Docker
- NVIDIA driver compatible with the CUDA runtime in the image
- NVIDIA Container Toolkit so `docker run --gpus all` works
- access to the Docker daemon via either:
  - `docker`
  - or `sudo -n docker`

Python environment requirements on the host:
- the `lf` conda environment must exist
- `conda-pack` must be available on the host, or `CODING_AGENT_CONDA_PACK_BIN` must point to it

## Build

Build the image with:

```bash
bash coding-agent/docker/coding_agent_lf/build_image.sh coding-agent-lf:latest
```

The build script will:
1. Copy editable sources for `google_adk` and `llamafactory`
2. Pack the current host `lf` environment into `.artifacts/lf-env.tar.gz`
3. Build the Docker image

The build script automatically tries:
- `docker`
- then `sudo -n docker`

## Verify The Environment

Check the Python path and visible GPUs:

```bash
sudo -n docker run --rm --gpus all coding-agent-lf:latest \
  bash -lc 'which python && python -V && nvidia-smi -L'
```

Expected:
- `python` resolves to the `lf` environment inside the container
- GPUs are listed by `nvidia-smi`

Check JAX sees CUDA devices:

```bash
sudo -n docker run --rm --gpus all coding-agent-lf:latest \
  bash -lc 'python - <<'"'"'\"'"'"'PY'"'"'\"'"'"'
import jax
print(jax.__version__)
print(jax.devices())
PY'
```

Expected:
- JAX imports successfully
- `jax.devices()` returns `CudaDevice(...)` entries

Check editable sources resolve correctly:

```bash
sudo -n docker run --rm coding-agent-lf:latest \
  bash -lc 'python - <<'"'"'\"'"'"'PY'"'"'\"'"'"'
import inspect
import google.adk
import llamafactory
print(inspect.getfile(google.adk))
print(inspect.getfile(llamafactory))
PY'
```

Expected:
- both imports resolve into the editable source trees copied by `build_image.sh`

## Use With The Attempt Runner

The recommended entrypoint is:
- [run_attempt_in_docker.py](../../scripts/run_attempt_in_docker.py)

Prebuilt torch-to-jax attempt wrappers are available in:
- [attempt_scripts](./attempt_scripts)

Their corresponding prompt files are in:
- [prompts](./prompts)

Available wrappers:
- `kill_all_attempts.sh`
- `run_all_dpo_ppo_sft_once.sh`
- `run_attempt_background.sh`
- `run_dpo_gemma3.sh`
- `run_dpo_llava_next.sh`
- `run_dpo_qwen3_vl.sh`
- `run_ppo_gemma3.sh`
- `run_ppo_llava_next.sh`
- `run_ppo_qwen3_vl.sh`
- `run_sft_gemma3.sh`
- `run_sft_llava_next.sh`
- `run_sft_qwen3_vl.sh`

Each wrapper:
- uses its matching prompt from `coding-agent/docker/coding_agent_lf/prompts/`
- runs `client_agent`
- mounts exactly one `dpo/ppo/sft/<model>/torch_orig`
- mounts `eval`
- requests two GPUs by default via `--gpus device=0,1`

Example:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_ppo_gemma3.sh
```

Run with manually selected GPUs:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_ppo_gemma3.sh --gpus device=2,3
```

Run all `run_dpo*`, `run_ppo*`, and `run_sft*` scripts once in batches of 4,
with non-overlapping 2-GPU assignments per batch:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_all_dpo_ppo_sft_once.sh
```

Preview the launch order without running anything:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_all_dpo_ppo_sft_once.sh --dry-run
```

Stop all currently running Docker attempts for this image:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/kill_all_attempts.sh
```

Preview what would be stopped:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/kill_all_attempts.sh --dry-run
```

Start an attempt in the background and store logs under `.attempt_docker/`:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_attempt_background.sh dpo gemma3
```

Start in the background on specific GPUs:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_attempt_background.sh \
  dpo gemma3 \
  --gpus device=2,3
```

Use a custom log file:

```bash
coding-agent/docker/coding_agent_lf/attempt_scripts/run_attempt_background.sh \
  dpo gemma3 \
  --log-file dataset/.attempt_docker/dpo_gemma3_custom.log
```

Direct example with the Python runner:

```bash
python scripts/run_attempt_in_docker.py \
  --image coding-agent-lf:latest \
  --workspace dataset \
  --mount ppo \
  --mount eval \
  --prompt-file prompt.md \
  --agent client_agent \
  --verbose
```

The attempt runner will:
- pass through almost all host environment variables into the container
  - except a small set of shell/runtime path variables that would break the container environment
- mount host Hugging Face cache and config when present:
  - `~/.cache/huggingface`
  - `~/.huggingface`
- set container-friendly Hugging Face paths under the container user's home directory
- mount the repo read-only in the container
- mount the selected workspace paths under the container workspace root
- create attempt artifacts under `<workspace>/.attempt_docker/...`
- store agent state for that specific run under the same attempt directory:
  - `.coding_agent_memory/`
  - `.adk/`
  - `.attempt/`
  - `.tmp/`
- run `call_agent.py` inside the container
- use the configured `lf` environment Python inside the container
- pass `--gpus all` by default

## GPU Selection With The Attempt Runner

By default, the attempt runner passes:

```bash
--gpus all
```

To disable GPU explicitly:

```bash
... --gpus none
```

To expose a single specific GPU:

```bash
... --gpus '"device=0"'
```

To expose multiple specific GPUs:

```bash
... --gpus '"device=0,1"'
```

The `--gpus` value is passed through directly to Docker's `--gpus` flag, so it
should follow Docker's native GPU selection syntax.

## Direct Interactive Use

If you want a shell in the container:

```bash
sudo -n docker run --rm -it --gpus all coding-agent-lf:latest bash
```

If you want the current repo root mounted read-only into the container, bind-mount `$PWD` to a directory inside the container when launching `docker run`.

## Rebuild Policy

This image is a snapshot of the current host environment. Rebuild it when:
- the host `lf` environment changes
- `google_adk` editable code changes
- `LlamaFactory` editable code changes
- CUDA/JAX-related packages in `lf` change

If the host environment changes, the existing image will not update itself.

## Where Outputs And Agent State Go

For each run, the attempt runner creates:

- `<workspace>/.attempt_docker/<timestamp>/prompt.md`
- `<workspace>/.attempt_docker/<timestamp>/docker_command.sh`
- `<workspace>/.attempt_docker/<timestamp>/agent_history/`

When you mount only selected workspace subdirectories, the agent's writable
state is still persisted to the host, but it is isolated per run under:

- `<workspace>/.attempt_docker/<timestamp>/.coding_agent_memory/`
- `<workspace>/.attempt_docker/<timestamp>/.adk/`
- `<workspace>/.attempt_docker/<timestamp>/.attempt/`
- `<workspace>/.attempt_docker/<timestamp>/.tmp/`

This keeps memory, todos, temporary files, and nested attempt state scoped to a
single Docker run instead of mixing them into the workspace root.

## Portability Notes

This image is more portable than reusing the host environment directly, but it
is not universally portable.

What usually carries over:
- Python package set
- editable source layout inside the container
- agent runtime behavior

What may break on another machine:
- GPU runtime compatibility
- Docker GPU passthrough
- NVIDIA driver compatibility
- architecture mismatch

On a new host, validate at least:
- `which python`
- `python -V`
- `nvidia-smi -L`
- `python -c "import jax; print(jax.devices())"`

## Troubleshooting

If build fails with Docker socket permission errors:
- verify `docker ps` works
- or verify `sudo -n docker ps` works

If container starts but GPU is missing:
- check `docker run --gpus all ... nvidia-smi -L`
- verify NVIDIA Container Toolkit is installed on the host

If Python path is wrong:
- run:
  - `which python`
  - `echo $CONDA_PREFIX`
- expected interpreter:
  - the `python` from the active `lf` environment

If imports fail for editable packages:
- verify the editable source roots configured in `build_image.sh` exist
- verify the copied sources were included during image build
