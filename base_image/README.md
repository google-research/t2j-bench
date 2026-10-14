# Base Images

The images every per-task Harbor image builds on. Each one builds with `base_image/` as
its docker context, so this directory alone is enough to produce them.

```bash
base_image/build_base_images.sh                     # training bases + every model in models.txt
base_image/build_base_images.sh bloom qwen2         # training bases + the named models
base_image/build_base_images.sh --bases-only        # just the two training bases
base_image/build_base_images.sh --host-seed         # no docker: populate /data on this host

base_image/build_inference_bases.sh                 # the two inference bases
base_image/build_inference_bases.sh --prove-guard   # also check the torch-free assert works
```

`./build_harbor_images.sh --with-bases <task>` (repo root) runs these for the models its
tasks need. Without `--with-bases` it only checks that the base images exist. Gated models
(gemma3, llama3, paligemma) need a Hugging Face token in `HF_TOKEN`, `HF_TOKEN_FILE` or
`~/.cache/huggingface/token`.

## Training stack (jax 0.6.2 / flax 0.10.7)

```
t2j-agent-base:latest     Dockerfile.agent    jax + flax + optax + torch + transformers
  └─ t2j-model:{model}    Dockerfile.model    + that model's weights at /data/{repo_id}
       └─ t2j-agent:{method}-{model}          tasks/{task}/environment/Dockerfile
t2j-verifier-base:latest  Dockerfile.grade    torch-free grade env
  └─ t2j-verifier-model:{model}               Dockerfile.verifier-model
     │                                        + the same /data/{repo_id}, copied from t2j-model
     └─ t2j-verifier:{method}-{model}         tasks/{task}/tests/Dockerfile
```

The weight images are why task code needs no network: the loaders read
`/data/{repo_id}` and never download. They are shared by every task on a model. The
verifier copies its weights from `t2j-model:{model}` instead of downloading them again,
so both sides see identical files (including falcon's checkpoint, which
`prepare_model.py` converts from `.bin` to `.safetensors` using torch).

## Inference stack (jax 0.11.0 / flax 0.12.8)

```
t2j-inference-verifier-base   Dockerfile.inference-grade   torch-free grade env
  └─ t2j-inference-verifier:{task-with-hyphens}   tasks/{mode}_{model}/tests/Dockerfile
t2j-inference-agent-base      Dockerfile.inference-agent   the above + torch 2.10
  └─ t2j-inference-agent:{task-with-hyphens}      tasks/{mode}_{model}/environment/Dockerfile
```

Task image tags replace every underscore with a hyphen, for example
`core_gpt_oss` becomes `core-gpt-oss`.

A separate stack because the training goldens are frozen against the older jax. It differs
from the training stack in three ways:

- **python 3.12**: jaxlib 0.11.0 has no wheels for older Pythons.
- **Same weight images**: per-task images bring in weights with
  `COPY --from=t2j-model:{model} /data /data`, never `FROM`, since those images are built
  on the jax 0.6.2 base.
- **The agent base is built on the verifier base**, so both sides share one jax/flax install.

## Models

[`models.txt`](models.txt) lists each model and its Hugging Face repo: the 15 training
models plus `gpt_oss`, `qwen2_5_vl` and `gemma4` for the inference tasks. To add a model,
start by adding a line there. The repo id is also the weights path (`/data/{repo_id}`).

`prepare_model.py` downloads only the files the task loaders open. It mirrors
`model/loader.py::_iter_checkpoint_files`, so keep the two in sync.

## Running task code on the host

Outside a container, task code still expects weights at `/data/{repo_id}`. To populate it
from your local Hugging Face cache (hardlinks are used where possible):

```bash
sudo mkdir -p /data && sudo chown "$USER" /data
base_image/build_base_images.sh --host-seed
```

Checkpoint normalization can allocate additional disk space: `.bin` files are converted
to `.safetensors`, and tied tensors are materialized as separate arrays.
