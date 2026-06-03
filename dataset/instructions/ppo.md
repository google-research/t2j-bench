# PyTorch-to-JAX Conversion: PPO

## Goal

Convert the source PyTorch PPO repo into `jax_repo/` using a PyTorch-backed JAX wrapper. The converted repo must preserve forward behavior, PPO value computation, and gradient artifact compatibility.

## Shared Requirements

- Use a wrapper around the original PyTorch model.
- Keep the source repo layout as intact as possible.
- Preserve the PPO trainer and utilities from the source repo unless a listed rename or import fix is required.
- Ensure the wrapped model exposes `pt_model` directly.
- Keep the repo-local transformer package self-contained. Internal helper modules must be imported relatively so the same package works in every runtime entry point.
- The converted runtime must follow the same tokenizer, collator, model loader, PPO loss/value computation, and gradient interfaces as the original PyTorch pipeline.
- Match the exact interface names and value-output field names that the original PyTorch PPO path actually uses. Do not rely on alias compatibility if the original repo uses a different name.
- Derive the runtime contract from the original PyTorch path first, then implement the converted runtime to satisfy that contract exactly.
- Config classes used by the local model package must expose `get_text_config(decoder=None, **kwargs)` or an equivalent compatible method.
- If PPO needs a value head, add it on the PyTorch model before exporting params.
- When evaluation requests hidden states or value outputs, make sure the wrapper handles them explicitly.
- Normalize integer token and label tensors to `int64` on the PyTorch side. Do not leave indices as `int32` in loss or gather paths.
- Wrapper forward and apply paths must tolerate control kwargs such as `return_dict`, `use_cache`, and `output_hidden_states` without passing duplicates to the backend model.
- Return summary dicts for gradients instead of arrays.

## File Plan

- Copy the source repo into `jax_repo/`.
- Preserve PPO trainer and utilities unless an import, packaging, or runtime compatibility fix is required.
- Rewrite or create `model/loader.py`.
- Create `model/__init__.py` only if the package layout requires it.
- Copy any vendored local transformer or model package into `jax_repo/` and keep imports relative.
- Apply only the minimal import, path, or packaging fixes needed by the original repo.

## `model/loader.py` Requirements

- Load the vendored local model and its config.
- Resolve backend device safely.
- Load checkpoints from safetensors or other checkpoint formats already used by the source repo.
- Add a value head when PPO requires it.
- Preserve dtype handling for reduced precision and integer token inputs.
- Support `apply(...)` returning logits, loss, and values when applicable.
- Keep `output_hidden_states`, `return_dict`, and `use_cache` handling explicit so kwargs are not passed twice.
- If the backend model depends on runtime-only buffers or generated position metadata, rebuild or clear that state deterministically before runtime calls.
- Delegate standard model methods to `pt_model`.

## Skeleton

```python
def _ensure_local_transformers_package() -> str:
    # please complete this part


def _import_local_transformers_module(module_name: str):
    # please complete this part


def _hf_transformers_context():
    # please complete this part


def _load_state_dict(model_path):
    # please complete this part


def _resolve_runtime_dtype(config, requested_dtype):
    # please complete this part


def _torch_to_jax(tensor):
    # please complete this part


def _any_to_torch(value, device, key=None):
    # please complete this part


def _batch_to_torch(batch, device):
    # please complete this part


def _summarize_tensor(arr):
    # please complete this part


class ModelOutput:
    def __init__(self, logits, loss=None, values=None):
        # please complete this part


class ModelWrapper:
    def __init__(self, pt_model, backend_device, jax_device=None, add_valuehead=False):
        # please complete this part

    def device(self):
        # please complete this part

    def reset_runtime_state(self):
        # please complete this part

    def apply(self, variables, **kwargs):
        # please complete this part

    def gradient_artifact(self, batch, training_method="ppo", ignore_index=-100):
        # please complete this part

    def generate(self, **kwargs):
        # please complete this part

    def train(self, mode=True):
        # please complete this part

    def eval(self):
        # please complete this part

    def parameters(self):
        # please complete this part

    def named_parameters(self):
        # please complete this part

    def state_dict(self):
        # please complete this part

    def zero_grad(self):
        # please complete this part

    def modules(self):
        # please complete this part


def load_tokenizer(model_args):
    # please complete this part


def load_model(
    tokenizer,
    model_args,
    finetuning_args,
    is_trainable=True,
    add_valuehead=False,
    backend_device=None,
    jax_device=None,
):
    # please complete this part
```

## Behavioral Notes

- Keep tokenizer and processor loading aligned with the source repo.
- Freeze optional submodules only if the source finetuning args require it.
- If the PyTorch model lacks `generate()`, implement fallback greedy decoding in the wrapper.
- Preserve preprocessing and generation config setup expected by the original repo.
- Expose helper methods or delegated attributes required by collators, trainers, or evaluation.
- Ensure `output_hidden_states` is handled before forwarding kwargs when value-head logic depends on hidden states.

## Acceptance Criteria

- PPO trainer imports and runs against `jax_repo/`.
- Wrapper returns logits plus value outputs when required.
- Gradient artifact returns JSON-serializable summaries only.
- The converted runtime uses the same loader, collator, forward, value-head, and loss interfaces, including the same field names actually consumed by the original pipeline.
- Evaluation can run end to end without duplicate-kwarg, dtype, or runtime-state failures.
