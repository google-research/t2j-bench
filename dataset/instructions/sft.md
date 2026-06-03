# PyTorch-to-JAX Conversion: SFT

## Goal

Convert the source PyTorch SFT repo into `jax_repo/` using a PyTorch-backed JAX wrapper. The converted repo must keep the original model implementation and produce numerically aligned forward and gradient artifacts.

## Shared Requirements

- Use a wrapper pattern. Do not rewrite the model in native JAX/Flax.
- Keep the data pipeline unchanged unless a source-repo compatibility fix is required.
- Load the vendored local model implementation, not HuggingFace `AutoModel*`, unless the source repo already does so.
- Keep the repo-local transformer package self-contained. Internal helper modules must be imported relatively so the same package works in every runtime entry point.
- Expose the same public entry points expected by evaluation:
  - `config.get_train_args(...)`
  - `model.loader.load_tokenizer(...)`
  - `model.loader.load_model(...)`
  - dataset, collator, template, and processor exports already present in the source repo
- The converted runtime must follow the same tokenizer, processor, collator, model loader, forward, gradient, and generation interfaces as the original PyTorch pipeline.
- Match the interface names and output field names that the original PyTorch pipeline actually uses. Do not introduce alternate names and do not rely on alias compatibility if the original repo uses a different name.
- Derive the runtime contract from the original PyTorch path first, then implement the converted runtime to satisfy that contract exactly.
- Config classes used by the local model package must expose `get_text_config(decoder=None, **kwargs)` or an equivalent compatible method when required by the source stack.
- Preserve dtype behavior, especially reduced-precision runtime paths such as `bfloat16`.
- Normalize integer token and label tensors to `int64` on the PyTorch side. Do not leave indices as `int32` in loss or gather paths.
- Wrapper forward and apply paths must tolerate control kwargs such as `return_dict`, `use_cache`, and `output_hidden_states` without passing duplicates to the backend model.
- Return gradient summaries as plain Python metadata dicts, not raw arrays.

## File Plan

- Copy the source repo into `jax_repo/`.
- Keep non-model files unchanged unless an import, packaging, or runtime compatibility fix is required.
- Rewrite or create `model/loader.py`.
- Copy any vendored local transformer or model package into `jax_repo/` and keep imports relative.
- Add only the minimal compatibility shims required by the original runtime contract.
- If the wrapped model has no native `generate()`, implement deterministic fallback generation in the wrapper.

## `model/loader.py` Requirements

- Register the vendored transformers package under a private import name.
- Use a temporary context when importing the real HuggingFace `transformers` package for tokenizer or processor loading.
- Load checkpoints from sharded or single-file safetensors or PyTorch checkpoints.
- Convert JAX and NumPy inputs to PyTorch tensors on the backend device.
- Convert PyTorch outputs back to JAX arrays, preserving reduced-precision dtypes where applicable.
- Expose a wrapper with:
  - `pt_model`
  - `config`
  - `device`
  - `apply(...)`
  - `gradient_artifact(...)`
  - `generate(...)`
  - delegate methods such as `train`, `eval`, `parameters`, `named_parameters`, `state_dict`, `zero_grad`, `modules`
- Reset or rebuild runtime-only state deterministically before runtime calls when required by the backend model.
- If the backend model depends on generated position metadata or similar transient buffers, rebuild or clear that state instead of reusing stale values from earlier forwards.

## Skeleton

```python
def _ensure_local_transformers_package() -> str:
    # please complete this part


def _import_local_transformers_module(module_name: str):
    # please complete this part


def _hf_transformers_context():
    # please complete this part


def _iter_checkpoint_files(model_path):
    # please complete this part


def _load_state_dict_file(path):
    # please complete this part


def _load_state_dict(model_path):
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
    def __init__(self, logits, loss=None):
        # please complete this part


class ModelWrapper:
    def __init__(self, pt_model, backend_device, jax_device=None):
        # please complete this part

    def device(self):
        # please complete this part

    def reset_runtime_state(self):
        # please complete this part

    def apply(self, variables, **kwargs):
        # please complete this part

    def gradient_artifact(self, batch, training_method="sft", ignore_index=-100):
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
    backend_device=None,
    jax_device=None,
):
    # please complete this part
```

## Behavioral Notes

- Keep tokenizer and processor loading aligned with the source repo.
- Preserve runtime dtype resolution so reduced precision remains consistent when the backend supports it.
- Expose any delegated methods or helper attributes that collators, processors, or trainers expect.
- Preserve adapter or wrapper behavior required by the source repo before exporting params.

## Acceptance Criteria

- `jax_repo/` imports cleanly.
- Tokenizer and processor load successfully.
- Forward output and gradient artifact run without shape or dtype mismatches.
- The converted runtime uses the same loader, collator, forward, gradient, and generation interfaces, including the same field names actually consumed by the original pipeline.
- Evaluation can call the wrapper without serialization failures.
