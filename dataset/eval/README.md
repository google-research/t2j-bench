# eval

`eval` is a reference-first evaluator for checking whether a JAX training repo is equivalent to a PyTorch training repo.

The evaluator is designed for repo-to-repo comparison, not for model demos or benchmark scoring. Its job is to answer a narrow question:

`Does this JAX translation implement the same training semantics as the PyTorch reference?`

The intended workflow is:

1. treat `torch_orig` as the contract source
2. validate that `torch_orig` is internally coherent
3. compare the JAX translation against the reference in ordered stages

The framework is written for LlamaFactory-like `SFT`, `DPO`, and `PPO` pipelines, but the core idea is broader: the JAX repo does not need to copy the PyTorch file layout, it does need to match the PyTorch training behavior.

## Design Philosophy

The evaluator follows five rules.

### 1. Reference first

The PyTorch repo is the contract source. If `torch_orig` cannot produce a valid parameter export, batch, forward pass, gradients, or method loss artifact in the requested precision, then it is not a valid reference and cross-framework comparison must stop.

### 2. Ordered stages

Equivalence is not a single boolean computed from one forward pass. The evaluator runs ordered stages:

1. `reference`
2. `spec`
3. `numeric`
4. `behavioral`

Later stages are blocked if earlier stages fail.

### 3. Artifact-driven comparison

The evaluator compares concrete artifacts produced by both repos:

- parameter trees
- collated batches
- forward logits
- optional hidden states
- method-specific loss artifacts
- gradients and gradient norms
- learning-rate schedules
- short behavioral artifacts

The comparison is intentionally explicit. If something fails, the report should tell you which artifact diverged.

### 4. No model-specific evaluator logic

The core evaluator contains training-method-specific logic, but not model-family-specific logic.

Allowed:

- `SFT`, `DPO`, and `PPO` each having different comparison formulas
- generic runtime normalization that applies to both sides equally
- generic support for repos that expose PyTorch or Flax/JAX loaders

Not allowed:

- branching on model family names such as `qwen3_vl`, `gemma3`, or `llava_next`
- passing one model only because the evaluator knows its quirks
- hidden adapters that require the JAX repo to mimic evaluator internals instead of PyTorch semantics

Current audit result:

- there is no model-name-specific logic in `eval/lib/`
- model names appear in README examples and external configs only

### 5. Practical precision and runtime policy

The main acceptance paths are:

- `fp16_compare`
- `bf16_compare`

Full-model `fp32` parity is intentionally out of scope. In practice it is too expensive and often not representative of the target training setup.

`NaN`, `Inf`, and OOM-like failures are hard failures.

## What Gets Evaluated

### `reference`

Checks that the PyTorch reference is usable.

Shared checks:

- non-empty parameter export
- non-empty collated batch
- finite forward outputs
- finite method loss artifact
- finite gradients
- finite learning-rate schedule

Method-specific checks:

- `SFT`: generation when the runtime supports generic generation
- `DPO`: reference-model path when the training setup uses one
- `PPO`: value outputs must be present

Important detail:

After `reference` validation passes, the evaluator rebuilds a fresh reference runtime before `spec` and `numeric`. This prevents preflight execution from mutating runtime caches and contaminating the actual comparison.

### `spec`

Checks structural agreement between the reference and candidate repos:

- parameter names and shapes
- collated batch keys, shapes, and dtypes

`spec` is intentionally structural. It does not prove semantic equivalence, but it catches missing parameters, wrong batch formats, and schema drift early.

### `numeric`

Checks math-level agreement:

- forward logits
- optional hidden states
- forward loss when both sides expose it
- method-specific loss artifacts
- gradients
- gradient norm
- learning-rate schedule
- gradient-accumulation invariant

### `behavioral`

Checks short-run behavior:

- short loss-trajectory replay under evaluator-owned step semantics
- generation parity for `SFT`

The evaluator should not silently skip behavioral checks. If a behavioral artifact cannot be produced, that is a failure to fix, not an acceptable final state.

## Method Focus

### `SFT`

Primary signals:

- parameter loading parity
- batch collation parity
- forward logits
- causal-loss parity
- gradients
- generation when available

### `DPO`

Primary signals:

- chosen/rejected batch collation parity
- policy logits
- policy log-probs
- reference-model log-probs when used
- preference-loss parity
- gradients

### `PPO`

Primary signals:

- forward logits and value outputs
- token log-probs
- GAE-related artifacts
- PPO-style loss decomposition under evaluator-owned formulas
- gradients

The PPO path intentionally uses evaluator-owned deterministic rewards for parity checks. The goal is to compare training semantics, not to compare an external reward service.

## Repo Contract

The evaluator assumes a LlamaFactory-like repo shape and reuses repo-owned entrypoints instead of injecting model-specific glue.

Expected modules:

- `config.py`
- `model/loader.py`
- `data/loader.py`
- `data/collator.py`
- optionally `data/template.py`
- optionally trainer-specific modules such as `dpo/trainer.py` or `ppo/trainer.py`

Required entrypoints:

- `config.get_train_args(config_dict)`
- `model.loader.load_tokenizer(model_args)`
- `model.loader.load_model(...)`
- `data.loader.get_dataset(...)`

Supported implementation styles:

1. native PyTorch repo
2. JAX/Flax-style repo whose model loader returns either:
   - `(model, params, frozen_prefixes)`
   - or a model object exposing `.apply(...)`

## Passing Checklist

If you want a JAX translation to pass the evaluator reliably, these are the main requirements.

### 1. Match the PyTorch batch exactly

This is not just about key names. The candidate must consume the same batch values:

- same token ids
- same masks
- same multimodal tensors
- same position inputs
- same auxiliary fields such as `rope_deltas`

If the JAX repo reinterprets or reconstructs these differently, `forward_logits` will drift even when parameter shapes match.

### 2. Match runtime-only state, not just checkpoint state

Some models contain deterministic runtime buffers that are not persistent checkpoint weights. Rotary-frequency buffers are the main example.

A candidate can have:

- matching parameter trees
- matching LoRA weights
- matching value-head weights

and still fail `forward_logits` because a runtime buffer such as `inv_freq` was left uninitialized or corrupted after load.

The evaluator now normalizes such deterministic rotary buffers on both sides after model load. A translation should still avoid depending on accidental garbage-state behavior.

### 3. Keep forward and gradient semantics consistent

For torch-backed JAX wrappers, forward-only and gradient paths must represent the same model semantics.

Typical mistakes:

- forward in eval mode but gradient in a different effective model state
- dropout or stochastic layers enabled in one path but not the other
- different dtype handling for multimodal float inputs

### 4. Use the same training-method formulas

For `DPO` and `PPO`, matching logits alone is not enough. The candidate must match the PyTorch method semantics:

- log-prob extraction
- chosen/rejected splitting for `DPO`
- GAE and returns for `PPO`
- the same masking rules, including `IGNORE_INDEX`

### 5. Put JAX and backend PyTorch on separate GPUs when needed

If a Flax/JAX candidate is implemented as a JAX wrapper over a PyTorch backend, do not keep both JAX state and backend PyTorch state on the same GPU unless you know the memory is sufficient.

Use:

- `device`: where evaluator-side JAX arrays live
- `backend_device`: where the candidate's backend PyTorch runtime lives

This matters especially for large multimodal PPO and DPO pipelines.

### 6. Do not rely on preflight-mutated state

The evaluator may run forward, loss, or gradient checks during `reference` validation. A correct implementation must not require prior calls to "warm up" hidden state in order to compare correctly.

The evaluator now rebuilds a clean reference runtime after preflight. Candidate repos should be equally robust.

## Common Pitfalls

These are the failure modes that repeatedly caused false negatives, unstable bring-up, or misleading reports during real repo conversions.

### 1. Local vendored `transformers` shadows Hugging Face `transformers`

Many repos in this tree vendor a local `transformers/` package. If the evaluator or a repo loader imports Hugging Face APIs without temporarily removing the repo-local package from `sys.path` and `sys.modules`, you can get errors such as:

- missing `AutoTokenizer`
- missing `_LazyModule`
- partial imports that succeed once and fail later

This is especially easy to miss because the first import may appear to work, while a later lazy import fails.

Practical rule:

- keep the entire `from_pretrained(...)` call inside the Hugging Face import isolation context
- do not restore the vendored `transformers` package until tokenizer/model loading is finished

### 2. Deterministic runtime buffers can break parity even when weights match

Not every numerically relevant tensor is a persistent checkpoint weight.

The most important example is rotary-frequency state such as:

- `inv_freq`
- `original_inv_freq`

If these buffers are stale, corrupted, or materialized from bad state, you can see:

- exact parameter-tree parity
- exact batch-schema parity
- large forward-logit drift
- `NaN` in later stages

Practical rule:

- rebuild deterministic rotary buffers after model load on both sides
- do not assume a successful checkpoint load implies a correct runtime state

### 3. Generation often needs more inputs than forward comparison

`SFT` generation failures were repeatedly caused by dropping fields that are not always required for plain forward loss but are required by the model during generation.

For Gemma3, the important example is:

- `token_type_ids`

If generation only forwards `input_ids` and `attention_mask`, the repo can pass numeric forward checks and still fail behavioral generation.

Practical rule:

- generation should forward every model-relevant input from the collated batch, not just the minimum text-only subset

### 4. One runtime can poison the next if you reuse mutated state

Reference validation performs real forward and gradient work. Some repos mutate runtime caches, rope state, or internal buffers as part of normal execution.

If the evaluator keeps using the same runtime object after reference checks, later stages may compare:

- a warmed-up reference
- against a clean candidate

That is a bad experiment.

Practical rule:

- always recreate a fresh reference runtime after reference validation
- treat preflight and comparison as separate executions

### 5. JAX/XLA preallocation can look like a model OOM

When the candidate is a Flax/JAX runtime or a JAX-facing wrapper, XLA may preallocate large amounts of GPU memory before the actual comparison starts.

This can produce apparent model OOMs even when:

- the model fits
- the PyTorch reference fits
- the candidate backend fits on a separate GPU

Practical rule:

- set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- do not interpret every early OOM as a model-side memory bug

### 6. Wrapper candidates must separate evaluator device and backend device

If a Flax/JAX candidate is internally backed by a PyTorch model, there are really two device placements:

- `device`: where evaluator-side JAX arrays and callbacks live
- `backend_device`: where the internal PyTorch model lives

Putting both on the same GPU can cause:

- OOM
- unstable timings
- misleading numeric failures caused by device pressure

Practical rule:

- use separate GPUs for `device` and `backend_device` when the candidate is wrapper-based

### 7. `IGNORE_INDEX` handling must be identical in every numeric path

Several false failures came from masking bugs rather than real model differences.

The same `IGNORE_INDEX` convention must be applied consistently in:

- method loss
- log-prob extraction
- gradient formulas
- PPO reward / advantage formulas

If one path masks labels correctly and another gathers on raw `-100`, you can get:

- device asserts
- non-finite outputs
- gradient mismatch with otherwise correct logits

Practical rule:

- normalize masking logic once per formula family and reuse it everywhere

### 8. Forward path and gradient path must use the same effective model

For wrapper-style candidates, it is easy to accidentally compare:

- forward from one runtime mode
- gradients from another runtime mode

Examples:

- eval mode vs train mode mismatch
- different dtype handling
- different cache usage
- different hidden input reconstruction

Practical rule:

- treat forward and backward as two views of the same underlying model state
- if you add a shortcut gradient path, validate it against the forward path before trusting evaluator results

### 9. Reusing an output directory can hide whether a fix really worked

If a long-running process is interrupted and a previous `new_eval_summary.json` remains in place, it is easy to mistake stale results for current ones.

Practical rule:

- use a fresh output directory for each debugging run
- or delete the previous output directory before rerunning

### 10. Do not report skipped checks as success

A skipped or placeholder behavioral check is not a passing result.

Practical rule:

- if a check matters, implement it
- if it cannot run, treat that as evaluator work remaining, not as a green status

## Data Sources

The evaluator supports two ways to build evaluation batches.

### Repo-owned dataset path

If the repo already has a valid dataset setup, define the usual keys in `shared_config`, for example:

- `dataset`
- `dataset_dir`
- `template`
- `max_samples`

The evaluator will call the repo's own `data.loader.get_dataset(...)`.

### Inline `eval_examples`

If you want a deterministic comparison without relying on external dataset files, provide `eval_examples` directly in the evaluator config.

These examples must already be in the repo's aligned intermediate format. The evaluator will pass them through the repo's own `data.processor` path before collation.

Example:

```yaml
shared_config:
  model_name_or_path: some/model
  template: some_template
  eval_examples:
    - _prompt:
        - role: user
          content: "Say hello in one short sentence."
      _response:
        - role: assistant
          content: "Hello."
      _system: ""
      _tools: ""
      _images: null
      _videos: null
      _audios: null
```

## Config Format

Minimal config:

```yaml
training_method: dpo
reference_impl: torch_orig
candidate_impl: jax_repo
precision_profile: fp16_compare
runtime_profile: single_gpu
output_dir: ../outputs/my_run
shared_config:
  model_name_or_path: some/model
  template: some_template
  eval_examples: [...]
implementations:
  - name: torch_orig
    framework: pytorch
    repo_dir: ../../dpo/some_model/torch_orig
  - name: jax_repo
    framework: flax
    repo_dir: ../../dpo/some_model/jax_repo
```

Important fields:

- `training_method`: `sft`, `dpo`, or `ppo`
- `reference_impl`: implementation name used as the contract source
- `candidate_impl`: implementation name compared against the reference
- `shared_config`: base config merged into both repos before `get_train_args`
- `implementations[].overrides`: per-repo config overrides
- `implementations[].device`: evaluator-side device for that implementation
- `implementations[].backend_device`: optional backend runtime device for wrapper-style candidates
- `output_dir`: where reports are written

## Commands

The evaluator still exposes CLI entry points, but the primary execution path is now `pytest`.

The same config file works for both `pytest` and the CLI.

Validate config files only:

```bash
python -m lib.validate_configs
python -m lib.validate_configs configs/example_dpo.yaml
```

Run evaluator stages through `pytest` for one config:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_eval_pipeline.py   --eval-config configs/example_dpo.yaml
```

Run multiple configs in one `pytest` invocation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_eval_pipeline.py   --eval-config configs/sft_gemma3.yaml   --eval-config configs/dpo_qwen3_vl.yaml
```

Run every config under `configs/`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_eval_pipeline.py --all-eval-configs
```

Run configs selected by glob:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_eval_pipeline.py   --eval-config-glob 'configs/sft_*.yaml'
```

Apply the same override to every selected config:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_eval_pipeline.py   --eval-config configs/example_dpo.yaml   --eval-override batch_size=2   --eval-override shared_config.learning_rate=1e-5
```

Write `pytest` outputs to a dedicated relative directory:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_eval_pipeline.py   --eval-config configs/example_dpo.yaml   --eval-output-root .pytest_eval_outputs
```

Validate only the PyTorch reference contract:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m lib.torch_orig_preflight configs/example_dpo.yaml
```

Run the full evaluation without `pytest`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1 python -m lib.run_eval configs/example_dpo.yaml
```

Programmatic usage:

```python
from lib.config import load_eval_config
from lib.runner import run_evaluation

cfg = load_eval_config("configs/example_dpo.yaml")
cfg.implementation_map()["jax_repo"].backend_device = "cuda:4"
report = run_evaluation(cfg)
print(report.summary())
```

## Outputs

Each run writes two files under `output_dir`:

- `new_eval_report.json`
- `new_eval_summary.json`

The full report includes the compared artifacts for each completed stage.

Typical artifacts include:

- forward outputs
- method-specific loss artifacts
- gradients
- learning-rate schedules
- behavioral artifacts

## Failure Interpretation

Use the stage ordering to decide what to fix.

If `reference` fails:

- fix the PyTorch repo first
- do not trust any cross-framework conclusion

If `spec` fails:

- fix parameter export or batch construction
- do not start tuning thresholds

If `numeric` fails with finite tensors:

- inspect the exact diverging artifact
- the most common causes are wrong batch values, wrong runtime buffers, wrong masking, or wrong method formulas

If `numeric` fails with `nonfinite`:

- treat it as a real runtime problem
- common causes are invalid rotary buffers, bad dtype paths, or broken device placement

If `numeric` fails with `oom`:

- separate JAX and backend PyTorch devices
- reduce duplicate full-model residency
- prefer runtime-specific gradient shortcuts over generic heavyweight fallback paths

If `behavioral` fails:

- first check whether the failure is a real semantic mismatch or an input-forwarding bug
- confirm generation is receiving the full batch context it needs
- confirm short replay uses the same optimizer-step semantics on both sides

## Current Status

The rebuilt framework has been verified at the package level and through real repo bring-up. That includes:

- config validation through `lib.validate_configs`
- reference-only validation through `lib.torch_orig_preflight`
- full staged comparison through `lib.run_eval`
- explicit handling for runtime-state normalization after load
- real behavioral generation comparison for `SFT`
- real short loss-trajectory replay for evaluator-controlled `SFT` bring-up
