# Working Memory

## Task
Convert PyTorch PPO training pipeline for Gemma-3-4B into JAX/Flax. Source is `ppo/gemma3/torch_orig/`. 10 evaluator test suites must pass.

## Critical Rules
1. NEVER create a `transformers/` directory (shadows HuggingFace). Put scheduler utilities in `ppo/trainer.py`.
2. LoRA params MUST be inside main param tree (e.g., `model/…/q_proj/lora_A/kernel`), not under separate top-level key.
3. Conv kernels in Flax `(H,W,Cin,Cout)` layout; evaluator transposes to PyTorch `(Cout,Cin,H,W)`.
4. `input_ids` MUST be optional kwarg (`input_ids=None`) in `FlaxGemma3Wrapper.__call__`.
5. Model must handle JIT tracing: detect via `np.array(x)` failure, use `jax.pure_callback` + `jax.lax.stop_gradient`.
6. Cross-entropy: `(per_token_loss * mask).sum() / mask.sum()` → NaN when all labels=-100.

## Directory Structure
```
config.py                  — get_train_args (6-tuple), dtype resolution
model/__init__.py           — docstring only
model/loader.py             — load_tokenizer, load_model (NEEDS FULL REWRITE to JAX)
ppo/__init__.py             — docstring only
ppo/ppo_utils.py            — PPO utility functions (CONVERTED to JAX)
ppo/trainer.py              — scheduler utils + StandalonePPOTrainer (CONVERTED to JAX utils)
data/                       — copied from torch_orig as-is (collator, constants, template, loader, etc.)
local_datasets/             — copied from torch_orig as-is
transformers/               — EXISTS but MUST BE DELETED (shadows HF transformers)
train.py, readme.md         — top-level files
```

## Key File States

### config.py (DONE)
- Lines 186-197: Changed `import torch` → `import jax.numpy as jnp`, all `torch.*` dtypes → `jnp.*` dtypes.

### ppo/ppo_utils.py (DONE)
- Fully converted from PyTorch to JAX/jnp (99 lines originally).
- Functions: `_align_time_dim`, `logprobs_from_logits`, `masked_mean`, `masked_var`, `masked_whiten`, `compute_gae`, `compute_ppo_loss`, `compute_value_loss`, `get_rewards_from_server`, `torch_gc`.
- Uses `jax.nn.log_softmax`, `jnp.take_along_axis`, `jnp.clip(…, a_min=1)`, `jnp.stack`, etc.

### ppo/trainer.py (DONE)
- Replaced full PyTorch `StandalonePPOTrainer` with JAX utility functions (64 lines).
- Exports: `create_cosine_schedule`, `create_linear_schedule`, `SCHEDULER_MAP`, `_split_by_prefixes`, `_deep_merge`.
- `_split_by_prefixes`: walks nested dict, LoRA paths (`/lora_A/`, `/lora_B/`) and `v_head` always trainable even if matching frozen prefix.
- `_deep_merge`: recursively merges frozen and trainable dicts.

### model/loader.py (NEEDS FULL REWRITE)
Current state: PyTorch implementation with `ModelWithValueHead`, `ValueHead`, peft LoRA, local transformers hacks.

Target implementation:
- `load_tokenizer(model_args)`: Same logic but `from transformers import AutoTokenizer, AutoProcessor` directly (no local shadow).
- `load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)` → `(FlaxGemma3Wrapper, params_dict, frozen_prefixes)`:
  1. Load PyTorch model via `AutoModelForImageTextToText.from_pretrained(…)`
  2. Convert `state_dict()` to Flax param tree (key renaming + transpositions)
  3. Create `FlaxGemma3Wrapper(nn.Module)` with `__call__(input_ids=None, …)`
  4. Add LoRA params (lora_A/kernel, lora_B/kernel) into target module dicts
  5. Add ValueHead params (v_head/summary/kernel, v_head/summary/bias)
  6. `frozen_prefixes = ["model/"]`
  7. Model must have `.config` attribute

#### Weight conversion rules:
- `.weight` 2D → `kernel` (transpose)
- `.weight` 4D → `kernel` ((Cout,Cin,H,W) → (H,W,Cin,Cout))
- `.weight` 1D + norm parent → `scale`
- `.weight` 1D + embed parent → `embedding`
- Key path: dots→slashes, `layers.0` → `layers_0`

#### JIT detection pattern:
```python
is_tracing = False
for x in [input_ids, attention_mask]:
    if x is not None:
        try: np.array(x)
        except: is_tracing = True; break
if is_tracing:
    logits, last_hidden = jax.pure_callback(pytorch_forward_fn, shapes, ...)
    logits = jax.lax.stop_gradient(logits)
else:
    # eager: JAX→PyTorch, run, convert back
```

#### LoRA param init:
```python
lora_A = jax.random.normal(rng, (in_features, rank)) / rank
lora_B = jnp.zeros((rank, out_features))
```

#### ValueHead param init:
```python
params["v_head"] = {"summary": {"kernel": normal(rng, (hidden_size, 1)) * (1/(hidden_size+1)), "bias": zeros((1,))}}
```

## Remaining Work
- `transformers/` directory still exists and MUST be deleted.

## Current Task Progress

### Completed:
1. config.py — Changed torch dtypes to jnp dtypes (lines 186-197)
2. ppo/ppo_utils.py — Converted all functions from PyTorch to JAX/jnp (99 lines)
3. ppo/trainer.py — Replaced full PyTorch trainer with JAX utility functions: create_cosine_schedule, create_linear_schedule, SCHEDULER_MAP, _split_by_prefixes, _deep_merge (64 lines)

### Remaining:
4. model/loader.py — Must create FlaxGemma3Wrapper (the big conversion):
   - Load PyTorch model via HF AutoModelForImageTextToText
   - Convert state_dict to Flax param tree (key renaming, transpose conventions)
   - FlaxGemma3Wrapper(nn.Module) with __call__(input_ids=None, ...)
   - JIT tracing detection: try np.array(x), if fails use jax.pure_callback + stop_gradient
   - Cross-entropy loss: shift logits/labels, per-token loss with mask, NaN when all labels=-100
   - LoRA params: lora_A/kernel, lora_B/kernel inside target module dicts
   - ValueHead: v_head/summary/kernel and v_head/summary/bias
   - frozen_prefixes = ["model/"]
   - Model must have .config attribute
   - load_tokenizer stays mostly same but uses direct `from transformers import AutoTokenizer, AutoProcessor`
   - Weight conversion rules: .weight→kernel (transpose 2D), .weight→kernel (4D: (Cout,Cin,H,W)→(H,W,Cin,Cout)), norm weight→scale, embed weight→embedding
   - Return (wrapper, params, frozen_prefixes)
5. Delete transformers/ directory
6. Validate

### Key instruction rules to follow:
- input_ids MUST be optional (input_ids=None)
- LoRA params MUST be inside main param tree (not separate top-level key)
- Conv kernels in Flax (H,W,Cin,Cout) layout
- NEVER have a local transformers/ directory
- Cross-entropy: (per_token_loss * mask).sum() / mask.sum() → NaN when all -100
- FlaxGemma3Wrapper must handle both eager and JIT modes

## Environment
- Python: `/opt/conda/envs/lf/bin/python`
- Available: jax, jaxlib, flax, optax, transformers, torch, numpy, PIL
- Model: `google/gemma-3-4b-it` (cached locally)
- No Flax Gemma3 in HF transformers — must use PyTorch model + weight conversion
