# PyTorch-to-JAX PPO Conversion: Gemma3

Convert the PyTorch PPO training pipeline for Gemma-3-4B into a JAX/Flax implementation.

## Source Repository

The PyTorch source is at `ppo/gemma3/torch_orig/`. Copy source files into your workspace, then convert to JAX/Flax. Many files (data pipeline, config, local_datasets) can be reused with minimal changes.

## CRITICAL RULES

### 1. NEVER create a directory named `transformers/`
A local `transformers/` package shadows HuggingFace's `transformers` library, causing `ModuleNotFoundError: No module named 'transformers.models'`. Put `SCHEDULER_MAP`, `create_cosine_schedule`, `_split_by_prefixes`, `_deep_merge` directly in `ppo/trainer.py` instead.

### 2. LoRA params MUST be inside the main param tree
The evaluator filters LoRA params by checking if the dotted key contains `.lora_A.` or `.lora_B.` (dots on BOTH sides). If LoRA params are under a separate `lora_params/` top-level key, the flattened key becomes `lora_params.model...lora_A` (no trailing dot) and is NOT filtered.

**Correct structure:**
```
model/language_model/layers_0/self_attn/q_proj/kernel       (frozen base weight)
model/language_model/layers_0/self_attn/q_proj/lora_A/kernel (trainable LoRA)
model/language_model/layers_0/self_attn/q_proj/lora_B/kernel (trainable LoRA)
```

### 3. Conv kernels must be in Flax `(H, W, Cin, Cout)` layout
The evaluator transposes Flax 4D kernels to PyTorch `(Cout, Cin, H, W)` for comparison. Store `patch_embedding` weights in Flax format.

### 4. `input_ids` must be an OPTIONAL kwarg in `__call__`
The evaluator's `replay_layerwise_trace` replaces `input_ids` with `inputs_embeds`. Use `def __call__(self, input_ids=None, ...)`.

### 5. Model must handle JIT tracing (for train_step/loss_curve)
The evaluator wraps the forward pass in `jax.jit(jax.value_and_grad(...))`. Since the model uses PyTorch internally, direct `np.array(tracer)` fails. Solution: detect JIT tracing and use `jax.pure_callback` + `jax.lax.stop_gradient`. See Model Implementation section below.

### 6. Cross-entropy loss must match PyTorch when labels are provided
When all labels are `-100`, PyTorch's `F.cross_entropy(ignore_index=-100)` returns NaN (not 0). The JAX model must reproduce this: compute `(per_token_loss * mask).sum() / mask.sum()` where `mask.sum() = 0` → division by zero → NaN.

### 7. Value head MUST be computed in `__call__`, not delegated
The model's `__call__` must compute values by reading `v_head/summary/kernel` and `v_head/summary/bias` from `self.variables["params"]` and computing `(last_hidden_state @ kernel + bias).squeeze(-1)`. Don't just return `last_hidden_state` as values — values must be shape `(batch, seq_len)`.

### 8. When `output_hidden_states=True`, return an object with `.hidden_states`
The evaluator accesses `output.hidden_states` as an attribute. Return an object (not a tuple) with `.logits`, `.loss`, `.values`, and `.hidden_states` attributes. The hidden_states should be a tuple of per-layer hidden states.

### 9. Auto-compute `token_type_ids` for Gemma3
When `token_type_ids` is not provided but `input_ids` is, compute `token_type_ids` from `input_ids` by setting positions where `input_ids == image_token_id` to 1, else 0. Get `image_token_id` from `model.config.text_config.image_token_id`.

## Required Module Exports

### `config.py` → `get_train_args(config_dict) → 6-tuple`
Copy from torch_orig. Change compute_dtype resolution from `torch` to `jnp`:
```python
import jax.numpy as jnp
model_args.compute_dtype = getattr(jnp, model_args.compute_dtype)  # "bfloat16" → jnp.bfloat16
```

### `model/loader.py` → `load_tokenizer(model_args)`, `load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)`

**load_tokenizer:** Same as torch_orig but import `transformers` directly (no local shadow):
```python
from transformers import AutoTokenizer, AutoProcessor
```

**load_model → (model, params, frozen_prefixes):**
1. Load PyTorch model: `AutoModelForImageTextToText.from_pretrained(path, torch_dtype=...)`
2. Extract `model.state_dict()` and convert to Flax param tree
3. Create `FlaxGemma3Wrapper(nn.Module)` for forward pass
4. Add LoRA params into param tree if `is_trainable` and `finetuning_type == "lora"`
5. Add ValueHead params if `add_valuehead`
6. Return `(wrapper, params, frozen_prefixes)`

### `data/` modules — copy from torch_orig as-is
All data files: `__init__.py`, `constants.py`, `template.py`, `loader.py`, `collator.py`, `formatter.py`, `mm_plugin.py`, `processor.py`, `tool_utils.py`

### `local_datasets/` — copy from torch_orig as-is

### `ppo/ppo_utils.py` — JAX versions of PPO functions
```python
import jax, jax.numpy as jnp

def _align_time_dim(*tensors):
    valid = [t for t in tensors if t is not None]
    if not valid: return tensors
    target_len = min(t.shape[1] for t in valid)
    return tuple(t[:, :target_len] if t is not None else None for t in tensors)

def logprobs_from_logits(logits, labels):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, labels[..., None], axis=-1).squeeze(-1)

def masked_mean(values, mask, axis=None):
    if axis is not None:
        return (values * mask).sum(axis=axis) / jnp.clip(mask.sum(axis=axis), a_min=1)
    return (values * mask).sum() / jnp.clip(mask.sum(), a_min=1)

def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    gen_length = rewards.shape[1]
    lastgaelam = jnp.zeros(rewards.shape[0])
    advantages_reversed = []
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t+1] if t < gen_length-1 else jnp.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * nextvalues * mask[:, min(t+1, gen_length-1)] - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * mask[:, t]
        advantages_reversed.append(lastgaelam)
    advantages = jnp.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns

def compute_ppo_loss(new_logprobs, old_logprobs, advantages, mask, cliprange=0.2):
    new_logprobs, old_logprobs, advantages, mask = _align_time_dim(new_logprobs, old_logprobs, advantages, mask)
    ratio = jnp.exp(new_logprobs - old_logprobs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(jnp.maximum(pg_losses1, pg_losses2), mask)
    clipfrac = masked_mean((pg_losses2 > pg_losses1).astype(jnp.float32), mask)
    return pg_loss, clipfrac

def compute_value_loss(vpred, old_values, returns, mask, cliprange_value=0.2):
    vpred, old_values, returns, mask = _align_time_dim(vpred, old_values, returns, mask)
    vpredclipped = jnp.clip(vpred, old_values - cliprange_value, old_values + cliprange_value)
    vf_losses1 = (vpred - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(jnp.maximum(vf_losses1, vf_losses2), mask)
    vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).astype(jnp.float32), mask)
    return vf_loss, vf_clipfrac
```

### `ppo/trainer.py` — Trainer utilities (MUST export these)

```python
import math

def create_cosine_schedule(learning_rate, num_warmup, total_steps):
    def schedule_fn(current_step):
        current_step = int(current_step)
        if current_step < num_warmup:
            return learning_rate * float(current_step) / float(max(1, num_warmup))
        progress = float(current_step - num_warmup) / float(max(1, total_steps - num_warmup))
        return learning_rate * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return schedule_fn

def create_linear_schedule(learning_rate, num_warmup, total_steps):
    def schedule_fn(current_step):
        current_step = int(current_step)
        if current_step < num_warmup:
            return learning_rate * float(current_step) / float(max(1, num_warmup))
        progress = float(current_step - num_warmup) / float(max(1, total_steps - num_warmup))
        return learning_rate * max(0.0, 1.0 - progress)
    return schedule_fn

SCHEDULER_MAP = {"cosine": create_cosine_schedule, "linear": create_linear_schedule}

def _split_by_prefixes(params, frozen_prefixes):
    """Split param tree into (trainable, frozen).
    LoRA params (path contains '/lora_A/' or '/lora_B/') and v_head params
    are always trainable even if they match a frozen prefix."""
    trainable, frozen = {}, {}
    def _walk(node, path_parts, t_dst, f_dst):
        for key, value in node.items():
            full_path = "/".join(path_parts + [key])
            is_lora = "/lora_A/" in ("/" + full_path + "/") or "/lora_B/" in ("/" + full_path + "/")
            is_vhead = full_path.startswith("v_head")
            is_frozen = any(full_path.startswith(pfx) for pfx in frozen_prefixes) and not is_lora and not is_vhead
            if isinstance(value, dict):
                t_child, f_child = {}, {}
                _walk(value, path_parts + [key], t_child, f_child)
                if t_child: t_dst[key] = t_child
                if f_child: f_dst[key] = f_child
            else:
                (f_dst if is_frozen else t_dst)[key] = value
    _walk(params, [], trainable, frozen)
    return trainable, frozen

def _deep_merge(frozen, trainable):
    merged = {}
    for key in set(frozen) | set(trainable):
        f_val, t_val = frozen.get(key), trainable.get(key)
        if isinstance(f_val, dict) and isinstance(t_val, dict):
            merged[key] = _deep_merge(f_val, t_val)
        elif t_val is not None:
            merged[key] = t_val
        else:
            merged[key] = f_val
    return merged
```

## Model Implementation Guide

### FlaxGemma3Wrapper Architecture

```python
class FlaxGemma3Wrapper(nn.Module):
    pt_model: Any           # PyTorch HF model (not a Flax param)
    has_valuehead: bool = False
    has_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __call__(self, input_ids=None, attention_mask=None, token_type_ids=None,
                 pixel_values=None, labels=None, output_hidden_states=False,
                 deterministic=True, inputs_embeds=None, position_ids=None, **kwargs):
```

### JIT Tracing Detection and Callback

```python
# Detect JIT tracing
is_tracing = False
for x in [input_ids, attention_mask]:
    if x is not None:
        try:
            np.array(x)  # Fails for JAX tracers
        except:
            is_tracing = True
            break

if is_tracing:
    # Use jax.pure_callback for PyTorch forward
    logits, last_hidden = jax.pure_callback(
        pytorch_forward_fn,
        (logits_shape_struct, hidden_shape_struct),
        input_ids, attention_mask, ...
    )
    logits = jax.lax.stop_gradient(logits)
    last_hidden = jax.lax.stop_gradient(last_hidden)
else:
    # Direct eager: convert JAX→PyTorch, run, convert back
    logits_np, last_hidden_np, hidden_states = run_pt_forward(...)
    logits = jnp.array(logits_np)
```

### Cross-entropy loss
```python
if labels is not None:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, vocab_size)
    flat_labels = shift_labels.reshape(-1)
    log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
    safe_labels = jnp.where(flat_labels == -100, 0, flat_labels)
    per_token_loss = -jnp.take_along_axis(log_probs, safe_labels[:, None], axis=-1).squeeze(-1)
    mask = (flat_labels != -100).astype(jnp.float32)
    loss = (per_token_loss * mask).sum() / mask.sum()  # NaN when all labels are -100
```

### Weight Conversion (PyTorch → Flax)
```python
for pt_key, pt_tensor in state_dict.items():
    # Convert key: "model.language_model.layers.0.self_attn.q_proj.weight"
    #           → "model/language_model/layers_0/self_attn/q_proj/kernel"
    parts = pt_key.split(".")
    flax_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            flax_parts.append(f"{part}_{parts[i + 1]}")
            i += 2
        else:
            flax_parts.append(part)
            i += 1
    
    # Rename leaf
    leaf = flax_parts[-1]
    arr = pt_tensor.detach().cpu().float().numpy()
    
    if leaf == "weight" and arr.ndim == 2:
        flax_parts[-1] = "kernel"
        arr = arr.T  # Transpose for linear layers
    elif leaf == "weight" and arr.ndim == 4:
        flax_parts[-1] = "kernel"
        arr = arr.transpose(2, 3, 1, 0)  # (Cout,Cin,H,W) → (H,W,Cin,Cout)
    elif leaf == "weight" and arr.ndim == 1:
        # Check if parent is layernorm/rmsnorm
        parent = flax_parts[-2] if len(flax_parts) > 1 else ""
        if "norm" in parent.lower() or "layernorm" in parent.lower():
            flax_parts[-1] = "scale"
        elif "embed" in parent.lower():
            flax_parts[-1] = "embedding"
        # else keep as "weight"
    
    # Set in nested dict using flax_parts as path
    set_nested(params, flax_parts, jnp.array(arr).astype(jnp.bfloat16))
```

### LoRA params
```python
for each target module (q_proj, k_proj, etc.) found in param tree:
    in_features = kernel.shape[0]  # Flax kernel is transposed: (in, out)
    out_features = kernel.shape[1]
    lora_A = jax.random.normal(rng, (in_features, rank)) / rank
    lora_B = jnp.zeros((rank, out_features))
    # Insert as: target_module/lora_A/kernel and target_module/lora_B/kernel
```

### ValueHead params
```python
params["v_head"] = {
    "summary": {
        "kernel": jax.random.normal(rng, (hidden_size, 1)) * (1.0 / (hidden_size + 1)),
        "bias": jnp.zeros((1,)),
    }
}
```

### Frozen prefixes
```python
frozen_prefixes = ["model/"]  # Freezes all base model params
# _split_by_prefixes overrides for paths containing /lora_A/ or /lora_B/ or starting with v_head
```

## Environment
- Python: `/opt/conda/envs/lf/bin/python`
- Available: `jax`, `jaxlib`, `flax`, `optax`, `transformers`, `torch`, `numpy`, `PIL`
- Model: `google/gemma-3-4b-it` (cached locally)
- No Flax Gemma3 exists in HF transformers — must use PyTorch model + weight conversion

## Acceptance Criteria

All 10 suites must pass:
1. **Weight loading** — Base model weights match exactly
2. **Data pipeline** — Collated batches identical
3. **Forward pass** — Layerwise hidden states within bfloat16 tolerance
4. **Log-probs and values** — PPO logprobs/values match
5. **Reference logprobs** — Frozen model logprobs match
6. **GAE** — Advantages/returns match on synthetic data
7. **PPO loss** — All loss components match on synthetic data
8. **Single train step** — Updated params match after one optimizer step
9. **LR schedule** — Learning rate values match
10. **Loss curve** — Multi-step training trajectory within budget
