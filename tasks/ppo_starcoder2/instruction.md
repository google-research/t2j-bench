# PyTorch-to-JAX Conversion: PPO

## Goal

Reimplement the PyTorch PPO repo in `torch_orig/` as a **native JAX/Flax** repo in
`jax_repo/` with the same training semantics. Mirror `torch_orig`'s structure and public
API — same modules, same class and function names and signatures — deviating only where
JAX forces it. Reimplement the Starcoder2 model in `flax.linen`, the PPO trainer
(rollout, GAE, clipped policy/value loss) + optimizer/schedule in pure JAX/`optax`, keep
the data pipeline framework-neutral, and port the weights yourself.

`jax_repo/` MUST NOT import `torch`, `torchvision`, `torchaudio`, `functorch` or
`torch_xla` **anywhere** — no offline exception, not even to read the checkpoint. The
grading environment has no torch, and no `transformers` and no HF `datasets` either, so
whatever `torch_orig` imports from those your repo has to carry itself.

## Instance contract

### Model
- `model_name_or_path`: `bigcode/starcoder2-3b`
- `template`: `default`
- training method: `ppo`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/bigcode/starcoder2-3b` — `config.json`, the tokenizer files, and the weights.
Nothing is downloaded at train time or at grade time, so this directory is the only
copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: a single `model.safetensors` (483 tensors, 11.3 GiB), all FP32
- no `lm_head` tensor in the file: the output head is tied to the embedding table

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[1, 16] dtype=int64
  input_ids: shape=[1, 16] dtype=int64
  labels: shape=[1, 16] dtype=int64
```

### Compute and time budget
You work in a container with one visible H100 GPU, 24 CPU cores and
200 GiB of RAM, and have up to **2 hours** of wall-clock time to produce `jax_repo/` —
enough to write, run and debug the port. That budget is for building the port; the
end-to-end training the grader runs afterwards is separate and much shorter (see **How
you are graded**).

## API contract

Your repo is imported and driven through the names below, so these are fixed — including
the `data/` path, which is what has to rebuild the batch above. The rest of the layout
you take from `torch_orig`.

```
config.py        get_train_args(args=None)
                     -> (model_args, data_args, training_args, finetuning_args, *method_args)
model/loader.py  load_tokenizer(model_args)
                     -> {"tokenizer": ..., "processor": ...}
                 load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=False)
                     -> (module, params, frozen_prefixes)
                 convert_checkpoint(model_name_or_path, config, *, dtype=None)
                     -> the parameter pytree load_model hands to module.apply
train.py         run_ppo(config=None), the CLI entry point mirroring torch_orig
data/            template.get_template_and_fix_tokenizer
                 processor.UnsupervisedDatasetProcessor
                 collator.MultiModalDataCollatorForSeq2Seq
```

## Requirements

- **Convert the checkpoint yourself.** `model/loader.py` exposes `convert_checkpoint`,
  and `load_model` gets its weights from it. Read the safetensors with
  **`safetensors.flax.load_file`** (or `safetensors.safe_open(..., framework="flax")`):
  `safetensors.torch` and `torch.load` are torch imports even though the file they read
  is a PyTorch checkpoint, and the flax backend hands back `jax.Array`s directly (this
  checkpoint carries no bf16, so `safetensors.numpy` would read it too, but into numpy
  arrays you would then have to convert). Follow `model.safetensors.index.json` when
  the checkpoint is sharded. Nothing is downloaded; the directory named above is the
  only copy.
- **Naming is yours; correspondence is not.** Parameter paths are your own to choose,
  but every leaf must map onto exactly one parameter of the torch checkpoint and carry
  that parameter's value — so keep the module names recognizable
  (`model.layers.N.self_attn.q_proj`, …). A rank-2 flax kernel may be the transpose of
  torch's `[out, in]` weight.
- **Synthesize what the checkpoint does not carry** — a tied output head (the contract
  above says whether this checkpoint carries its own), any buffer the file omits such as
  the rotary `inv_freq`, and the value head, which no checkpoint has — each with the
  same deterministic initialization `torch_orig` gives it.
- **Keep the objective and the trainer semantics**: rollout generation, reward scoring,
  whitened advantages via GAE, the clipped surrogate policy loss, the clipped value loss
  and the value head on top of the frozen backbone, with AdamW + schedule + grad-clip as
  the source configures them. Do not fall back to plain cross-entropy.
- **Parameters are external.** `load_model` returns a module plus a pytree, never a
  stateful model; the forward is `module.apply({"params": p}, …)` and must trace under
  `jax.jit` with no host callbacks. Optimizer and schedule are `optax`; the step is
  `jax.value_and_grad` + `optax.apply_updates`.
- **Do not hard-code a compute dtype.** It comes from the config, and the module has to
  compute correctly in whatever dtype the parameters it is handed carry.
- **Match the reference's config defaults.** The run's hyperparameters live in
  `config.py`, and with no overrides `get_train_args` has to resolve the same ones
  `torch_orig` declares — learning rate, scheduler and warmup, batch and
  gradient-accumulation sizes, seed, the LoRA settings, and any knobs the training
  objective itself carries — so a default that drifts from the reference skews the
  whole run.
- **The data pipeline is framework-neutral** — numpy arrays (`int64` for token, label
  and mask entries), never `torch.Tensor`. Reuse the checkpoint's own tokenizer files
  through a torch-free tokenizer library rather than `transformers`.
- **LoRA, when the config selects it**, lives in `model/loader.py` (`torch_orig` applies
  it via PEFT): add the adapter pair to the pytree as `lora_a`/`lora_b` leaves and
  report the frozen base weights through `frozen_prefixes`, instead of wrapping the
  module.

## How you are graded

`jax_repo/` is graded in an offline, torch-blocked container against reference artifacts
frozen from a run of `torch_orig` on a fixed batch. Grading is a **continuous fidelity
score**, not pass/fail: the closer your port comes to reproducing the reference, the
higher it scores (even a faithful fp32 port lands just short of the top — bit-for-bit is
not expected). Structure (`spec`) is the one pass/fail gate; `numeric` and `behavioral`
then score how closely your port matches. Three dimensions:

- **spec** — independence and structure. No torch anywhere in the tree (a static scan of
  the whole repo plus a runtime import check); a `flax.linen` module over `jax.Array`
  parameters with a pure, deterministic forward that traces to real XLA ops with no host
  callbacks; a parameter pytree whose leaves each correspond to a checkpoint tensor
  (shape-for-shape, up to a rank-2 transpose); and your own data pipeline rebuilding the
  frozen batch exactly — its schema plus its exact integer content (token ids, masks,
  label spans), so a weights-only port with no runnable pipeline fails here. Your repo's
  public surface is checked statically against `torch_orig` too: the required modules
  and layout (no leftover empty / namespace-package directories), the class/function
  names and their kinds, the entry points accepting the calls the harness makes, the
  shared class methods, and load-bearing constants at the reference's values (e.g.
  `IGNORE_INDEX`). Separately, the training-config defaults your `config.py`
  resolves under no overrides have to match the reference's.
- **numeric** — how closely your forward pass, losses and LR schedule match the
  reference's, and how faithfully your converted weights carry the checkpoint's values;
  plus gradient authenticity (the grader takes the analytic gradient through your own
  model and cross-checks it against a finite-difference estimate), within the precision
  profile named above.
- **behavioral** — your training loop runs for real: a short replay stays finite and
  does not run away (the RL objective is not monotone), and your `run_ppo` entry point
  runs end to end under its `config.py` defaults, its held-out validation loss and final
  training loss compared to the reference's (so it has to train to completion within a
  runtime budget comparable to the reference's). Repeated runs are deterministic, and
  the model holds the basic properties of a causal LM (causal masking, padding
  invariance, a normalized softmax, no leakage between rows of a batch).

This instance additionally compares individual functions and classes against their torch
counterparts one at a time, so port them faithfully rather than folding them into a
differently-shaped equivalent.
