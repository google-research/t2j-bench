# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone SFT ``Trainer`` (torch-free) — mirror of ``torch_orig/sft/trainer.py``.

Same ``Trainer`` methods, same ``TrainOutput``, same call order out of
``train.py::run_sft``. It replaces the dependency on ``transformers.Trainer`` with a small
self-contained loop, the SFT analogue of the ``dpo``/``ppo`` trainer packages. JAX
adjustments over the torch original:

1. **The model is a ``(module, params)`` pair.** Flax modules are stateless, so
   ``load_model`` returns the module and its params separately. ``self.model`` keeps the
   pair (torch's attribute survives) and ``self.module`` / ``self.params`` name the halves;
   the loop threads params through functionally rather than mutating them in place.
   ``model.train()`` / ``model.zero_grad()`` have no analogue and are dropped: dropout is
   off because the module is applied without an RNG, and optax grads are never accumulated
   into the params.
2. **optax replaces ``torch.optim`` + ``LambdaLR``.** ``create_optimizer`` returns an
   ``optax.GradientTransformation`` with the LR schedule already baked in, so it needs the
   horizon up front and takes ``num_training_steps`` (torch's takes none);
   ``create_scheduler`` returns the ``optax`` schedule callable, queried as
   ``scheduler(step)`` where torch calls ``scheduler.get_last_lr()[0]``, and accepts
   ``optimizer`` only for signature parity. torch's ``clip_grad_norm_`` before
   ``optimizer.step()`` becomes ``optax.clip_by_global_norm`` chained ahead of AdamW.
3. **No ``DataLoader``.** ``get_train_dataloader`` / ``get_eval_dataloader`` return fresh
   generators that slice the dataset into ``per_device_train_batch_size`` chunks and run
   the collator. Order is sequential, not ``RandomSampler``'s shuffle, so a run is
   reproducible against the frozen goldens; ``len(dataloader)`` is unavailable, so the step
   count comes from ``len(self.train_dataset)``.
4. **Flax models return logits, not a loss**, so torch's ``outputs.loss`` is computed here
   by ``_causal_lm_loss``. Gradients are taken over the whole batch in one jitted step, so
   ``gradient_accumulation_steps`` is not emulated (these configs all leave it at 1).
5. **``save_model`` writes a flat ``safetensors`` file** of the params (``/``-joined
   pytree paths) instead of calling ``save_pretrained``.
"""
import functools
import json
import logging
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Full-f32 matmuls (disable TF32) so the training loss curve matches torch's f32
# reference. JAX defaults to TF32 (~bf16 mantissa) for f32 dots on GPU; that ~0.03
# per-logit error is within the forward tolerance but compounds across optimizer
# steps and diverges the deterministic train_trajectory golden.
jax.config.update("jax_default_matmul_precision", "highest")

logger = logging.getLogger(__name__)


def _decay_mask(params):
    """optax label tree: ``True`` for params that receive weight decay (exclude bias /
    LayerNorm scale+bias), matching torch's ``no_decay`` name filter."""
    no_decay = ("bias", "layer_norm", "layernorm", "ln_", "scale")

    def leaf(path, _):
        name = "/".join(str(getattr(k, "key", k)).lower() for k in path)
        return not any(nd in name for nd in no_decay)

    return jax.tree_util.tree_map_with_path(leaf, params)


def _causal_lm_loss(logits, labels, ignore_index: int = -100):
    """Shifted cross-entropy loss. Private (leading underscore): a flax module returns
    logits, not a loss, so computing it is the trainer's job — but torch_orig has no such
    module-level symbol, so the public interface stays ``Trainer`` + ``TrainOutput``."""
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:]
    mask = shift_labels != ignore_index
    safe = jnp.where(mask, shift_labels, 0)
    logp = jax.nn.log_softmax(shift_logits, axis=-1)
    gathered = jnp.take_along_axis(logp, safe[..., None], axis=-1).squeeze(-1)
    return -(gathered * mask).sum() / jnp.maximum(mask.sum(), 1)


class Trainer:
    """Minimal trainer for supervised fine-tuning."""

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        **kwargs,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # torch moves the model onto CUDA here because model/loader.py leaves it on the
        # CPU. jax arrays are already on the default device (the GPU when one is visible),
        # so the only thing left to unpack is the (module, params) pair load_model returns.
        self.module, self.params = model if isinstance(model, tuple) else (model, None)
        # torch reads it off ``model.config``; here the caller passes the label index that
        # data/constants.py defines, so the trainer needs no import from the data package.
        self.ignore_index = kwargs.get("ignore_index", -100)
        self.global_step = 0
        self.epoch = 0.0

    def get_train_dataloader(self):
        return self._iter_batches(self.train_dataset, getattr(self.args, "per_device_train_batch_size", 1))

    def get_eval_dataloader(self, dataset=None):
        dataset = dataset if dataset is not None else self.eval_dataset
        if dataset is None:
            return None
        return self._iter_batches(dataset, getattr(self.args, "per_device_eval_batch_size", 1))

    def _iter_batches(self, dataset, batch_size):
        """Sequential batches of `batch_size`, collated and converted to jnp arrays.

        The JAX stand-in for ``DataLoader``: a fresh generator per call, so ``train()`` can
        re-iterate it once per epoch the way torch re-iterates its dataloader.

        Every batch in the split is right-padded to a single fixed sequence length (the
        split's longest batch). The collator pads only to the longest sample *within* a
        batch, so at ``batch_size=1`` each step would otherwise carry a distinct shape and
        the jitted train/eval step would recompile on every step — XLA compilation, not the
        GPU, then dominates the run (minutes become hours). A fixed shape compiles the step
        exactly once. The extra positions are numerically inert: they carry
        ``attention_mask=0`` and ``labels=IGNORE_INDEX``, so with right-padding and a causal
        mask the real tokens never attend to them and never enter the loss average — the
        losses are identical to pad-to-longest, only the wall-clock differs.
        """
        batch_size = int(batch_size or 1) or 1
        n = len(dataset)
        keys = ("input_ids", "attention_mask", "labels")
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        pad_values = {"input_ids": pad_id, "attention_mask": 0, "labels": self.ignore_index}

        # Collate every batch up front so the whole split shares one padded width.
        collated = []
        max_len = 1
        for i in range(0, n, batch_size):
            feats = [dataset[j] for j in range(i, min(i + batch_size, n))]
            raw = self.data_collator(feats)
            raw = raw.data if hasattr(raw, "data") else raw
            batch = {k: np.asarray(raw[k]) for k in keys if k in raw}
            if "input_ids" in batch:
                max_len = max(max_len, batch["input_ids"].shape[-1])
            collated.append(batch)

        def gen():
            for batch in collated:
                out = {}
                for k, v in batch.items():
                    pad_width = max_len - v.shape[-1]
                    if pad_width > 0:
                        widths = [(0, 0)] * (v.ndim - 1) + [(0, pad_width)]
                        v = np.pad(v, widths, mode="constant", constant_values=pad_values.get(k, 0))
                    out[k] = jnp.asarray(v)
                yield out

        return gen()

    def create_optimizer(self, num_training_steps):
        # optax AdamW with the LR schedule baked in (so it needs the horizon up front) +
        # gradient clipping. HF ``Trainer`` clips to ``max_grad_norm`` (default 1.0) before
        # each step; mirror it with ``optax.clip_by_global_norm`` chained ahead of AdamW so
        # the loss curve matches the reference and stays stable for models with large
        # gradients (unclipped bf16 steps can diverge to NaN).
        schedule = self.create_scheduler(None, num_training_steps)
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 1.0) or 0.0)
        adamw = optax.adamw(
            schedule,
            weight_decay=float(getattr(self.args, "weight_decay", 0.0) or 0.0),
            mask=_decay_mask,
        )
        if max_grad_norm > 0:
            return optax.chain(optax.clip_by_global_norm(max_grad_norm), adamw)
        return adamw

    def create_scheduler(self, optimizer, num_training_steps):
        del optimizer  # optax bakes the schedule into the optimizer; see adjustment 2.
        args = self.args
        base_lr = float(getattr(args, "learning_rate", 1e-6))
        warmup = int(getattr(args, "warmup_steps", 0) or 0)
        if warmup == 0:
            warmup = int(num_training_steps * float(getattr(args, "warmup_ratio", 0.0) or 0.0))
        if str(getattr(args, "lr_scheduler_type", "cosine")) == "linear":
            decay = optax.linear_schedule(base_lr, 0.0, max(1, num_training_steps - warmup))
        else:
            decay = optax.cosine_decay_schedule(base_lr, max(1, num_training_steps - warmup))
        if warmup > 0:
            warmup_fn = optax.linear_schedule(0.0, base_lr, warmup)
            return optax.join_schedules([warmup_fn, decay], [warmup])
        return decay

    def train(self, resume_from_checkpoint=None):
        """Run training loop."""
        args = self.args

        num_epochs = getattr(args, "num_train_epochs", 3.0)
        batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1) or 1
        logging_steps = getattr(args, "logging_steps", 10)
        save_steps = getattr(args, "save_steps", 500)
        output_dir = getattr(args, "output_dir", "output")

        epochs = int(math.ceil(float(num_epochs)))
        num_steps_per_epoch = max(1, math.ceil(len(self.train_dataset) / batch_size))
        total_steps = num_steps_per_epoch * epochs

        # LoRA if — and only if — ``load_model`` put adapters in the tree
        # (finetuning_type == "lora"): the base stays frozen and only the adapters get an
        # optimizer state, the mirror of torch's ``get_peft_model``. ``split`` finds nothing
        # under full finetuning, so every line below is then the path it always was.
        from model.loader import combine, has_adapters, split

        use_lora = has_adapters(self.params)
        base, state = split(self.params) if use_lora else (None, self.params)

        optimizer = self.create_optimizer(total_steps)
        scheduler = self.create_scheduler(optimizer, total_steps)
        # torch/PEFT keeps LoRA adapters + Adam moments in fp32 over a bf16 base; upcast the
        # trainable set so Adam's beta2 does not underflow in bf16 (mirrors ppo/trainer.py).
        if use_lora:
            state = jax.tree_util.tree_map(lambda x: jnp.asarray(x).astype(jnp.float32), state)
        opt_state = optimizer.init(state)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {num_epochs}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Total optimization steps = {total_steps}")

        # JIT the step with params + opt_state donated so XLA reuses their buffers (no
        # transient doubling of the Adam state) — this is what keeps a 7B model on a single
        # GPU. Module + optimizer are closed over; only params/opt_state/batch are traced.
        # ``base_`` is an argument rather than a closure so the frozen checkpoint stays a
        # runtime buffer instead of being baked into the executable as a constant.
        module = self.module
        ignore_index = self.ignore_index

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def jit_step(p, o, batch, base_):
            def loss_fn(trainable):
                # Under LoRA (``base_`` given) differentiate only the adapters — the frozen
                # base never gets a cotangent, the mirror of torch's ``get_peft_model``. Full
                # finetuning passes ``base_ is None`` and differentiates the whole tree.
                full = combine(base_, trainable) if base_ is not None else trainable
                out = module.apply({"params": full}, input_ids=batch["input_ids"],
                                   attention_mask=batch.get("attention_mask"))
                return _causal_lm_loss(out["logits"], batch["labels"], ignore_index)

            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, o = optimizer.update(grads, o, p)
            return optax.apply_updates(p, updates), o, loss

        params = state
        self.global_step = 0
        total_loss = 0.0
        cumulative_loss = 0.0
        start_time = time.time()

        for epoch in range(epochs):
            self.epoch = float(epoch)
            for batch in self.get_train_dataloader():
                params, opt_state, loss = jit_step(
                    params, opt_state, batch, base if use_lora else None)
                loss = float(loss)

                total_loss += loss
                cumulative_loss += loss
                self.global_step += 1

                if logging_steps and self.global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    elapsed = time.time() - start_time
                    logger.info(
                        "Step %d/%d, Loss: %.4f, LR: %.2e, Time: %.1fs",
                        self.global_step, total_steps, avg_loss,
                        # torch reads the LR back off the scheduler *after* stepping it;
                        # an optax schedule is a pure function of the step index instead.
                        float(scheduler(self.global_step - 1)), elapsed,
                    )
                    total_loss = 0.0

                if save_steps and save_steps > 0 and self.global_step % save_steps == 0:
                    # Back to one tree for the checkpoint: under LoRA the loop advanced only
                    # the adapters, and torch's checkpoint holds base + adapters together.
                    self.params = combine(base, params) if use_lora else params
                    self.save_model(os.path.join(output_dir, f"checkpoint-{self.global_step}"))

        self.params = combine(base, params) if use_lora else params
        self.epoch = float(epochs)
        avg_train_loss = cumulative_loss / max(self.global_step, 1)
        metrics = {"train_loss": avg_train_loss, "train_steps": self.global_step}
        return TrainOutput(global_step=self.global_step, training_loss=avg_train_loss, metrics=metrics)

    def evaluate(self, metric_key_prefix="eval"):
        """Run evaluation."""
        if self.eval_dataset is None:
            return {}

        # No ``torch.no_grad()`` needed: nothing here is differentiated, and the module is
        # applied without an RNG so dropout stays off — the JAX form of ``model.eval()``.
        # jit the forward+loss: ``_iter_batches`` gives every eval batch one fixed shape, so
        # this compiles once and the rest of the split is pure GPU (no per-batch retrace).
        module = self.module
        ignore_index = self.ignore_index

        @jax.jit
        def eval_step(params, input_ids, attention_mask, labels):
            out = module.apply({"params": params}, input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            return _causal_lm_loss(logits, labels, ignore_index)

        total_loss = 0.0
        num_steps = 0
        for batch in self.get_eval_dataloader():
            loss = eval_step(self.params, batch["input_ids"], batch.get("attention_mask"), batch["labels"])
            total_loss += float(loss)
            num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        return metrics

    def save_model(self, output_dir=None):
        """Save model checkpoint."""
        output_dir = output_dir or getattr(self.args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        # torch calls ``save_pretrained``/``torch.save(state_dict())``; the flax equivalent
        # is one safetensors file keyed by the "/"-joined pytree path. Use the flax backend:
        # it round-trips bfloat16, which ``safetensors.numpy`` cannot represent at all.
        from safetensors.flax import save_file

        flat = {"/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path): jnp.asarray(leaf)
                for path, leaf in jax.tree_util.tree_flatten_with_path(self.params)[0]}
        save_file(flat, os.path.join(output_dir, "model.safetensors"))

        # Save tokenizer
        if self.tokenizer and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

    def log_metrics(self, split, metrics):
        logger.info(f"***** {split} metrics *****")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")

    def save_metrics(self, split, metrics):
        output_dir = getattr(self.args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{split}_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def save_state(self):
        pass


class TrainOutput:
    def __init__(self, global_step, training_loss, metrics):
        self.global_step = global_step
        self.training_loss = training_loss
        self.metrics = metrics
