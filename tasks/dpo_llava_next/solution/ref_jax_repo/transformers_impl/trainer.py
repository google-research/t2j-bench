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

"""Minimal Trainer implementation for LLaVA-Next SFT.

Mirror of ``torch_orig/transformers_impl/trainer.py`` — same ``Trainer`` methods, same
``TrainOutput``. As in ``torch_orig``, nothing in this repo imports it: the DPO loop
lives in ``dpo/trainer.py``, and this module is the SFT-shaped trainer the family ships
alongside it. JAX adjustments:

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
   by ``causal_lm_loss``. Gradients are taken over the whole batch in one jitted step, so
   ``gradient_accumulation_steps`` is not emulated (these configs all leave it at 1).
5. **``save_model`` writes a flat ``safetensors`` file** of the params (``/``-joined
   pytree paths) instead of calling ``save_pretrained``.
6. **LoRA is discovered from the pytree.** torch's trainer needs no LoRA branch because
   PEFT has already frozen the base and ``param.requires_grad`` carries the answer. The
   flax analogue is ``_split_trainable``: if ``model/loader.py`` added adapters, split them
   off and hand the optimizer only those, so the frozen base gets neither gradients nor
   Adam moments — which is what keeps this model on one GPU.
"""
import functools
import json
import logging
import math
import os
import time

import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)


def build_schedule(training_args, num_steps: int):
    base_lr = float(getattr(training_args, "learning_rate", 1e-6))
    warmup = int(getattr(training_args, "warmup_steps", 0) or 0)
    if warmup == 0:
        warmup = int(num_steps * float(getattr(training_args, "warmup_ratio", 0.0) or 0.0))
    if str(getattr(training_args, "lr_scheduler_type", "cosine")) == "linear":
        decay = optax.linear_schedule(base_lr, 0.0, max(1, num_steps - warmup))
    else:
        decay = optax.cosine_decay_schedule(base_lr, max(1, num_steps - warmup))
    if warmup > 0:
        warmup_fn = optax.linear_schedule(0.0, base_lr, warmup)
        return optax.join_schedules([warmup_fn, decay], [warmup])
    return decay


def build_optimizer(training_args, num_steps: int = 8):
    """optax AdamW + LR schedule from get_train_args (matches the reference optimizer).

    HF ``Trainer`` clips gradients to ``max_grad_norm`` (default 1.0) before each
    step; mirror it with ``optax.clip_by_global_norm`` so the loss curve matches the
    reference and stays stable for models with large gradients (unclipped bf16 steps
    can diverge to NaN).
    """
    schedule = build_schedule(training_args, num_steps)
    max_grad_norm = float(getattr(training_args, "max_grad_norm", 1.0) or 0.0)
    adamw = optax.adamw(schedule)
    if max_grad_norm > 0:
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), adamw)
    return adamw


def causal_lm_loss(logits, labels, ignore_index: int = -100):
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:]
    mask = shift_labels != ignore_index
    safe = jnp.where(mask, shift_labels, 0)
    logp = jax.nn.log_softmax(shift_logits, axis=-1)
    gathered = jnp.take_along_axis(logp, safe[..., None], axis=-1).squeeze(-1)
    return -(gathered * mask).sum() / jnp.maximum(mask.sum(), 1)


def train_step(module, optimizer, params, opt_state, batch, ignore_index: int = -100):
    def loss_fn(p):
        out = module.apply({"params": p}, input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        return causal_lm_loss(out["logits"], batch["labels"], ignore_index)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


def train_step_lora(module, optimizer, base_params, trainable, opt_state, batch, ignore_index: int = -100):
    """``train_step`` for the LoRA path — differentiate the adapters only.

    ``base_params`` is threaded through as an argument rather than captured in a closure so
    that under ``jax.jit`` the frozen base stays an ordinary (non-donated) input instead of
    being hoisted into the executable as a constant. The recombined pytree is built inside
    ``loss_fn``, so ``value_and_grad`` sees adapters as the only differentiable leaves: no
    grads and no Adam moments for the base, which is torch's PEFT ``requires_grad=False``.
    """
    from model.loader import combine

    def loss_fn(tr):
        p = combine(base_params, tr)
        out = module.apply({"params": p}, input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        return causal_lm_loss(out["logits"], batch["labels"], ignore_index)

    loss, grads = jax.value_and_grad(loss_fn)(trainable)
    updates, opt_state = optimizer.update(grads, opt_state, trainable)
    return optax.apply_updates(trainable, updates), opt_state, loss


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

        # Flax modules are stateless, so torch's single ``model`` is the (module, params)
        # pair load_model returns; unpack it into the two halves the loop threads through.
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
        """
        batch_size = int(batch_size or 1) or 1
        n = len(dataset)

        def gen():
            for i in range(0, n, batch_size):
                feats = [dataset[j] for j in range(i, min(i + batch_size, n))]
                raw = self.data_collator(feats)
                raw = raw.data if hasattr(raw, "data") else raw
                yield {k: jnp.asarray(raw[k]) for k in ("input_ids", "attention_mask", "labels") if k in raw}

        return gen()

    def create_optimizer(self, num_training_steps):
        return build_optimizer(self.args, num_training_steps)

    def create_scheduler(self, optimizer, num_training_steps):
        del optimizer  # optax bakes the schedule into the optimizer; see adjustment 2.
        return build_schedule(self.args, num_training_steps)

    def _split_trainable(self, params):
        """Return ``(frozen_base, trainable, use_lora)`` for `params`.

        torch's trainer needs no equivalent: PEFT has already set ``requires_grad=False`` on
        the base and ``model.parameters()`` carries the answer. Here the same question is
        asked of the pytree — did ``model/loader.py`` add LoRA adapters? — so that the
        optimizer only ever sees the leaves that are meant to move.
        """
        from model.loader import has_adapters, split

        if not has_adapters(params):
            return None, params, False
        base, trainable = split(params)
        return base, trainable, True

    @staticmethod
    def _merge(base_params, trainable, use_lora):
        """Inverse of ``_split_trainable``: the full pytree, adapters included."""
        if not use_lora:
            return trainable
        from model.loader import combine

        return combine(base_params, trainable)

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

        optimizer = self.create_optimizer(total_steps)
        scheduler = self.create_scheduler(optimizer, total_steps)
        base_params, trainable, use_lora = self._split_trainable(self.params)
        opt_state = optimizer.init(trainable)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {num_epochs}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Total optimization steps = {total_steps}")

        # JIT the step with params + opt_state donated so XLA reuses their buffers (no
        # transient doubling of the Adam state) — this is what keeps a 7B model on a single
        # GPU. Module + optimizer are closed over; only params/opt_state/batch are traced.
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def jit_step(p, o, batch):
            return train_step(self.module, optimizer, p, o, batch, self.ignore_index)

        # Same, for the LoRA path: adapters + Adam state are donated, the frozen base is not.
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def jit_step_lora(tr, o, base, batch):
            return train_step_lora(self.module, optimizer, base, tr, o, batch, self.ignore_index)

        self.global_step = 0
        total_loss = 0.0
        cumulative_loss = 0.0
        start_time = time.time()

        for epoch in range(epochs):
            self.epoch = float(epoch)
            for batch in self.get_train_dataloader():
                if use_lora:
                    trainable, opt_state, loss = jit_step_lora(trainable, opt_state, base_params, batch)
                else:
                    trainable, opt_state, loss = jit_step(trainable, opt_state, batch)
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
                    self.params = self._merge(base_params, trainable, use_lora)
                    self.save_model(os.path.join(output_dir, f"checkpoint-{self.global_step}"))

        # Fold the trained adapters back in so ``evaluate`` and ``save_model`` see them.
        self.params = self._merge(base_params, trainable, use_lora)
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
        total_loss = 0.0
        num_steps = 0
        for batch in self.get_eval_dataloader():
            out = self.module.apply(
                {"params": self.params}, input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
            )
            logits = out["logits"] if isinstance(out, dict) else out.logits
            total_loss += float(causal_lm_loss(logits, batch["labels"], self.ignore_index))
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
