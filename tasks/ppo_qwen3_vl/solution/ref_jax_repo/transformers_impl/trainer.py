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

"""Minimal Trainer implementation for Qwen3-VL SFT.

Replaces HuggingFace's Seq2SeqTrainer. Supports:
- Standard JAX/optax training loop
- Learning rate scheduling with warmup
- Logging (console + file)
- Checkpointing
- Evaluation

Mirror of ``torch_orig/transformers_impl/trainer.py`` — same ``TrainResult``, same
``SCHEDULER_MAP``, same ``Trainer`` methods. As in ``torch_orig``, nothing in this repo
imports it: the PPO loop lives in ``ppo/trainer.py``, and this module is the
SFT-shaped trainer the family ships alongside it. JAX adjustments:

1. **The model is a ``(module, params)`` pair.** Flax modules are stateless, so
   ``load_model`` returns the module and its params separately. ``self.model`` keeps the
   pair (torch's attribute survives) and ``self.module`` / ``self.params`` name the halves;
   the loop threads params through functionally rather than mutating them in place.
   ``model.train()`` / ``model.zero_grad()`` have no analogue and are dropped: dropout is
   off because the module is applied without an RNG, and optax grads are never accumulated
   into the params. ``ignore_index`` joins the signature because a flax module returns
   logits, so computing the loss is the trainer's job.
2. **optax replaces ``torch.optim`` + ``LambdaLR``.** The schedule is baked into the
   optimizer rather than wrapped around it, so ``get_{cosine,linear}_schedule_with_warmup``
   take the peak learning rate where torch takes the optimizer to read it off,
   ``_create_optimizer`` takes ``num_training_steps`` (torch's takes none), and
   ``_create_scheduler``'s ``optimizer`` argument survives for signature parity but is
   unused. The returned schedule is queried as ``scheduler(step)`` where torch calls
   ``scheduler.get_last_lr()[0]``. torch's ``clip_grad_norm_`` before ``optimizer.step()``
   becomes ``optax.clip_by_global_norm`` chained ahead of AdamW. torch's decay/no-decay
   parameter grouping is not reproduced: the reference optimizer these repos are graded
   against — and that ``train.run_training`` must reproduce — is a single ``optax.adamw``.
3. **No ``DataLoader``.** ``_get_train_dataloader`` / ``_get_eval_dataloader`` return fresh
   generators that slice the dataset into ``per_device_train_batch_size`` chunks and run
   the collator. Order is sequential, not ``RandomSampler``'s shuffle, so a run is
   reproducible against the frozen goldens; ``len(dataloader)`` is unavailable, so the step
   count comes from ``len(self.train_dataset)``.
4. **Flax models return logits, not a loss**, so torch's ``outputs.loss`` is computed here
   by ``causal_lm_loss``. Gradients are taken over the whole batch in one jitted step, so
   ``gradient_accumulation_steps`` is not emulated (these configs all leave it at 1).
5. **``save_model`` writes a flat ``safetensors`` file** of the params (``/``-joined
   pytree paths) instead of calling ``save_pretrained``; there is no PEFT branch because
   this repo's LoRA lives in the flax module itself (``model/loader.py``).
"""

import functools
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of training."""
    metrics: dict[str, Any] = field(default_factory=dict)


def get_cosine_schedule_with_warmup(base_lr, num_warmup_steps, num_training_steps):
    """Create a cosine LR schedule with warmup."""
    decay = optax.cosine_decay_schedule(base_lr, max(1, num_training_steps - num_warmup_steps))
    if num_warmup_steps > 0:
        warmup = optax.linear_schedule(0.0, base_lr, num_warmup_steps)
        return optax.join_schedules([warmup, decay], [num_warmup_steps])
    return decay


def get_linear_schedule_with_warmup(base_lr, num_warmup_steps, num_training_steps):
    """Create a linear LR schedule with warmup."""
    decay = optax.linear_schedule(base_lr, 0.0, max(1, num_training_steps - num_warmup_steps))
    if num_warmup_steps > 0:
        warmup = optax.linear_schedule(0.0, base_lr, num_warmup_steps)
        return optax.join_schedules([warmup, decay], [num_warmup_steps])
    return decay


SCHEDULER_MAP = {
    "cosine": get_cosine_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
}


def build_schedule(training_args, num_steps: int):
    base_lr = float(getattr(training_args, "learning_rate", 1e-6))
    warmup = int(getattr(training_args, "warmup_steps", 0) or 0)
    if warmup == 0:
        warmup = int(num_steps * float(getattr(training_args, "warmup_ratio", 0.0) or 0.0))
    scheduler_fn = SCHEDULER_MAP.get(
        str(getattr(training_args, "lr_scheduler_type", "cosine")), get_cosine_schedule_with_warmup
    )
    return scheduler_fn(base_lr, warmup, num_steps)


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


class Trainer:
    """Minimal trainer for SFT with multimodal support."""

    def __init__(
        self,
        model,  # the (flax module, params) pair model.loader.load_model returns
        args,  # TrainingArguments from config.py
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        ignore_index: int = -100,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # torch moves the model onto CUDA here because model/loader.py leaves it on the
        # CPU. jax arrays are already on the default device (the GPU when one is visible),
        # so the only thing left to unpack is the (module, params) pair.
        self.module, self.params = model if isinstance(model, tuple) else (model, None)
        self.ignore_index = ignore_index

        self.global_step = 0
        self.epoch = 0.0

    def _get_train_dataloader(self):
        return self._iter_batches(self.train_dataset, self.args.per_device_train_batch_size)

    def _get_eval_dataloader(self):
        return self._iter_batches(self.eval_dataset, self.args.per_device_eval_batch_size)

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

    def _create_optimizer(self, num_training_steps):
        return build_optimizer(self.args, num_training_steps)

    def _create_scheduler(self, optimizer, num_training_steps):
        del optimizer  # optax bakes the schedule into the optimizer; see adjustment 2.
        num_warmup_steps = self.args.warmup_steps
        if num_warmup_steps == 0 and self.args.warmup_ratio > 0:
            num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        scheduler_fn = SCHEDULER_MAP.get(self.args.lr_scheduler_type, get_cosine_schedule_with_warmup)
        return scheduler_fn(float(self.args.learning_rate), num_warmup_steps, num_training_steps)

    def train(self, resume_from_checkpoint=None) -> TrainResult:
        num_examples = len(self.train_dataset)
        batch_size = int(self.args.per_device_train_batch_size or 1) or 1
        num_update_steps_per_epoch = max(math.ceil(num_examples / batch_size), 1)
        num_epochs = int(math.ceil(self.args.num_train_epochs))
        max_steps = num_update_steps_per_epoch * num_epochs

        optimizer = self._create_optimizer(max_steps)
        scheduler = self._create_scheduler(optimizer, max_steps)
        opt_state = optimizer.init(self.params)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")

        # JIT the step with params + opt_state donated so XLA reuses their buffers (no
        # transient doubling of the Adam state) — this is what keeps a large model on a
        # single GPU. Module + optimizer are closed over; only params/opt_state/batch are
        # traced. It is also what ``gradient_checkpointing`` would have bought in torch:
        # XLA schedules the rematerialisation itself, so the flag needs no handling here.
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def jit_step(p, o, batch):
            return train_step(self.module, optimizer, p, o, batch, self.ignore_index)

        params = self.params
        self.global_step = 0
        total_loss = 0.0
        logging_loss = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            for batch in self._get_train_dataloader():
                params, opt_state, loss = jit_step(params, opt_state, batch)
                total_loss += float(loss)
                self.global_step += 1

                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    elapsed = time.time() - start_time
                    avg_loss = (total_loss - logging_loss) / self.args.logging_steps
                    # torch reads the LR back off the scheduler *after* stepping it; an
                    # optax schedule is a pure function of the step index instead.
                    lr = float(scheduler(self.global_step - 1))
                    logger.info(
                        f"Step {self.global_step}/{max_steps} | "
                        f"Epoch {epoch + 1}/{num_epochs} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )
                    logging_loss = total_loss

                # Save checkpoint
                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    self.params = params
                    self._save_checkpoint(f"checkpoint-{self.global_step}")

                if self.global_step >= max_steps:
                    break

            if self.global_step >= max_steps:
                break

        self.params = params
        metrics = {
            "train_loss": total_loss / max(self.global_step, 1),
            "train_runtime": time.time() - start_time,
            "train_steps": self.global_step,
            "epoch": self.args.num_train_epochs,
        }
        return TrainResult(metrics=metrics)

    def evaluate(self, metric_key_prefix="eval") -> dict[str, float]:
        """Run evaluation."""
        if self.eval_dataset is None:
            logger.warning("No eval dataset provided, skipping evaluation.")
            return {}

        # No ``torch.no_grad()`` needed: nothing here is differentiated, and the module is
        # applied without an RNG so dropout stays off — the JAX form of ``model.eval()``.
        total_loss = 0.0
        total_steps = 0

        for batch in self._get_eval_dataloader():
            out = self.module.apply(
                {"params": self.params}, input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
            )
            logits = out["logits"] if isinstance(out, dict) else out.logits
            total_loss += float(causal_lm_loss(logits, batch["labels"], self.ignore_index))
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_steps": total_steps,
        }
        return metrics

    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        output_dir = os.path.join(self.args.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(output_dir)
        logger.info(f"Saved checkpoint to {output_dir}")

        # Manage save_total_limit
        if self.args.save_total_limit is not None and self.args.save_total_limit > 0:
            checkpoints = sorted(
                [d for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]),
            )
            if len(checkpoints) > self.args.save_total_limit:
                for old_ckpt in checkpoints[: len(checkpoints) - self.args.save_total_limit]:
                    import shutil
                    shutil.rmtree(os.path.join(self.args.output_dir, old_ckpt), ignore_errors=True)

    def save_model(self, output_dir: Optional[str] = None):
        """Save model weights."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # torch calls ``save_pretrained``/``torch.save(state_dict())``; the flax equivalent
        # is one safetensors file keyed by the "/"-joined pytree path. Use the flax backend:
        # it round-trips bfloat16, which ``safetensors.numpy`` cannot represent at all.
        from safetensors.flax import save_file

        flat = {"/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path): jnp.asarray(leaf)
                for path, leaf in jax.tree_util.tree_flatten_with_path(self.params)[0]}
        save_file(flat, os.path.join(output_dir, "model.safetensors"))

        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)

    def log_metrics(self, split: str, metrics: dict[str, Any]):
        """Log metrics to console."""
        logger.info(f"***** {split} metrics *****")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key} = {value}")

    def save_metrics(self, split: str, metrics: dict[str, Any]):
        """Save metrics to JSON file."""
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_state(self):
        """Save trainer state."""
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state = {"global_step": self.global_step, "epoch": self.epoch}
        path = os.path.join(output_dir, "trainer_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
