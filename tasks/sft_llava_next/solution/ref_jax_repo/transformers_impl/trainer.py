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
``TrainOutput``, same call order out of ``train.py::run_sft``. JAX adjustments:

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
import numpy as np
import optax

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
        """Sequential batches of `batch_size`, collated (JAX stand-in for ``DataLoader``).

        A fresh generator per call, so ``train()`` can re-iterate it once per epoch the way
        torch re-iterates its dataloader.

        Every batch in the split is right-padded on the token axis to a single fixed sequence
        length (the split's longest batch): input_ids / attention_mask / labels. The collator
        pads only to the longest sample *within* a batch, so at ``batch_size=1`` each step would
        otherwise carry a distinct shape and the jitted train/eval step would recompile on every
        step — XLA compilation, not the GPU, then dominates the run (minutes become hours). A
        fixed shape compiles the step exactly once. The extra positions are numerically inert:
        they carry ``attention_mask=0`` and ``labels=IGNORE_INDEX``, so with right-padding and a
        causal mask the real tokens never attend to them and never enter the loss average — the
        losses are identical to pad-to-longest, only the wall-clock differs.

        Modality tensors (``pixel_values`` / ``image_sizes``) are passed through untouched — the
        collator produces them (``data/collator.py`` ``batch.update(mm_inputs)``) and the trainer
        turns them into a fixed-width frozen-vision ``image_features`` block on the host (see
        ``_precompute_image_features``), so they never ride into the jitted step and never force a
        per-image recompile of the any-res packing. Batches are yielded as the collator's arrays
        (numpy); the jitted step converts what it consumes.
        """
        batch_size = int(batch_size or 1) or 1
        n = len(dataset)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        seq_pad = {"input_ids": pad_id, "attention_mask": 0, "labels": self.ignore_index}

        # Collate every batch up front so the whole split shares one padded token width.
        collated = []
        max_len = 1
        for i in range(0, n, batch_size):
            feats = [dataset[j] for j in range(i, min(i + batch_size, n))]
            raw = self.data_collator(feats)
            raw = dict(raw.data if hasattr(raw, "data") else raw)
            collated.append(raw)
            if "input_ids" in raw:
                max_len = max(max_len, int(np.asarray(raw["input_ids"]).shape[-1]))

        def gen():
            for b in collated:
                if "input_ids" not in b:
                    yield b
                    continue
                seq_len = int(np.asarray(b["input_ids"]).shape[-1])
                if max_len <= seq_len:
                    yield b
                    continue
                out = dict(b)
                for k in list(out):
                    if k in seq_pad:
                        a = np.asarray(out[k])
                        if a.ndim >= 1 and a.shape[-1] == seq_len:
                            widths = [(0, 0)] * (a.ndim - 1) + [(0, max_len - seq_len)]
                            out[k] = np.pad(a, widths, mode="constant", constant_values=seq_pad[k])
                yield out

        return gen()

    def _precompute_image_features(self, batches, params):
        """Precompute the frozen any-res vision feature block for each batch, padded to one
        fixed width across the set (mirror of the dpo_llava_next trainer).

        The vision tower + multi-modal projector are frozen (``freeze_vision_tower`` /
        ``freeze_multi_modal_projector`` — no LoRA adapter, no gradient), so their flat
        ``(num_image_tokens, hidden)`` output is CONSTANT across every step and epoch.
        Computing it inside the jitted step would force ``image_sizes`` to ride in as a
        per-image static key (the any-res packing does host-side ``int(...)`` control flow),
        recompiling the WHOLE fwd+bwd for every distinct image (~180 unique -> hours). Instead
        compute it ONCE here, pad every block to the split-wide max, and feed it into the step
        as a plain ``image_features`` array — the step then has one static shape and compiles
        ONCE. Numerically identical: ``_scatter_image_features`` gathers only the first
        (#placeholder) rows, so the zero pad rows are never read (mirror of torch's flattened
        ``torch.cat(image_features)``).

        The 24-layer CLIP tower is the cost; run eager (op-by-op ``apply``) it dispatches one op
        at a time with the GPU mostly idle. So the shape-stable tower+projector half is jitted
        ONCE over a patch dim padded to the set-wide max (``return_patch_features``); only the
        tiny ``image_sizes``-dependent any-res packing stays on the host. Patches ride the
        tower's batch dim and never mix, so the pad patches are inert and the sliced result
        equals the eager path (up to op fusion). The numeric golden path never calls this method.
        """
        from transformers_impl.modeling import _num_image_patches, _pack_one_image

        cfg = self.module.config
        # Frozen ``image_newline`` param (wherever the checkpoint placed it in the tree).
        newline = None
        for path, leaf in jax.tree_util.tree_leaves_with_path(params):
            if any(getattr(k, "key", None) == "image_newline" for k in path):
                newline = leaf
                break

        # Host pass: real patches + any-res metadata per batch; set-wide max patch count.
        metas, max_p = [], 0
        for b in batches:
            pv = b.get("pixel_values") if isinstance(b, dict) else None
            isz = b.get("image_sizes") if isinstance(b, dict) else None
            if pv is None or isz is None:
                metas.append(None)
                continue
            isz_np = np.asarray(isz)
            sizes = [(int(isz_np[i][0]), int(isz_np[i][1])) for i in range(isz_np.shape[0])]
            nump = [_num_image_patches(h, w, cfg.image_grid_pinpoints, cfg.vision_image_size)
                    for h, w in sizes]
            pv_np = np.asarray(pv)
            if pv_np.ndim == 5:
                pv_np = np.concatenate([pv_np[i, :nump[i]] for i in range(len(sizes))], axis=0)
            metas.append((pv_np, sizes, nump))
            max_p = max(max_p, pv_np.shape[0])
        if max_p == 0:
            return [None] * len(batches)

        # Jitted, shape-stable tower+projector over a fixed-width patch batch (compiles once).
        if getattr(self, "_patch_fn", None) is None:
            _ids = jnp.zeros((1, 1), jnp.int32)
            self._patch_fn = jax.jit(
                lambda p, pv: self.module.apply(
                    {"params": p}, input_ids=_ids, pixel_values=pv, return_patch_features=True
                )
            )

        feats = []
        for m in metas:
            if m is None:
                feats.append(None)
                continue
            pv_np, sizes, nump = m
            total = pv_np.shape[0]
            if total < max_p:
                pad = np.zeros((max_p - total,) + pv_np.shape[1:], dtype=pv_np.dtype)
                pv_pad = np.concatenate([pv_np, pad], axis=0)
            else:
                pv_pad = pv_np
            raw = self._patch_fn(params, jnp.asarray(pv_pad))[:total]  # (total, T, D), concrete
            packed, off = [], 0
            for i, npi in enumerate(nump):
                packed.append(
                    _pack_one_image(raw[off:off + npi], sizes[i][0], sizes[i][1], newline, cfg)
                )
                off += npi
            feats.append(np.asarray(jnp.concatenate(packed, axis=0)))

        present = [f for f in feats if f is not None]
        if not present:
            return [None] * len(feats)
        max_len = max(f.shape[0] for f in present)
        d = present[0].shape[1]
        dt = present[0].dtype
        padded = []
        for f in feats:
            if f is None:
                padded.append(np.zeros((max_len, d), dtype=dt))
            elif f.shape[0] < max_len:
                padded.append(np.pad(f, ((0, max_len - f.shape[0]), (0, 0))))
            else:
                padded.append(f)
        return padded

    def create_optimizer(self, num_training_steps):
        """optax AdamW + LR schedule from ``self.args`` (mirrors torch's ``create_optimizer``).

        HF ``Trainer`` clips gradients to ``max_grad_norm`` (default 1.0) before each step;
        mirror it with ``optax.clip_by_global_norm`` so the loss curve matches the reference
        and stays stable for models with large gradients (unclipped bf16 steps can diverge).
        """
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
        # torch/PEFT keeps LoRA adapters + Adam moments in fp32 over a bf16 base; upcast the
        # trainable set so Adam's beta2 does not underflow in bf16 (mirrors ppo/trainer.py).
        if use_lora:
            trainable = jax.tree_util.tree_map(lambda x: jnp.asarray(x).astype(jnp.float32), trainable)
        opt_state = optimizer.init(trainable)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {num_epochs}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Total optimization steps = {total_steps}")

        module = self.module
        ignore_index = self.ignore_index
        from model.loader import combine

        # Materialize the training batches once (sequential order, no shuffle -> reproducible),
        # then precompute the frozen any-res vision features once (constant across steps -- see
        # _precompute_image_features) so the jitted step is fed a fixed-width ``image_features``
        # array and compiles ONCE instead of recompiling the whole fwd+bwd per distinct image.
        batches = list(self.get_train_dataloader())
        train_feats = self._precompute_image_features(batches, self.params)

        # JITTED step. The any-res vision features are precomputed once (frozen tower, constant
        # across steps) and passed in as a fixed-width ``image_features`` array, so this step
        # carries no ``image_sizes``-dependent static arg and compiles ONCE -- instead of
        # recompiling the whole fwd+bwd per distinct image (the packing's host-side ``int(...)``
        # control flow otherwise forces ``image_sizes`` in as a per-image static key -> ~180
        # compiles = hours). params+opt_state are donated so XLA reuses their buffers (no
        # transient Adam-state doubling), which is what keeps a 7B model on a single GPU.
        # Numerically identical to running the tower in-step: the frozen tower output is the same,
        # and ``_scatter_image_features`` reads only the first (#placeholder) rows so pad rows
        # never enter the loss.
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def jit_step(p, o, batch, image_features):
            def loss_fn(trainable):
                # Thread every batch key except the label tensor, plus the precomputed frozen
                # ``image_features`` block, so the vision tokens are attended during training.
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                model_inputs["image_features"] = image_features
                out = module.apply({"params": trainable}, run_vision=True, **model_inputs)
                logits = out["logits"] if isinstance(out, dict) else out.logits
                return _causal_lm_loss(logits, batch["labels"], ignore_index)

            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, o = optimizer.update(grads, o, p)
            return optax.apply_updates(p, updates), o, loss

        # Same, for the LoRA path: adapters + Adam state are donated, the frozen base is not.
        # ``base`` is an argument, not a closure, so the frozen checkpoint stays a runtime
        # buffer; the recombined tree is built inside ``loss_fn`` so ``value_and_grad`` sees
        # only the adapters as differentiable — torch's PEFT ``requires_grad=False``. The frozen
        # vision block rides in as ``image_features`` (precomputed, fixed width -> one compile).
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def jit_step_lora(tr, o, base, batch, image_features):
            def loss_fn(trainable):
                full = combine(base, trainable)
                # Thread every batch key except the label tensor, plus the precomputed frozen
                # ``image_features`` block, so the vision tokens are attended during training.
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                model_inputs["image_features"] = image_features
                out = module.apply({"params": full}, run_vision=True, **model_inputs)
                logits = out["logits"] if isinstance(out, dict) else out.logits
                return _causal_lm_loss(logits, batch["labels"], ignore_index)

            loss, grads = jax.value_and_grad(loss_fn)(tr)
            updates, o = optimizer.update(grads, o, tr)
            return optax.apply_updates(tr, updates), o, loss

        self.global_step = 0
        total_loss = 0.0
        cumulative_loss = 0.0
        start_time = time.time()

        for epoch in range(epochs):
            self.epoch = float(epoch)
            for bi, batch in enumerate(batches):
                # Drop the raw modality tensors -- the frozen vision block is precomputed and
                # passed in as ``image_features`` (fixed width -> the step compiles once).
                batch_t = {k: v for k, v in batch.items()
                           if k not in ("image_sizes", "pixel_values")}
                if use_lora:
                    trainable, opt_state, loss = jit_step_lora(
                        trainable, opt_state, base_params, batch_t, train_feats[bi])
                else:
                    trainable, opt_state, loss = jit_step(
                        trainable, opt_state, batch_t, train_feats[bi])
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
        module = self.module
        ignore_index = self.ignore_index

        # Materialize the eval batches, then precompute the frozen any-res vision features once
        # (constant across steps -- the tower/projector are frozen, so taking them here is
        # identical to taking them during train()). The jitted eval forward is then fed a
        # fixed-width ``image_features`` block, so it attends the image tokens (matching torch's
        # ``model(**batch)`` eval, which sees ``pixel_values``) and compiles ONCE -- no
        # per-image recompile of the any-res packing.
        eval_batches = list(self.get_eval_dataloader())
        eval_feats = self._precompute_image_features(eval_batches, self.params)

        @jax.jit
        def eval_step(params, batch, image_features):
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            model_inputs["image_features"] = image_features
            out = module.apply({"params": params}, run_vision=True, **model_inputs)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            return _causal_lm_loss(logits, batch["labels"], ignore_index)

        total_loss = 0.0
        num_steps = 0
        for batch, imgf in zip(eval_batches, eval_feats):
            # Drop the raw modality tensors -- the frozen block rides in as ``image_features``.
            batch_t = {k: v for k, v in batch.items()
                       if k not in ("image_sizes", "pixel_values")}
            loss = eval_step(self.params, batch_t, imgf)
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
