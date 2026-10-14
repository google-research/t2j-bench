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

"""Standalone PPO trainer (torch-free JAX port of ``torch_orig/ppo/trainer.py``).

Faithful mirror of ``StandalonePPOTrainer``: the same rollout -> reward -> logprobs/
values -> KL -> GAE -> clipped PPO update loop, with JAX adjustments:
- params are held **functionally** (``self.module`` + ``self.params``); every forward
  is ``module.apply({"params": params}, ...)`` and the update is ``optax`` +
  ``jax.value_and_grad`` (no ``.backward()`` / ``optimizer.step()``).
- the reference log-probs (the KL baseline) are the base policy under
  ``jax.lax.stop_gradient`` (torch runs ``self.model.pretrained_model`` with the
  adapter disabled).
- generation rollout is an explicit ``jax.random`` sampling loop (temperature / top-k /
  top-p) rather than ``model.generate``.
- weight-decay excludes bias/LayerNorm params via an ``optax.masked`` label tree.

Like the torch original, the default reward is a **label reward** -- token-F1 of the
rolled-out answer against the dataset's gold tokens (``token_f1_reward``); an
``api``/``full`` reward model is still selectable. The training rollout is stochastic,
so the loop itself is not part of the (deterministic, golden-based) numeric grade;
``evaluate`` (L6) is the cross-framework scalar -- ``-mean_reward`` (negative mean
token-F1 reward) over a greedy decode. The trainer exists so the reference is a complete,
runnable migration.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from transformers_impl import ppo_compat as _ppo_compat

GenerationConfig = _ppo_compat.GenerationConfig
get_scheduler = _ppo_compat.get_scheduler

from .ppo_utils import (
    compute_gae,
    compute_ppo_loss,
    compute_value_loss,
    logprobs_from_logits,
    masked_mean,
    masked_whiten,
    token_f1_reward,
    torch_gc,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Every model forward in the PPO step (decode rollout, old-logprob/value/reference
# forwards, and the value_and_grad update) is fed a variable-width sequence, and XLA
# recompiles per distinct shape. At the run's batch size of 1 that means one compile
# per *distinct prompt length* -- ~once per example -- which dominated the rollout
# (see COMPARISON.md "Speed"). Rounding every such width UP to a multiple of
# _SEQ_BUCKET collapses those hundreds of shapes to a handful, so XLA compiles a few
# times and then reuses. It is numerically transparent: causal masking plus the
# attention mask make the logits at every real position independent of the trailing
# pad, so the sampled tokens and the sliced logprobs/values are identical to a tight
# buffer -- only the (masked-out) pad columns differ.
_SEQ_BUCKET = 64


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def _unbundle(model):
    """Accept the ``(module, params, frozen)`` tuple from ``load_model`` (or an object
    with ``.module``/``.params``) and return ``(module, params)``."""
    if isinstance(model, (tuple, list)):
        return model[0], model[1]
    return getattr(model, "module", model), getattr(model, "params", None)


def _decay_mask(params):
    """optax label tree: ``True`` for params that receive weight decay (exclude bias /
    LayerNorm scale+bias), matching torch's ``no_decay`` name filter."""
    no_decay = ("bias", "layer_norm", "layernorm", "ln_", "scale")

    def leaf(path, _):
        name = "/".join(str(getattr(k, "key", k)).lower() for k in path)
        return not any(nd in name for nd in no_decay)

    return jax.tree_util.tree_map_with_path(leaf, params)


def _set_schedule_lr(opt_state, lr):
    """Return ``opt_state`` with the injected ``learning_rate`` replaced by ``lr``.

    Functional, not in-place: the state is donated to a jitted update. Duck-typed on
    ``hyperparams`` because optax renamed the state class (``InjectHyperparamsState``
    -> ``InjectStatefulHyperparamsState``); the dtype is preserved so the update is
    not retraced each rollout.
    """

    def _walk(node):
        hyperparams = getattr(node, "hyperparams", None)
        if isinstance(hyperparams, dict) and "learning_rate" in hyperparams:
            updated = dict(hyperparams)
            updated["learning_rate"] = jnp.asarray(
                lr, dtype=hyperparams["learning_rate"].dtype
            )
            return node._replace(hyperparams=updated)
        if isinstance(node, tuple):
            walked = [_walk(child) for child in node]
            return type(node)(*walked) if hasattr(node, "_fields") else tuple(walked)
        return node

    return _walk(opt_state)


class StandalonePPOTrainer:
    """Standalone PPO trainer for generic JAX/Flax models, independent of TRL."""

    def __init__(
        self,
        model,
        ref_model,
        reward_model,
        tokenizer,
        processor,
        data_collator,
        train_dataset,
        training_args,
        ppo_args,
        generating_args,
        finetuning_args,
        eval_dataset=None,
    ):
        self.module, self.params = _unbundle(model)
        self.ref_module, self.ref_params = _unbundle(ref_model) if ref_model is not None else (None, None)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.ppo_args = ppo_args
        self.generating_args = generating_args
        self.finetuning_args = finetuning_args
        self.reward_model_type = getattr(ppo_args, "reward_model_type", "api")

        eos_ids = [getattr(self.tokenizer, "eos_token_id", None)]
        extra_eos = getattr(self.tokenizer, "additional_special_tokens_ids", None)
        if extra_eos:
            eos_ids += list(extra_eos)
        self.generation_config = GenerationConfig(
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=eos_ids,
            **generating_args.to_dict(),
        )

        # bf16 params are upcast to fp32 for the optimizer (matching the SFT/DPO trainers).
        self.compute_dtype = jnp.float32
        self.rng = jax.random.PRNGKey(int(getattr(training_args, "seed", 42)))

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler (optax; JAX adjustment of torch AdamW + LambdaLR).
    # ------------------------------------------------------------------ #
    def _create_optimizer(self, num_training_steps: int):
        args = self.training_args
        warmup_steps = args.warmup_steps or 0
        warmup_ratio = args.warmup_ratio or 0.0
        if warmup_steps == 0 and warmup_ratio > 0:
            warmup_steps = int(num_training_steps * warmup_ratio)
        schedule = get_scheduler(
            args.lr_scheduler_type,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            base_lr=args.learning_rate,
        )
        # Injected, not baked in: optax would otherwise advance the schedule on every
        # minibatch inside the ppo_epochs loop. ppo_train sets the rate per rollout.
        self._lr_schedule = schedule
        adam = optax.inject_hyperparams(optax.adamw)(
            learning_rate=schedule(0),
            weight_decay=args.weight_decay,
            mask=_decay_mask,
        )
        if args.max_grad_norm and args.max_grad_norm > 0:
            return optax.chain(optax.clip_by_global_norm(args.max_grad_norm), adam)
        return adam

    # ------------------------------------------------------------------ #
    # Forward helpers.
    # ------------------------------------------------------------------ #
    def _forward(self, params, input_ids, attention_mask, **kwargs):
        # JIT the forward so XLA frees intermediate activations after each call — the
        # rollout (generation + old-logprob/value/reference forwards) otherwise accumulates
        # eager activations and exhausts the GPU before the update. (Text path only; the mm
        # path with extra kwargs stays eager since the run uses text-only examples.)
        if not kwargs:
            if getattr(self, "_fwd_jit", None) is None:
                self._fwd_jit = jax.jit(
                    lambda p, ids, m: self.module.apply(
                        {"params": p}, input_ids=ids, attention_mask=m, use_cache=False
                    )
                )
            out = self._fwd_jit(params, input_ids, attention_mask)
        else:
            out = self.module.apply(
                {"params": params},
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
        logits = out["logits"] if isinstance(out, dict) else out.logits
        values = out["values"] if isinstance(out, dict) else getattr(out, "values", None)
        return logits, values

    def _next_key(self):
        self.rng, sub = jax.random.split(self.rng)
        return sub

    def _sample_token(self, logits_row: jnp.ndarray, generation_config=None) -> int:
        """Sample one token from a (vocab,) logits row with temperature / top-k / top-p."""
        gc = generation_config if generation_config is not None else self.generation_config
        logits = logits_row.astype(jnp.float32)
        if not gc.do_sample:
            return int(jnp.argmax(logits))
        if gc.temperature and gc.temperature != 1.0:
            logits = logits / gc.temperature
        if gc.top_k and gc.top_k > 0:
            k = min(gc.top_k, logits.shape[-1])
            kth = jnp.sort(logits)[-k]
            logits = jnp.where(logits < kth, -jnp.inf, logits)
        if gc.top_p and gc.top_p < 1.0:
            order = jnp.argsort(logits)[::-1]
            sorted_logits = logits[order]
            probs = jax.nn.softmax(sorted_logits)
            cdf = jnp.cumsum(probs)
            keep = cdf <= gc.top_p
            keep = keep.at[0].set(True)
            masked = jnp.where(keep, sorted_logits, -jnp.inf)
            full = jnp.full_like(logits, -jnp.inf).at[order].set(masked)
            logits = full
        return int(jax.random.categorical(self._next_key(), logits))

    def _generate_responses(self, batch, generation_config=None, seed_offset=None):
        """Autoregressive sampling rollout. Returns lists of (query, response) int arrays.

        The prompt is taken from where target labels begin (``labels != -100``), mirroring
        torch's supervised-style PPO batches. ``generation_config`` overrides the default
        sampling config (a greedy config for evaluation); ``seed_offset`` fixes the sampling
        RNG for a reproducible rollout (greedy decode is deterministic regardless).
        """
        gen_config = generation_config if generation_config is not None else self.generation_config
        if seed_offset is not None:
            self.rng = jax.random.PRNGKey(int(getattr(self.training_args, "seed", 42)) + int(seed_offset))
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        input_ids = np.asarray(batch["input_ids"])
        attention_mask = np.asarray(batch["attention_mask"])
        labels = np.asarray(batch["labels"]) if "labels" in batch else None

        queries, responses = [], []
        for row in range(input_ids.shape[0]):
            nonpad = np.nonzero(attention_mask[row])[0]
            start = int(nonpad[0]) if len(nonpad) else 0
            end = int(nonpad[-1]) + 1 if len(nonpad) else input_ids.shape[1]
            prompt_end = end
            if labels is not None:
                sup = np.nonzero(labels[row] != -100)[0]
                if len(sup):
                    prompt_end = max(start + 1, int(sup[0]))
            prompt = input_ids[row, start:prompt_end]
            if prompt.size == 0:
                prompt = input_ids[row, start:end]
            queries.append(np.asarray(prompt))

            # Fixed-shape decode: preallocate a (1, total) buffer and forward it whole each
            # step (the attention mask gates unfilled positions). ``total`` rounds the full
            # decode width (prompt + max_new_tokens) UP to a multiple of _SEQ_BUCKET, so the
            # forward takes one of only a handful of shapes across the whole run -- and the
            # same multiples the compute/update forwards below bucket to, so they share XLA
            # variants. The earlier ``total = plen + max_new`` discarded the collator's
            # padding and recompiled per example; padding all the way to cutoff_len (as SFT/
            # DPO do) would instead over-compute every short-prompt forward, since the rollout
            # forwards the buffer max_new_tokens times per example. Bucketing keeps each
            # forward within _SEQ_BUCKET tokens of the true prompt (see _SEQ_BUCKET above for
            # why the sampled tokens are identical to a tight buffer).
            max_new = int(gen_config.max_new_tokens or 0)
            plen = int(prompt.shape[0])
            total = ((plen + max_new + _SEQ_BUCKET - 1) // _SEQ_BUCKET) * _SEQ_BUCKET
            buf = np.full((1, total), pad_id, dtype=np.int64)
            buf[0, :plen] = np.asarray(prompt)
            mask = np.zeros((1, total), dtype=np.int64)
            mask[0, :plen] = 1
            gen = []
            for step in range(max_new):
                pos = plen + step
                logits, _ = self._forward(self.params, jnp.asarray(buf), jnp.asarray(mask))
                tok = self._sample_token(logits[0, pos - 1], gen_config)
                gen.append(tok)
                buf[0, pos] = tok
                mask[0, pos] = 1
                if eos_id is not None and tok == eos_id:
                    break
            responses.append(np.asarray(gen if gen else [eos_id if eos_id is not None else pad_id]))
        return queries, responses

    def _compute_rewards(self, queries, responses, gold_responses=None):
        """Reward scores for query+response pairs.

        In ``label`` mode the reward is the token-F1 of each generated response against its
        gold answer; otherwise the reward is zero (no reward server in offline grading).
        """
        if self.reward_model_type == "label":
            return self._compute_label_rewards(responses, gold_responses)
        # No reward server configured (e.g. offline reference run): zero reward.
        return jnp.zeros((len(queries),), dtype=jnp.float32)

    def _compute_label_rewards(self, responses, gold_responses):
        """Token-F1 overlap between each generated response and its gold answer.

        A decode-order-invariant signal in [0, 1]: the multiset overlap of the response
        token ids with the supervised answer token ids. Mirrors torch_orig.
        """
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        gold_responses = gold_responses if gold_responses is not None else [None] * len(responses)
        rewards = []
        for response, gold in zip(responses, gold_responses):
            pred_ids = [int(t) for t in np.asarray(response).tolist() if t != pad_id]
            gold_ids = [int(t) for t in (gold or []) if t != pad_id]
            rewards.append(token_f1_reward(pred_ids, gold_ids))
        return jnp.asarray(rewards, dtype=jnp.float32)

    def _gold_response_tokens(self, labels) -> list:
        """Per-row supervised answer token ids (the non-ignored positions of labels)."""
        gold = []
        for row in np.asarray(labels):
            gold.append([int(t) for t in np.asarray(row).tolist() if t != IGNORE_INDEX])
        return gold

    def _pad_sequences(self, sequences, pad_value, padding_side="right", pad_to_multiple=None):
        max_len = max(s.shape[0] for s in sequences)
        if pad_to_multiple:
            # Round the padded width up to a bucket so the compute/update forwards over
            # these ids take one of only a handful of XLA shapes (see _SEQ_BUCKET). The
            # extra columns are pad; every downstream slice is bounded by the true content
            # length (``max_seq_len`` below), so they are never read and never contribute.
            max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        out = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
        for i, seq in enumerate(sequences):
            if padding_side == "right":
                out[i, : seq.shape[0]] = seq
            else:
                out[i, max_len - seq.shape[0] :] = seq
        return out

    def _compute_logprobs_and_values(self, params, full_ids, full_mask):
        logits, values = self._forward(params, full_ids, full_mask)
        logprobs = logprobs_from_logits(logits[:, :-1, :], full_ids[:, 1:])
        return logprobs, values

    def _reference_params(self):
        """The KL anchor: an explicit reference model if one was loaded, else the
        adapter-disabled base -- never the live policy, whose KL against itself is zero.
        Mirrors torch's ``pretrained.disable_adapter()``."""
        # Local import, as in ppo_train: model.loader pulls in the flax model.
        from model.loader import disable_adapters, has_adapters

        if self.ref_params is not None:
            return self.ref_params
        if has_adapters(self.params):
            return disable_adapters(self.params)
        return self.params

    def _compute_ref_logprobs(self, full_ids, full_mask):
        ref_params = self._reference_params()
        logits, _ = self._forward(jax.lax.stop_gradient(ref_params), full_ids, full_mask)
        return logprobs_from_logits(logits[:, :-1, :], full_ids[:, 1:])

    def _iter_batches(self, batch_size):
        """Torch-free replacement for the shuffled, drop_last DataLoader."""
        n = len(self.train_dataset)
        order = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            features = [self.train_dataset[int(i)] for i in idx]
            yield self.data_collator(features)

    def ppo_train(self):
        """Main PPO training loop (faithful to torch ``ppo_train``)."""
        args = self.training_args
        np.random.seed(int(getattr(args, "seed", 42)))
        ppo_args = self.ppo_args
        batch_size = args.per_device_train_batch_size

        n = len(self.train_dataset)
        steps_per_epoch = max(1, (n // (batch_size * ppo_args.ppo_buffer_size)))
        max_steps = int(args.num_train_epochs * steps_per_epoch)
        num_train_epochs = int(math.ceil(args.num_train_epochs))

        optimizer = self._create_optimizer(max_steps)
        # Train in the configured compute dtype (mirrors torch_orig / bf16 — keeps a 7B
        # model on one GPU). Defaults to float32.
        _dt = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}.get(
            getattr(self.training_args, "compute_dtype", "float32") or "float32", jnp.float32
        )
        params = jax.tree_util.tree_map(lambda x: x.astype(_dt), self.params)
        # LoRA (mirror of torch_orig's PEFT LoRA): when ``model.loader`` put in-forward
        # adapters in the tree, train only those — the frozen base gets no cotangent and no
        # Adam state. The value head is the exception: ``load_model`` creates it fresh and
        # zero-initialised rather than reading it from the checkpoint, and torch attaches it
        # after ``get_peft_model`` so it keeps ``requires_grad``. Full finetuning (the grader
        # path) finds no adapters and is unchanged.
        from model.loader import combine, has_adapters, split

        _use_lora = has_adapters(params)
        _extra = ("v_head",)  # the PPO value head is trained alongside the adapters
        base_params, state = split(params, _extra) if _use_lora else (None, params)
        # Upcast the *trainable* params to float32 for the optimizer -- the "bf16 params are
        # upcast to fp32 for the optimizer" the __init__ dtype note promises. Adam is unsafe
        # in bfloat16: beta2=0.999 rounds to 1.0 in bf16, so (1 - beta2) underflows to 0, the
        # second-moment accumulator nu stays 0, and the bias-corrected nu/(1 - beta2**t) is a
        # 0/0 that makes the very first update NaN -- which then poisons every subsequent
        # rollout (this is the JAX PPO NaN divergence). The frozen base stays in the compute
        # dtype for memory, and the module casts these leaves back to the compute dtype for
        # every forward, so the rollout/loss numerics are otherwise unchanged. Under LoRA the
        # trainable set is just the adapters + value head, so the fp32 copy is tiny.
        state = jax.tree_util.tree_map(lambda x: jnp.asarray(x).astype(jnp.float32), state)
        params = combine(base_params, state, _extra) if _use_lora else state
        opt_state = optimizer.init(state)

        # JIT the optimizer update with params + opt_state donated so XLA reuses their
        # buffers instead of transiently doubling the Adam state (keeps large models on a
        # single GPU). The loss+grad stays eager (per-minibatch tensors + response slicing
        # remain plain Python, out of the compiled constants).
        import functools as _functools

        @_functools.partial(jax.jit, donate_argnums=(1, 2))
        def _apply_update(grads, opt_state, params):
            updates, new_opt = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt

        loss_meter, reward_meter, kl_meter = AverageMeter(), AverageMeter(), AverageMeter()
        g_losses, g_rewards, g_kls = [], [], []
        global_step = 0

        for epoch in range(num_train_epochs):
            for batch in self._iter_batches(batch_size * ppo_args.ppo_buffer_size):
                if global_step >= max_steps:
                    break

                # ------- ROLLOUT + REWARD ------- #
                all_queries, all_responses, all_rewards = [], [], []
                n_examples = int(np.asarray(batch["input_ids"]).shape[0])
                for idx in range(0, n_examples, batch_size):
                    mb = {
                        "input_ids": np.asarray(batch["input_ids"])[idx : idx + batch_size],
                        "attention_mask": np.asarray(batch["attention_mask"])[idx : idx + batch_size],
                        "labels": np.asarray(batch["labels"])[idx : idx + batch_size] if "labels" in batch else None,
                    }
                    gold_responses = None
                    if self.reward_model_type == "label" and mb.get("labels") is not None:
                        gold_responses = self._gold_response_tokens(mb["labels"])
                    queries, responses = self._generate_responses(mb)
                    rewards = self._compute_rewards(queries, responses, gold_responses=gold_responses)
                    all_queries.extend(queries)
                    all_responses.extend(responses)
                    all_rewards.append(jnp.atleast_1d(rewards))
                rewards = jnp.concatenate(all_rewards, axis=0)
                bs = len(all_queries)

                # ------- COMPUTE: logprobs / values / KL / GAE ------- #
                full_ids = self._pad_sequences(
                    [np.concatenate([q, r]) for q, r in zip(all_queries, all_responses)],
                    pad_value=getattr(self.tokenizer, "pad_token_id", 0) or 0,
                    padding_side="right",
                    pad_to_multiple=_SEQ_BUCKET,
                )
                pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
                full_mask = (full_ids != pad_id).astype(np.int64)
                full_ids_j = jnp.asarray(full_ids)
                full_mask_j = jnp.asarray(full_mask)

                query_lens = np.asarray([len(q) for q in all_queries])
                response_lens = np.asarray([len(r) for r in all_responses])
                # True content width (max over the batch of len(q)+len(r)) -- NOT the bucketed
                # tensor width. Every clamp/slice below (``min(..., max_seq_len - 1/2)``) uses
                # this, so it indexes exactly the tokens it did before the bucketing pad was
                # added; the trailing pad columns are masked out of the KL/loss and never read.
                max_seq_len = int((query_lens + response_lens).max())

                old_logprobs, old_values = self._compute_logprobs_and_values(params, full_ids_j, full_mask_j)
                ref_logprobs = self._compute_ref_logprobs(full_ids_j, full_mask_j)
                kl = old_logprobs - ref_logprobs

                per_token_rewards = -ppo_args.kl_coef * kl
                per_token_rewards = np.array(per_token_rewards)  # writable copy (jnp -> numpy)
                for i in range(bs):
                    end_pos = min(query_lens[i] - 1 + response_lens[i] - 1, max_seq_len - 2)
                    per_token_rewards[i, end_pos] += float(rewards[i])

                max_response_len = int(response_lens.max())
                resp_rewards = np.zeros((bs, max_response_len), dtype=np.float32)
                resp_values = np.zeros((bs, max_response_len), dtype=np.float32)
                resp_mask = np.zeros((bs, max_response_len), dtype=np.float32)
                resp_old_logprobs = np.zeros((bs, max_response_len), dtype=np.float32)
                old_values_np = np.asarray(old_values)
                old_logprobs_np = np.asarray(old_logprobs)
                for i in range(bs):
                    start = query_lens[i] - 1
                    end = min(start + response_lens[i], max_seq_len - 1)
                    actual = end - start
                    resp_rewards[i, :actual] = per_token_rewards[i, start:end]
                    resp_values[i, :actual] = old_values_np[i, start:end]
                    resp_mask[i, :actual] = 1.0
                    resp_old_logprobs[i, :actual] = old_logprobs_np[i, start:end]

                resp_rewards_j = jnp.asarray(resp_rewards)
                resp_mask_j = jnp.asarray(resp_mask)
                if ppo_args.ppo_whiten_rewards:
                    resp_rewards_j = masked_whiten(resp_rewards_j, resp_mask_j, shift_mean=False)
                advantages, returns = compute_gae(
                    resp_rewards_j, jnp.asarray(resp_values), resp_mask_j,
                    gamma=ppo_args.gamma, lam=ppo_args.lam,
                )
                advantages = masked_whiten(advantages, resp_mask_j)

                kl_meter.update(float(masked_mean(kl, jnp.asarray(full_mask[:, 1:] if full_mask.shape[1] > 1 else full_mask))), n=bs)
                reward_meter.update(float(jnp.mean(rewards)), n=bs)
                g_kls.append(kl_meter.val)
                g_rewards.append(reward_meter.val)

                # ------- OPTIMIZATION: PPO epochs ------- #
                # One scheduler step per rollout, as torch does -- not per minibatch.
                opt_state = _set_schedule_lr(opt_state, self._lr_schedule(global_step))
                resp_old_logprobs_j = jnp.asarray(resp_old_logprobs)
                resp_values_j = jnp.asarray(resp_values)
                for _ppo_epoch in range(ppo_args.ppo_epochs):
                    perm = np.random.permutation(bs)
                    for mb_start in range(0, bs, batch_size):
                        mb_inds = perm[mb_start : mb_start + batch_size]
                        if len(mb_inds) == 0:
                            continue
                        mb_ids = full_ids_j[mb_inds]
                        mb_mask = full_mask_j[mb_inds]
                        mb_q = query_lens[mb_inds]
                        mb_r = response_lens[mb_inds]

                        def loss_fn(s_):
                            p = combine(base_params, s_, _extra) if _use_lora else s_
                            logits, vpred = self._forward(p, mb_ids, mb_mask)
                            new_logprobs = logprobs_from_logits(logits[:, :-1, :], mb_ids[:, 1:])
                            mb_new = jnp.zeros((len(mb_inds), max_response_len), dtype=jnp.float32)
                            mb_vpred = jnp.zeros((len(mb_inds), max_response_len), dtype=jnp.float32)
                            for j in range(len(mb_inds)):
                                start = int(mb_q[j]) - 1
                                end = min(start + int(mb_r[j]), max_seq_len - 1)
                                actual = end - start
                                mb_new = mb_new.at[j, :actual].set(new_logprobs[j, start:end])
                                mb_vpred = mb_vpred.at[j, :actual].set(vpred[j, start:end])
                            pg_loss, _ = compute_ppo_loss(
                                mb_new, resp_old_logprobs_j[mb_inds], advantages[mb_inds],
                                resp_mask_j[mb_inds], ppo_args.cliprange,
                            )
                            vf_loss, _ = compute_value_loss(
                                mb_vpred, resp_values_j[mb_inds], returns[mb_inds],
                                resp_mask_j[mb_inds], ppo_args.cliprange_value,
                            )
                            return pg_loss + ppo_args.vf_coef * vf_loss

                        loss, grads = jax.value_and_grad(loss_fn)(state)
                        state, opt_state = _apply_update(grads, opt_state, state)
                        # Keep the full pytree in sync so the next rollout/logprobs use the
                        # updated adapters + value head. This is not just freshness: the
                        # rollout above reads ``params``, and ``_apply_update`` donates
                        # ``state`` -- whose leaves are the very arrays ``base_params`` still
                        # points at -- so without this rebind the next iteration hands XLA a
                        # donated-and-deleted buffer. The pre-LoRA loop got this for free by
                        # rebinding ``params`` in the update itself.
                        params = combine(base_params, state, _extra) if _use_lora else state
                        loss_meter.update(float(loss), n=len(mb_inds))
                        g_losses.append(float(loss))

                self.params = params
                global_step += 1
                torch_gc()

                if global_step % args.logging_steps == 0:
                    logs = {
                        "step": global_step,
                        "loss": round(loss_meter.avg, 4),
                        "reward": round(reward_meter.avg, 4),
                        "kl": round(kl_meter.avg, 4),
                        "epoch": round(global_step / steps_per_epoch, 2),
                    }
                    logger.info(str(logs))
                    loss_meter.reset()
                    reward_meter.reset()
                    kl_meter.reset()

            if global_step >= max_steps:
                break

        # ``params`` already carries the trained adapters (+ value head) folded back in.
        self.params = params
        # Write training metrics (JAX analogue of HF Trainer.save_metrics) so a run
        # harness can read train_results.json, mirroring run_sft/run_dpo.
        import json as _json

        _mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        out_dir = getattr(self.training_args, "output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        metrics = {
            "train_loss": _mean(g_losses),
            "train_steps": len(g_losses),
            "train_reward": _mean(g_rewards),
            "train_kl": _mean(g_kls),
            "global_step": global_step,
        }
        with open(os.path.join(out_dir, "train_results.json"), "w") as _f:
            _json.dump(metrics, _f, indent=2)
        logger.info("PPO training complete. " + str(metrics))

    def _greedy_generation_config(self):
        """A deterministic (greedy) copy of the generation config for evaluation."""
        gen = dict(self.generating_args.to_dict())
        gen["do_sample"] = False
        for key in ("temperature", "top_p", "top_k"):
            gen.pop(key, None)
        eos_ids = [getattr(self.tokenizer, "eos_token_id", None)]
        extra_eos = getattr(self.tokenizer, "additional_special_tokens_ids", None)
        if extra_eos:
            eos_ids += list(extra_eos)
        return GenerationConfig(
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=eos_ids,
            **gen,
        )

    def evaluate(self, eval_dataset=None, metric_key_prefix: str = "eval") -> dict:
        """Held-out validation metric: the negative label reward on the eval split.

        Returns ``-reward`` -- the negated mean generation-based label reward (mean
        token-F1 of the greedily decoded response against the gold answer); lower is
        better, so minimizing it maximizes the held-out reward. Falls back to the
        teacher-forced KL(policy || reference) when the reward is not label-based.
        JAX mirror of ``torch_orig``'s ``evaluate``.
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {}

        batch_size = self.training_args.per_device_eval_batch_size
        kl_meter = AverageMeter()
        reward_meter = AverageMeter()
        include_reward = self.reward_model_type == "label"
        greedy_config = self._greedy_generation_config() if include_reward else None
        n = len(dataset)
        for start in range(0, n, batch_size):
            features = [dataset[int(i)] for i in range(start, min(start + batch_size, n))]
            batch = self.data_collator(features)
            full_ids = jnp.asarray(np.asarray(batch["input_ids"]))
            full_mask = jnp.asarray(np.asarray(batch["attention_mask"]))
            labels = np.asarray(batch["labels"]) if "labels" in batch else None

            policy_logprobs, _ = self._compute_logprobs_and_values(self.params, full_ids, full_mask)
            ref_logprobs = self._compute_ref_logprobs(full_ids, full_mask)

            # logprob[:, i] scores full_ids[:, i + 1], so a position counts when its
            # target token is a response token; shift the mask by one to align.
            # When labels share input_ids' length the shifted != -100 mask lines up
            # directly; when image-token expansion makes input_ids (hence
            # policy_logprobs) longer than labels (VL), fall back to marking the
            # trailing n_resp positions as the response, mirroring torch_orig's
            # evaluate() mask.
            L = int(policy_logprobs.shape[1])
            if labels is not None and labels.shape[1] == int(full_ids.shape[1]):
                mask = jnp.asarray((labels[:, 1:] != -100).astype(np.float32))
            else:
                if labels is not None:
                    n_resp = (labels != -100).sum(axis=1)
                else:
                    n_resp = np.asarray(full_mask[:, 1:]).sum(axis=1)
                n_resp = np.minimum(n_resp, L)
                positions = np.arange(L)[None, :]
                mask = jnp.asarray((positions >= (L - n_resp)[:, None]).astype(np.float32))
            n_tokens = int(mask.sum())
            if n_tokens > 0:
                mean_kl = masked_mean(policy_logprobs - ref_logprobs, mask)
                # Weight by token count so the average is batch-size invariant.
                kl_meter.update(float(mean_kl), n=n_tokens)

            if include_reward and labels is not None:
                gen_batch = {
                    "input_ids": np.asarray(batch["input_ids"]),
                    "attention_mask": np.asarray(batch["attention_mask"]),
                    "labels": labels,
                }
                queries, responses = self._generate_responses(
                    gen_batch, generation_config=greedy_config, seed_offset=0
                )
                gold_responses = self._gold_response_tokens(labels)
                rewards = self._compute_rewards(queries, responses, gold_responses=gold_responses)
                reward_meter.update(float(jnp.mean(rewards)), n=int(np.asarray(rewards).shape[0]))

        mean_kl = kl_meter.avg
        mean_reward = reward_meter.avg
        eval_loss = -mean_reward if include_reward else mean_kl
        return {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_kl": mean_kl,
            f"{metric_key_prefix}_reward": mean_reward,
            f"{metric_key_prefix}_tokens": kl_meter.count,
        }

    def log_metrics(self, split, metrics):
        logger.info("***** %s metrics *****", split)
        for key, value in metrics.items():
            logger.info("  %s = %s", key, value)

    def save_metrics(self, split, metrics):
        out_dir = getattr(self.training_args, "output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{split}_results.json"), "w") as file:
            json.dump(metrics, file, indent=2)

    def save_model(self, output_dir: Optional[str] = None):
        """Save the flax params to ``jax_params.npz`` (torch-free)."""
        if output_dir is None:
            output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        flat = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(self.params):
            key = "/".join(str(getattr(k, "key", k)) for k in path)
            flat[key] = np.asarray(leaf)
        np.savez(os.path.join(output_dir, "jax_params.npz"), **flat)
        logger.info(f"Model saved to {output_dir}")
