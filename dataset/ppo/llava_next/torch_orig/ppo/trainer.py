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

"""Standalone PPO trainer for generic torch_orig models.

No TRL dependency -- implements the full PPO training loop from scratch,
following the same structure as LlamaFactory's CustomPPOTrainer.
"""

import importlib.util
import logging
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_IMPL_ROOT = Path(__file__).resolve().parents[1]
_PPO_COMPAT_PATH = _IMPL_ROOT / "transformers_impl" / "ppo_compat.py"
_PPO_COMPAT_MODULE = "_generic_ppo_torch_orig_ppo_compat"


def _load_local_ppo_compat():
    if _PPO_COMPAT_MODULE in sys.modules:
        return sys.modules[_PPO_COMPAT_MODULE]

    spec = importlib.util.spec_from_file_location(_PPO_COMPAT_MODULE, _PPO_COMPAT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local PPO compat module from {_PPO_COMPAT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PPO_COMPAT_MODULE] = module
    spec.loader.exec_module(module)
    return module


_ppo_compat = _load_local_ppo_compat()
GenerationConfig = _ppo_compat.GenerationConfig
get_scheduler = _ppo_compat.get_scheduler

from .ppo_utils import (
    compute_gae,
    compute_ppo_loss,
    compute_value_loss,
    get_rewards_from_server,
    logprobs_from_logits,
    masked_mean,
    masked_whiten,
    torch_gc,
)

if TYPE_CHECKING:
    from config import FinetuningArguments, GeneratingArguments, PPOArguments, TrainingArguments
    from model.loader import ModelWithValueHead

logger = logging.getLogger(__name__)


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


class StandalonePPOTrainer:
    """Standalone PPO trainer for generic torch_orig models, independent of TRL.

    Implements the PPO training loop:
    1. Rollout: generate responses from prompts
    2. Reward: score responses using reward model (API or full model)
    3. Compute: old logprobs, ref logprobs, KL penalties, GAE advantages
    4. Optimize: PPO policy + value loss with clipping
    """

    def __init__(
        self,
        model: "ModelWithValueHead",
        ref_model: Optional["ModelWithValueHead"],
        reward_model,  # str (API URL) or ModelWithValueHead
        tokenizer,
        processor,
        data_collator,
        train_dataset,
        training_args: "TrainingArguments",
        ppo_args: "PPOArguments",
        generating_args: "GeneratingArguments",
        finetuning_args: "FinetuningArguments",
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.training_args = training_args
        self.ppo_args = ppo_args
        self.generating_args = generating_args
        self.finetuning_args = finetuning_args

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Move models to device
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
        if isinstance(self.reward_model, torch.nn.Module):
            self.reward_model.to(self.device)

        # Generation config
        eos_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "additional_special_tokens_ids"):
            eos_ids += self.tokenizer.additional_special_tokens_ids
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            **generating_args.to_dict(),
        )

        # Compute dtype for mixed precision
        self.compute_dtype = getattr(model.config, "torch_dtype", torch.float32)
        if training_args.bf16:
            self.compute_dtype = torch.bfloat16
        elif training_args.fp16:
            self.compute_dtype = torch.float16

        # Enable gradient checkpointing if requested
        if training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        decay_params = []
        nodecay_params = []
        no_decay_names = {"bias", "layer_norm", "layernorm", "ln_"}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay_names):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.training_args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.training_args.learning_rate,
        )
        return optimizer

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        warmup_steps = self.training_args.warmup_steps
        if warmup_steps == 0 and self.training_args.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.training_args.warmup_ratio)

        scheduler = get_scheduler(
            self.training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    @torch.no_grad()
    def _generate_responses(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate responses given query batch.

        Returns lists of (query, response) tensors with padding removed.
        """
        self.model.eval()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")

        # Collect multimodal inputs for generation
        gen_kwargs = {}
        for key in ["pixel_values", "token_type_ids"]:
            if key in batch:
                gen_kwargs[key] = batch[key]

        # For supervised-style PPO batches, derive the rollout prompt from the
        # point where target labels begin. This keeps the stored batch format
        # compatible with SFT-style encoding while still generating from the
        # prompt boundary.
        prompt_inputs = []
        prompt_masks = []
        prompt_token_types = [] if "token_type_ids" in gen_kwargs else None
        for row_idx in range(input_ids.size(0)):
            nonpad = attention_mask[row_idx].nonzero()
            if len(nonpad) > 0:
                start_index = nonpad[0].item()
                end_index = nonpad[-1].item() + 1
            else:
                start_index = 0
                end_index = input_ids.size(1)

            prompt_end = end_index
            if labels is not None:
                supervised_positions = (labels[row_idx] != -100).nonzero()
                if len(supervised_positions) > 0:
                    prompt_end = max(start_index + 1, supervised_positions[0].item())

            row_input_ids = input_ids[row_idx, start_index:prompt_end]
            row_attention_mask = attention_mask[row_idx, start_index:prompt_end]
            if row_input_ids.numel() == 0:
                row_input_ids = input_ids[row_idx, start_index:end_index]
                row_attention_mask = attention_mask[row_idx, start_index:end_index]

            prompt_inputs.append(row_input_ids)
            prompt_masks.append(row_attention_mask)
            if prompt_token_types is not None:
                prompt_token_types.append(gen_kwargs["token_type_ids"][row_idx, start_index:prompt_end])

        max_prompt_len = max(seq.size(0) for seq in prompt_inputs)
        padded_input_ids = input_ids.new_full((len(prompt_inputs), max_prompt_len), self.tokenizer.pad_token_id)
        padded_attention_mask = attention_mask.new_zeros((len(prompt_inputs), max_prompt_len))
        padded_token_types = (
            gen_kwargs["token_type_ids"].new_zeros((len(prompt_inputs), max_prompt_len))
            if prompt_token_types is not None
            else None
        )
        left_padding = getattr(self.tokenizer, "padding_side", "right") == "left"
        for row_idx, row_input_ids in enumerate(prompt_inputs):
            row_len = row_input_ids.size(0)
            offset = max_prompt_len - row_len if left_padding else 0
            padded_input_ids[row_idx, offset : offset + row_len] = row_input_ids
            padded_attention_mask[row_idx, offset : offset + row_len] = prompt_masks[row_idx]
            if padded_token_types is not None:
                padded_token_types[row_idx, offset : offset + row_len] = prompt_token_types[row_idx]

        input_ids = padded_input_ids
        attention_mask = padded_attention_mask
        if padded_token_types is not None:
            gen_kwargs["token_type_ids"] = padded_token_types

        # Generate
        generate_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            **gen_kwargs,
        )

        # Split into queries and responses, remove padding
        query = input_ids.detach().cpu()
        response = generate_output[:, input_ids.size(-1):].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()
            if len(query_start_index) > 0:
                query_start_index = query_start_index[0].item()
            else:
                query_start_index = 0

            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()
            if len(response_indexes) == 0:
                response_length = 1  # allow empty response
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                response_length = response_indexes[-1].item() + 2  # include eos
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])
            responses.append(response[i, :response_length])

        return queries, responses

    @torch.no_grad()
    def _compute_rewards(
        self,
        queries: list[torch.Tensor],
        responses: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute reward scores for query-response pairs.

        Returns: (batch_size,) tensor of rewards on CPU.
        """
        if isinstance(self.reward_model, str):
            # API-based reward model
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            rewards = get_rewards_from_server(self.reward_model, messages)
            return rewards
        elif isinstance(self.reward_model, torch.nn.Module):
            # Full reward model
            input_ids = self._pad_sequences(
                [torch.cat((q, r), dim=-1) for q, r in zip(queries, responses)],
                pad_value=self.tokenizer.pad_token_id,
                padding_side="left",
            ).to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            with torch.amp.autocast(self.device.type, dtype=self.compute_dtype):
                _, _, values = self.reward_model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            seq_lengths = attention_mask.sum(dim=-1, keepdim=True) - 1
            rewards = values.gather(dim=-1, index=seq_lengths).squeeze(-1)
            return rewards.float().cpu()
        else:
            raise ValueError(f"Unknown reward model type: {type(self.reward_model)}")

    def _pad_sequences(
        self,
        sequences: list[torch.Tensor],
        pad_value: int,
        padding_side: str = "right",
    ) -> torch.Tensor:
        """Pad a list of variable-length tensors to the same length."""
        max_len = max(s.size(0) for s in sequences)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            if padding_side == "right":
                padded[i, :seq.size(0)] = seq
            else:
                padded[i, max_len - seq.size(0):] = seq
        return padded

    @torch.no_grad()
    def _compute_logprobs_and_values(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probabilities and values for the full sequence.

        Returns:
            logprobs: (batch, seq_len-1) log-probs of each next token
            values: (batch, seq_len) value predictions
        """
        with torch.amp.autocast(self.device.type, dtype=self.compute_dtype):
            logits, _, values = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        return logprobs, values

    @torch.no_grad()
    def _compute_ref_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute reference model log-probabilities."""
        if self.ref_model is not None:
            # Separate reference model
            with torch.amp.autocast(self.device.type, dtype=self.compute_dtype):
                logits, _, _ = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **kwargs,
                )
        else:
            # LoRA: disable adapter to get base model behavior
            pretrained = self.model.pretrained_model
            with pretrained.disable_adapter():
                with torch.amp.autocast(self.device.type, dtype=self.compute_dtype):
                    base_output = pretrained(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=False,
                        **kwargs,
                    )
                    logits = base_output.logits

        ref_logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        return ref_logprobs

    def _compute_token_type_ids(
        self,
        input_ids: torch.Tensor,
    ) -> dict:
        """Compute token_type_ids for generic torch_orig models (0 for text, 1 for image tokens)."""
        extra_kwargs = {}
        image_token_id = getattr(self.processor, "image_token_id",
                                 getattr(self.tokenizer, "image_token_id", None))
        if image_token_id is not None:
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids == image_token_id] = 1
            extra_kwargs["token_type_ids"] = token_type_ids
        return extra_kwargs

    def ppo_train(self):
        """Main PPO training loop."""
        args = self.training_args
        ppo_args = self.ppo_args
        batch_size = args.per_device_train_batch_size

        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size * ppo_args.ppo_buffer_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )

        # Compute training steps
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = max_steps
        else:
            max_steps = int(args.num_train_epochs * len(dataloader))
            num_train_epochs = int(math.ceil(args.num_train_epochs))

        # Create optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, max_steps)

        # Logging
        logger.info("***** Running PPO training *****")
        logger.info(f"  Num examples = {len(self.train_dataset):,}")
        logger.info(f"  Num epochs = {num_train_epochs}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  PPO buffer size = {ppo_args.ppo_buffer_size}")
        logger.info(f"  PPO epochs per batch = {ppo_args.ppo_epochs}")
        logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
        logger.info(f"  Max training steps = {max_steps}")
        logger.info(f"  Learning rate = {args.learning_rate}")

        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        kl_meter = AverageMeter()
        global_step = 0

        for epoch in range(num_train_epochs):
            dataiter = iter(dataloader)
            for step_in_epoch, batch in enumerate(
                tqdm(dataiter, desc=f"Epoch {epoch}", disable=False)
            ):
                if global_step >= max_steps:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Extract multimodal inputs from batch
                batch_mm_inputs = {}
                for key in ["pixel_values", "token_type_ids"]:
                    if key in batch:
                        batch_mm_inputs[key] = batch[key]

                # =========================================================
                # ROLLOUT PHASE: Generate responses and compute rewards
                # =========================================================
                self.model.eval()

                all_queries, all_responses, all_rewards = [], [], []

                for idx in range(0, batch["input_ids"].size(0), batch_size):
                    mini_batch = {
                        "input_ids": batch["input_ids"][idx:idx + batch_size],
                        "attention_mask": batch["attention_mask"][idx:idx + batch_size],
                    }
                    for key in batch_mm_inputs:
                        if key in batch:
                            mini_batch[key] = batch[key] if batch[key].size(0) <= batch_size else batch[key][idx:idx + batch_size]

                    queries, responses = self._generate_responses(mini_batch)
                    rewards = self._compute_rewards(queries, responses)
                    all_queries.extend(queries)
                    all_responses.extend(responses)
                    if rewards.dim() == 0:
                        all_rewards.append(rewards.unsqueeze(0))
                    else:
                        all_rewards.append(rewards)

                rewards = torch.cat(all_rewards, dim=0)
                bs = len(all_queries)

                # =========================================================
                # COMPUTE PHASE: logprobs, ref logprobs, KL, advantages
                # =========================================================
                # Pad query+response sequences
                full_ids = self._pad_sequences(
                    [torch.cat((q, r), dim=-1) for q, r in zip(all_queries, all_responses)],
                    pad_value=self.tokenizer.pad_token_id,
                    padding_side="right",
                ).to(self.device)
                full_mask = (full_ids != self.tokenizer.pad_token_id).long().to(self.device)

                # Compute query lengths for masking
                query_lens = torch.tensor([len(q) for q in all_queries])
                response_lens = torch.tensor([len(r) for r in all_responses])
                max_seq_len = full_ids.size(1)

                # Create response mask: 1 for response tokens, 0 for query/padding
                response_mask = torch.zeros(bs, max_seq_len - 1, device=self.device)
                for i in range(bs):
                    q_len = query_lens[i].item()
                    r_len = response_lens[i].item()
                    start = q_len - 1
                    end = start + r_len
                    end = min(end, max_seq_len - 1)
                    response_mask[i, start:end] = 1.0

                # Compute token_type_ids for the full sequence
                extra_kwargs = self._compute_token_type_ids(full_ids)

                # Compute old logprobs and values
                old_logprobs, old_values = self._compute_logprobs_and_values(
                    self.model, full_ids, full_mask, response_mask, **extra_kwargs
                )

                # Compute reference logprobs
                ref_logprobs = self._compute_ref_logprobs(
                    full_ids, full_mask, **extra_kwargs
                )

                # Per-token KL divergence
                kl = (old_logprobs - ref_logprobs).detach()

                # Per-token rewards: -kl_coef * KL + final reward at end of response
                per_token_rewards = -ppo_args.kl_coef * kl
                for i in range(bs):
                    q_len = query_lens[i].item()
                    r_len = response_lens[i].item()
                    end_pos = q_len - 1 + r_len - 1
                    end_pos = min(end_pos, max_seq_len - 2)
                    per_token_rewards[i, end_pos] += rewards[i].to(self.device)

                # Extract response-only rewards and values for GAE
                max_response_len = response_lens.max().item()
                resp_rewards = torch.zeros(bs, max_response_len, device=self.device)
                resp_values = torch.zeros(bs, max_response_len, device=self.device)
                resp_mask = torch.zeros(bs, max_response_len, device=self.device)
                resp_old_logprobs = torch.zeros(bs, max_response_len, device=self.device)

                for i in range(bs):
                    q_len = query_lens[i].item()
                    r_len = response_lens[i].item()
                    start = q_len - 1
                    end = start + r_len
                    end = min(end, max_seq_len - 1)
                    actual_len = end - start
                    resp_rewards[i, :actual_len] = per_token_rewards[i, start:end]
                    resp_values[i, :actual_len] = old_values[i, start:end]
                    resp_mask[i, :actual_len] = 1.0
                    resp_old_logprobs[i, :actual_len] = old_logprobs[i, start:end]

                # Whiten rewards if configured
                if ppo_args.ppo_whiten_rewards:
                    resp_rewards = masked_whiten(resp_rewards, resp_mask, shift_mean=False)

                # GAE
                advantages, returns = compute_gae(
                    resp_rewards, resp_values, resp_mask,
                    gamma=ppo_args.gamma, lam=ppo_args.lam,
                )
                # Whiten advantages
                advantages = masked_whiten(advantages, resp_mask)

                # Track KL
                mean_kl = masked_mean(kl, response_mask).item()
                kl_meter.update(mean_kl, n=bs)
                reward_meter.update(rewards.mean().item(), n=bs)

                # =========================================================
                # OPTIMIZATION PHASE: PPO update
                # =========================================================
                self.model.train()
                torch_gc()

                for ppo_epoch in range(ppo_args.ppo_epochs):
                    indices = np.random.permutation(bs)
                    for mb_start in range(0, bs, batch_size):
                        mb_inds = indices[mb_start:mb_start + batch_size]
                        if len(mb_inds) == 0:
                            continue

                        mb_ids = full_ids[mb_inds]
                        mb_mask = full_mask[mb_inds]

                        # Filter extra_kwargs for mini-batch
                        mb_kwargs = {}
                        for key, val in extra_kwargs.items():
                            if torch.is_tensor(val):
                                if val.size(0) == bs:
                                    mb_kwargs[key] = val[mb_inds]
                                else:
                                    mb_kwargs[key] = val
                            else:
                                mb_kwargs[key] = val

                        # Forward pass
                        with torch.amp.autocast(self.device.type, dtype=self.compute_dtype):
                            logits, _, vpred = self.model(
                                input_ids=mb_ids,
                                attention_mask=mb_mask,
                                use_cache=False,
                                **mb_kwargs,
                            )

                        new_logprobs = logprobs_from_logits(logits[:, :-1, :], mb_ids[:, 1:])

                        # Extract response-only new logprobs and values
                        mb_new_logprobs = torch.zeros(len(mb_inds), max_response_len, device=self.device)
                        mb_vpred = torch.zeros(len(mb_inds), max_response_len, device=self.device)

                        for j, orig_idx in enumerate(mb_inds):
                            q_len = query_lens[orig_idx].item()
                            r_len = response_lens[orig_idx].item()
                            start = q_len - 1
                            end = min(start + r_len, max_seq_len - 1)
                            actual_len = end - start
                            mb_new_logprobs[j, :actual_len] = new_logprobs[j, start:end]
                            mb_vpred[j, :actual_len] = vpred[j, start:end]

                        # Compute losses
                        pg_loss, clipfrac = compute_ppo_loss(
                            mb_new_logprobs,
                            resp_old_logprobs[mb_inds],
                            advantages[mb_inds],
                            resp_mask[mb_inds],
                            ppo_args.cliprange,
                        )
                        vf_loss, vf_clipfrac = compute_value_loss(
                            mb_vpred,
                            resp_values[mb_inds],
                            returns[mb_inds],
                            resp_mask[mb_inds],
                            ppo_args.cliprange_value,
                        )

                        total_loss = pg_loss + ppo_args.vf_coef * vf_loss

                        # Backward
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            args.max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                        loss_meter.update(total_loss.item(), n=len(mb_inds))

                scheduler.step()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logs = {
                        "step": global_step,
                        "loss": round(loss_meter.avg, 4),
                        "reward": round(reward_meter.avg, 4),
                        "kl": round(kl_meter.avg, 4),
                        "lr": current_lr,
                        "epoch": round(global_step / len(dataloader), 2),
                    }
                    logger.info(str(logs))
                    tqdm.write(str(logs))
                    loss_meter.reset()
                    reward_meter.reset()
                    kl_meter.reset()

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    self.save_model(
                        os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    )

            if global_step >= max_steps:
                break

        logger.info("PPO training complete.")

    def save_model(self, output_dir: Optional[str] = None):
        """Save model checkpoint."""
        if output_dir is None:
            output_dir = self.training_args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Save model state dict
        state_dict = self.model.state_dict()
        save_path = os.path.join(output_dir, "model.safetensors")
        try:
            from safetensors.torch import save_file
            save_file(state_dict, save_path)
        except ImportError:
            save_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(state_dict, save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")
