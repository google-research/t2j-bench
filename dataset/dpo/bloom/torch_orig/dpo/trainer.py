# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone DPO Trainer for Gemma3.

Implements Direct Preference Optimization without dependency on TRL or HF Trainer.
Supports:
- Standard DPO (sigmoid) with reference model
- ORPO (reference-free, odds ratio)
- SimPO (reference-free, simplified)
- Auxiliary SFT loss (ftx)
- LoRA adapter disabling for reference model
"""

import json
import logging
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class TrainResult:
    """Result of training."""
    metrics: dict[str, Any] = field(default_factory=dict)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a cosine LR schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a linear LR schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            ),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


SCHEDULER_MAP = {
    "cosine": get_cosine_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
}


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities of labels under the given logits.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with IGNORE_INDEX for non-target tokens
        label_pad_token_id: Token ID to mask (default: IGNORE_INDEX)

    Returns:
        logps: (batch_size,) sum of log probabilities
        valid_length: (batch_size,) number of non-masked tokens
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :].float()  # cast to float32 for numerical stability
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    valid_length = loss_mask.sum(-1)
    logps = (per_token_logps * loss_mask).sum(-1)
    return logps, valid_length


class DPOTrainer:
    """Standalone DPO trainer for Qwen3-VL."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        args,  # TrainingArguments
        dpo_args,  # DPOArguments
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.args = args
        self.dpo_args = dpo_args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # DPO hyperparameters
        self.beta = dpo_args.pref_beta
        self.loss_type = dpo_args.pref_loss
        self.ftx_gamma = dpo_args.pref_ftx
        self.label_smoothing = dpo_args.dpo_label_smoothing
        self.simpo_gamma = dpo_args.simpo_gamma
        self.use_ref_model = dpo_args.use_ref_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        if not isinstance(model, nn.DataParallel):
            self.model = self.model.to(self.device)

        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()

        # Disable dropout
        if dpo_args.disable_dropout:
            self._disable_dropout(self.model)
            if self.ref_model is not None:
                self._disable_dropout(self.ref_model)

        self.global_step = 0
        self.epoch = 0.0

    def _disable_dropout(self, model: nn.Module):
        """Disable all dropout layers in the model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _get_eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=SequentialSampler(self.eval_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _create_optimizer(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped, lr=self.args.learning_rate)

    def _create_scheduler(self, optimizer, num_training_steps):
        num_warmup_steps = self.args.warmup_steps
        if num_warmup_steps == 0 and self.args.warmup_ratio > 0:
            num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        scheduler_fn = SCHEDULER_MAP.get(
            self.args.lr_scheduler_type, get_cosine_schedule_with_warmup
        )
        return scheduler_fn(optimizer, num_warmup_steps, num_training_steps)

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass on concatenated chosen+rejected batch.

        The batch has 2*n examples: first n are chosen, last n are rejected.

        Returns:
            chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps_avg
        """
        labels = batch.pop("labels")

        # Forward pass
        all_logits = model(
            **batch, return_dict=True, use_cache=False
        ).logits

        # Compute log probabilities
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=labels)

        # For ORPO/SimPO: use average log probs
        if self.loss_type in ("orpo", "simpo"):
            all_logps_normalized = all_logps / valid_length
        else:
            all_logps_normalized = all_logps

        # Split into chosen and rejected
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps_normalized.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type in ("orpo", "simpo"):
            chosen_logps_avg = chosen_logps
        else:
            chosen_logps_avg = all_logps[:batch_size] / chosen_length

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps_avg,
        }

    def compute_reference_log_probs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute reference model log probabilities."""
        if not self.use_ref_model:
            return None, None

        if self.ref_model is None:
            # LoRA case: disable adapter
            ref_model = self.model
            if (
                hasattr(self.model, "disable_adapter")
                and self.model.__class__.__name__ == "PeftModel"
            ):
                ref_context = self.model.disable_adapter()
            else:
                ref_context = nullcontext()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        # Clone batch to avoid mutation
        ref_batch = {
            k: v.detach().clone() if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.no_grad(), ref_context:
            ref_output = self.concatenated_forward(ref_model, ref_batch)
            return ref_output["chosen_logps"], ref_output["rejected_logps"]

    def dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute standard DPO (sigmoid) loss."""
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        # Reward computation
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        logits = chosen_logratios - rejected_logratios

        if self.label_smoothing > 0:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(self.beta * logits)

        return losses, chosen_rewards, rejected_rewards

    def orpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ORPO (Odds Ratio Preference Optimization) loss."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps))
            - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        return sft_loss + self.beta * odds_ratio_loss

    def simpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SimPO loss."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        return -F.logsigmoid(self.beta * logits)

    def compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: Optional[torch.Tensor],
        ref_rejected_logps: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute preference loss based on loss type."""
        if not self.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.orpo_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

            chosen_rewards = self.beta * policy_chosen_logps.detach()
            rejected_rewards = self.beta * policy_rejected_logps.detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        batch: dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss and metrics for a single batch."""
        metrics = {}

        # Clone batch for reference model forward
        batch_for_ref = {
            k: v.detach().clone() if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # Policy model forward
        model_output = self.concatenated_forward(self.model, batch)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]
        policy_chosen_logits = model_output["chosen_logits"]
        policy_rejected_logits = model_output["rejected_logits"]
        policy_chosen_logps_avg = model_output["chosen_logps_avg"]

        # Reference model forward
        ref_chosen_logps, ref_rejected_logps = self.compute_reference_log_probs(
            batch_for_ref
        )

        # Compute preference loss
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        # Auxiliary SFT loss
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses = losses + self.ftx_gamma * sft_loss

        # Log metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (
            (chosen_rewards > rejected_rewards).float().mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()

        return losses.mean(), metrics

    def train(self, resume_from_checkpoint=None) -> TrainResult:
        """Run DPO training loop."""
        train_dataloader = self._get_train_dataloader()
        num_examples = len(self.train_dataset)
        total_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = max(
            len(train_dataloader) // self.args.gradient_accumulation_steps, 1
        )
        num_epochs = int(math.ceil(self.args.num_train_epochs))
        max_steps = int(num_update_steps_per_epoch * self.args.num_train_epochs)

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, max_steps)

        logger.info("***** Running DPO training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  DPO loss type = {self.loss_type}")
        logger.info(f"  DPO beta = {self.beta}")
        logger.info(f"  Use reference model = {self.use_ref_model}")

        if self.args.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.model.train()
        self.model.zero_grad()
        total_loss = 0.0
        logging_loss = 0.0
        all_metrics = defaultdict(list)
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                loss, metrics = self.get_batch_loss_metrics(batch, train_eval="train")

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                total_loss += loss.item()

                for k, v in metrics.items():
                    all_metrics[k].append(v)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(train_dataloader):
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        elapsed = time.time() - start_time
                        avg_loss = (total_loss - logging_loss) / self.args.logging_steps
                        lr = scheduler.get_last_lr()[0]
                        avg_metrics = {
                            k: sum(v[-self.args.logging_steps:]) / len(v[-self.args.logging_steps:])
                            for k, v in all_metrics.items()
                        }
                        logger.info(
                            f"Step {self.global_step}/{max_steps} | "
                            f"Epoch {epoch + 1}/{num_epochs} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                            f"Acc: {avg_metrics.get('rewards/accuracies', 0):.3f} | "
                            f"Margin: {avg_metrics.get('rewards/margins', 0):.3f} | "
                            f"Time: {elapsed:.1f}s"
                        )
                        logging_loss = total_loss

                    # Save checkpoint
                    if (
                        self.args.save_steps > 0
                        and self.global_step % self.args.save_steps == 0
                    ):
                        self._save_checkpoint(f"checkpoint-{self.global_step}")

                    if self.global_step >= max_steps:
                        break

            if self.global_step >= max_steps:
                break

        metrics = {
            "train_loss": total_loss / max(self.global_step, 1),
            "train_runtime": time.time() - start_time,
            "train_steps": self.global_step,
            "epoch": self.args.num_train_epochs,
        }
        # Add average DPO metrics
        for k, v in all_metrics.items():
            metrics[f"train_{k}"] = sum(v) / len(v) if v else 0

        return TrainResult(metrics=metrics)

    @torch.no_grad()
    def evaluate(self, metric_key_prefix="eval") -> dict[str, float]:
        """Run evaluation."""
        if self.eval_dataset is None:
            logger.warning("No eval dataset provided, skipping evaluation.")
            return {}

        eval_dataloader = self._get_eval_dataloader()
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        eval_metrics = defaultdict(list)

        for batch in eval_dataloader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            loss, metrics = self.get_batch_loss_metrics(batch, train_eval="eval")
            total_loss += loss.item()
            total_steps += 1
            for k, v in metrics.items():
                eval_metrics[k].append(v)

        self.model.train()
        avg_loss = total_loss / max(total_steps, 1)
        result = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_steps": total_steps,
        }
        for k, v in eval_metrics.items():
            result[f"{metric_key_prefix}_{k}"] = sum(v) / len(v) if v else 0

        return result

    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        output_dir = os.path.join(self.args.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(output_dir)
        logger.info(f"Saved checkpoint to {output_dir}")

        # Manage save_total_limit
        if self.args.save_total_limit is not None and self.args.save_total_limit > 0:
            checkpoints = sorted(
                [
                    d
                    for d in os.listdir(self.args.output_dir)
                    if d.startswith("checkpoint-")
                ],
                key=lambda x: int(x.split("-")[-1]),
            )
            if len(checkpoints) > self.args.save_total_limit:
                import shutil
                for old_ckpt in checkpoints[
                    : len(checkpoints) - self.args.save_total_limit
                ]:
                    shutil.rmtree(
                        os.path.join(self.args.output_dir, old_ckpt),
                        ignore_errors=True,
                    )

    def save_model(self, output_dir: Optional[str] = None):
        """Save model weights."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if (
            hasattr(self.model, "save_pretrained")
            and self.model.__class__.__name__ == "PeftModel"
        ):
            self.model.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            return

        unwrapped_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(output_dir)
        else:
            torch.save(
                unwrapped_model.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin"),
            )

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
