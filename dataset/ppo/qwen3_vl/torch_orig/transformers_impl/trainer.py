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
- Standard PyTorch training loop
- Gradient accumulation
- Learning rate scheduling with warmup
- Logging (console + file)
- Checkpointing
- Evaluation
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of training."""
    metrics: dict[str, Any] = field(default_factory=dict)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a cosine LR schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a linear LR schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


SCHEDULER_MAP = {
    "cosine": get_cosine_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
}


class Trainer:
    """Minimal trainer for SFT with multimodal support."""

    def __init__(
        self,
        model: nn.Module,
        args,  # TrainingArguments from config.py
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(model, nn.DataParallel):
            self.model = self.model.to(self.device)

        self.global_step = 0
        self.epoch = 0.0

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

        scheduler_fn = SCHEDULER_MAP.get(self.args.lr_scheduler_type, get_cosine_schedule_with_warmup)
        return scheduler_fn(optimizer, num_warmup_steps, num_training_steps)

    def train(self, resume_from_checkpoint=None) -> TrainResult:
        train_dataloader = self._get_train_dataloader()
        num_examples = len(self.train_dataset)
        total_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(len(train_dataloader) // self.args.gradient_accumulation_steps, 1)
        num_epochs = int(math.ceil(self.args.num_train_epochs))
        max_steps = int(num_update_steps_per_epoch * self.args.num_train_epochs)

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, max_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}")

        if self.args.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.model.train()
        self.model.zero_grad()
        total_loss = 0.0
        logging_loss = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                total_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        elapsed = time.time() - start_time
                        avg_loss = (total_loss - logging_loss) / self.args.logging_steps
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step}/{max_steps} | "
                            f"Epoch {epoch + 1}/{num_epochs} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )
                        logging_loss = total_loss

                    # Save checkpoint
                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
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

        for batch in eval_dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            total_loss += loss.item()
            total_steps += 1

        self.model.train()
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

        # Save PEFT model if applicable
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(output_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                return
        except ImportError:
            pass

        # Save full model
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(output_dir)
        else:
            torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

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
