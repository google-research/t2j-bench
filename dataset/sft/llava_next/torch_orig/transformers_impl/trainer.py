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

"""Minimal Trainer implementation for LLaVA-Next SFT."""
import json
import logging
import math
import os
import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


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

    def get_train_dataloader(self):
        batch_size = getattr(self.args, "per_device_train_batch_size", 1)
        sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=getattr(self.args, "dataloader_num_workers", 0),
            pin_memory=True,
        )

    def get_eval_dataloader(self, dataset=None):
        dataset = dataset or self.eval_dataset
        if dataset is None:
            return None
        batch_size = getattr(self.args, "per_device_eval_batch_size", 1)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=getattr(self.args, "dataloader_num_workers", 0),
        )

    def create_optimizer(self):
        lr = getattr(self.args, "learning_rate", 1e-4)
        wd = getattr(self.args, "weight_decay", 0.0)
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def create_scheduler(self, optimizer, num_training_steps):
        warmup_steps = getattr(self.args, "warmup_steps", 0)
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.0)
        if warmup_steps == 0 and warmup_ratio > 0:
            warmup_steps = int(num_training_steps * warmup_ratio)

        scheduler_type = getattr(self.args, "lr_scheduler_type", "cosine")

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if scheduler_type == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:  # linear
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train(self, resume_from_checkpoint=None):
        """Run training loop."""
        model = self.model
        args = self.args
        train_dataloader = self.get_train_dataloader()

        num_epochs = getattr(args, "num_train_epochs", 3.0)
        grad_accum_steps = getattr(args, "gradient_accumulation_steps", 1)
        max_grad_norm = getattr(args, "max_grad_norm", 1.0)
        logging_steps = getattr(args, "logging_steps", 10)
        save_steps = getattr(args, "save_steps", 500)
        output_dir = getattr(args, "output_dir", "output")
        gradient_checkpointing = getattr(args, "gradient_checkpointing", False)

        num_steps_per_epoch = len(train_dataloader) // grad_accum_steps
        total_steps = int(num_steps_per_epoch * num_epochs)

        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer, total_steps)

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        model.train()
        global_step = 0
        total_loss = 0.0
        cumulative_loss = 0.0
        start_time = time.time()

        logger.info(f"***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {num_epochs}")
        logger.info(f"  Batch size = {getattr(args, 'per_device_train_batch_size', 1)}")
        logger.info(f"  Gradient Accumulation steps = {grad_accum_steps}")
        logger.info(f"  Total optimization steps = {total_steps}")

        for epoch in range(int(num_epochs)):
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / grad_accum_steps
                loss.backward()

                total_loss += loss.item()
                cumulative_loss += loss.item()

                if (step + 1) % grad_accum_steps == 0:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        avg_loss = total_loss / logging_steps
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Step {global_step}/{total_steps}, "
                            f"Loss: {avg_loss:.4f}, "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}, "
                            f"Time: {elapsed:.1f}s"
                        )
                        total_loss = 0.0

                    if save_steps > 0 and global_step % save_steps == 0:
                        self.save_model(os.path.join(output_dir, f"checkpoint-{global_step}"))

        avg_train_loss = cumulative_loss / max(global_step, 1)
        metrics = {"train_loss": avg_train_loss, "train_steps": global_step}
        return TrainOutput(global_step=global_step, training_loss=avg_train_loss, metrics=metrics)

    def evaluate(self, metric_key_prefix="eval"):
        """Run evaluation."""
        if self.eval_dataset is None:
            return {}

        model = self.model
        model.eval()
        eval_dataloader = self.get_eval_dataloader()

        total_loss = 0.0
        num_steps = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                total_loss += loss.item()
                num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        return metrics

    def save_model(self, output_dir=None):
        """Save model checkpoint."""
        output_dir = output_dir or getattr(self.args, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

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
