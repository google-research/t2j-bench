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

"""Standalone PPO trainer."""

import inspect
import json
import logging
import math
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers_impl import ppo_compat as _ppo_compat
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

GenerationConfig = _ppo_compat.GenerationConfig
get_scheduler = _ppo_compat.get_scheduler

if TYPE_CHECKING:
    from config import FinetuningArguments, GeneratingArguments, PPOArguments, TrainingArguments
    from model.loader import ModelWithValueHead

import contextlib as _contextlib

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@_contextlib.contextmanager
def _deterministic_fp32_eval(*models):
    """Deterministic, float32 context for evaluation.

    Casts the given model(s) to float32 and forces deterministic reductions so the
    evaluation loss is free of bf16 rounding and reproducible run-to-run, restoring
    the original parameter dtypes and global flags on exit.
    """
    models = [m for m in models if m is not None]
    dtypes = [next(m.parameters()).dtype for m in models]
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for m in models:
        m.float()
    try:
        yield
    finally:
        for m, dtype in zip(models, dtypes):
            m.to(dtype)
        torch.backends.cudnn.benchmark = prev_cudnn_bench
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.use_deterministic_algorithms(prev_det, warn_only=True)


def _eval_batch(batch, device):
    """Move ``batch`` to ``device``, upcasting float tensors to float32 for evaluation."""
    moved = {}
    for _k, _v in batch.items():
        if torch.is_tensor(_v):
            _v = _v.to(device)
            if _v.is_floating_point():
                _v = _v.float()
        moved[_k] = _v
    return moved


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
    """Standalone PPO trainer, independent of TRL."""

    def __init__(
        self,
        model: "ModelWithValueHead",
        ref_model: Optional["ModelWithValueHead"],
        reward_model,
        tokenizer,
        processor,
        data_collator,
        train_dataset,
        training_args: "TrainingArguments",
        ppo_args: "PPOArguments",
        generating_args: "GeneratingArguments",
        finetuning_args: "FinetuningArguments",
        eval_dataset=None,
    ):
        self.model = model
        self.ref_model = ref_model
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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
        if isinstance(self.reward_model, torch.nn.Module):
            self.reward_model.to(self.device)

        eos_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "additional_special_tokens_ids"):
            eos_ids += self.tokenizer.additional_special_tokens_ids
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            **generating_args.to_dict(),
        )

        self.compute_dtype = getattr(model.config, "torch_dtype", torch.float32)
        if training_args.bf16:
            self.compute_dtype = torch.bfloat16
        elif training_args.fp16:
            self.compute_dtype = torch.float16

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
        warmup_steps = self.training_args.warmup_steps or 0
        warmup_ratio = self.training_args.warmup_ratio or 0.0
        if warmup_steps == 0 and warmup_ratio > 0:
            warmup_steps = int(num_training_steps * warmup_ratio)

        scheduler = get_scheduler(
            self.training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    def _base_model(self):
        """The model underneath the value head and any PEFT wrappers."""
        model = getattr(self.model, "pretrained_model", self.model)
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()

        return model

    def _mm_token_type_ids(self, input_ids: torch.Tensor) -> dict:
        """Per-position modality -- 0 text, 1 image, 2 video -- when the model asks for it."""
        forward = getattr(self._base_model(), "forward", None)
        if forward is None or "mm_token_type_ids" not in inspect.signature(forward).parameters:
            return {}

        modality = torch.zeros_like(input_ids)
        for value, attribute in ((1, "image_token_id"), (2, "video_token_id")):
            token_id = getattr(self.processor, attribute, getattr(self.tokenizer, attribute, None))
            if token_id is not None:
                modality[input_ids == token_id] = value

        return {"mm_token_type_ids": modality}

    def _supported_generation_kwargs(self, gen_kwargs: dict) -> dict:
        """Drop the media kwargs this model's ``generate`` would reject."""
        model = self._base_model()
        prepare = getattr(model, "prepare_inputs_for_generation", None)
        if prepare is None:
            return gen_kwargs

        accepted = set(inspect.signature(prepare).parameters)
        forward_params = set(inspect.signature(model.forward).parameters)
        if accepted & {"kwargs", "model_kwargs"}:
            accepted |= forward_params

        submodel = getattr(model, "thinker", None)
        if submodel is not None and forward_params <= {"self", "args", "kwargs"}:
            accepted |= set(inspect.signature(submodel.prepare_inputs_for_generation).parameters)
            accepted |= set(inspect.signature(submodel.forward).parameters)
        dropped = sorted(key for key in gen_kwargs if key not in accepted)
        if dropped:
            logger.info("%s does not accept %s during generation; dropping",
                        type(model).__name__, dropped)

        return {key: value for key, value in gen_kwargs.items() if key in accepted}

    @torch.no_grad()
    def _generate_responses(
        self,
        batch: dict[str, torch.Tensor],
        generation_config=None,
        seed_offset: Optional[int] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate responses given query batch."""
        self.model.eval()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")

        gen_kwargs = {}
        for key in [
            "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
            "image_sizes", "second_per_grid_ts", "token_type_ids",
            "input_features", "feature_attention_mask",
        ]:
            if key in batch:
                gen_kwargs[key] = batch[key]

        gen_kwargs = self._supported_generation_kwargs(gen_kwargs)

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
        gen_kwargs.update(self._mm_token_type_ids(input_ids))

        gen_kwargs = self._mm_inputs_present_in(input_ids, gen_kwargs)

        _gen_config = generation_config if generation_config is not None else self.generation_config
        gen_params = {k: v for k, v in _gen_config.to_dict().items() if v is not None}
        generate_params = inspect.signature(self._base_model().generate).parameters
        gen_params.update({
            f"thinker_{key}": value
            for key, value in gen_params.items()
            if f"thinker_{key}" in generate_params
        })
        if seed_offset is None:
            self._rollout_step = getattr(self, "_rollout_step", 0) + 1
            _seed_step = self._rollout_step
        else:
            _seed_step = seed_offset
        torch.manual_seed(int(getattr(self.training_args, "seed", 42)) + _seed_step)
        generate_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_params,
            **gen_kwargs,
        )

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
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                response_length = response_indexes[-1].item() + 2
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
        gold_responses: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Compute reward scores for query-response pairs."""
        if self.reward_model_type == "label":
            return self._compute_label_rewards(responses, gold_responses)
        if isinstance(self.reward_model, torch.nn.Module):
            input_ids = self._pad_sequences(
                [torch.cat((q, r), dim=-1) for q, r in zip(queries, responses)],
                pad_value=self.tokenizer.pad_token_id,
                padding_side="left",
            ).to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            with torch.amp.autocast(self.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
                _, _, values = self.reward_model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            seq_lengths = attention_mask.sum(dim=-1, keepdim=True) - 1
            rewards = values.gather(dim=-1, index=seq_lengths).squeeze(-1)
            return rewards.float().cpu()
        else:
            raise ValueError(f"Unknown reward model type: {type(self.reward_model)}")

    def _compute_label_rewards(
        self,
        responses: list[torch.Tensor],
        gold_responses: Optional[list[list[int]]],
    ) -> torch.Tensor:
        """Token-F1 overlap between each generated response and its gold answer.

        A decode-order-invariant signal in [0, 1]: the multiset overlap of the
        response token ids with the supervised answer token ids.
        """
        pad_id = self.tokenizer.pad_token_id
        gold_responses = gold_responses if gold_responses is not None else [None] * len(responses)
        rewards = []
        for response, gold in zip(responses, gold_responses):
            pred_ids = [int(t) for t in response.tolist() if t != pad_id]
            gold_ids = [int(t) for t in (gold or []) if t != pad_id]
            rewards.append(token_f1_reward(pred_ids, gold_ids))
        return torch.tensor(rewards, dtype=torch.float32)

    def _gold_response_tokens(self, labels) -> list[list[int]]:
        """Per-row supervised answer token ids (the non-ignored positions of labels)."""
        gold = []
        for row in labels:
            row = row.tolist() if torch.is_tensor(row) else list(row)
            gold.append([int(t) for t in row if t != IGNORE_INDEX])
        return gold

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
        """Compute log-probabilities and values for the full sequence."""
        with torch.amp.autocast(self.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
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
            with torch.amp.autocast(self.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
                logits, _, _ = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **kwargs,
                )
        else:
            import contextlib

            pretrained = self.model.pretrained_model
            adapter_ctx = (
                pretrained.disable_adapter()
                if hasattr(pretrained, "disable_adapter")
                else contextlib.nullcontext()
            )
            with adapter_ctx:
                with torch.amp.autocast(self.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
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
        """Compute token_type_ids (0 for text, 1 for image tokens)."""
        extra_kwargs = {}
        image_token_id = getattr(self.processor, "image_token_id",
                                 getattr(self.tokenizer, "image_token_id", None))
        if image_token_id is not None:
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[input_ids == image_token_id] = 1
            extra_kwargs["token_type_ids"] = token_type_ids
        return extra_kwargs

    def _mm_inputs_present_in(self, input_ids: torch.Tensor, mm_inputs: dict) -> dict:
        """Keep only the media inputs whose placeholder tokens appear in ``input_ids``."""
        kept = dict(mm_inputs)
        for attribute, keys in (
            ("image_token_id", ("pixel_values", "image_grid_thw", "image_sizes")),
            ("video_token_id", ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts")),
            ("audio_token_id", ("input_features", "feature_attention_mask")),
        ):
            token_id = getattr(self.processor, attribute,
                               getattr(self.tokenizer, attribute, None))
            if token_id is None or bool((input_ids == token_id).any()):
                continue

            for key in keys:
                kept.pop(key, None)

        return kept

    def ppo_train(self):
        """Main PPO training loop."""
        args = self.training_args
        ppo_args = self.ppo_args
        batch_size = args.per_device_train_batch_size

        # Seed the rollout shuffle and the PPO minibatch permutation so the
        # training order is reproducible run-to-run.
        _seed = int(getattr(args, "seed", 42))
        np.random.seed(_seed)
        _rollout_generator = torch.Generator()
        _rollout_generator.manual_seed(_seed)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size * ppo_args.ppo_buffer_size,
            shuffle=True,
            generator=_rollout_generator,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )

        _max_steps = getattr(args, "max_steps", 0) or 0
        if _max_steps > 0:
            max_steps = _max_steps
            num_train_epochs = max_steps
        else:
            max_steps = int(args.num_train_epochs * len(dataloader))
            num_train_epochs = int(math.ceil(args.num_train_epochs))

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, max_steps)

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

                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                batch_mm_inputs = {}
                for key in ["pixel_values", "image_grid_thw", "pixel_values_videos",
                            "video_grid_thw", "image_sizes", "second_per_grid_ts", "token_type_ids",
                            "input_features", "feature_attention_mask"]:
                    if key in batch:
                        batch_mm_inputs[key] = batch[key]

                self.model.eval()

                all_queries, all_responses, all_rewards = [], [], []

                n_examples = batch["input_ids"].size(0)
                for idx in range(0, n_examples, batch_size):
                    mini_batch = {
                        "input_ids": batch["input_ids"][idx:idx + batch_size],
                        "attention_mask": batch["attention_mask"][idx:idx + batch_size],
                    }
                    for key, value in batch_mm_inputs.items():
                        if torch.is_tensor(value) and value.size(0) == n_examples:
                            mini_batch[key] = value[idx:idx + batch_size]
                        else:
                            mini_batch[key] = value

                    gold_responses = None
                    if self.reward_model_type == "label" and "labels" in batch:
                        mini_labels = batch["labels"][idx:idx + batch_size]
                        # Handing labels to generation truncates the input to the prompt
                        # (the gold continuation is dropped), so the rollout is a fresh
                        # answer that can be scored against that gold.
                        mini_batch["labels"] = mini_labels
                        gold_responses = self._gold_response_tokens(mini_labels)

                    queries, responses = self._generate_responses(mini_batch)
                    rewards = self._compute_rewards(queries, responses, gold_responses=gold_responses)
                    all_queries.extend(queries)
                    all_responses.extend(responses)
                    if rewards.dim() == 0:
                        all_rewards.append(rewards.unsqueeze(0))
                    else:
                        all_rewards.append(rewards)

                rewards = torch.cat(all_rewards, dim=0)
                bs = len(all_queries)

                full_ids = self._pad_sequences(
                    [torch.cat((q, r), dim=-1) for q, r in zip(all_queries, all_responses)],
                    pad_value=self.tokenizer.pad_token_id,
                    padding_side="right",
                ).to(self.device)
                full_mask = (full_ids != self.tokenizer.pad_token_id).long().to(self.device)

                query_lens = torch.tensor([len(q) for q in all_queries])
                response_lens = torch.tensor([len(r) for r in all_responses])
                max_seq_len = full_ids.size(1)

                response_mask = torch.zeros(bs, max_seq_len - 1, device=self.device)
                for i in range(bs):
                    q_len = query_lens[i].item()
                    r_len = response_lens[i].item()
                    start = q_len - 1
                    end = start + r_len
                    end = min(end, max_seq_len - 1)
                    response_mask[i, start:end] = 1.0

                extra_kwargs = self._compute_token_type_ids(full_ids)
                extra_kwargs.update(self._mm_token_type_ids(full_ids))
                extra_kwargs.update(self._mm_inputs_present_in(full_ids, batch_mm_inputs))

                old_logprobs, old_values = self._compute_logprobs_and_values(
                    self.model, full_ids, full_mask, response_mask, **extra_kwargs
                )

                ref_logprobs = self._compute_ref_logprobs(
                    full_ids, full_mask, **extra_kwargs
                )

                kl = (old_logprobs - ref_logprobs).detach()

                per_token_rewards = -ppo_args.kl_coef * kl
                for i in range(bs):
                    q_len = query_lens[i].item()
                    r_len = response_lens[i].item()
                    end_pos = q_len - 1 + r_len - 1
                    end_pos = min(end_pos, max_seq_len - 2)
                    per_token_rewards[i, end_pos] += rewards[i].to(self.device)

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

                if ppo_args.ppo_whiten_rewards:
                    resp_rewards = masked_whiten(resp_rewards, resp_mask, shift_mean=False)

                advantages, returns = compute_gae(
                    resp_rewards, resp_values, resp_mask,
                    gamma=ppo_args.gamma, lam=ppo_args.lam,
                )
                advantages = masked_whiten(advantages, resp_mask)

                mean_kl = masked_mean(kl, response_mask).item()
                kl_meter.update(mean_kl, n=bs)
                reward_meter.update(rewards.mean().item(), n=bs)

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

                        mb_kwargs = {}
                        for key, val in extra_kwargs.items():
                            if torch.is_tensor(val):
                                if val.size(0) == bs:
                                    mb_kwargs[key] = val[mb_inds]
                                else:
                                    mb_kwargs[key] = val
                            else:
                                mb_kwargs[key] = val

                        with torch.amp.autocast(self.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
                            logits, _, vpred = self.model(
                                input_ids=mb_ids,
                                attention_mask=mb_mask,
                                use_cache=False,
                                **mb_kwargs,
                            )

                        new_logprobs = logprobs_from_logits(logits[:, :-1, :], mb_ids[:, 1:])

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

                if global_step % args.save_steps == 0:
                    self.save_model(
                        os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    )

            if global_step >= max_steps:
                break

        logger.info("PPO training complete.")

    def _greedy_generation_config(self):
        """A deterministic (greedy) copy of the generation config for evaluation."""
        gen = dict(self.generating_args.to_dict())
        gen["do_sample"] = False
        for key in ("temperature", "top_p", "top_k"):
            gen.pop(key, None)
        eos_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "additional_special_tokens_ids"):
            eos_ids += self.tokenizer.additional_special_tokens_ids
        return GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            **gen,
        )

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, metric_key_prefix: str = "eval") -> dict:
        """Validation metric: the negated mean held-out label reward on the eval split (lower is better)."""
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {}

        loader = DataLoader(
            dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=self.training_args.dataloader_num_workers,
        )

        self.model.eval()
        kl_meter = AverageMeter()
        reward_meter = AverageMeter()
        include_reward = self.reward_model_type == "label"
        greedy_config = self._greedy_generation_config() if include_reward else None
        _mm_keys = [
            "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
            "image_sizes", "second_per_grid_ts", "token_type_ids",
            "input_features", "feature_attention_mask",
        ]
        _prev_compute_dtype = self.compute_dtype
        with _deterministic_fp32_eval(self.model, self.ref_model):
            # fp32 forwards: disable the training autocast during evaluation.
            self.compute_dtype = torch.float32
            try:
                for batch in loader:
                    batch = _eval_batch(batch, self.device)
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = (
                        batch["labels"]
                        if "labels" in batch and torch.is_tensor(batch["labels"])
                        else None
                    )

                    policy_logprobs, _ = self._compute_logprobs_and_values(
                        self.model, input_ids, attention_mask, attention_mask
                    )
                    ref_logprobs = self._compute_ref_logprobs(input_ids, attention_mask)

                    L = policy_logprobs.size(1)
                    if labels is not None and labels.size(1) == input_ids.size(1):
                        mask = (labels[:, 1:] != -100).to(policy_logprobs.dtype)
                    else:
                        if labels is not None:
                            n_resp = (labels != -100).sum(dim=1)
                        else:
                            n_resp = attention_mask[:, 1:].sum(dim=1)
                        n_resp = n_resp.clamp(max=L)
                        positions = torch.arange(L, device=policy_logprobs.device).unsqueeze(0)
                        mask = (positions >= (L - n_resp).unsqueeze(1)).to(policy_logprobs.dtype)
                    n_tokens = int(mask.sum().item())
                    if n_tokens > 0:
                        mean_kl = masked_mean(policy_logprobs - ref_logprobs, mask)
                        kl_meter.update(float(mean_kl.item()), n=n_tokens)

                    if include_reward and labels is not None:
                        gen_batch = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        }
                        for key in _mm_keys:
                            if key in batch:
                                gen_batch[key] = batch[key]
                        queries, responses = self._generate_responses(
                            gen_batch, generation_config=greedy_config, seed_offset=0
                        )
                        gold_responses = self._gold_response_tokens(labels)
                        rewards = self._compute_rewards(
                            queries, responses, gold_responses=gold_responses
                        )
                        reward_meter.update(float(rewards.mean().item()), n=rewards.numel())
            finally:
                self.compute_dtype = _prev_compute_dtype

        self.model.train()
        mean_kl = kl_meter.avg
        mean_reward = reward_meter.avg
        eval_loss = -mean_reward if include_reward else mean_kl
        return {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_kl": mean_kl,
            f"{metric_key_prefix}_reward": mean_reward,
            f"{metric_key_prefix}_tokens": kl_meter.count,
        }

    def log_metrics(self, split: str, metrics: dict):
        logger.info(f"***** {split} metrics *****")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")

    def save_metrics(self, split: str, metrics: dict):
        output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{split}_results.json"), "w") as file:
            json.dump(metrics, file, indent=2)

    def save_model(self, output_dir: Optional[str] = None):
        """Save model checkpoint."""
        if output_dir is None:
            output_dir = self.training_args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        saved = False
        if hasattr(self.model, "save_pretrained"):
            try:
                self.model.save_pretrained(output_dir)
                saved = True
            except Exception as error:
                logger.warning(
                    "save_pretrained failed (%s); falling back to a state dict dump.",
                    error,
                )

        if not saved:
            state_dict = self.model.state_dict()
            save_path = os.path.join(output_dir, "model.safetensors")
            try:
                from safetensors.torch import save_file
                save_file(state_dict, save_path)
            except Exception:
                save_path = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(state_dict, save_path)

        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")
