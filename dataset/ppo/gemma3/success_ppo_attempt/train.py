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
Main entry point for Gemma3 PPO training.

Usage:
    python train.py config.yaml [overrides...]

Example:
    python train.py examples/gemma3_ppo.yaml learning_rate=1e-6 ppo_epochs=2
"""

import logging
from typing import Optional

import torch

from config import get_train_args
from data.collator import MultiModalDataCollatorForSeq2Seq
from data.loader import get_dataset
from data.template import get_template_and_fix_tokenizer
from model.loader import load_model, load_tokenizer
from ppo.trainer import StandalonePPOTrainer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_ppo(config: Optional[dict] = None):
    """Run PPO training for Gemma3."""
    # 1. Parse configuration
    model_args, data_args, training_args, finetuning_args, ppo_args, generating_args = get_train_args(config)

    # 2. Load tokenizer and processor
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    # 3. Get template and fix tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 4. Load dataset (unsupervised/prompt-only for PPO)
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        processor=processor,
    )

    # 5. Load model with value head
    logger.info("Loading model with value head...")
    model = load_model(
        tokenizer, model_args, finetuning_args,
        is_trainable=True, add_valuehead=True,
    )

    # 6. Create reference model
    ref_model = None
    if finetuning_args.finetuning_type != "lora":
        # Full fine-tuning needs a separate frozen reference model
        logger.info("Loading reference model...")
        from dataclasses import replace
        ref_finetuning_args = replace(finetuning_args, finetuning_type="full")
        if ppo_args.ref_model is not None:
            ref_model_args = replace(model_args, model_name_or_path=ppo_args.ref_model)
        else:
            ref_model_args = model_args
        ref_model = load_model(
            tokenizer, ref_model_args, ref_finetuning_args,
            is_trainable=False, add_valuehead=True,
        )
    else:
        logger.info("Using LoRA adapter disable trick for reference model (no extra memory).")

    # 7. Set up reward model
    if ppo_args.reward_model_type == "api":
        if ppo_args.reward_model is None:
            raise ValueError("reward_model (API URL) must be specified for api reward_model_type.")
        reward_model = ppo_args.reward_model  # API URL string
        logger.info(f"Using API reward model at: {reward_model}")
    elif ppo_args.reward_model_type == "full":
        if ppo_args.reward_model is None:
            raise ValueError("reward_model path must be specified for full reward_model_type.")
        logger.info(f"Loading full reward model from: {ppo_args.reward_model}")
        from dataclasses import replace
        reward_model_args = replace(model_args, model_name_or_path=ppo_args.reward_model)
        reward_finetuning_args = replace(finetuning_args, finetuning_type="full")
        reward_model = load_model(
            tokenizer, reward_model_args, reward_finetuning_args,
            is_trainable=False, add_valuehead=True,
        )
    else:
        raise ValueError(f"Unknown reward_model_type: {ppo_args.reward_model_type}")

    # 8. Create data collator (left-padding for generation)
    tokenizer.padding_side = "left"
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=model,
        pad_to_multiple_of=training_args.pad_to_multiple_of if training_args.do_train else None,
        label_pad_token_id=-100,
        tokenizer=tokenizer,
        processor=processor,
    )

    # 9. Initialize PPO trainer
    trainer = StandalonePPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        processor=processor,
        data_collator=data_collator,
        train_dataset=dataset_module["train_dataset"],
        training_args=training_args,
        ppo_args=ppo_args,
        generating_args=generating_args,
        finetuning_args=finetuning_args,
    )

    # 10. Train
    if training_args.do_train:
        logger.info("*** Starting PPO training ***")
        trainer.ppo_train()
        trainer.save_model()
        logger.info("*** PPO training complete ***")


if __name__ == "__main__":
    run_ppo()
