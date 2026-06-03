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

"""
Main entry point for Qwen3-VL SFT training.

Usage:
    python train.py config.yaml [overrides...]

Example:
    python train.py examples/qwen3_vl_sft.yaml learning_rate=2e-5 num_train_epochs=1
"""

import logging
from typing import Optional

import torch
from transformers_impl.trainer import Trainer

from config import get_train_args
from data.collator import SFTDataCollatorWith4DAttentionMask
from data.constants import IGNORE_INDEX
from data.loader import get_dataset
from data.template import get_template_and_fix_tokenizer
from model.loader import load_model, load_tokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_sft(config: Optional[dict] = None):
    """Run supervised fine-tuning for Qwen3-VL."""
    # 1. Parse configuration
    model_args, data_args, training_args_cfg, finetuning_args = get_train_args(config)

    # 2. Load tokenizer and processor
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    # 3. Get template and fix tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 4. Load dataset
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args_cfg,
        tokenizer=tokenizer,
        processor=processor,
    )

    # 5. Load model
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args_cfg.do_train)

    # 6. Create data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args_cfg.predict_with_generate else None,
        pad_to_multiple_of=training_args_cfg.pad_to_multiple_of if training_args_cfg.do_train else None,
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", "eager"),
        compute_dtype=model_args.compute_dtype or torch.bfloat16,
        tokenizer=tokenizer,
        processor=processor,
    )

    # 7. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args_cfg,
        data_collator=data_collator,
        tokenizer=tokenizer,
        **dataset_module,
    )

    # 8. Training
    if training_args_cfg.do_train:
        logger.info("*** Starting training ***")
        train_result = trainer.train(resume_from_checkpoint=training_args_cfg.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 9. Evaluation
    if training_args_cfg.do_eval:
        logger.info("*** Starting evaluation ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    run_sft()
