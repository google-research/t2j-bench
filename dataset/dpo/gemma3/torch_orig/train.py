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

"""Main entry point for DPO training."""

import inspect
import logging
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_train_args
from data.collator import PairwiseDataCollatorWithPadding
from data.constants import IGNORE_INDEX
from data.loader import get_dataset
from data.template import get_template_and_fix_tokenizer
from dpo.trainer import DPOTrainer
from model.loader import load_model, load_ref_model, load_tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**accepted)


def run_dpo(config: Optional[dict] = None):
    """Run DPO training."""
    model_args, data_args, training_args, finetuning_args, dpo_args = get_train_args(config)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        processor=processor,
    )

    model = _call_with_supported_kwargs(
        load_model,
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train,
        disable_dropout=dpo_args.disable_dropout,
    )
    ref_model = _call_with_supported_kwargs(
        load_ref_model,
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        dpo_args=dpo_args,
    )

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=training_args.pad_to_multiple_of if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX,
        compute_dtype=model_args.compute_dtype or torch.bfloat16,
        tokenizer=tokenizer,
        processor=processor,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        dpo_args=dpo_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        **dataset_module,
    )

    if training_args.do_train:
        logger.info("*** Starting DPO training ***")
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Starting evaluation ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    run_dpo()
