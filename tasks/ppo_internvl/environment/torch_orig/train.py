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

"""Main entry point for PPO training."""

import logging
from typing import Optional

import numpy as np
import torch
import os
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
    """Run PPO training."""
    model_args, data_args, training_args, finetuning_args, ppo_args, generating_args = get_train_args(config)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _seed = int(getattr(training_args, "seed", 42))
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        processor=processor,
    )

    model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=training_args.do_train,
        add_valuehead=True,
    )

    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=model,
        pad_to_multiple_of=getattr(training_args, "pad_to_multiple_of", None) if training_args.do_train else None,
        label_pad_token_id=-100,
        tokenizer=tokenizer,
        processor=processor,
        compute_dtype=model_args.compute_dtype,
    )

    trainer = StandalonePPOTrainer(
        model=model,
        ref_model=None,
        reward_model=ppo_args.reward_model,
        tokenizer=tokenizer,
        processor=processor,
        data_collator=data_collator,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module.get("eval_dataset"),
        training_args=training_args,
        ppo_args=ppo_args,
        generating_args=generating_args,
        finetuning_args=finetuning_args,
    )

    if training_args.do_eval:
        logger.info("*** Evaluating held-out validation loss (pre-training) ***")
        _pre_eval = trainer.evaluate(metric_key_prefix="eval_pre")
        trainer.log_metrics("eval_pre", _pre_eval)
        trainer.save_metrics("eval_pre", _pre_eval)

    if training_args.do_train:
        logger.info("*** Starting PPO training ***")
        trainer.ppo_train()
        trainer.save_model()
        logger.info("*** PPO training complete ***")

    if training_args.do_eval:
        logger.info("*** Evaluating held-out validation loss ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    run_ppo()
