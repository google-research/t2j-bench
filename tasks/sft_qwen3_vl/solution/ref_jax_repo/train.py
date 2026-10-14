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

"""Native-JAX SFT entrypoint (torch-free) — mirror of ``torch_orig/train.py``.

Same public surface as ``torch_orig/train.py`` (``run_sft`` only). ``run_sft`` builds the
same pipeline torch does and hands model + collator + datasets to
``transformers_impl.trainer.Trainer``, which owns the optimizer, the schedule and the loop
just as the torch trainer does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Re-exported, not redefined: the trainer must drive the same optimizer, schedule and
# step torch does, and torch keeps ``Trainer`` in ``transformers_impl/trainer.py``.
from transformers_impl.trainer import Trainer

# Full-f32 matmuls (disable TF32) so the training loss curve matches torch's f32
# reference. JAX defaults to TF32 (~bf16 mantissa) for f32 dots on GPU; that ~0.03
# per-logit error is within the forward tolerance but compounds across optimizer
# steps and diverges the deterministic train_trajectory golden.
jax.config.update("jax_default_matmul_precision", "highest")


def run_sft(config=None):
    """Full SFT training entry — mirror of ``torch_orig/train.py::run_sft``.

    Loads the dataset via the data pipeline (template -> SupervisedDatasetProcessor ->
    SFTDataCollatorWith4DAttentionMask), then hands model + collator + datasets to
    ``transformers_impl.trainer.Trainer`` in torch's order — ``train`` / ``save_model`` /
    ``log_metrics`` / ``save_metrics`` / ``save_state``, then ``evaluate`` — which is what
    writes ``{train,eval}_results.json`` to ``output_dir``. Heavy imports are local so
    ``import train`` stays lightweight.
    """
    import logging

    # torch_orig's Trainer logs one line per ``logging_steps``; this loop honours the same
    # key so the two runs are comparable step for step (and so a long run shows progress).
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    from config import get_train_args
    from data.collator import SFTDataCollatorWith4DAttentionMask
    from data.constants import IGNORE_INDEX
    from data.loader import get_dataset
    from data.template import get_template_and_fix_tokenizer
    from model.loader import load_model, load_tokenizer

    model_args, data_args, training_args, finetuning_args = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, tokenizer, processor)
    module, params, _ = load_model(
        tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train
    )
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        label_pad_token_id=IGNORE_INDEX,
        # torch's train.py passes this in from the training args; the collator itself
        # defaults it to None, so omitting it silently drops the padding policy.
        pad_to_multiple_of=(
            getattr(training_args, "pad_to_multiple_of", None) if training_args.do_train else None
        ),
        compute_dtype=model_args.compute_dtype,
    )

    # Train in the configured compute dtype. torch_orig's model/loader.py hands the Trainer
    # a model already in compute_dtype (``torch_dtype=``); the JAX loader returns fp32
    # params, so the cast belongs here, before the trainer takes ownership of them. bf16
    # keeps a 7B model on a single GPU (params + Adam state + grads all bf16).
    _dt = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}.get(
        getattr(model_args, "compute_dtype", "float32") or "float32", jnp.float32
    )
    params = jax.tree_util.tree_map(lambda x: x.astype(_dt), params)

    # LoRA: the Trainer trains the adapters and keeps the base frozen. Mirror of torch's
    # ``get_peft_model`` + ``_log_param_counts``; ``split`` finds nothing under full
    # finetuning, so that path reports 100% trainable exactly as before.
    from model.loader import has_adapters, split

    _use_lora = has_adapters(params)
    _trainable = split(params)[1] if _use_lora else params
    _total = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    _train_n = sum(int(x.size) for x in jax.tree_util.tree_leaves(_trainable))
    logger.info(
        "Model parameters: %s total, %s trainable (%.2f%%) [%s]",
        f"{_total:,}", f"{_train_n:,}", 100.0 * _train_n / max(1, _total),
        "lora" if _use_lora else "full",
    )

    trainer = Trainer(
        # Flax modules are stateless, so torch's single ``model`` is the (module, params)
        # pair here; ``ignore_index`` is the one added keyword, because a flax module
        # returns logits and computing the loss is the trainer's job.
        model=(module, params),
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        ignore_index=IGNORE_INDEX,
        **dataset_module,
    )

    if training_args.do_eval:
        logger.info("*** Evaluating held-out validation loss (pre-training) ***")
        _pre_eval = trainer.evaluate(metric_key_prefix="eval_pre")
        trainer.log_metrics("eval_pre", _pre_eval)
        trainer.save_metrics("eval_pre", _pre_eval)

    if training_args.do_train:
        logger.info("*** Starting training ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
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
    run_sft()
