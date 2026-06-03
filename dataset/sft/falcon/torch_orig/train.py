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

"""Generated pure-Torch SFT entrypoint."""
from pathlib import Path
import sys

_SFT_ROOT = Path(__file__).resolve().parents[2]
if str(_SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SFT_ROOT))

import logging

from config import get_train_args
from data.collator import SFTDataCollatorWith4DAttentionMask
from data.constants import IGNORE_INDEX
from data.loader import get_dataset
from data.template import get_template_and_fix_tokenizer
from model.loader import load_model, load_tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sft(config=None):
    model_args, data_args, training_args, finetuning_args = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    processor = tokenizer_module['processor']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, tokenizer, processor)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=training_args.do_train)
    predict_with_generate = getattr(training_args, 'predict_with_generate', False)
    pad_to_multiple_of = getattr(training_args, 'pad_to_multiple_of', None)
    data_collator = SFTDataCollatorWith4DAttentionMask(template=template, model=model if not predict_with_generate else None, pad_to_multiple_of=pad_to_multiple_of if training_args.do_train else None, label_pad_token_id=IGNORE_INDEX, block_diag_attn=getattr(model_args, 'block_diag_attn', False), attn_implementation=getattr(model_args, 'attn_implementation', 'eager'), compute_dtype=model_args.compute_dtype, tokenizer=tokenizer, processor=processor)
    from transformers import Trainer
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, processing_class=processor or tokenizer, **dataset_module)
    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics('train', result.metrics)
        trainer.save_metrics('train', result.metrics)
        trainer.save_state()
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

if __name__ == '__main__':
    run_sft()
