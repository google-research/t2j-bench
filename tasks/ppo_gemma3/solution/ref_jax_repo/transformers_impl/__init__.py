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

"""Standalone native-JAX (flax) modules for Gemma3 — decoupled from HuggingFace.

Mirror of ``torch_orig/transformers_impl/__init__.py``, re-export list included.
JAX adjustments are documented per module: flax ``__call__`` over an external
parameter pytree, and numpy in place of ``torch.Tensor`` in the processors,
the collator and the trainer.

The flax model is the text tower this benchmark grades, so torch's ``PreTrainedConfig``,
``Gemma3TextConfig``, ``SiglipVisionConfig``, ``PreTrainedModel``,
``SiglipVisionModel``, ``Gemma3MultiModalProjector`` and ``DynamicCache`` have no
counterpart here and are not re-exported.
"""

from .configuration import Gemma3Config, load_config
from .modeling import Gemma3TextModel, Gemma3ForConditionalGeneration
from .tokenization import Tokenizer
from .image_processing import Gemma3ImageProcessor
from .processing import Gemma3Processor
from .collator import DataCollatorForSeq2Seq
from .trainer import Trainer
from .utils import ACT2FN, ModelOutput, BatchFeature
