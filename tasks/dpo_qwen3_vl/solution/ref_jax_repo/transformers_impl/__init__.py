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

"""Minimal native-JAX (flax) transformers module for Qwen3-VL.

Mirror of ``torch_orig/transformers_impl/__init__.py``, re-export list included.
JAX adjustments are documented per module: flax ``__call__`` over an external
parameter pytree, and numpy in place of ``torch.Tensor`` in the processors,
the collator and the trainer.

The flax model is the text tower this benchmark grades, so torch's ``PreTrainedConfig``,
``Qwen3VLTextConfig``, ``Qwen3VLVisionConfig``, ``AutoModelForImageTextToText`` and
``PreTrainedModel`` have no counterpart here and are not re-exported.
"""

from .configuration import AutoConfig, Qwen3VLConfig
from .tokenization import AutoTokenizer, Tokenizer, load_tokenizer
from .image_processing import Qwen2VLImageProcessor
from .video_processing import Qwen3VLVideoProcessor
from .processing import AutoProcessor, Qwen3VLProcessor
from .modeling import Qwen3VLForConditionalGeneration
from .collator import DataCollatorForSeq2Seq
from .trainer import Trainer
from .utils import BatchFeature, ModelOutput, is_valid_image
