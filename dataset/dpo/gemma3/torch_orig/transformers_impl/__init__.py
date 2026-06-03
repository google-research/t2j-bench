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

"""Standalone transformers modules for Gemma3 - decoupled from HuggingFace."""
from .configuration import (
    PreTrainedConfig,
    Gemma3Config,
    Gemma3TextConfig,
    SiglipVisionConfig,
    load_config,
)
from .modeling import (
    PreTrainedModel,
    SiglipVisionModel,
    Gemma3TextModel,
    Gemma3ForConditionalGeneration,
    Gemma3MultiModalProjector,
    DynamicCache,
)
from .tokenization import Tokenizer
from .image_processing import Gemma3ImageProcessor
from .processing import Gemma3Processor
from .collator import DataCollatorForSeq2Seq
from .trainer import Trainer
from .utils import ACT2FN, ModelOutput, BatchFeature
