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

"""Standalone transformers implementation for LLaVA-Next."""
from .configuration import CLIPVisionConfig, LlamaConfig, LlavaNextConfig, load_config
from .modeling import (
    CLIPVisionModel,
    LlamaModel,
    LlavaNextModel,
    LlavaNextForConditionalGeneration,
    PreTrainedModel,
)
from .tokenization import Tokenizer, AutoTokenizer
from .image_processing import LlavaNextImageProcessor
from .processing import LlavaNextProcessor
from .trainer import Trainer, TrainOutput
from .collator import DataCollatorForSeq2Seq
from .utils import ACT2FN, ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
