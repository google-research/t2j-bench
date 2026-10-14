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

"""Checkpoint resolution and conversion.

torch_orig exports `load_weights`, which loads INTO a torch module. The JAX side
splits the same job in two -- `convert_checkpoint` produces the parameter set and
`apply_params` writes it in -- because the conversion is the graded artifact
(EVAL_FRAMEWORK_DESIGN §4.3): the grader re-drives it on the real checkpoint
without ever constructing a model.
"""

from model.loader import (
    STACKED_PARAMS,
    apply_params,
    convert_checkpoint,
    load_tokenizer,
    resolve_model_dir,
)

__all__ = [
    "STACKED_PARAMS",
    "apply_params",
    "convert_checkpoint",
    "load_tokenizer",
    "resolve_model_dir",
]
