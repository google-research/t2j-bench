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

from pathlib import Path
import importlib.util
import sys

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SFT_ROOT = Path(__file__).resolve().parents[2]
if str(_SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SFT_ROOT))

_MODEL_META_PATH = _REPO_ROOT / 'model_meta.py'
_MODEL_META_SPEC = importlib.util.spec_from_file_location(f'{__name__}_model_meta', _MODEL_META_PATH)
_MODEL_META = importlib.util.module_from_spec(_MODEL_META_SPEC)
_MODEL_META_SPEC.loader.exec_module(_MODEL_META)

from _torch_sft_shared.config_base import build_get_train_args
DEFAULT_TEMPLATE = _MODEL_META.DEFAULT_TEMPLATE

get_train_args = build_get_train_args(DEFAULT_TEMPLATE)
