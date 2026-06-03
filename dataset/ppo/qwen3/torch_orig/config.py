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
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SFT_ROOT = Path(__file__).resolve().parents[2]
if str(_SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SFT_ROOT))

_MODEL_META_PATH = _REPO_ROOT / "model_meta.py"
_MODEL_META_SPEC = importlib.util.spec_from_file_location(f"{__name__}_model_meta", _MODEL_META_PATH)
_MODEL_META = importlib.util.module_from_spec(_MODEL_META_SPEC)
_MODEL_META_SPEC.loader.exec_module(_MODEL_META)

from _torch_sft_shared.config_base import _populate_dataclass, build_get_train_args, read_args

DEFAULT_TEMPLATE = _MODEL_META.DEFAULT_TEMPLATE
get_train_args = build_get_train_args(DEFAULT_TEMPLATE)
_SFT_GET_TRAIN_ARGS = get_train_args


@dataclass
class PPOArguments:
    ppo_epochs: int = 4
    ppo_buffer_size: int = 1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    kl_coef: float = 0.05
    ppo_target: float = 6.0
    ppo_score_norm: bool = False
    ppo_whiten_rewards: bool = False
    reward_model: Optional[str] = None
    reward_model_type: Literal["api", "full"] = "api"
    ref_model: Optional[str] = None


@dataclass
class GeneratingArguments:
    do_sample: bool = True
    temperature: float = 0.95
    top_p: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_train_args(args: Optional[dict[str, Any]] = None):
    config = read_args(args)
    if isinstance(config, list):
        raise ValueError("Please provide a YAML or JSON configuration file.")

    model_args, data_args, training_args, finetuning_args = _SFT_GET_TRAIN_ARGS(config)
    ppo_args = _populate_dataclass(PPOArguments, config)
    generating_args = _populate_dataclass(GeneratingArguments, config)
    return model_args, data_args, training_args, finetuning_args, ppo_args, generating_args
