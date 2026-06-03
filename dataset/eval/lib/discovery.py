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

"""Repository import helpers for LlamaFactory-derived repos."""

from __future__ import annotations

import importlib
import sys
import enum
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator


@dataclass(slots=True)
class RepoModules:
    repo_dir: str
    config: ModuleType
    model_loader: ModuleType
    data_loader: ModuleType
    data_collator: ModuleType | None
    data_template: ModuleType | None
    trainer: ModuleType | None
    extra: dict[str, ModuleType]


def _safe_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@contextmanager
def repo_imports(repo_dir: str) -> Iterator[RepoModules]:
    repo_path = str(Path(repo_dir).resolve())
    before_modules = set(sys.modules)
    if not hasattr(enum, "StrEnum"):
        class _CompatStrEnum(str, enum.Enum):
            pass

        enum.StrEnum = _CompatStrEnum  # type: ignore[attr-defined]
    sys.path.insert(0, repo_path)
    try:
        config = importlib.import_module("config")
        model_loader = importlib.import_module("model.loader")
        data_loader = importlib.import_module("data.loader")
        data_collator = _safe_import("data.collator")
        data_template = _safe_import("data.template")
        trainer = None
        extra: dict[str, ModuleType] = {}
        for candidate in (
            "transformers.trainer",
            "dpo.trainer",
            "ppo.trainer",
            "ppo.ppo_utils",
        ):
            module = _safe_import(candidate)
            if module is None:
                continue
            extra[candidate] = module
            if trainer is None and candidate.endswith(".trainer"):
                trainer = module
        yield RepoModules(
            repo_dir=repo_path,
            config=config,
            model_loader=model_loader,
            data_loader=data_loader,
            data_collator=data_collator,
            data_template=data_template,
            trainer=trainer,
            extra=extra,
        )
    finally:
        sys.path = [item for item in sys.path if item != repo_path]
        for module_name, module in list(sys.modules.items()):
            if module_name in before_modules:
                continue
            module_file = getattr(module, "__dict__", {}).get("__file__")
            if module_file is None:
                continue
            try:
                resolved = str(Path(module_file).resolve())
            except OSError:
                continue
            if resolved.startswith(repo_path):
                sys.modules.pop(module_name, None)
