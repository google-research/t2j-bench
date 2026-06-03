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

"""Shared helpers for running evaluator configs through pytest."""

from __future__ import annotations

import copy
import glob
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import EvalConfig, load_eval_config
from .results import CheckResult, EvalReport
from .runner import run_evaluation


@dataclass(frozen=True, slots=True)
class EvalCase:
    config_path: str
    overrides: tuple[str, ...] = ()

    @property
    def display_path(self) -> str:
        path = Path(self.config_path).resolve()
        root = Path.cwd().resolve()
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    @property
    def case_id(self) -> str:
        suffix = ""
        if self.overrides:
            digest = hashlib.sha1("|".join(self.overrides).encode("utf-8")).hexdigest()[:8]
            suffix = f"::{digest}"
        return f"{self.display_path}{suffix}"


def collect_eval_cases(
    *,
    config_paths: Iterable[str] | None = None,
    config_globs: Iterable[str] | None = None,
    all_eval_configs: bool = False,
    overrides: Iterable[str] | None = None,
) -> list[EvalCase]:
    root = Path.cwd()
    discovered: list[str] = []
    for path_str in config_paths or ():
        path = Path(path_str)
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        discovered.append(str(path))
    for pattern in config_globs or ():
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No evaluator configs matched glob: {pattern}")
        for match in matches:
            discovered.append(str((root / match).resolve()))
    if all_eval_configs:
        for match in sorted(glob.glob("configs/*.yaml")):
            discovered.append(str((root / match).resolve()))
    unique = sorted(dict.fromkeys(discovered))
    return [EvalCase(config_path=path, overrides=tuple(overrides or ())) for path in unique]


def prepare_pytest_config(case: EvalCase, output_root: str) -> EvalConfig:
    config = load_eval_config(case.config_path, list(case.overrides))
    configured = copy.deepcopy(config)
    config_path = Path(case.config_path)
    run_name = config_path.stem
    if case.overrides:
        digest = hashlib.sha1("|".join(case.overrides).encode("utf-8")).hexdigest()[:8]
        run_name = f"{run_name}_{digest}"
    configured.output_dir = str(Path(output_root) / run_name)
    return configured


def run_eval_case(case: EvalCase, output_root: str) -> EvalReport:
    config = prepare_pytest_config(case, output_root)
    return run_evaluation(config)


def stage_failures(report: EvalReport, stage: str) -> list[CheckResult]:
    items = getattr(report, stage)
    return [item for item in items if not item.passed]


def describe_failures(report: EvalReport, stage: str) -> str:
    failures = stage_failures(report, stage)
    if not failures:
        return ""
    parts = []
    for item in failures:
        kind = f" [{item.failure_kind}]" if item.failure_kind else ""
        parts.append(f"{item.name}{kind}: {item.summary}")
    return "; ".join(parts)
