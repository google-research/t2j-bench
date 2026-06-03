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

"""Configuration loading for the rebuilt evaluator."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("PyYAML is required to load evaluator configs.") from exc
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _apply_override(target: dict[str, Any], key: str, value: Any) -> None:
    cursor = target
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


@dataclass(slots=True)
class ImplementationConfig:
    name: str
    repo_dir: str
    framework: str
    device: str | None = None
    backend_device: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalConfig:
    training_method: str
    reference_impl: str
    candidate_impl: str
    implementations: list[ImplementationConfig]
    shared_config: dict[str, Any]
    output_dir: str
    precision_profile: str = "fp16_compare"
    runtime_profile: str = "single_gpu"
    seed: int = 42
    batch_size: int = 2
    max_eval_examples: int = 2
    loss_trajectory_steps: int = 3
    generation_max_new_tokens: int = 16
    compare_hidden_states: bool = True
    suites: list[str] | None = None

    def implementation_map(self) -> dict[str, ImplementationConfig]:
        return {impl.name: impl for impl in self.implementations}


def _resolve_repo_dir(base_dir: Path, repo_dir: str) -> str:
    candidate = Path(repo_dir)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


def _coerce_implementations(base_dir: Path, raw_items: list[dict[str, Any]]) -> list[ImplementationConfig]:
    items: list[ImplementationConfig] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Each implementations entry must be a mapping.")
        try:
            name = str(raw["name"])
            repo_dir = _resolve_repo_dir(base_dir, str(raw["repo_dir"]))
            framework = str(raw["framework"])
        except KeyError as exc:
            raise ValueError(f"Missing implementation field: {exc}") from exc
        overrides = raw.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError("implementation overrides must be a mapping")
        items.append(
            ImplementationConfig(
                name=name,
                repo_dir=repo_dir,
                framework=framework,
                device=raw.get("device"),
                backend_device=raw.get("backend_device"),
                overrides=overrides,
            )
        )
    return items


def _coerce_eval_config(base_dir: Path, raw: dict[str, Any]) -> EvalConfig:
    implementations = _coerce_implementations(base_dir, raw.get("implementations", []))
    if not implementations:
        raise ValueError("Config must define at least one implementation.")
    output_dir = raw.get("output_dir")
    if not output_dir:
        raise ValueError("Config must define output_dir.")
    output_dir = str((base_dir / output_dir).resolve()) if not Path(output_dir).is_absolute() else str(Path(output_dir))
    shared_config = raw.get("shared_config", {})
    if not isinstance(shared_config, dict):
        raise ValueError("shared_config must be a mapping.")
    suites = raw.get("suites")
    if suites is not None and not isinstance(suites, list):
        raise ValueError("suites must be a list when provided.")
    config = EvalConfig(
        training_method=str(raw["training_method"]),
        reference_impl=str(raw["reference_impl"]),
        candidate_impl=str(raw["candidate_impl"]),
        implementations=implementations,
        shared_config=shared_config,
        output_dir=output_dir,
        precision_profile=str(raw.get("precision_profile", "fp16_compare")),
        runtime_profile=str(raw.get("runtime_profile", "single_gpu")),
        seed=int(raw.get("seed", 42)),
        batch_size=int(raw.get("batch_size", 2)),
        max_eval_examples=int(raw.get("max_eval_examples", 2)),
        loss_trajectory_steps=int(raw.get("loss_trajectory_steps", 3)),
        generation_max_new_tokens=int(raw.get("generation_max_new_tokens", 16)),
        compare_hidden_states=bool(raw.get("compare_hidden_states", True)),
        suites=suites,
    )
    names = {impl.name for impl in implementations}
    if config.reference_impl not in names:
        raise ValueError(f"reference_impl {config.reference_impl!r} is not in implementations.")
    if config.candidate_impl not in names:
        raise ValueError(f"candidate_impl {config.candidate_impl!r} is not in implementations.")
    if config.precision_profile not in {"fp16_compare", "bf16_compare"}:
        raise ValueError("precision_profile must be fp16_compare or bf16_compare.")
    if config.runtime_profile not in {"single_gpu", "multi_gpu"}:
        raise ValueError("runtime_profile must be single_gpu or multi_gpu.")
    if config.training_method not in {"sft", "dpo", "ppo"}:
        raise ValueError("training_method must be one of sft, dpo, ppo.")
    return config


def load_eval_config(path: str, overrides: list[str] | None = None) -> EvalConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if config_path.suffix in {".yaml", ".yml"}:
        raw = _load_yaml(config_path)
    elif config_path.suffix == ".json":
        raw = json.loads(config_path.read_text())
    else:
        raise ValueError("Evaluator configs must be .yaml, .yml, or .json.")
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}.")
        key, raw_value = item.split("=", 1)
        _apply_override(raw, key, _parse_scalar(raw_value))
    return _coerce_eval_config(config_path.parent, raw)
