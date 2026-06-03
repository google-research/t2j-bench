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

"""Structured result types for evaluator runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CheckResult:
    name: str
    stage: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    failure_kind: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalReport:
    config: dict[str, Any]
    reference: list[CheckResult] = field(default_factory=list)
    spec: list[CheckResult] = field(default_factory=list)
    numeric: list[CheckResult] = field(default_factory=list)
    behavioral: list[CheckResult] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "reference": [item.to_dict() for item in self.reference],
            "spec": [item.to_dict() for item in self.spec],
            "numeric": [item.to_dict() for item in self.numeric],
            "behavioral": [item.to_dict() for item in self.behavioral],
            "artifacts": self.artifacts,
        }

    def summary(self) -> dict[str, Any]:
        def pack(stage_items: list[CheckResult]) -> dict[str, Any]:
            return {
                "passed": all(item.passed for item in stage_items) if stage_items else True,
                "checks": [item.to_dict() for item in stage_items],
            }

        overall = all(
            pack(items)["passed"]
            for items in (self.reference, self.spec, self.numeric, self.behavioral)
        )
        return {
            "overall": "PASS" if overall else "FAIL",
            "reference": pack(self.reference),
            "spec": pack(self.spec),
            "numeric": pack(self.numeric),
            "behavioral": pack(self.behavioral),
        }
