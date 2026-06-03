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

from __future__ import annotations

import pytest

from lib.pytest_support import EvalCase, describe_failures


pytestmark = [pytest.mark.eval]


def _case_label(eval_case: EvalCase | None) -> str:
    return eval_case.case_id if eval_case is not None else "<no-eval-config>"


def test_reference_contract(eval_case: EvalCase | None, eval_report) -> None:
    summary = eval_report.summary()
    assert summary["reference"]["passed"], f"{_case_label(eval_case)} reference failed: {describe_failures(eval_report, 'reference')}"


def test_spec_stage(eval_case: EvalCase | None, eval_report) -> None:
    summary = eval_report.summary()
    assert summary["spec"]["passed"], f"{_case_label(eval_case)} spec failed: {describe_failures(eval_report, 'spec')}"


def test_numeric_stage(eval_case: EvalCase | None, eval_report) -> None:
    summary = eval_report.summary()
    assert summary["numeric"]["passed"], f"{_case_label(eval_case)} numeric failed: {describe_failures(eval_report, 'numeric')}"


def test_behavioral_stage(eval_case: EvalCase | None, eval_report) -> None:
    summary = eval_report.summary()
    assert summary["behavioral"]["passed"], f"{_case_label(eval_case)} behavioral failed: {describe_failures(eval_report, 'behavioral')}"
