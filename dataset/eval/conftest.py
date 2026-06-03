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

from pathlib import Path

import pytest

from lib.pytest_support import EvalCase, collect_eval_cases, run_eval_case


DEFAULT_OUTPUT_ROOT = ".pytest_eval_outputs"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("new-eval")
    group.addoption(
        "--eval-config",
        action="append",
        default=[],
        help="Relative path to an evaluator config. Repeat to run more than one.",
    )
    group.addoption(
        "--eval-config-glob",
        action="append",
        default=[],
        help="Glob of evaluator configs to run, for example configs/*.yaml.",
    )
    group.addoption(
        "--all-eval-configs",
        action="store_true",
        default=False,
        help="Run every config under configs/.",
    )
    group.addoption(
        "--eval-override",
        action="append",
        default=[],
        help="Apply a dotted key=value override to every selected evaluator config.",
    )
    group.addoption(
        "--eval-output-root",
        action="store",
        default=DEFAULT_OUTPUT_ROOT,
        help="Relative directory used for pytest-generated evaluator outputs.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "eval: repo-to-repo evaluator tests")


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "eval_case" not in metafunc.fixturenames:
        return
    pytest_config = metafunc.config
    cases = collect_eval_cases(
        config_paths=pytest_config.getoption("eval_config"),
        config_globs=pytest_config.getoption("eval_config_glob"),
        all_eval_configs=pytest_config.getoption("all_eval_configs"),
        overrides=pytest_config.getoption("eval_override"),
    )
    if not cases:
        metafunc.parametrize("eval_case", [None], ids=["no-eval-config"])
        return
    metafunc.parametrize("eval_case", cases, ids=[case.case_id for case in cases])


@pytest.fixture(scope="session")
def eval_report_cache() -> dict[tuple[str, tuple[str, ...]], object]:
    return {}


@pytest.fixture(scope="session")
def eval_output_root(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("eval_output_root")


@pytest.fixture()
def eval_report(eval_case: EvalCase | None, eval_output_root: str, eval_report_cache: dict[tuple[str, tuple[str, ...]], object]):
    if eval_case is None:
        pytest.skip("No evaluator config selected. Use --eval-config, --eval-config-glob, or --all-eval-configs.")
    key = (eval_case.config_path, eval_case.overrides)
    if key not in eval_report_cache:
        Path(eval_output_root).mkdir(parents=True, exist_ok=True)
        eval_report_cache[key] = run_eval_case(eval_case, eval_output_root)
    return eval_report_cache[key]
