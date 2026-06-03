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

"""CLI entry point for full torch-vs-JAX evaluation."""

from __future__ import annotations

import argparse
import json

from .config import load_eval_config
from .runner import run_evaluation
from .utils import jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layered torch-to-JAX equivalence evaluation.")
    parser.add_argument("config", help="Path to evaluator yaml/json config.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted key=value overrides.")
    args = parser.parse_args()
    config = load_eval_config(args.config, args.overrides)
    report = run_evaluation(config)
    print(json.dumps(jsonable(report.summary()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
