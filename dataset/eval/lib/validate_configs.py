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

"""Validate evaluator configs without running model code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_eval_config


def _iter_configs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.suffix in {".yaml", ".yml", ".json"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluator configs.")
    parser.add_argument("paths", nargs="*", help="Explicit config paths. Defaults to ./configs if it exists.")
    args = parser.parse_args()
    if args.paths:
        config_paths = [Path(item).resolve() for item in args.paths]
    else:
        default_root = Path.cwd() / "configs"
        config_paths = _iter_configs(default_root) if default_root.exists() else []
    results = []
    for path in config_paths:
        try:
            load_eval_config(str(path))
            results.append({"config": str(path), "passed": True})
        except Exception as exc:
            results.append({"config": str(path), "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
