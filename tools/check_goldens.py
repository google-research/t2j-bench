#!/usr/bin/env python3
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

"""Check a task's frozen references before building or running its verifier.

Exit status: 0 complete, 1 missing artifacts or unresolved LFS pointers,
2 the manifest/configuration could not be checked. Missing tokens are written
to stdout; configuration errors are written to stderr.
"""

import importlib.util
import inspect
import sys
from pathlib import Path


class Unavailable(Exception):
    """The check could not be run at all -- exit 2, never exit 0."""


def missing_tokens(task_dir):
    tests = task_dir / "tests"
    manifest_py = tests / "_t2j" / "golden_manifest.py"
    instance_yaml = tests / "instance.yaml"
    if not manifest_py.is_file():
        raise Unavailable(f"no golden manifest at {manifest_py}")
    if not instance_yaml.is_file():
        raise Unavailable(f"no instance config at {instance_yaml}")
    try:
        import yaml
    except ImportError as e:
        raise Unavailable("python has no `yaml` module (use .venv-inference/bin/python)") from e
    try:
        cfg = yaml.safe_load(instance_yaml.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Unavailable(f"cannot read configuration at {instance_yaml}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Unavailable(f"{instance_yaml} must contain a YAML mapping")
    # `mode` for the inference class, `training_method` for the training class.
    key = cfg.get("mode") or cfg.get("training_method")
    if not key:
        raise Unavailable(f"{instance_yaml} has neither `mode` nor `training_method`")
    spec = importlib.util.spec_from_file_location("_t2j_golden_manifest", manifest_py)
    if spec is None or spec.loader is None:
        raise Unavailable(f"cannot import {manifest_py}")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    if not getattr(gm, "REQUIRED_GOLDENS", {}).get(str(key).lower()):
        # An unknown mode would report nothing missing; treat it as a config error.
        raise Unavailable(
            f"{manifest_py.parent.parent.name}: manifest has no entry for '{key}' -- it "
            f"speaks only for {sorted(getattr(gm, 'REQUIRED_GOLDENS', {}))}")
    kwargs = {}
    if "include_preserved" in inspect.signature(gm.missing_required).parameters:
        kwargs["include_preserved"] = True
    return list(gm.missing_required(tests / "golden", key, **kwargs))


def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/check_goldens.py tasks/<task>", file=sys.stderr)
        return 2
    task_dir = Path(sys.argv[1])
    try:
        missing = missing_tokens(task_dir)
    except Unavailable as exc:
        print(f"golden completeness check: cannot check {task_dir}: {exc}", file=sys.stderr)
        return 2
    # Existence alone is insufficient: a clone may contain LFS pointers instead of data.
    golden_dir = task_dir / "tests" / "golden"
    for path in sorted(golden_dir.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    missing.append("lfs-pointer:" + str(path.relative_to(golden_dir)))
    if missing:
        print(" ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
