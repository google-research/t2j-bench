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

"""Apply the release's idempotent GPU and artifact-scrubber patches to Harbor 0.20.0.

Each source anchor is checked before replacement. The patch fails if the installed
package layout differs from the supported version.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def harbor_root():
    """Where the running `harbor` actually imports from, not where it might be."""
    exe = shutil.which("harbor")
    if not exe:
        sys.exit("ERROR: harbor not on PATH (uv tool install harbor==0.20.0)")
    out = subprocess.run(
        [str(Path(exe).resolve().parent / "python"), "-c",
         "import harbor, pathlib; print(pathlib.Path(harbor.__file__).parent)"],
        capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip())
    for cand in Path(exe).resolve().parent.parent.glob("lib/python*/site-packages/harbor"):
        return cand
    sys.exit("ERROR: could not locate the harbor package")


def _require_unique(src, anchor, path, what):
    n = src.count(anchor)
    if n != 1:
        sys.exit(
            f"ERROR: cannot {what} in {path}: expected exactly one occurrence of the "
            f"anchor, found {n}. Harbor's source moved; update the patch in "
            f"tools/patch_harbor.py rather than hand-editing site-packages.")


# --- patch 1: GPU allocation + device pinning (environments/docker/*) ---
GPU_MARKER = "T2J_GPU_DEVICE_IDS"
RESERVATION_OLD = """    if gpus is not None and gpus > 0:
        reservations["devices"] = [
            {"driver": "nvidia", "count": gpus, "capabilities": ["gpu"]}
        ]"""
RESERVATION_NEW = '''    if gpus is not None and gpus > 0:
        # Pin to specific devices when the caller leased them. `count: N` is resolved by
        # the NVIDIA runtime to devices 0..N-1 unconditionally, so concurrent trials would
        # otherwise share one card while the rest of the host sits idle. `device_ids` and
        # `count` are mutually exclusive in the compose schema -- emit exactly one.
        _pinned = [
            d.strip()
            for d in os.environ.get("T2J_GPU_DEVICE_IDS", "").split(",")
            if d.strip()
        ]
        if _pinned:
            reservations["devices"] = [
                {"driver": "nvidia", "device_ids": _pinned, "capabilities": ["gpu"]}
            ]
        else:
            reservations["devices"] = [
                {"driver": "nvidia", "count": gpus, "capabilities": ["gpu"]}
            ]'''
SIG_PARAM = "    gpus: int | None = None,\n) -> Path:"
SIG_PARAM_MISSING_AFTER = "    memory_limit_mb: int | None = None,\n) -> Path:"
SIG_PARAM_REPLACEMENT = (
    "    memory_limit_mb: int | None = None,\n    gpus: int | None = None,\n) -> Path:")
CAPS_OLD = "            gpus=True,"
CAPS_ANCHOR = """    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
"""
CAPS_REPLACEMENT = CAPS_ANCHOR + "            gpus=True,\n"
CALL_OLD = "            gpus=self._effective_gpus,"
CALL_ANCHOR = """            memory_limit_mb=self._resource_limit_value(
                "memory", auto_mode=ResourceMode.LIMIT
            ),
        )
"""
CALL_REPLACEMENT = """            memory_limit_mb=self._resource_limit_value(
                "memory", auto_mode=ResourceMode.LIMIT
            ),
            gpus=self._effective_gpus,
        )
"""


def patch_init(path, apply):
    src = path.read_text()
    todo = []
    if "import os\n" not in src.split("def ")[0]:
        todo.append("add `import os`")
        if apply:
            src = src.replace("import json\n", "import json\nimport os\n", 1)
    if SIG_PARAM not in src:
        if SIG_PARAM_MISSING_AFTER not in src:
            sys.exit(f"ERROR: unrecognised write_resources_compose_file signature in {path}")
        todo.append("add `gpus` parameter")
        if apply:
            src = src.replace(SIG_PARAM_MISSING_AFTER, SIG_PARAM_REPLACEMENT, 1)
    if GPU_MARKER not in src:
        if RESERVATION_OLD in src:
            todo.append("make the device reservation honour T2J_GPU_DEVICE_IDS")
            if apply:
                src = src.replace(RESERVATION_OLD, RESERVATION_NEW, 1)
        else:
            todo.append("emit the device reservation (with T2J_GPU_DEVICE_IDS support)")
            if apply:
                anchor = "    if reservations:\n"
                if anchor not in src:
                    sys.exit(f"ERROR: cannot find the reservations block in {path}")
                src = src.replace(anchor, RESERVATION_NEW + "\n\n" + anchor, 1)
    if apply and todo:
        path.write_text(src)
    return todo


def patch_docker(path, apply):
    src = path.read_text()
    todo = []
    if CAPS_OLD not in src:
        todo.append("set gpus=True in the capabilities property")
        if apply:
            _require_unique(src, CAPS_ANCHOR, path, "set gpus=True in `capabilities`")
            src = src.replace(CAPS_ANCHOR, CAPS_REPLACEMENT, 1)
    if CALL_OLD not in src:
        todo.append("pass gpus=self._effective_gpus to the resources writer")
        if apply:
            _require_unique(src, CALL_ANCHOR, path,
                            "forward gpus to write_resources_compose_file")
            src = src.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)
    if apply and todo:
        path.write_text(src)
    return todo


# --- patch 2: minimum-secret-length guard on the artifact scrubber (trial/trial.py) ---
SCRUB_MARKER = "T2J_MIN_SECRET_LEN"
MIN_SECRET_LEN = 12
SCRUB_ANCHOR = (
    "                        value = resolve_env_vars({key: value})[key]\n"
    "                        if value:\n"
    "                            secrets.add(value)\n")
SCRUB_REPLACEMENT = (
    "                        value = resolve_env_vars({key: value})[key]\n"
    "                        # T2J_MIN_SECRET_LEN -- a short value here drives a blind\n"
    "                        # str.replace over every text file in the trial dir, so it\n"
    "                        # corrupts artifacts rather than protecting anything: e.g. an\n"
    "                        # env var whose name matches AUTH but whose value is `1` would\n"
    "                        # redact every digit 1 and leave reward.json as invalid JSON.\n"
    f"                        if value and len(value) >= {MIN_SECRET_LEN}:\n"
    "                            secrets.add(value)\n")


def patch_trial(path, apply):
    src = path.read_text()
    todo = []
    if SCRUB_MARKER not in src:
        todo.append(f"require len(secret) >= {MIN_SECRET_LEN} before blind redaction")
        if apply:
            _require_unique(src, SCRUB_ANCHOR, path, "add the minimum-secret-length guard")
            src = src.replace(SCRUB_ANCHOR, SCRUB_REPLACEMENT, 1)
            path.write_text(src)
    return todo



def main():
    root = harbor_root()
    init_py = root / "environments" / "docker" / "__init__.py"
    docker_py = root / "environments" / "docker" / "docker.py"
    trial_py = root / "trial" / "trial.py"
    for p in (init_py, docker_py, trial_py):
        if not p.is_file():
            sys.exit(f"ERROR: missing {p}")

    todo = patch_init(init_py, apply=True)
    todo += patch_docker(docker_py, apply=True)
    todo += patch_trial(trial_py, apply=True)

    # Re-read to confirm the writes landed.
    remaining = (patch_init(init_py, apply=False)
                 + patch_docker(docker_py, apply=False)
                 + patch_trial(trial_py, apply=False))
    if remaining:
        sys.exit("ERROR: harbor patch did not fully apply: " + ", ".join(remaining))
    if todo:
        print(">> patched installed harbor: " + "; ".join(todo))
    else:
        print(">> harbor already patched (GPU allocation + device pinning + scrub guard)")


if __name__ == "__main__":
    main()
