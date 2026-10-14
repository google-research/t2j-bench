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

"""CPU-only regression tests for release utilities; no checkpoints or Docker required."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def golden_task(tmp_path):
    tests = tmp_path / "tests"
    (tests / "_t2j").mkdir(parents=True)
    (tests / "golden").mkdir()
    (tests / "instance.yaml").write_text("training_method: sft\n")
    (tests / "_t2j" / "golden_manifest.py").write_text(
        "REQUIRED_GOLDENS = {'sft': ['meta.json']}\n"
        "def missing_required(directory, key, include_preserved=False):\n"
        "    assert include_preserved\n"
        "    return [] if (directory / 'meta.json').is_file() else ['meta.json']\n"
    )
    return tmp_path


@pytest.mark.parametrize("state, expected", [("complete", 0), ("missing", 1), ("pointer", 1)])
def test_golden_check_distinguishes_data_from_lfs_pointers(golden_task, state, expected):
    target = golden_task / "tests" / "golden" / "meta.json"
    if state == "complete":
        target.write_text("{}\n")
    elif state == "pointer":
        target.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 3\n")
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_goldens.py"), str(golden_task)],
                            capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    if state == "pointer":
        assert "lfs-pointer:meta.json" in result.stdout


def test_unknown_golden_mode_is_a_configuration_error(golden_task):
    (golden_task / "tests/instance.yaml").write_text("mode: unknown\n")
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_goldens.py"), str(golden_task)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "manifest has no entry" in result.stderr


@pytest.mark.parametrize("contents", ["training_method: [\n", "- sft\n"])
def test_invalid_yaml_reports_a_configuration_error(golden_task, contents):
    (golden_task / "tests/instance.yaml").write_text(contents)
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_goldens.py"), str(golden_task)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "cannot check" in result.stderr
    assert "Traceback" not in result.stderr


def test_runner_help_and_unknown_task_do_not_require_harbor_or_a_gpu():
    for args, expected, text in [( ["--help"], 0, "Usage:"),
                                  (["missing_task"], 2, "task not found"),
                                  (["dpo_bloom", "--debug"], 2, "must follow '--'")]:
        result = subprocess.run(["bash", str(ROOT / "run_eval.sh"), *args],
                                capture_output=True, text=True)
        assert result.returncode == expected
        assert text in result.stdout + result.stderr


def test_harbor_scrubber_patch_is_idempotent_and_rejects_unknown_source(tmp_path):
    module = load("patch_harbor")
    source = tmp_path / "trial.py"
    source.write_text(module.SCRUB_ANCHOR)
    assert module.patch_trial(source, apply=True)
    patched = source.read_bytes()
    assert module.patch_trial(source, apply=True) == []
    assert source.read_bytes() == patched
    source.write_text("unexpected layout\n")
    with pytest.raises(SystemExit, match="expected exactly one occurrence"):
        module.patch_trial(source, apply=True)
    assert source.read_text() == "unexpected layout\n"


def test_failed_lfs_download_removes_partial_file_and_reports_protocol_error(tmp_path):
    # Load without a real bucket client; the injected client fails after writing a partial file.
    import types

    hub = types.ModuleType("huggingface_hub")
    hub.batch_bucket_files = lambda *args, **kwargs: None
    hub.get_bucket_paths_info = lambda *args, **kwargs: []

    def failing_download(bucket, pairs, **kwargs):
        Path(pairs[0][1]).write_bytes(b"partial")
        raise OSError("interrupted transfer")

    hub.download_bucket_files = failing_download
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        module = load("lfs_hf_bucket")

    messages = io.StringIO()
    request = json.dumps({"event": "download", "oid": "a" * 64, "size": 7}) + "\n"
    with patch.object(module, "lfs_tmp_dir", return_value=str(tmp_path)), \
         patch.object(module.os, "dup", return_value=99), \
         patch.object(module.os, "dup2"), \
         patch.object(module.os, "fdopen", return_value=messages), \
         patch.object(module.sys, "stdin", io.StringIO(request)):
        module.serve()
    assert list(tmp_path.iterdir()) == []
    response = json.loads(messages.getvalue())
    assert response["event"] == "complete"
    assert "interrupted transfer" in response["error"]["message"]
