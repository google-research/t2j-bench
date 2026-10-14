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

"""Check inference grading preconditions before pytest starts.

The pinned GPU and _harness_checks failures are fatal: a host or harness that
cannot support the frozen contract must not produce a candidate score.
Driver/XLA provenance drift produces a warning and provenance_check.json;
it is not treated as a measured candidate failure. The device-matching rule is
shared with conftest, and per-task collection expectations live in instance.yaml.

Usage:
    python _grade_precheck.py --golden /tests/golden --out /logs/verifier
    python _grade_precheck.py --golden ... --fake-device 'NVIDIA A100-SXM4-80GB'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _conftest():
    """The grader's own ``conftest`` as a plain module, for its device matcher.

    Loaded by path rather than imported by name so this file works whether it is
    run from ``/tests`` or from a checkout, and so it does not depend on pytest
    having already put the directory on ``sys.path``. ``conftest`` imports cleanly
    headless (no GPU touched) -- it only defines hooks.
    """
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    spec = importlib.util.spec_from_file_location("_t2j_conftest", _HERE / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nvml_facts() -> tuple[str | None, str | None]:
    """``(gpu_model, driver_version)`` from NVML, or ``(None, None)``.

    NVML is preferred over ``jax.devices()`` for the SKU because it answers
    WITHOUT initialising a CUDA context -- this runs before pytest and must not
    leave a claimed device behind -- and because it is the only source for the
    driver version, which jax does not expose. ``nvidia-ml-py`` is a pin of the
    verifier base, so it is present by construction.
    """
    try:
        import pynvml
    except ImportError:
        return None, None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - no driver, no GPU, no NVML: all "unknown"
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        return (
            name.decode() if isinstance(name, bytes) else str(name),
            driver.decode() if isinstance(driver, bytes) else str(driver),
        )
    except Exception:  # noqa: BLE001
        return None, None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def jax_device_kind() -> str | None:
    """The SKU jax itself reports -- the fallback, and it DOES initialise CUDA."""
    try:
        import jax

        devices = jax.devices()
        return str(devices[0].device_kind) if devices else None
    except Exception:  # noqa: BLE001
        return None


def jax_version() -> str | None:
    try:
        import jax

        return str(jax.__version__)          # no backend init: attribute only
    except Exception:  # noqa: BLE001
        return None


#: Distributions whose build determines what XLA actually executes. RECORDED, not
#: compared -- see :func:`jax_build_facts`.
_BUILD_DISTS = ("jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
                "nvidia-cublas-cu12", "nvidia-cudnn-cu12")


def jax_build_facts() -> dict:
    """The versions that decide what XLA runs, RECORDED rather than compared.

    WHAT THE DRIFT CHECK ABOVE CAN AND CANNOT SEE, stated plainly because the name
    ``jax_xla_version`` promises more than it delivers.
    ``meta.provenance.jax_xla_version`` is a **hardcoded literal** in the freezer,
    compared against ``jax.__version__``. So:

    * it DOES catch a deliberate pin change -- someone moves the class to jax 0.12
      and re-runs a grade against 0.11-era goldens;
    * it does NOT catch a wheel-level rebuild at the same version number: a new
      ``jaxlib`` build, a new ``jax-cuda12-plugin``, or a bumped ``nvidia-*-cu12``.
      That is precisely the drift the carry was worried about, and the frozen field
      cannot express it.

    Closing that needs the FREEZER to stamp these versions, which moves a golden and
    is not this task's to do. What is free is recording them at grade time, so a
    later reader comparing two runs' ``provenance_check.json`` can see a rebuild even
    though nothing in the golden pinned it. No value here is ever compared to
    anything; a missing distribution is reported as ``None``, not as drift.
    """
    from importlib import metadata

    out: dict = {}
    for name in _BUILD_DISTS:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 - absence is a value
            out[name] = None
    return out


def harness_health(golden_dir: Path) -> list[tuple[str, str]]:
    """The benchmark's own self-checks. ``[]`` means healthy; see ``_harness_checks``."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import _harness_checks
    from fixtures import load_golden

    return _harness_checks.run(load_golden(golden_dir))


def _prove_guards() -> int:
    """Break each precondition on a THROWAWAY golden and require it to fail.

    Standing rule 4: a guard that has only ever been exercised in the passing
    direction is indistinguishable from a comment. ``--fake-device`` already proves
    the SKU pin; this proves the harness checks, which cannot be provoked by an
    argument because their subject is the golden tree itself.

    The golden is mirrored as a SYMLINK FARM (33 MB of artifacts, none of them
    copied) so each mutation is one file rewritten in a temp directory and the real
    tree is never touched.

    A NEGATIVE CONTROL RUNS FIRST. Without it, "the check failed" could just as
    easily mean the symlink farm itself is broken, and every proof below would be
    vacuous.
    """
    import shutil
    import tempfile

    real = _HERE / "golden"
    rc = 0

    def farm(root: Path) -> Path:
        out = root / "golden"
        out.mkdir(parents=True)
        for src in sorted(real.iterdir()):
            (out / src.name).symlink_to(src.resolve())
        return out

    def expect(label: str, golden_dir: Path, want: str | None,
               contains: str | None = None) -> None:
        """``contains`` pins WHICH CLAUSE of a multi-clause check fired.

        Two checks here have more than one ``assert``, and matching on the check
        NAME alone cannot tell them apart -- a row aimed at the second clause would
        report "proven" if the first one tripped instead, which is the passing
        direction of a different guard wearing this row's label. Where a clause is
        the point, the row names a substring of its own failure message.
        """
        nonlocal rc
        try:
            failures = harness_health(golden_dir)
        except Exception as exc:  # noqa: BLE001 - a raise is also a failed expectation
            print(f"!! {label}: raised {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
            return
        names = [name for name, _ in failures]
        if want is None:
            if names:
                print(f"!! {label}: expected a clean bill of health, got {names}",
                      file=sys.stderr)
                rc = 1
            else:
                print(f"   {label}: OK (no findings)")
        elif want not in names:
            print(f"!! {label}: check_{want} did NOT fail; findings were {names}. "
                  "The guard is not enforcing anything.", file=sys.stderr)
            rc = 1
        elif contains is not None and not any(
                contains in msg for name, msg in failures if name == want):
            print(f"!! {label}: check_{want} failed, but on a DIFFERENT clause -- "
                  f"no finding mentions {contains!r}, so the clause this row aims "
                  "at is still unproven. Findings: "
                  f"{[msg for name, msg in failures if name == want]}",
                  file=sys.stderr)
            rc = 1
        else:
            print(f"   {label}: proven -- check_{want} fails")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # [1/9] negative control: the untouched golden is healthy through the farm.
        control = farm(tmp / "control")
        expect("[1/9] negative control", control, None)

        # [2/9] a required WITHIN-FILE token goes missing.
        #
        # NOT a deleted .npz, and the difference is the whole reason this proof was
        # worth writing. Deleting `sampler.npz` never reaches `check_golden_manifest`:
        # `fixtures.load_golden` opens every file-level catalog artifact eagerly and
        # raises `FileNotFoundError` first (which is what the check's
        # own docstring says). Dropping the `logits` ARRAY out of
        # `fp32_prefill_forward.npz` leaves a file that loads fine and only the
        # manifest's token grammar can see the hole -- which is one of exactly two
        # detections this check has to itself on a `core` instance.
        import numpy as np

        token = farm(tmp / "token")
        (token / "fp32_prefill_forward.npz").unlink()
        with np.load(real / "fp32_prefill_forward.npz") as src:
            kept = {k: src[k] for k in src.files if k != "logits"}
        np.savez(token / "fp32_prefill_forward.npz", **kept)
        expect("[2/9] dropped golden array (within-file token)", token,
               "golden_manifest")

        # [3/9] a threshold BAND misspelled by one character -- the failure mode that
        # silently loosens fp32 abs_tol 1.0e-4 -> 1.0e-3.
        typo = farm(tmp / "typo")
        (typo / "meta.json").unlink()
        meta = json.loads((real / "meta.json").read_text())
        meta["fp32_threshold"] = meta.pop("fp32_thresholds")
        (typo / "meta.json").write_text(json.dumps(meta))
        expect("[3/9] misspelled threshold band", typo,
               "threshold_extras_name_something_a_check_reads",
               contains="missing_bands")

        # [4/9] an applicability gate keyed on a name no module defines. The TWO-SIDED
        # typo -- renamed in `conftest._APPLICABLE`, in `_DESELECTION_KIND` and in
        # `_APPLICABLE_OWNER` -- because that is the realistic single-edit mistake and
        # it is the only one that reaches the AST clause. A one-sided typo trips the
        # cheaper attribution clause and would leave the per-module walk unproven.
        import _harness_checks
        import conftest
        gate = farm(tmp / "gate")
        tables = (conftest._APPLICABLE, conftest._DESELECTION_KIND,
                  _harness_checks._APPLICABLE_OWNER)
        saved = [dict(t) for t in tables]
        try:
            for table in tables:
                table["test_hidden_state"] = table.pop("test_hidden_states")
            expect("[4/9] typo'd applicability gate (two-sided)", gate,
                   "applicability_table_names_real_checks")
        finally:
            for table, backup in zip(tables, saved):
                table.clear()
                table.update(backup)

        # [5/9] a STALE golden set: a meta frozen against a different instance.yaml.
        # The sampler grid is the illustrative case -- rolling the
        # goldens back to a smaller previous grid against a larger instance.yaml
        # grades the reference as fully passing on a grid with zero
        # discriminating mutations. The meta is rewritten here rather than the yaml
        # because the yaml is the CONTRACT: this stands in for "the goldens are old",
        # which is the reachable direction.
        stale = farm(tmp / "stale")
        (stale / "meta.json").unlink()
        stale_meta = json.loads((real / "meta.json").read_text())
        stale_meta["sampler_grid"] = (stale_meta.get("sampler_grid") or [])[:-1]
        (stale / "meta.json").write_text(json.dumps(stale_meta))
        expect("[5/9] stale golden (meta frozen against another instance.yaml)",
               stale, "meta_mirrors_instance_yaml")

        # [6/9] a scenario key in NEITHER the input allow-list nor the output
        # deny-list. The realistic shape is a future freezer growing one key, so
        # the mutation adds one rather than removing anything. Left ungated, the
        # same condition raises out of `fixtures.build_candidate` -- the
        # SESSION-SCOPED `cand` fixture -- and errors every candidate-facing
        # check at the non-importing floor, for a defect in the
        # freeze. Proven here, at the severity that writes no reward.json at all.
        unclassified = farm(tmp / "unclassified")
        (unclassified / "scenario.json").unlink()
        scenario = json.loads((real / "scenario.json").read_text())
        scenario["prefill_chunk_budget"] = 4        # a plausible future INPUT key
        (unclassified / "scenario.json").write_text(json.dumps(scenario))
        expect("[6/9] an unclassified scenario key", unclassified,
               "scenario_keys_are_classified", contains="unclassified")

        # [7/9] the input/output partition is not a partition: ONE KEY IN BOTH LISTS.
        #
        # `check_scenario_keys_are_classified` has two asserts and [6/9] only ever
        # reached the second, so `assert not overlap` was a comment by standing
        # rule 4 -- the only clause here without a row of its own.
        #
        # The mutation is the one that matters rather than an arbitrary collision:
        # `decode_token_ids` is the ONLY declared output, and it is declared one
        # because it is the reference's own greedy continuation. Put it in the input
        # list too and `fixtures.build_candidate`'s
        # `{k: golden.scenario[k] for k in _SCENARIO_INPUT_KEYS ...}` copies the
        # frozen answer into the sandbox UNDER ITS OWN NAME, while the same file
        # goes on declaring that no frozen output crosses. A port could then read
        # the answer key. That is the overlap this clause exists to refuse, and it
        # is a plausible single edit: the payload already carries these ids, renamed
        # to `decode_input_token_ids` for teacher forcing, so "add it to the inputs"
        # looks like tidying.
        #
        # In memory, not on disk: the partition is a pair of module-level tuples in
        # `fixtures`, not anything the golden tree can express. Restored in a
        # `finally` for the same reason [4/9]'s tables are -- this process goes on
        # to run [8/9].
        import fixtures
        overlap = farm(tmp / "overlap")
        _saved_inputs = fixtures._SCENARIO_INPUT_KEYS
        try:
            fixtures._SCENARIO_INPUT_KEYS = tuple(_saved_inputs) + ("decode_token_ids",)
            expect("[7/9] a scenario key in BOTH the input and output lists",
                   overlap, "scenario_keys_are_classified", contains="in_both")
        finally:
            fixtures._SCENARIO_INPUT_KEYS = _saved_inputs

        # [8/9] a frozen decode run too short for the steady-state compilation leg.
        # ONE id, not zero, because one is the case that passes VACUOUSLY
        # (the counted region runs zero iterations and `decode_steady_state == 0`
        # holds for any candidate); zero would additionally trip the probe's other
        # guard. Without this guard, `test_compilation_is_cached` SKIPS on this, taking the
        # check out of both terms of passed/(passed+failed+error) and scoring a
        # re-jitting port full marks.
        short = farm(tmp / "short_decode")
        (short / "scenario.json").unlink()
        short_scenario = json.loads((real / "scenario.json").read_text())
        short_scenario["decode_token_ids"] = (
            list(short_scenario.get("decode_token_ids") or [0])[:1])
        (short / "scenario.json").write_text(json.dumps(short_scenario))
        expect("[8/9] decode run too short to grade the compilation leg", short,
               "scenario_carries_a_gradable_decode_run")

        # [9/9] a CALIBRATED THRESHOLD A GRADED MODULE DEMANDS IS ABSENT FROM META.
        #
        # The other direction of the same check, and the one every clause [3/9]
        # proves is blind to: those validate the keys meta HAS, so a key it LACKS
        # is legal by vacuity. `test_behavioral` reads five of them with a
        # KeyError-loud `golden.fp32_extras[...]` on purpose -- a calibrated bound
        # has no honest default -- and that KeyError lands inside a GRADED check,
        # which `summary_to_reward` counts as an `error` and charges to the port.
        #
        # With `held_out_seq_atol` deleted from
        # BOTH `instance.yaml` and `meta.json`, all six self-checks PASS
        # (`check_threshold_extras_name_something_a_check_reads` iterates the keys
        # meta carries; `check_meta_mirrors_instance_yaml` compares yaml -> meta
        # only, and the yaml no longer had it either) and `test_held_out_logprobs`
        # ERRORS, for an instance-authoring omission.
        #
        # DELETED FROM META ONLY here, which additionally trips
        # `check_meta_mirrors_instance_yaml` -- `expect` tolerates extra findings
        # and `contains` pins the clause this row is aimed at. Deleting it from the
        # yaml too is the state seen, but the yaml is the CONTRACT and
        # this prover farms goldens, not trees.
        uncalibrated = farm(tmp / "uncalibrated")
        (uncalibrated / "meta.json").unlink()
        thin = json.loads((real / "meta.json").read_text())
        thin["fp32_thresholds"] = {k: v for k, v in thin["fp32_thresholds"].items()
                                   if k != "held_out_seq_atol"}
        (uncalibrated / "meta.json").write_text(json.dumps(thin))
        expect("[9/9] a calibrated threshold key no meta carries", uncalibrated,
               "threshold_extras_name_something_a_check_reads",
               contains="fp32_thresholds.held_out_seq_atol")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(">> harness guards proven" if rc == 0 else "!! harness guard proof FAILED",
          file=sys.stderr if rc else sys.stdout)
    return rc


def _harness_only(golden_dir: Path) -> int:
    """The benchmark self-checks alone (``_harness_checks.CHECKS``) -- the
    automated inference-checks path.

    No GPU, no candidate, no device pin: just ``golden/``, ``conftest`` and a few AST
    walks, so it runs on an authoring box in about a second. This exists because
    moving those checks out of the graded suite left them running in no developer
    path at all, and "invisible until somebody does a container grade" is a feedback
    loop measured in minutes rather than seconds.

    TWO SKIPS, BOTH LOUD, BOTH EXIT 0, and the reason each is a skip rather than a
    failure is that neither is a statement about the benchmark:

    * NO GOLDENS. They are git-ignored, so a fresh clone genuinely has none and CI
      must stay green through this target. The Harbor path is unaffected -- there a
      missing golden is fatal, and the message names the freezer.
    * WRONG INTERPRETER. The graded tree needs the grade env (``dataclass(slots=True)``
      is 3.10+, and this repo's system ``python3`` is 3.9), and a developer target that
      hard-failed on the default interpreter would just be turned off.

    A golden that is PRESENT and unhealthy exits 1. That is the case this exists for.
    """
    meta = golden_dir / "meta.json"
    if not meta.is_file():
        print(f"[harness] SKIPPED: no goldens at {golden_dir} (they are git-ignored). "
              "Nothing was checked. Fetch or re-freeze them to run this.")
        return 0
    try:
        findings = harness_health(golden_dir)
    except (ImportError, TypeError, SyntaxError) as exc:
        print(f"[harness] SKIPPED: this interpreter cannot import the graded tree "
              f"({type(exc).__name__}: {exc}). It needs the grade environment "
              "(Python 3.12); re-run with GRADE_PYTHON=.venv-inference/bin/python. "
              "Nothing was checked.")
        return 0
    except Exception as exc:            # noqa: BLE001 - ANY unloadable golden
        # WIDENED, the same fix `_t2j/tests/conftest.py` carries at its own
        # `load_golden` call. `(OSError, ValueError, KeyError)` was too narrow:
        # `fixtures._guard` raises `RuntimeError` on a Git LFS pointer and
        # `np.load` raises `zipfile.BadZipFile` on a truncated `.npz`, and
        # neither subclasses those three -- so an unloadable golden escaped this
        # handler as an unhandled traceback instead of the exit code that names
        # the repair. `BaseException` is deliberately NOT caught:
        # KeyboardInterrupt/SystemExit must still propagate, so an interrupted
        # run stops rather than being recorded as a broken golden set. The
        # narrower clause above still runs first, so an interpreter that cannot
        # import the tree keeps its own SKIPPED message.
        print(f"[harness] FAIL: the golden set could not be loaded at all -- "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if findings:
        print(f"[harness] FAIL: {len(findings)} benchmark self-check(s) failed. These "
              "describe the TASK, not a candidate.", file=sys.stderr)
        for name, message in findings:
            print(f"   --- check_{name} ---\n   {message}", file=sys.stderr)
        return 1
    import _harness_checks
    print(f"[harness] ok    {golden_dir.parent.parent.name} "
          f"({len(_harness_checks.CHECKS)} self-checks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", default=str(_HERE / "golden"))
    ap.add_argument("--out", default=None, help="where to write provenance_check.json")
    ap.add_argument("--fake-device", default=None,
                    help="pretend the host reports this SKU (guard proof only)")
    ap.add_argument("--prove-guards", action="store_true",
                    help="break each harness precondition and require it to fail")
    ap.add_argument("--harness-only", action="store_true",
                    help="run only the benchmark self-checks (no GPU, no candidate)")
    args = ap.parse_args()

    if args.prove_guards:
        return _prove_guards()
    if args.harness_only:
        return _harness_only(Path(args.golden))

    golden_dir = Path(args.golden)
    meta_path = golden_dir / "meta.json"
    if not meta_path.is_file():
        print(f"!! PRECHECK FAIL: no golden at {meta_path}. The verifier image did "
              "not receive tests/golden/ -- check .dockerignore handling in "
              "build_harbor_images.sh.", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text())
    prov = meta.get("provenance") or {}

    want_gpu = str(prov.get("gpu_model") or "").strip()
    nvml_gpu, driver = nvml_facts()
    got_gpu = args.fake_device or nvml_gpu or jax_device_kind()

    # ---- the SKU pin: fatal ------------------------------------------------
    if not want_gpu:
        print("!! PRECHECK FAIL: golden/meta.json carries no provenance.gpu_model, "
              "so the pinned-device precondition cannot be verified. An "
              "unverifiable pin is not a satisfied pin.", file=sys.stderr)
        return 2
    matches = _conftest()._device_matches(want_gpu, got_gpu or "")
    if not matches:
        print(f"!! PRECHECK FAIL: this grade must run on {want_gpu!r}; this host "
              f"reports {got_gpu!r}. Every tier -- Spec included -- is calibrated "
              "against that SKU, so running elsewhere produces numbers that cannot "
              "be compared. No reward.json is written: the grade did not happen.",
              file=sys.stderr)
        return 2

    # ---- provenance drift: warn -------------------------------------------
    want_driver = str(prov.get("cuda_driver") or "") or None
    want_xla = str(prov.get("jax_xla_version") or "") or None
    got_xla = jax_version()
    drift = {}
    if want_driver and driver and want_driver != driver:
        drift["cuda_driver"] = {"frozen_under": want_driver, "grading_on": driver}
    if want_xla and got_xla and want_xla != got_xla:
        drift["jax_xla_version"] = {"frozen_under": want_xla, "grading_on": got_xla}

    report = {
        "gpu_model": {"pinned": want_gpu, "observed": got_gpu, "matches": True},
        "cuda_driver": {"frozen_under": want_driver, "grading_on": driver},
        "jax_xla_version": {"frozen_under": want_xla, "grading_on": got_xla},
        # Recorded, never compared -- the golden pins no build to compare against.
        # See jax_build_facts for what the comparison above does and does not detect.
        "runtime_build": jax_build_facts(),
        "drift": drift,
        "severity": "warn",
    }
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "provenance_check.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f">> device pin OK: {got_gpu!r} satisfies {want_gpu!r}")
    if drift:
        for key, pair in sorted(drift.items()):
            print(f"!! WARNING: {key} DRIFT -- goldens frozen under "
                  f"{pair['frozen_under']!r}, grading on {pair['grading_on']!r}. "
                  "This is reported, not enforced: no measurement in this phase "
                  "shows that this pair moves a graded number. If a tier fails "
                  "unexpectedly, suspect this first.")
    else:
        print(f">> provenance OK: cuda_driver {driver!r}, jax {got_xla!r} match "
              "what the goldens were frozen under (jaxlib "
              f"{report['runtime_build'].get('jaxlib')} recorded, not pinned)")

    # ---- the benchmark's own health: fatal, no reward file ------------------
    #
    # A GOLDEN FILE THAT IS SIMPLY ABSENT NEVER REACHES THE HARNESS CHECKS.
    # `fixtures.load_golden` opens every file-level catalog artifact eagerly and
    # raises first (proven by `--prove-guards`), which is why its [2/9] case
    # deletes an ARRAY rather than a file. That raise is the louder answer (its
    # message names the artifact and the re-freeze command), so it is reported here
    # rather than allowed to escape as a traceback: same severity, same "no
    # reward.json", and an exit code instead of an unhandled exception.
    try:
        findings = harness_health(golden_dir)
    except Exception as exc:            # noqa: BLE001 - ANY unloadable golden
        # WIDENED, exactly as the sibling handler in `_harness_only` above and
        # as `_t2j/tests/conftest.py` already was: `(OSError, ValueError,
        # KeyError)` misses `fixtures._guard`'s `RuntimeError` (a Git LFS
        # pointer) and `zipfile.BadZipFile` (a truncated `.npz`), and this is
        # the handler that decides whether the grade happens at all -- an
        # escaping traceback here means no `reward.json` and no exit code that
        # names the repair. `BaseException` is deliberately NOT caught:
        # KeyboardInterrupt/SystemExit must still propagate.
        print(f"!! PRECHECK FAIL: the golden set could not be loaded at all -- "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if findings:
        print(f"!! PRECHECK FAIL: {len(findings)} harness check(s) failed. These "
              "grade the BENCHMARK, not the candidate -- a golden set with a hole "
              "in it, a gate that deselects nothing, a threshold key no check "
              "reads, a scenario key nobody classified, or a frozen decode run too "
              "short for the check that grades it. No reward.json is written: the "
              "grade did not happen, and the repair is to the task, not to the "
              "port.", file=sys.stderr)
        for name, message in findings:
            print(f"   --- check_{name} ---\n   {message}", file=sys.stderr)
        return 2
    import _harness_checks
    print(f">> harness OK: all {len(_harness_checks.CHECKS)} benchmark self-checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
