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

"""Convert the inference grader's summary to Harbor reward metrics.

The reward is the Spec pass fraction multiplied by an equal-weighted mean of
Numeric and Behavioral similarity scores, renormalized over dimensions with
scored checks. Binary stage gates, counts, and pooled pass_rate are diagnostics;
retired performance-stage results do not contribute to the reward.

The verifier runs this root-owned reducer after the unprivileged grading phase.
This protects the reducer and published reward from direct candidate writes,
but does not authenticate measurements in eval_summary.json: candidate workers
share the grading user's permissions. A separate trusted measurement boundary
would be needed to protect against forged grading inputs.
"""

import json
import sys

_STAGES = ("spec", "numeric", "behavioral")

# Equal weight per SCORED dimension, renormalized over the dimensions that actually ran (so
# a task missing one is scored on the one it has, not capped below the ceiling). Spec is NOT
# in this map: it is the multiplicative gate, not a weighted contributor. Mirrors the
# training class's ``_W_NUM`` / ``_W_BEH`` in ``tasks/*/tests/summary_to_reward.py``.
_W_NUM = 0.5
_W_BEH = 0.5


def _counts(v) -> dict:
    if isinstance(v, dict):
        return {k: int(v.get(k, 0) or 0) for k in ("passed", "failed", "skipped", "xfailed", "error")}
    return {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0}


def _stage_pass(c: dict) -> int:
    # A stage passes iff no failures/errors and at least one check ran.
    return 1 if (c["failed"] == 0 and c["error"] == 0 and c["passed"] > 0) else 0


def _rate(c: dict) -> float:
    ran = c["passed"] + c["failed"] + c["error"]
    return round(c["passed"] / ran, 4) if ran else 0.0


def _rate_raw(c: dict) -> float:
    # Unrounded pass fraction for the reward math; the rounded ``_rate`` feeds the
    # per-stage diagnostic keys.
    ran = c["passed"] + c["failed"] + c["error"]
    return c["passed"] / ran if ran else 0.0


def _f(v) -> float | None:
    """One finite float from ``v`` (a number), else ``None``."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not (f == f) or f in (float("inf"), float("-inf")):
        return None
    return f


def _dim_n(s: dict, name: str) -> int:
    """How many checks were SCORED in dimension ``name`` (``<name>.n``).

    0 if the block is absent or has no scored checks. This is the count of
    per-check similarities the conftest banked, NOT the binary pass/fail tally -- a
    behavioral stage whose six gate checks all passed but scored nothing has ``n`` from
    its eight fidelity checks only.
    """
    block = s.get(name)
    if not isinstance(block, dict):
        return 0
    try:
        return int(block.get("n", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _dim_score(s: dict, name: str, top_key: str) -> float:
    """The continuous [0,1] similarity score the grader emitted for dimension ``name``.

    Prefers the top-level ``top_key`` scalar; falls back to the ``name`` block's ``mean``.
    Missing / non-finite -> 0.0, clamped to [0,1].
    """
    block = s.get(name) if isinstance(s.get(name), dict) else {}
    for value in (s.get(top_key), block.get("mean")):
        f = _f(value)
        if f is not None:
            return min(1.0, max(0.0, f))
    return 0.0


def _reward(s: dict, stages: dict) -> float:
    """Spec-gated, dimension-weighted reward over the continuous numeric + behavioral means.

    ``spec_pass_rate`` multiplies a weighted mean of the numeric and behavioral SIMILARITY
    scores, the weights renormalized over the dimensions that ran. A dimension "ran" iff at
    least one of its checks was SCORED (``<dim>.n > 0``) -- so a behavioral stage whose six
    gate checks all passed but scored nothing (all eight fidelity checks deselected) does
    not contribute a phantom 0.0, it renormalizes OUT. Spec == 0 (including "no spec check
    ran") gates the reward to 0; no scored dimension running also yields 0.
    """
    spec_rate = _rate_raw(stages["spec"])
    numer = den = 0.0
    # Include a dimension only when at least one similarity score was recorded.
    if _dim_n(s, "numeric") > 0:
        numer += _W_NUM * _dim_score(s, "numeric", "numeric_score")
        den += _W_NUM
    # Behavioral: continuous similarity too. The eight fidelity checks are scored; the six
    # gate checks bank no score (a gate FAIL files a 0.0). "Ran" iff at least one was scored.
    if _dim_n(s, "behavioral") > 0:
        numer += _W_BEH * _dim_score(s, "behavioral", "behavior_score")
        den += _W_BEH
    dims = (numer / den) if den else 0.0
    return round(spec_rate * dims, 6)


def main() -> int:
    summary_path, reward_path = sys.argv[1], sys.argv[2]
    try:
        s = json.loads(open(summary_path).read())
    except (OSError, ValueError):
        s = {}

    stages = {name: _counts(s.get(name)) for name in _STAGES}
    total = {k: sum(st[k] for st in stages.values()) for k in ("passed", "failed", "skipped", "xfailed", "error")}
    overall = 1 if s.get("overall") == "PASS" else 0

    reward = {
        # Primary scalar: spec-gated, continuous numeric + behavioral similarity (docstring).
        "reward": _reward(s, stages),
        # Binary all-or-nothing gate, retained as a secondary metric.
        "overall": overall,
        # Per-stage 0/1 gates (back-compat).
        "spec": _stage_pass(stages["spec"]),
        "numeric": _stage_pass(stages["numeric"]),
        "behavioral": _stage_pass(stages["behavioral"]),
        # The continuous dimension scores and how many checks each aggregates.
        "numeric_score": _dim_score(s, "numeric", "numeric_score"),
        "numeric_n": _dim_n(s, "numeric"),
        "behavior_score": _dim_score(s, "behavioral", "behavior_score"),
        "behavior_n": _dim_n(s, "behavioral"),
        # Fine-grained totals so partial success is visible in result.json.
        "n_passed": total["passed"],
        "n_failed": total["failed"],
        "n_skipped": total["skipped"],
        "n_xfailed": total["xfailed"],
        "n_error": total["error"],
        # The old flat pooled pass rate over all three stages, retained as a diagnostic;
        # it does not contribute to the reward.
        "pass_rate": _rate(total),
    }
    for name in _STAGES:  # per-stage counts + (binary) pass rates
        c = stages[name]
        reward[f"{name}_passed"] = c["passed"]
        reward[f"{name}_failed"] = c["failed"]
        reward[f"{name}_skipped"] = c["skipped"]
        reward[f"{name}_pass_rate"] = _rate(c)

    with open(reward_path, "w") as f:
        json.dump(reward, f)
    print(reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
