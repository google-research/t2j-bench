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

"""Convert the training grader's summary to Harbor reward metrics.

The reward is the Spec pass fraction multiplied by an equal-weighted mean of
Numeric and Behavioral similarity scores. Weights are renormalized over dimensions
with scored checks; no scored dimension yields zero. Dimension scores are reduced
by the grader before this script runs. Binary overall status is diagnostic only.
Per-check scores are flattened because Harbor accepts scalar reward values.
A missing or invalid summary produces zero reward.
"""

import json
import sys

# The single documented global dial: equal weight per scored dimension, renormalized over
# the dimensions that actually ran a check (so they need not sum to 1, and a missing
# dimension does not cap the reward). This replaces the retired per-stage pass-rate weights.
_W_NUM = 0.5
_W_BEH = 0.5


def _f(x, default: float = 0.0) -> float:
    """Coerce to a finite float in [0, 1]; anything else -> ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not (v == v) or v in (float("inf"), float("-inf")):
        return default
    return min(1.0, max(0.0, v))


def _dimension(block, top_score) -> tuple[float, int, dict]:
    """(mean_score, n_checks, scores_map) for a scored dimension.

    Prefers the block's own ``mean``/``n``/``scores`` (written by conftest); falls back to
    the top-level convenience score and, ultimately, to recomputing the mean from the
    per-check map so a partially-written summary still scores sanely.
    """
    block = block if isinstance(block, dict) else {}
    scores = block.get("scores") if isinstance(block.get("scores"), dict) else {}
    n = block.get("n")
    n = int(n) if isinstance(n, int) else len(scores)
    if "mean" in block:
        mean = _f(block.get("mean"))
    elif top_score is not None:
        mean = _f(top_score)
    elif scores:
        mean = _f(sum(_f(v) for v in scores.values()) / len(scores))
    else:
        mean = 0.0
    return mean, n, {k: _f(v) for k, v in scores.items()}


def _spec_pass_rate(summary: dict) -> float:
    """Binary Spec pass fraction: passed / (passed + failed + error), 0.0 if none ran.

    Prefers the top-level ``spec_pass_rate``; recomputes from the spec block otherwise.
    """
    if "spec_pass_rate" in summary:
        return _f(summary.get("spec_pass_rate"))
    spec = summary.get("spec") if isinstance(summary.get("spec"), dict) else {}
    ran = 0
    for k in ("passed", "failed", "error"):
        v = spec.get(k, 0)
        ran += v if isinstance(v, int) and not isinstance(v, bool) else 0
    passed = spec.get("passed", 0)
    passed = passed if isinstance(passed, int) and not isinstance(passed, bool) else 0
    return round(passed / ran, 6) if ran else 0.0


def _reward(spec_pass_rate: float, num: tuple, beh: tuple) -> float:
    """spec-gated, dimension-weighted mean of the numeric/behavior similarity means."""
    num_mean, num_n, _ = num
    beh_mean, beh_n, _ = beh
    numer = den = 0.0
    if num_n > 0:
        numer += _W_NUM * num_mean
        den += _W_NUM
    if beh_n > 0:
        numer += _W_BEH * beh_mean
        den += _W_BEH
    dims = (numer / den) if den else 0.0
    return round(spec_pass_rate * dims, 6)


def main() -> int:
    summary_path, reward_path = sys.argv[1], sys.argv[2]
    try:
        s = json.loads(open(summary_path).read())
        if not isinstance(s, dict):
            s = {}
    except (OSError, ValueError):
        s = {}

    spec_pass_rate = _spec_pass_rate(s)
    num = _dimension(s.get("numeric"), s.get("numeric_score"))
    beh = _dimension(s.get("behavioral"), s.get("behavior_score"))
    num_mean, num_n, num_scores = num
    beh_mean, beh_n, beh_scores = beh

    reward = {
        # Primary scalar: spec-gated, dimension-weighted similarity.
        "reward": _reward(spec_pass_rate, num, beh),
        # The three dimension scores the reward is built from.
        "spec_pass_rate": spec_pass_rate,
        "numeric_score": num_mean,
        "behavior_score": beh_mean,
        # Checks that ran per scored dimension (the denominators of the means).
        "n_numeric": num_n,
        "n_behavioral": beh_n,
        # Retired-but-kept coarse diagnostic; the reward does NOT depend on it.
        "overall": 1 if s.get("overall") == "PASS" else 0,
    }
    # Per-check similarity, FLATTENED to scalar top-level keys so a weak module stays
    # visible in Harbor's result.json. Harbor's ``VerifierResult.rewards`` is typed
    # ``dict[str, float | int]`` and rejects nested dicts, so the per-check maps cannot be
    # emitted as ``numeric_scores`` / ``behavioral_scores`` sub-objects. The check node-ids
    # (``test_numeric.py::...`` / ``test_behavioral.py::...``) are already unique and do not
    # collide with the summary keys above. The reward scalar is unchanged by this shape.
    reward.update(num_scores)
    reward.update(beh_scores)

    with open(reward_path, "w") as f:
        json.dump(reward, f)
    print(reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
