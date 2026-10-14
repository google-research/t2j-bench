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
"""Parent-side REDUCTION for the perf stage.

NOT a driver. `_CandidateRuntime.perf_probe` already ran the rounds, inside the single
sandbox worker `Candidate._ensure_run` spawns -- the parent receives every probe's result
in one message and cannot invoke one round at a time. So everything here is arithmetic
over a structure that already exists.

That division is `jaxopt`'s own rule, arrived at by a different route: children emit raw
(K, seconds) pairs and nothing else, and every reduction happens parent-side, so a
candidate cannot compute any number reported about itself.

UNMEASURABLE FAILS, and the reason lives in the caller (`test_perf.py`), not here. This
module's job is to say WHAT happened -- a slope, or the reason there isn't one -- with
enough detail attached that the failure is diagnosable offline instead of by re-running
the grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .stats import fit_step_time, round_median

#: Rounds that must resolve their slope at 2 sigma. Two of three, not three of three: at
#: three of three a single scheduler hiccup fails a correct port, and the median of two
#: is the mean of two, which is what `round_median` already does.
_QUORUM = 2


@dataclass
class PerfResult:
    step_ms: float | None = None
    intercept_s: float | None = None
    rounds_used: int = 0
    rounds_detail: list[dict] = field(default_factory=list)
    tokens: Any = None
    reason: str | None = None
    throttle_discards: int = 0


def reduce(probe: dict) -> PerfResult:
    """Fit, median and quorum over `perf_probe`'s returned rounds.

    `probe` is the dict the worker produced. A probe that reports `supported: False`
    is a CANDIDATE failure and is named as one -- distinct from a probe that ran but
    whose rounds would not resolve, which is a MEASUREMENT failure. The two get the
    same verdict and different reasons, so the logs never conflate a broken port with
    a busy card.
    """
    out = PerfResult()
    if not probe or not probe.get("supported"):
        err = (probe or {}).get("error") or "the candidate did not produce a perf probe"
        out.reason = f"candidate: {err}"
        return out

    out.tokens = probe.get("tokens")
    slopes: list[float] = []
    intercepts: list[float] = []
    requested = 0

    for rnd in probe.get("rounds") or []:
        detail = dict(rnd)
        if rnd.get("discarded"):
            out.throttle_discards += 1
            out.rounds_detail.append(detail)
            continue
        requested += 1
        try:
            fit = fit_step_time(list(rnd.get("calls") or []))
            slopes.append(fit.step_ms)
            intercepts.append(fit.intercept_s)
            detail["step_ms"] = fit.step_ms
            detail["fit"] = fit.as_dict()
        except (ValueError, TypeError, IndexError) as exc:
            # A round that cannot distinguish its slope from zero has NO step time.
            # Recorded with the reason rather than dropped, so "unmeasurable" stays
            # distinguishable from "not measured".
            detail["step_ms"] = None
            detail["fit_error"] = f"{type(exc).__name__}: {exc}"
        out.rounds_detail.append(detail)

    out.rounds_used = len(slopes)
    if out.rounds_used >= _QUORUM:
        out.step_ms = round_median(slopes)
        out.intercept_s = round_median(intercepts)
    else:
        out.reason = (
            f"quorum not met: {out.rounds_used} of {requested} attempted round(s) "
            f"resolved a slope at 2 sigma (needed {_QUORUM}); "
            f"{out.throttle_discards} discarded for throttling")
    return out
