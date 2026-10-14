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

"""Parent-side reductions.

Lives in the parent so a candidate cannot compute any number reported about
itself: children emit raw (K, seconds) pairs and nothing else.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# How many standard errors the fitted slope must clear before the round is believed.
# NOT a tuning knob for strictness -- it is the question "can this round distinguish its
# own slope from zero at all". A round that cannot has no step time, and reporting one
# anyway produces the phantom-speedup failure `fit_step_time` guards against.
_SLOPE_SIGMA = 2.0


@dataclass(frozen=True)
class StepFit:
    """A fitted `T(K) = C + K*s`, WITH the evidence that it is a fit rather than a solve.

    `se_rel` and `residual_rel` are `nan` at n == 2, and that is the whole point: two
    points determine a line exactly, so there is nothing to report and nothing to check.
    """

    step_ms: float
    intercept_s: float
    n: int
    #: SE(slope) / slope. The round's OWN precision, available without a second round.
    se_rel: float
    #: max |residual| / (slope * K-span) -- how far the worst point sits off the line,
    #: measured against the signal the slope is extracted from rather than against T,
    #: which is dominated by the constant and would make everything look excellent.
    residual_rel: float
    #: Residual standard deviation, in SECONDS. This is a direct measurement of the
    #: per-call constant's jitter AT THE EPOCHS ACTUALLY RUN, which is what the epoch
    #: solver needs and what a trial triple can only guess at.
    resid_sd_s: float
    ks: tuple
    seconds: tuple

    def as_dict(self) -> dict:
        return {"step_ms": self.step_ms, "intercept_s": self.intercept_s, "n": self.n,
                "resid_sd_s": self.resid_sd_s,
                "se_rel": None if math.isnan(self.se_rel) else self.se_rel,
                "residual_rel": None if math.isnan(self.residual_rel) else self.residual_rel,
                "ks": list(self.ks), "seconds": list(self.seconds)}


def fit_step_time(calls: list[tuple[int, float]], *, gate: bool = True) -> StepFit:
    """Least-squares `T(K) = C + K*s` over every call after the first.

    WHY NOT THE LAST TWO. The two-point form this replaces solved for the slope instead
    of fitting it, and an exactly-determined system cannot be contradicted by its own
    data: every arrangement of two positive times is consistent with some (C, s),
    including physically impossible ones. A single corrupt round can make a task
    ungradeable: a candidate whose call was inflated by a late recompile (a fusion that
    recompiled in call 2 instead of call 1) reports an implausibly small step time on
    byte-identical code, driving the median-of-5 standard error high enough that the task
    is cut as ungradeable -- and with only two points there is no residual in which it
    could show up.

    Two guards were tried against it at n == 2 and BOTH were verified not to fire, which
    is why neither is here and why the fix had to be a third fitting point:
      * `slope_13 >= slope_23`, on the reasoning that call 1's extra constant drags its
        long-span slope below the true one. False in this regime: when `C_load` dominates
        the signal, `slope_13` is negative on a healthy child too.
      * `C_load >= 0 and C_compile >= 0`. The corrupt round satisfies both -- inflating
        one call just moves mass between the two constants.

    With three fitting points there is one degree of freedom, and both failures above
    become visible in it: a single inflated call cannot be absorbed by the constant
    without leaving a residual proportional to the inflation.

    Call 1 is still excluded rather than fitted with its own dummy variable. Adding a
    `C_load` term would consume the degree of freedom this exists to buy, which is the
    same trap in a new shape.

    Raises `ValueError` when the round has no usable step time. `runner._metric_value`
    turns that into "drop this round from BOTH arms", which is the correct disposition:
    an unusable measurement must not become a number, and it must not become a number on
    one arm only.
    """
    if len(calls) < 3:
        raise ValueError(
            f"need at least three calls to fit a slope with a residual, got {len(calls)}; "
            "the workload must declare four epoch counts (one warm-up + three fitting "
            "points) -- see Workload.epochs"
        )
    pts = [(int(k), float(t)) for k, t in calls[1:]]
    ks = [k for k, _ in pts]
    ts = [t for _, t in pts]
    if len(set(ks)) < 2:
        raise ValueError(f"need distinct step counts to regress a slope, got {ks}")

    n = len(pts)
    kbar, tbar = statistics.fmean(ks), statistics.fmean(ts)
    sxx = sum((k - kbar) ** 2 for k in ks)
    sxy = sum((k - kbar) * (t - tbar) for k, t in pts)
    slope = sxy / sxx                                     # seconds per step
    intercept = tbar - slope * kbar

    resid = [t - (intercept + slope * k) for k, t in pts]
    dof = n - 2
    if dof > 0:
        s_err = math.sqrt(max(sum(r * r for r in resid), 0.0) / dof)
        se_slope = s_err / math.sqrt(sxx)
    else:                                                 # n == 2: exactly determined
        se_slope = float("nan")

    span = abs(slope) * (max(ks) - min(ks))
    residual_rel = (max(abs(r) for r in resid) / span) if span > 0 else float("inf")
    se_rel = abs(se_slope / slope) if slope != 0 and not math.isnan(se_slope) else float("nan")

    fit = StepFit(step_ms=slope * 1000.0, intercept_s=intercept, n=n,
                  se_rel=se_rel, residual_rel=residual_rel,
                  resid_sd_s=0.0 if math.isnan(se_slope) else s_err,
                  ks=tuple(ks), seconds=tuple(ts))
    if not gate:
        # The UNGATED fit exists for one caller: the jaxopt calibration tooling, which on a
        # failed fit needs `resid_sd_s` -- the jitter measured at the epochs actually run
        # -- to widen the span and try again. Refusing to hand back the diagnosis would
        # force it to re-derive the regression, i.e. to keep a second copy of this.
        return fit
    if slope <= 0.0:
        # T(K) must increase with K: more steps cannot take less wall clock. A descending
        # fit means the measurement is corrupt (a foreign load spike on the shared GPU, a
        # mid-run throttle change) or the entry short-circuited the longer call -- which
        # is exactly the signature the work gate exists to catch, so it must not be
        # silently absorbed by `paired_ratio`'s positivity filter.
        raise ValueError(
            f"non-positive step-time slope over calls {pts}: {slope * 1000.0:.6f} ms/step. "
            "T(K) must increase with K."
        )
    if not math.isnan(se_slope) and slope <= _SLOPE_SIGMA * se_slope:
        # The round cannot tell its own slope from zero. Reporting a number here is
        # strictly worse than reporting none: on a null it is noise, but on a candidate a
        # slope that happens to land small produces an r_step that clips to the cap and
        # takes full marks on the step axis for nothing -- a defect that PAYS OUT.
        raise ValueError(
            f"step-time slope {slope * 1000.0:.4f} ms/step is within {_SLOPE_SIGMA} standard "
            f"errors of zero (SE {se_slope * 1000.0:.4f} ms/step, relative residual "
            f"{residual_rel:.2f}) over calls {pts}: this round cannot resolve a step time"
        )
    return fit


def step_time_ms(calls: list[tuple[int, float]]) -> float:
    """Steady-state step time in ms/step. Thin wrapper over `fit_step_time`.

    Kept as the name every caller already uses, and kept returning a bare float so
    `runner._metric_value` and the offline diagnosis tools are unchanged. Everything that
    wants the evidence calls `fit_step_time` directly.

    ---- the original two-point contract, retained for artifacts written before the
    fourth call existed ----

    Steady-state step time from a two-point regression over the LAST two calls.

    Both training entries build their jitted step inside the function (`run_sft`'s
    `_jit_step`, `DPOTrainer.train`'s `_train_step`), so every call is a fresh
    closure and recompiles — a warm-cache measurement is not available. Modelling
    T(K) = C + K*s, calls 2 and 3 share the same C (compile only, no import or
    weight load), so the slope cancels it. Call 1 is excluded because it also pays
    import + load.

    THE FIT HAS NO RESIDUAL, AND THAT IS A REAL LIMIT ON WHAT CAN BE CHECKED HERE.
    Three calls determine three unknowns — `C_load`, `C_compile` and `s` — exactly, so
    there is no spare degree of freedom to test the model against. Every arrangement of
    three positive times is consistent with SOME (C_load, C_compile, s), including
    arrangements that are physically impossible.

    A single corrupt round is undetectable here, and it can make a task ungradeable.
    If one call is inflated by a late recompile (a fusion that recompiled in call 2
    instead of call 1) or a call comes up short, the three times are still consistent
    with SOME (C_load, C_compile, s), so nothing flags it -- yet the resulting step
    time can be off several-fold, driving the per-round `sd(log)` and the median-of-5
    standard error high enough that the task cannot be graded at all.

    Two guards were tried against it and BOTH were verified not to fire, which is why
    neither is here:
      * `slope_13 >= slope_23`, on the reasoning that call 1 pays an extra constant so its
        long-span slope sits just below the true one. False in this regime: when `C_load`
        dominates the signal, `slope_13` is NEGATIVE on a healthy child too.
      * `C_load >= 0 and C_compile >= 0`. The corrupt round satisfies both, because
        inflating call 2 just moves mass between the two constants.

    The durable fix is a FOURTH call, which buys the one degree of freedom that makes the
    fit testable, at the cost of one more compile per child. Recorded rather than done:
    it is a workload change, so it means recalibrating all instances.
    Until then the exposure is bounded by reporting it — `rounds_detail` carries the raw
    (K, seconds) pairs for every call precisely so the next occurrence is diagnosable from
    the artifact instead of from a re-run.

    The DIRECTION is what makes this worth recording rather than shrugging at. On a null a
    phantom speedup is merely noise. On a candidate it is an `r_step` that clips straight to the
    cap and takes full marks on the step axis for nothing — a defect that pays out rather
    than penalises.
    """
    if len(calls) >= 3:
        return fit_step_time(calls).step_ms
    # ---- legacy two-point path ----
    # Reached only by an artifact written before the workload declared a fourth epoch
    # count: the arm-asymmetry diagnostic tooling and `precision_report.py` both read
    # `rounds_detail` blocks off disk that predate the change. Live measurement never
    # lands here, because `Workload.__post_init__` refuses a workload with fewer than
    # four epoch counts.
    if len(calls) < 2:
        raise ValueError(f"need at least two calls to regress a slope, got {len(calls)}")
    (k_a, t_a), (k_b, t_b) = calls[-2], calls[-1]
    if k_a == k_b:
        raise ValueError(f"need distinct step counts to regress a slope, got {k_a} twice")
    slope = (t_b - t_a) / (k_b - k_a) * 1000.0
    if slope <= 0.0:
        # T(K) must increase with K: more steps cannot take less wall clock. A
        # descending pair means the measurement is corrupt (a foreign load spike
        # on the shared GPU, a mid-run throttle change) or the entry short-
        # circuited the longer call. Raising surfaces it as "this round has no
        # usable step time" -- runner._metric_value catches ValueError and drops
        # the round from BOTH arms -- instead of feeding a negative number into
        # paired_ratio, whose positivity filter would silently discard it with no
        # trace. This became load-bearing once the work gate started checking
        # every call: a near-zero or negative slope is exactly the signature of
        # the short-circuit the gate is there to catch, and it must not be
        # invisible on the way through.
        raise ValueError(
            f"non-positive step-time slope from calls {(k_a, t_a)} -> {(k_b, t_b)}: "
            f"{slope:.6f} ms/step. T(K) must increase with K."
        )
    return slope


def round_median(xs: list[float]) -> float:
    """Plain median across paired-ratio rounds.

    No trimming: with >= 5 rounds the plain median is already unmoved by two
    arbitrary outliers, one at each extreme, because the median of an
    odd-length sorted sequence is exactly its middle value -- the same value
    you would get after dropping the smallest and largest first. An earlier
    "drop min and max, then median" formulation was checked in for that
    outlier rejection and was a mathematical no-op (removing exactly one min
    and one max from a sorted sequence never changes the median); do not
    reintroduce it.
    """
    if not xs:
        raise ValueError("round_median of an empty sequence")
    return statistics.median(xs)


def paired_ratio(baseline: list[float], candidate: list[float]) -> float:
    """Ratio taken WITHIN each round, then medianed across rounds.

    Pairing first is what makes this robust: AlphaTensor found GPU clock scaling
    made a fixed algorithm's apparent speedup range from 4% to 11.6%, and taking
    the ratio inside the pair cancels drift that per-arm medians do not.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"paired arms must be the same length, got {len(baseline)} and {len(candidate)}")
    if not baseline:
        raise ValueError("need at least one paired round")
    ratios = [b / c for b, c in zip(baseline, candidate) if b > 0 and c > 0]
    if not ratios:
        raise ValueError("no usable pairs (baseline and candidate values must both be positive)")
    return round_median(ratios)


def coefficient_of_variation(xs: list[float]) -> float:
    """Reported, never gated. CV catches variance, not bias."""
    if len(xs) < 2:
        return 0.0
    mean = statistics.fmean(xs)
    if mean == 0:
        return 0.0
    return statistics.stdev(xs) / abs(mean)
