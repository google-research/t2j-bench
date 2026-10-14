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
"""Step-time measurement for the `perf` stage.

`stats.py` and `nvml.py` are byte-for-byte copies of a shared step-time instrument.
They are the only step-time instrument in this repo with a measured precision record,
and their comments record real failure modes. `runner.py` is this class's own:
jaxopt pairs a live baseline against the candidate in one session, and this class
compares a live candidate against a FROZEN reference, which is a different orchestration
with a different failure surface.

UNUSED, NOT UNREACHABLE: `nvml.py`'s memory-sampling path has no caller here.
`device_used_bytes` (nvml.py:182), `compute_process_pids` (nvml.py:194),
`MemorySampler` (nvml.py:243-386), and the `_RECOVERY_GRACE_S` constant only
`MemorySampler` reads (nvml.py:78) exist in the byte copy and do nothing in this
class. This is NOT because the architecture rules them out -- the sandbox this
class runs a candidate in IS a spawned child (`Candidate._ensure_run` starts a
`multiprocessing.get_context("spawn")` worker, `_t2j/candidate.py:4032`), the same
shape jaxopt samples memory around. It is unused because this class grades no memory
at all: memory/peak-bytes gates are
explicitly out of scope -- "not measured today, and nothing in the class asks for
it." If a future slice adds a memory gate, `MemorySampler` already does the
parent-side sampling; do not write a second one.

MUST NOT BE USED FOR THE GATE: `paired_ratio` (stats.py:266) and
`coefficient_of_variation` (stats.py:283) assume both arms are measured in the
SAME session, so that taking the ratio inside a round cancels GPU-clock drift
shared by both arms. That assumption does not hold here. This class's comparison is
CROSS-TIME -- a reference frozen on one day against a candidate timed on another
-- so jaxopt's paired-session precision evidence
does NOT transfer to that shape, and the frozen-vs-live scatter has not
been characterised anywhere in this repo. The obvious move when computing
`perf_ratio` is to reach for the ratio helper already sitting in `_t2j.perf.stats`
-- don't. It would silently import an assumption that is unverified.
`perf_ratio` must be its own frozen-vs-live computation.

Also unused, for an unrelated reason: `step_time_ms` (stats.py:164-246) is
two-point legacy compat kept only for jaxopt-only calibration and diagnosis
tooling with no counterpart in
this class. This class calls `fit_step_time` directly; nothing here should call
`step_time_ms`.
"""
