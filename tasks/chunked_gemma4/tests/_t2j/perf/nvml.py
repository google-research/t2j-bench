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

"""Parent-side GPU memory sampling.

Deliberately NOT `jax.local_devices()[0].memory_stats()`: that reads the BFC pool,
which XLA_PYTHON_CLIENT_PREALLOCATE, ..._MEM_FRACTION and ..._ALLOCATOR all move
without changing the computation. Sampling from the parent puts the instrument out
of the candidate's reach.

TOTAL DEVICE MEMORY, MINUS A PRE-CHILD BASELINE -- not per-process attribution.
NVML's per-process `usedGpuMemory` is structurally 0 inside a container: the lever
is the PID namespace, not privileges. For a child that really allocated memory:

    docker run --gpus device=0            -> usedGpuMemory 0
    docker run --gpus all                 -> usedGpuMemory 0
    docker run --gpus all --privileged    -> usedGpuMemory 0
    docker run --pid=host ...             -> the real number

`--pid=host` is not something the verifier controls, so per-process attribution
pins `r_mem` at 1.0 forever and silently kills 30% of the score. The device total
does move (a real delta appears where per-process reads 0), and is sound here because
the verifier has ONE GPU and runs ONE child at a time: baseline is read
immediately before the spawn, the peak is sampled during the run, and the reported
number is `peak_device_used - baseline`.

Two guards keep a corrupted number from being reported as if it were real:

1. The pre-child baseline must be recovered after the child exits. If device
   memory stays above it, something else on the device grew during our window and
   the delta is not ours.
2. No compute process other than the child, this parent, or one already present at
   baseline may appear on the device.

Failing either marks the measurement UNUSABLE (`peak_bytes is None`), which is
distinct from a genuine measured zero (`peak_bytes == 0`). Callers must keep that
distinction: `runner._metric_value` uses `is not None`, not truthiness.

Degrades to peak_bytes=None on CPU-only hosts rather than raising; the perf stage
is then skipped and flagged while equivalence still grades normally.
"""

from __future__ import annotations

import os
import threading
import time

# How long, after the child has exited, to keep re-reading device memory while
# waiting for it to fall back to the pre-child baseline. The driver reclaims a
# dead process's memory asynchronously, so a strict read taken the instant
# `communicate()` returns can still see the child's allocation. This is a TIME
# allowance for that reclaim, deliberately NOT a byte tolerance on the comparison
# itself -- a byte tolerance would silently absorb exactly the foreign allocation
# this guard exists to catch.
#
# 15s, chosen so a large reclaim cannot overrun it. Reclaim of a ~9GB context can be
# instant under `--gpus all` but overrun a few seconds under `--gpus device=0` (the
# verifier's real `gpus=1` shape), and the failure CASCADES: the round that times
# out is correctly marked unusable, but it also returns while the memory is still
# resident, so the NEXT child's pre-spawn baseline is taken against a dirty device
# and its delta is garbage too. Blocking here is
# the only place that can stop it, and it costs nothing on the normal path where
# recovery lands in well under a second.
_RECOVERY_GRACE_S = 15.0


def _pynvml():
    try:
        import pynvml  # noqa: PLC0415 - optional, absent on CPU-only hosts
        return pynvml
    except Exception:  # noqa: BLE001
        return None


_init_lock = threading.Lock()
_initialized = False


def _ensure_init(nv) -> bool:
    """Call `nvmlInit()` at most once per process.

    NVML's init is refcounted, so calling it repeatedly isn't a leak, but each
    call is a real driver round-trip. The sampler polls at a ~20Hz default;
    doing that on every tick is pure overhead in the very process that's timing
    something. Public functions still work standalone (each still checks
    `_pynvml()` and still never raises).
    """
    global _initialized
    if _initialized:
        return True
    with _init_lock:
        if _initialized:
            return True
        try:
            nv.nvmlInit()
        except Exception:  # noqa: BLE001
            return False
        _initialized = True
        return True


def nvml_available() -> bool:
    nv = _pynvml()
    if nv is None:
        return False
    if not _ensure_init(nv):
        return False
    try:
        return nv.nvmlDeviceGetCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _visible_device_token() -> str | None:
    """The first entry of CUDA_VISIBLE_DEVICES, or None when it is unset/empty."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    first = raw.split(",")[0].strip()
    return first or None


def resolve_device_index() -> int | None:
    """NVML index of the ONE device the child will actually compute on.

    Scoping matters: summing across every visible GPU adds the same large
    constant to both arms on a multi-GPU host, which
    compresses `r_mem` toward 1.0 and hides real wins. JAX puts its default
    device -- and therefore the training step -- on CUDA ordinal 0 of whatever
    is visible.

    The mapping from CUDA ordinal 0 to an NVML index is: NVML enumerates in PCI
    bus order, and `runner._PIN` forces `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the
    child so its ordinals are in that same order. CUDA_VISIBLE_DEVICES then
    selects which physical device ordinal 0 lands on -- given as an index, or as
    a `GPU-<uuid>`/`MIG-<uuid>` string.

    Returns None when no device can be resolved, which makes the sampler degrade
    to `peak_bytes = None` rather than measure the wrong GPU.
    """
    nv = _pynvml()
    if nv is None or not _ensure_init(nv):
        return None
    try:
        count = int(nv.nvmlDeviceGetCount())
    except Exception:  # noqa: BLE001
        return None
    if count <= 0:
        return None
    token = _visible_device_token()
    if token is None:
        return 0
    if token.isdigit():
        index = int(token)
        return index if index < count else None
    for i in range(count):
        try:
            uuid = nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(i))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8", errors="replace")
        if uuid == token:
            return i
    return None


def device_used_bytes(index: int) -> int | None:
    """Total device memory in use on NVML device `index`. None if unreadable."""
    nv = _pynvml()
    if nv is None or not _ensure_init(nv):
        return None
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        return int(nv.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:  # noqa: BLE001
        return None


def compute_process_pids(index: int) -> list[int] | None:
    """PIDs of compute processes on NVML device `index`. None if unreadable.

    The per-process MEMORY figure is worthless in a container (see the module
    docstring) but the process LIST is still informative: it is how the sampler
    notices a foreign tenant arriving mid-run. It is not sufficient on its own --
    a process in another PID namespace may not be enumerable at all -- which is
    why the baseline-recovery check exists alongside it.
    """
    nv = _pynvml()
    if nv is None or not _ensure_init(nv):
        return None
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        return [int(p.pid) for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]
    except Exception:  # noqa: BLE001
        return None


def throttle_reasons(index: int | None = None) -> list[str]:
    """Active clock-throttle reasons on the measured device. Reported, never
    gated. Scoped to one device for the same reason the memory
    sampler is: another GPU's thermal state says nothing about ours."""
    nv = _pynvml()
    if nv is None or not _ensure_init(nv):
        return []
    if index is None:
        index = resolve_device_index()
    if index is None:
        return []
    names = {
        "nvmlClocksThrottleReasonSwPowerCap": "sw_power_cap",
        "nvmlClocksThrottleReasonHwThermalSlowdown": "hw_thermal_slowdown",
        "nvmlClocksThrottleReasonHwPowerBrakeSlowdown": "hw_power_brake_slowdown",
        "nvmlClocksThrottleReasonSwThermalSlowdown": "sw_thermal_slowdown",
    }
    out: list[str] = []
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        active = nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        for attr, label in names.items():
            mask = getattr(nv, attr, 0)
            if mask and (active & mask):
                out.append(f"gpu{index}:{label}")
    except Exception:  # noqa: BLE001
        return out
    return out


class MemorySampler:
    """Peak device memory attributable to one child, as a delta over a baseline.

    Usage is deliberately two-step, because the baseline must be read BEFORE the
    child exists:

        with MemorySampler() as sampler:
            proc = subprocess.Popen(...)
            sampler.watch(proc.pid)
            proc.communicate()
        sampler.peak_bytes   # int (>=0), or None when unusable/unmeasurable

    `watch()` is optional; without it the child's own PID is simply not on the
    allow-list for the foreign-process check.
    """

    def __init__(
        self,
        interval_s: float = 0.05,
        *,
        device_index: int | None = None,
        recovery_grace_s: float = _RECOVERY_GRACE_S,
    ) -> None:
        self._interval = interval_s
        self._device_index = device_index
        self._recovery_grace_s = recovery_grace_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._baseline: int | None = None
        self._baseline_pids: set[int] = set()
        # None means "no sample has succeeded yet"; a real measurement of 0
        # bytes over baseline must survive as 0, not collapse into "unmeasured".
        self._peak: int | None = None
        self._child_pid: int | None = None
        self._observed_pids: set[int] = set()
        self._foreign_pids: set[int] = set()
        self.peak_bytes: int | None = None
        self.unusable_reason: str | None = None
        self.device_index: int | None = None
        self.baseline_bytes: int | None = None

    def watch(self, pid: int) -> None:
        """Tell the sampler which PID is the child, so the foreign-process check
        does not flag our own measurement subject."""
        self._child_pid = int(pid)

    # --- internals ------------------------------------------------------- #
    def _allowed_pids(self) -> set[int]:
        allowed = set(self._baseline_pids)
        allowed.add(os.getpid())
        if self._child_pid is not None:
            allowed.add(self._child_pid)
        return allowed

    def _sample(self) -> None:
        index = self.device_index
        if index is None:
            return
        try:
            used = device_used_bytes(index)
            pids = compute_process_pids(index)
        except Exception:  # noqa: BLE001 - a broken sampler must not kill the thread
            return
        if used is not None:
            self._peak = used if self._peak is None else max(self._peak, used)
        if pids is not None:
            # Accumulate raw PIDs and difference them against the allow-list in
            # __exit__, not here: `watch()` is necessarily called a moment AFTER
            # the thread starts (the child's PID does not exist before the
            # spawn), so subtracting an allow-list that does not yet contain the
            # child would race and flag our own measurement subject.
            self._observed_pids |= set(pids)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)
        self._sample()

    def _wait_for_recovery(self) -> int | None:
        """Poll device memory until it falls back to the baseline, or the grace
        period runs out. Returns the last reading."""
        index = self.device_index
        assert index is not None
        deadline = time.monotonic() + self._recovery_grace_s
        final = device_used_bytes(index)
        while final is not None and self._baseline is not None and final > self._baseline:
            if time.monotonic() >= deadline:
                break
            time.sleep(min(self._interval, 0.1))
            final = device_used_bytes(index)
        return final

    # --- context manager -------------------------------------------------- #
    def __enter__(self) -> "MemorySampler":
        if not nvml_available():
            return self
        try:
            index = self._device_index if self._device_index is not None else resolve_device_index()
            if index is None:
                return self
            baseline = device_used_bytes(index)
            if baseline is None:
                return self
            baseline_pids = set(compute_process_pids(index) or ())
        except Exception:  # noqa: BLE001 - never let the instrument break the run
            return self
        self.device_index = index
        self._baseline = baseline
        self.baseline_bytes = baseline
        self._baseline_pids = baseline_pids
        self._peak = baseline
        self._enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self._enabled:
            self.peak_bytes = None
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._foreign_pids = self._observed_pids - self._allowed_pids()
        final = self._wait_for_recovery()
        if final is None:
            self.unusable_reason = "device memory became unreadable while the child ran"
        elif self._baseline is not None and final > self._baseline:
            self.unusable_reason = (
                f"device memory did not return to the pre-child baseline "
                f"({final} > {self._baseline} bytes on gpu{self.device_index}): another "
                "allocation on this device overlapped the measurement window"
            )
        elif self._foreign_pids:
            self.unusable_reason = (
                f"foreign compute process(es) {sorted(self._foreign_pids)} appeared on "
                f"gpu{self.device_index} during the measurement window"
            )
        if self.unusable_reason is not None or self._peak is None or self._baseline is None:
            self.peak_bytes = None
            return
        self.peak_bytes = max(0, self._peak - self._baseline)
