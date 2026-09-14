#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Shared subprocess helpers for the L3 perf plugin.

Two callers — :class:`L3PerfMeasurement` (continuous interval mode) and
:class:`PerfStatTelemetry` (point-in-time mode) — share availability
checks and CSV parsing.

Linux ``perf stat -x ,`` writes a CSV-ish format. With the ``-I N`` flag
each line carries a relative timestamp; without it, a single summary
line per event is produced after the workload exits. This module
parses both.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# No shipping x86 part clocks above this. Used only as a sanity ceiling for
# cycles/sec, so a generous bound is fine — the corruption it catches is off
# by seven orders of magnitude, not by a few percent.
_MAX_PLAUSIBLE_GHZ = 10.0


def _counts_are_physical(sums: Dict[str, float], duration_s: float) -> bool:
    """False when the counter sums cannot have come from this machine.

    Two independent impossibilities, either one damning:

    1. More cycles than the hardware can retire — ``cycles/s`` above
       ``logical_cpus x 10 GHz``.
    2. A miss rate above 100%, i.e. more cache misses than references or more
       branch misses than branches.

    Lives here rather than in either consumer because both perf callers need
    it: the continuous interval mode (per-interval salvage) and the
    point-in-time telemetry mode (whole-window reject).
    """
    cyc = sums.get("cycles", 0.0)
    if cyc > 0 and duration_s > 0:
        ceiling = (os.cpu_count() or 1) * _MAX_PLAUSIBLE_GHZ * 1e9
        if cyc / duration_s > ceiling:
            return False

    for total_key, miss_key in (
        ("cache-references", "cache-misses"),
        ("branch-instructions", "branch-misses"),
    ):
        total, miss = sums.get(total_key, 0.0), sums.get(miss_key, 0.0)
        if total > 0 and miss > total:
            return False

    return True


# Default events the harness uses. Available on most x86-64 CPUs and
# do not require vendor-specific PMU codes (which per the no-fabricated-PMU-codes
# rule in the README's *Integrity principles*, must NOT be fabricated for
# unknown SKUs).
DEFAULT_EVENTS = (
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branch-instructions",
    "branch-misses",
    "context-switches",
    "cpu-migrations",
    "page-faults",
)


def perf_available() -> bool:
    """Return True if ``perf`` runs and the paranoid level allows counters.

    A False return here is a normal degradation path on locked-down
    hosts (paranoid >= 2) — callers degrade to no-counter mode rather
    than failing the whole run.
    """
    try:
        r = subprocess.run(
            ["perf", "stat", "-e", "cycles", "--", "sleep", "0.001"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def parse_perf_csv_summary(text: str) -> Tuple[Dict[str, float], List[str]]:
    """Parse the trailing summary lines of a non-interval ``perf stat -x ,``.

    Each non-comment line looks like ``<value>,<unit>,<event>,<run>,...``.
    Returns ``(values, unmeasured)`` — the event-name → value map, plus the
    events perf reported as ``<not counted>`` / ``<not supported>``.

    Unmeasured events are deliberately kept OUT of ``values`` rather than
    recorded as 0.0. A counter that did not count is *unmeasured*, not zero,
    and a fabricated zero is indistinguishable downstream from a real reading
    below threshold: ``evaluate_thresholds`` would report "claimed
    optimization did not engage" for what is actually "this host could not
    measure it". Returning them separately lets the caller say which.
    """
    out: Dict[str, float] = {}
    unmeasured: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw_value = parts[0].strip()
        event = parts[2].strip()
        if not event:
            continue
        if raw_value in ("<not counted>", "<not supported>"):
            unmeasured.append(event)
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        out[event] = value
    return out, unmeasured


def parse_perf_csv_interval(
    text: str,
) -> List[Tuple[float, str, float]]:
    """Parse a ``perf stat -I N -x ,`` interval log.

    Each non-comment line looks like ``<rel_ts>,<value>,<unit>,<event>,...``.
    Returns a list of ``(rel_ts_seconds, event_name, value)`` tuples.
    Rows with ``<not counted>`` or unparseable values are dropped, not
    coerced to zero — silently zero-counting is the kind of "soft"
    behaviour ``docs/methodology/DESIGN.md §9`` warns against.
    """
    out: List[Tuple[float, str, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            ts = float(parts[0].strip())
        except ValueError:
            continue
        raw_value = parts[1].strip()
        event = parts[3].strip()
        if not event:
            continue
        if raw_value in ("<not counted>", "<not supported>"):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        out.append((ts, event, value))
    return out


def target_to_perf_args(target: Optional[str]) -> List[str]:
    """Translate an L3 ``target`` spec into perf-stat scoping flags.

    Accepted forms:

    - ``None`` or ``"system"`` — system-wide sampling (``perf stat -a``).
      Counts every CPU's activity in the window. On large machines this
      dilutes per-span IPC heavily; prefer narrower targets when the
      caller can identify them.
    - ``"self"`` — attach to the current Python process PID
      (``perf stat -p <os.getpid()>``). Useful for in-process workloads
      like the synthetic_cpu adapter, where the workload runs in the
      same process that opens the span.
    - ``"pid:<N>"`` — attach to PID ``N`` (``perf stat -p N``). Counts
      the named process plus any threads/children created after attach.
    - ``"cgroup:<PATH>"`` — attach to a cgroup (``perf stat --cgroup PATH``).
      Used by the harness to scope to a Docker container's cgroup.

    Unknown forms log a warning and degrade to system-wide. This is the
    one place "soft" behaviour is acceptable: the caller asked for a
    target the plugin doesn't understand, and refusing to sample at all
    would be worse than sampling the wrong scope and surfacing the
    warning. Per-span attribution still happens; it just covers the
    whole system.
    """
    if target is None or target == "system":
        return ["-a"]
    if target == "self":
        return ["-p", str(os.getpid())]
    if target.startswith("pid:"):
        return ["-p", target[4:]]
    if target.startswith("cgroup:"):
        return ["--cgroup", target[7:]]
    logger.warning(
        "L3 perf: unknown target spec %r — falling back to system-wide", target,
    )
    return ["-a"]


def run_perf_window(
    events: List[str],
    window_s: float,
    *,
    target: Optional[str] = None,
) -> Optional[str]:
    """Run ``perf stat`` for exactly ``window_s`` and return raw stderr.

    See :func:`target_to_perf_args` for accepted ``target`` forms. Returns
    the raw stderr text (perf writes counters to stderr) or ``None`` if
    perf isn't available or the run fails.

    Implementation note: ``perf stat -- sleep <window>`` is the canonical
    way to time-bound a counter sampling window without attaching to a
    target process; ``-p`` or ``--cgroup`` narrow it down.
    """
    cmd = ["perf", "stat", "-e", ",".join(events), "-x", ","]
    cmd += target_to_perf_args(target)
    cmd += ["--", "sleep", str(window_s)]

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=window_s + 10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("L3 perf: window run failed (%s); skipping", exc)
        return None
    if r.returncode != 0:
        logger.debug("L3 perf: returncode %d; stderr head=%r",
                     r.returncode, r.stderr[:200])
    return r.stderr


def detect_cpu_model() -> str:
    """Best-effort CPU model identifier for :class:`CounterReading.cpu_model`.

    Reads ``/proc/cpuinfo`` model name on Linux; returns ``""`` if the
    info isn't readable. The string is informational — never used to
    gate event selection (that is what
    :attr:`HardwareTelemetryPlugin.available_events` is for).
    """
    try:
        with Path("/proc/cpuinfo").open() as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""
