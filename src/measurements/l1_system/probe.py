#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""L1-system :class:`Measurement` plugin — node-level resource telemetry.

Distinct from ``l1_subspan`` (per-span psutil, "what did THIS span use").
``l1_system`` answers "what is the WHOLE NODE doing during this window" —
the signal a concurrency sweep needs: how the box behaves as N agents pile on.

This is the AgentSysPerf port of the colleague's mpstat/vmstat sampling
(agentic-benchmark-4 → ``harness/scripts/perf_sampler.py``). Because EMR does
not ship sysstat (no ``mpstat``/``iostat``) but the same numbers live in
``/proc``, the sampler reads ``/proc/stat`` + ``/proc/meminfo`` directly in a
background thread — dependency-free, with precise monotonic timestamps for
span-window attribution. ``vmstat`` itself reads ``/proc/stat``, so the
quantities are identical to the colleague's.

Lifecycle mirrors :class:`~src.measurements.l3_perf.L3PerfMeasurement`:

1. :meth:`start` — open a CSV, launch one background sampler thread that
   appends a timestamped row every ``sample_interval_s``.
2. :meth:`observe_span` — record the span's open time (monotonic-relative).
3. :meth:`finalize_span` — slice the samples to the span window, emit one
   :class:`MeasurementRecord` (layer ``"l1_system"``) with cpu avg/p50/p95/peak,
   runqueue avg/max, context-switches/s, iowait%, memory low-water mark.
4. :meth:`stop` — signal the thread to exit; close the CSV.

Per-cell granularity: in a density sweep each cell is one run and the per-task
span covers the whole cell, so the emitted record characterizes the node for
that density point.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.protocols import MeasurementRecord, NullMeasurement

logger = logging.getLogger(__name__)


def _read_proc_stat() -> Optional[Dict[str, float]]:
    """Read one snapshot from /proc/stat.

    Returns the aggregate cpu jiffies (for utilization deltas), the total
    context-switch counter, and the instantaneous run-queue depth.
    """
    try:
        with open("/proc/stat") as f:
            text = f.read()
    except OSError:
        return None

    out: Dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "cpu":
            # user nice system idle iowait irq softirq steal guest guest_nice
            vals = [float(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
            total = sum(vals)
            out["cpu_total"] = total
            out["cpu_idle"] = idle
            out["cpu_iowait"] = vals[4] if len(vals) > 4 else 0.0
        elif parts[0] == "ctxt":
            out["ctxt"] = float(parts[1])
        elif parts[0] == "procs_running":
            out["procs_running"] = float(parts[1])
    return out or None


def _read_mem_mb() -> Tuple[Optional[float], Optional[float]]:
    """Return (mem_available_mb, mem_used_mb) from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, _, rest = line.partition(":")
                info[k.strip()] = rest.strip()
    except OSError:
        return None, None
    try:
        total_kb = float(info["MemTotal"].split()[0])
        avail_kb = float(info["MemAvailable"].split()[0])
    except (KeyError, ValueError, IndexError):
        return None, None
    avail_mb = avail_kb / 1024.0
    used_mb = (total_kb - avail_kb) / 1024.0
    return avail_mb, used_mb


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list (pct in [0,100])."""
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * (pct / 100.0)), len(sorted_vals) - 1)
    return sorted_vals[idx]


class L1SystemMeasurement(NullMeasurement):
    """L1-system: node-level CPU / runqueue / context-switch / memory telemetry.

    One instance per run. Samples ``/proc`` in a background thread; attributes
    samples to spans by wall-clock overlap on :meth:`finalize_span`.
    """

    name: str = "l1_system"
    layer: str = "l1_system"

    def __init__(self, *, sample_interval_s: float = 1.0) -> None:
        super().__init__()
        self._interval = sample_interval_s
        self._logical_cpus = os.cpu_count() or 1
        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._t0: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        # rel_ts -> sample dict, kept in memory for window attribution.
        self._samples: List[dict] = []
        self._lock = threading.Lock()
        # span_id -> (open_rel_s, kind, node_id)
        self._open_spans: Dict[str, Tuple[float, str, str]] = {}
        self._closed: set = set()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self, *, run_id: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = output_dir / "l1_system_samples.csv"
        self._csv_file = self._csv_path.open("w")
        self._csv_file.write(
            "rel_ts,cpu_pct,iowait_pct,runqueue,ctx_sw_per_s,mem_avail_mb,mem_used_mb\n"
        )
        self._t0 = time.monotonic()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(
            "l1_system: sampling /proc every %.1fs (run=%s, logical_cpus=%d, csv=%s)",
            self._interval, run_id, self._logical_cpus, self._csv_path,
        )

    def _sample_loop(self) -> None:
        prev = _read_proc_stat()
        prev_t = time.monotonic()
        # Prime: skip the first delta (needs two reads to compute a rate).
        while not self._stop_evt.wait(self._interval):
            now = time.monotonic()
            cur = _read_proc_stat()
            if cur is None or prev is None:
                prev, prev_t = cur, now
                continue
            dt = max(now - prev_t, 1e-6)
            total_d = cur["cpu_total"] - prev["cpu_total"]
            idle_d = cur["cpu_idle"] - prev["cpu_idle"]
            iowait_d = cur.get("cpu_iowait", 0.0) - prev.get("cpu_iowait", 0.0)
            cpu_pct = 100.0 * (1.0 - idle_d / total_d) if total_d > 0 else 0.0
            iowait_pct = 100.0 * (iowait_d / total_d) if total_d > 0 else 0.0
            ctx_per_s = (cur.get("ctxt", 0.0) - prev.get("ctxt", 0.0)) / dt
            runq = cur.get("procs_running", 0.0)
            avail_mb, used_mb = _read_mem_mb()

            sample = {
                "rel_ts": now - self._t0,
                "cpu_pct": cpu_pct,
                "iowait_pct": iowait_pct,
                "runqueue": runq,
                "ctx_sw_per_s": ctx_per_s,
                "mem_avail_mb": avail_mb,
                "mem_used_mb": used_mb,
            }
            with self._lock:
                self._samples.append(sample)
                if self._csv_file is not None:
                    self._csv_file.write(
                        f"{sample['rel_ts']:.3f},{cpu_pct:.2f},{iowait_pct:.2f},"
                        f"{runq:.0f},{ctx_per_s:.0f},"
                        f"{(avail_mb or 0):.0f},{(used_mb or 0):.0f}\n"
                    )
                    self._csv_file.flush()
            prev, prev_t = cur, now

    def observe_span(self, *, span_id: str, kind: str, node_id: str) -> None:
        if self._t0 is None:
            return
        with self._lock:
            self._open_spans[span_id] = (time.monotonic() - self._t0, kind, node_id)

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        if self._t0 is None:
            return
        with self._lock:
            opened = self._open_spans.pop(span_id, None)
            if opened is None:
                return
            self._closed.add(span_id)
            open_rel, kind, node_id = opened
            close_rel = time.monotonic() - self._t0
            window = [s for s in self._samples if open_rel <= s["rel_ts"] <= close_rel]

        record = self._aggregate(
            span_id=span_id, kind=kind, node_id=node_id,
            window=window, duration_s=close_rel - open_rel,
        )
        if record is not None:
            yield record

    def stop(self) -> Iterable[MeasurementRecord]:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2.0)
            self._thread = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        logger.info("l1_system: stopped sampling (csv=%s)", self._csv_path)
        return ()

    # ── Internal: window → record ─────────────────────────────────────

    def _aggregate(
        self, *, span_id: str, kind: str, node_id: str,
        window: List[dict], duration_s: float,
    ) -> Optional[MeasurementRecord]:
        if not window:
            return None  # Span shorter than one sample interval.

        cpu_sorted = sorted(s["cpu_pct"] for s in window)
        runqs = [s["runqueue"] for s in window]
        ctx = [s["ctx_sw_per_s"] for s in window]
        iowaits = [s["iowait_pct"] for s in window]
        avail = [s["mem_avail_mb"] for s in window if s["mem_avail_mb"] is not None]
        used = [s["mem_used_mb"] for s in window if s["mem_used_mb"] is not None]
        n = len(window)

        payload: Dict[str, object] = {
            "kind": kind,
            "node_id": node_id,
            "duration_s": duration_s,
            "sample_count": n,
            "logical_cpus": self._logical_cpus,
            # CPU utilization (100 - idle), matching the colleague's stats.txt.
            "cpu_avg": sum(cpu_sorted) / n,
            "cpu_p50": _percentile(cpu_sorted, 50),
            "cpu_p95": _percentile(cpu_sorted, 95),
            "cpu_peak": cpu_sorted[-1],
            # Scheduler-oversubscription signals.
            "runqueue_avg": sum(runqs) / n,
            "runqueue_max": max(runqs),
            "ctx_sw_per_s_avg": sum(ctx) / n,
            # I/O-bound signal.
            "iowait_pct_avg": sum(iowaits) / n,
        }
        if avail:
            payload["mem_avail_mb_min"] = min(avail)  # memory-pressure low-water mark
        if used:
            payload["mem_used_mb_max"] = max(used)
        return MeasurementRecord(span_id=span_id, layer=self.layer, payload=payload)


__all__ = ["L1SystemMeasurement"]
