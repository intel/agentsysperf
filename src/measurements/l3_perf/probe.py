#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""L3 :class:`Measurement` plugin — per-span perf-counter rows.

Lifecycle:

1. :meth:`start` — boots a long-lived ``perf stat -I 100 -a`` process
   that streams interval-tagged counter values to a CSV file. This is
   the same continuous-sampling pattern ``harness/scripts/perf_sampler.py``
   uses; we just attach to the
   :class:`Measurement` Protocol so the records flow through the
   universal output schema.

2. :meth:`observe_span` — records the wall-clock time the span opened.
   Per-span counter sampling is **not** done by re-attaching perf;
   instead we attribute already-running interval samples to spans by
   wall-clock overlap on close. This keeps overhead at a single global
   perf process for the whole run regardless of how many spans open.

3. :meth:`finalize_span` — reads interval samples whose timestamps fall
   inside the span window, sums per-event counts, and emits one
   :class:`MeasurementRecord` per span with derived metrics (IPC,
   cache-miss%, branch-miss%, LLC-miss/s).

4. :meth:`stop` — sends SIGINT to the perf process so it flushes
   trailing samples; reads any final records that closed after the
   last :meth:`finalize_span` call.

Graceful degradation: if ``perf`` is unavailable or
``kernel.perf_event_paranoid`` is locked down, :meth:`start` logs a
warning and the plugin degrades to a no-op for the run. The benchmark
still produces results — just without L3 counters.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.measurements.l3_perf._perf_subprocess import (
    DEFAULT_EVENTS,
    _MAX_PLAUSIBLE_GHZ,
    _counts_are_physical,
    parse_perf_csv_interval,
    perf_available,
    target_to_perf_args,
)
from src.protocols import MeasurementRecord, NullMeasurement

logger = logging.getLogger(__name__)


class L3PerfMeasurement(NullMeasurement):
    """L3: hardware perf counters per span.

    One instance per benchmark run. Spawns one ``perf stat`` subprocess
    in :meth:`start`, attributes its interval samples to spans by
    wall-clock overlap on :meth:`finalize_span`.
    """

    name: str = "l3_perf"
    layer: str = "l3"

    def __init__(
        self,
        *,
        events: Optional[List[str]] = None,
        sample_interval_ms: int = 100,
        core_range: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        """Build an L3 plugin instance.

        Parameters
        ----------
        events:
            Override the default counter event list.
        sample_interval_ms:
            How often perf emits an interval row (lower = finer span
            attribution, higher = lower overhead). 100 ms matches the
            harness default.
        core_range:
            Restrict sampling to a CPU range (perf ``-C`` flag), e.g.
            ``"0-21"``. Ignored unless target is system-wide.
        target:
            Attribution scope. ``None`` / ``"system"`` is system-wide
            (legacy behaviour); ``"self"`` attaches to the current
            process's PID; ``"pid:<N>"`` and ``"cgroup:<PATH>"`` attach
            to specific PIDs or cgroups. See
            :func:`target_to_perf_args` for the full grammar. Use
            ``"self"`` for in-process benchmark adapters like
            synthetic_cpu — system-wide on a 344-core box dilutes IPC
            by a large factor.
        """
        super().__init__()
        self._events = list(events or DEFAULT_EVENTS)
        self._sample_interval_ms = sample_interval_ms
        self._core_range = core_range
        self._target = target
        self._proc: Optional[subprocess.Popen] = None
        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._t0_perf_clock: Optional[float] = None
        # span_id -> (open_rel_ts_s, kind, node_id)
        self._open_spans: Dict[str, Tuple[float, str, str]] = {}
        # span_id -> finalized record window so stop() doesn't re-emit
        self._closed: set = set()
        self._lock = threading.Lock()
        self._available = False

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self, *, run_id: str, output_dir: Path) -> None:
        """Spawn the long-lived perf-stat subprocess.

        On a host without perf access this logs a warning and leaves
        ``_available=False`` — every subsequent observe_span/finalize_span
        becomes a no-op and the run continues without L3 records.
        """
        if not perf_available():
            logger.warning(
                "L3 perf: perf binary unavailable or kernel.perf_event_paranoid "
                "blocks counter access; L3 records will be empty for run %s",
                run_id,
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = output_dir / "l3_perf_continuous.csv"
        self._csv_file = self._csv_path.open("w")

        cmd = [
            "perf", "stat",
            "-e", ",".join(self._events),
            "-I", str(self._sample_interval_ms),
            "-x", ",",
        ]
        cmd += target_to_perf_args(self._target)
        # ``-C <range>`` only makes sense for system-wide sampling; perf
        # rejects it alongside ``-p`` / ``--cgroup``. Honour it only in
        # the system-wide path so callers don't get a startup error.
        if self._core_range and (self._target is None or self._target == "system"):
            cmd += ["-C", self._core_range]

        # perf stat writes counter rows to stderr; stdout is unused.
        self._t0_perf_clock = time.monotonic()
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=self._csv_file,
        )
        self._available = True
        logger.info(
            "L3 perf: started continuous sampling (run=%s, events=%d, "
            "interval=%dms, target=%s, csv=%s)",
            run_id, len(self._events), self._sample_interval_ms,
            self._target or "system", self._csv_path,
        )

    def observe_span(
        self, *, span_id: str, kind: str, node_id: str,
    ) -> None:
        """Record the span's open time so finalize can attribute samples."""
        if not self._available or self._t0_perf_clock is None:
            return
        rel = time.monotonic() - self._t0_perf_clock
        with self._lock:
            self._open_spans[span_id] = (rel, kind, node_id)

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        """Slice the interval CSV to this span's window, emit one record."""
        if not self._available or self._t0_perf_clock is None:
            return
        with self._lock:
            opened = self._open_spans.pop(span_id, None)
            if opened is None:
                return
            self._closed.add(span_id)
        open_rel, kind, node_id = opened
        close_rel = time.monotonic() - self._t0_perf_clock

        record = self._emit_for_window(
            span_id=span_id, kind=kind, node_id=node_id,
            window_start_s=open_rel, window_end_s=close_rel,
        )
        if record is not None:
            yield record

    def stop(self) -> Iterable[MeasurementRecord]:
        """SIGINT the perf subprocess so it flushes; close the CSV.

        Does not retroactively emit per-span records that finalize_span
        missed — those would carry trailing samples we couldn't have
        attributed deterministically. Callers should ensure every
        opened span has a matching close before stop() runs.
        """
        if not self._available:
            return ()
        if self._proc is not None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # SIGINT didn't land (perf can ignore it while flushing a
                # large event set). Escalate, then reap: without a second
                # wait() the child stays a zombie and any thread joining on
                # it blocks forever, which hangs RunContext teardown.
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "L3 perf: perf stat (pid %s) survived SIGKILL; "
                        "leaving it unreaped rather than blocking teardown",
                        self._proc.pid,
                    )
            self._proc = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        logger.info(
            "L3 perf: stopped continuous sampling (csv=%s)", self._csv_path,
        )
        return ()

    # ── Internal: window → record ─────────────────────────────────────

    def _emit_for_window(
        self,
        *,
        span_id: str,
        kind: str,
        node_id: str,
        window_start_s: float,
        window_end_s: float,
    ) -> Optional[MeasurementRecord]:
        if self._csv_path is None or not self._csv_path.exists():
            return None
        # Re-read the CSV every finalize. The cost is dominated by
        # subprocess wakeup latency at the perf-stat sampler, not by
        # parsing a few KB of CSV per span — fine for the typical
        # ~10 spans/run at this scale. If span counts grow large we
        # can switch to a tail-follower; for now, simplicity wins.
        try:
            text = self._csv_path.read_text()
        except OSError:
            return None
        intervals = parse_perf_csv_interval(text)
        if not intervals:
            return None

        # Per-interval sanity gate, applied BEFORE summing. Corruption arrives
        # in bursts (54 contiguous blocks over one 21-min run), so summing first
        # would let a handful of ~1e17 intervals poison a span that is otherwise
        # 74% good. Drop the impossible intervals and keep the rest.
        interval_ceiling = (
            (os.cpu_count() or 1) * _MAX_PLAUSIBLE_GHZ * 1e9
            * (self._sample_interval_ms / 1000.0)
        )
        # When cycles is impossible at a timestamp, every event sampled at that
        # timestamp is suspect (they are corrupted together), so reject the
        # whole interval rather than just the one counter.
        bad_ts = {
            ts for ts, event, value in intervals
            if event == "cycles" and value > interval_ceiling
        }

        sums: Dict[str, float] = {}
        sample_counts: Dict[str, int] = {}
        for ts, event, value in intervals:
            if ts < window_start_s or ts > window_end_s:
                continue
            if ts in bad_ts:
                continue
            sums[event] = sums.get(event, 0.0) + value
            sample_counts[event] = sample_counts.get(event, 0) + 1

        if not sums:
            return None  # Window too short to capture any sample.

        dropped = sum(
            1 for ts, event, _ in intervals
            if event == "cycles" and ts in bad_ts
            and window_start_s <= ts <= window_end_s
        )
        if dropped:
            kept = sample_counts.get("cycles", 0)
            logger.warning(
                "L3 perf: span %s dropped %d of %d intervals as physically "
                "impossible (>%.3g cycles per %dms on %d CPUs); metrics "
                "derived from the remaining %d.",
                span_id, dropped, dropped + kept, interval_ceiling,
                self._sample_interval_ms, os.cpu_count() or 1, kept,
            )

        duration_s = max(window_end_s - window_start_s, 1e-9)

        # Counter-sanity check. `perf stat` can exit 0 and emit well-formed CSV
        # in which every event reads back a near-identical ~1e18 count, giving
        # IPC ~= 1.0 and miss rates ~= 100% (sometimes >100%) — numbers the
        # analyzers turn into confident, wrong verdicts. Two triggers observed
        # on a 288-core Clearwater Forest host:
        #   - an exclusive-PMU collector holding the counters (every interval corrupt)
        #   - heavy event multiplexing under contention for a limited number of
        #     PMU slots (bursty: 26% of intervals over a 21-min run, in 54
        #     contiguous blocks, all at a low counter-enabled fraction)
        # Reject on physical impossibility rather than trusting the exit code.
        if not _counts_are_physical(sums, duration_s):
            logger.warning(
                "L3 perf: discarding span %s — counter values are physically "
                "impossible (cycles=%.3g over %.1fs on %d CPUs). perf exited "
                "cleanly but the counts cannot be real. Usual causes: another "
                "tool holds the PMU (an exclusive-PMU collector cannot run "
                "alongside perf), or severe event multiplexing.",
                span_id, sums.get("cycles", 0.0), duration_s, os.cpu_count() or 1,
            )
            return None

        payload = {
            "kind": kind,
            "node_id": node_id,
            "duration_s": duration_s,
            "events": dict(sums),
            "sample_counts": dict(sample_counts),
        }
        # Derived metrics — only emit when the underlying counters
        # are present, never coerce missing data to zero.
        cyc = sums.get("cycles", 0.0)
        ins = sums.get("instructions", 0.0)
        if cyc > 0:
            payload["ipc"] = ins / cyc
        cache_refs = sums.get("cache-references", 0.0)
        cache_miss = sums.get("cache-misses", 0.0)
        if cache_refs > 0:
            payload["cache_miss_pct"] = 100.0 * cache_miss / cache_refs
        br = sums.get("branch-instructions", 0.0)
        br_miss = sums.get("branch-misses", 0.0)
        if br > 0:
            payload["branch_miss_pct"] = 100.0 * br_miss / br
        if cache_miss > 0 and duration_s > 0:
            payload["llc_miss_per_s"] = cache_miss / duration_s

        return MeasurementRecord(
            span_id=span_id,
            layer=self.layer,
            payload=payload,
        )


__all__ = ["L3PerfMeasurement"]
