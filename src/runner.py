#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Centralized measurement dispatch — the developer-facing run API.

This module provides the infrastructure that makes running benchmarks
with measurements trivially composable.  The developer experience::

    from src.runner import RunContext, track_span

    ctx = RunContext(
        measurements=[L3PerfMeasurement(target="self")],
        output_dir=Path(f"{_TMP}/my_run"),
    )

    with ctx:
        for task in adapter.list_tasks():
            with track_span(ctx, task.id, kind="synthetic_cpu", node_id=task.id):
                adapter.run_task(task, agent_invoker=invoker)

    for record in ctx.records:
        print(record.layer, record.payload.get("ipc"))

The :func:`track_span` context manager drives all registered
:class:`Measurement` plugins automatically — plugins never need to be
called manually.  This is the centralized dispatch that makes adding a
new measurement layer (L1, L2, L4, L5, VTune) a single-file exercise
for contributors.

Design: :class:`RunContext` is **scoped** (not process-global).  Tests
create their own context.  Concurrent runs don't interfere.  The sampler
(if L1 is registered) attaches to the context, not to the process.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.protocols import Measurement, MeasurementRecord
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

logger = logging.getLogger(__name__)

__all__ = [
    "RunContext",
    "track_span",
    "SpanRecord",
]


# =============================================================================
# SpanRecord — lightweight per-span aggregate container
# =============================================================================


@dataclass
class SpanRecord:
    """Tracks one in-flight span and sampler aggregates accumulated during it.

    Populated by :class:`PsutilSampler` (when active) between the span's
    open and close.  Read by L1SubSpanMeasurement in finalize_span.
    """

    span_id: str
    kind: str
    node_id: str
    tid: int
    started_at_ns: int
    phase: str = ""
    sample_count: int = 0
    # Time-weighted mean of proc.cpu_percent(), not the arithmetic mean of the
    # samples. The sampler aims for a fixed interval but does not achieve one —
    # a tick that lands late covers a longer window and must count for more.
    # Weighting by elapsed time makes the estimate insensitive to that jitter;
    # an unweighted mean is not (see _run).
    cpu_pct_mean: float = 0.0
    cpu_pct_peak: float = 0.0
    rss_kb_peak: int = 0
    num_threads_peak: int = 0
    # Accumulators behind cpu_pct_mean: sum(pct * dt) and sum(dt). Kept on the
    # record because the sampler is shared across concurrently-active spans, so
    # per-span state cannot live on the sampler.
    _pct_weighted_sum: float = 0.0
    _pct_weight_total: float = 0.0
    # Process-wide CPU seconds elapsed while this span was open, matching the
    # process-wide scope of cpu_pct_mean. On a multi-threaded workload this can
    # exceed the span's wall duration (that is what >1 core means), and spans
    # that overlap in time each count the same CPU seconds — so these do not
    # sum to a run total. Compare against duration_s, do not aggregate.
    cpu_time_s: float = 0.0
    closed_at_ns: int = 0

    def duration_us(self) -> int:
        if self.closed_at_ns and self.started_at_ns:
            return (self.closed_at_ns - self.started_at_ns) // 1000
        return 0


# =============================================================================
# SpanRegistry — thread-keyed map of active spans (for sampler reads)
# =============================================================================


class SpanRegistry:
    """Thread-keyed map of active spans within a RunContext."""

    def __init__(self) -> None:
        self._by_tid: Dict[int, List[SpanRecord]] = {}
        self._lock = threading.Lock()

    def push(self, record: SpanRecord) -> None:
        with self._lock:
            self._by_tid.setdefault(record.tid, []).append(record)

    def pop(self, span_id: str, tid: int) -> Optional[SpanRecord]:
        with self._lock:
            stack = self._by_tid.get(tid)
            if not stack:
                return None
            top = stack[-1]
            if top.span_id != span_id:
                logger.warning(
                    "span pop mismatch on tid %d: expected %s, got %s",
                    tid, span_id, top.span_id,
                )
            stack.pop()
            if not stack:
                self._by_tid.pop(tid, None)
            return top

    def current(self, tid: int) -> Optional[SpanRecord]:
        stack = self._by_tid.get(tid)
        return stack[-1] if stack else None

    def all_active(self) -> List[SpanRecord]:
        with self._lock:
            return [s for stack in self._by_tid.values() for s in stack]


# =============================================================================
# PsutilSampler — opt-in daemon thread for L1 resource metrics
# =============================================================================


class PsutilSampler:
    """Background daemon that polls psutil for CPU/RSS between ticks.

    Attaches to a :class:`SpanRegistry` and updates streaming aggregates
    on active :class:`SpanRecord` instances.  Opt-in: only started when
    a measurement plugin that needs it (e.g. L1) is present.

    ``sample_interval_ms`` is a LOWER BOUND, not a rate. This is a Python
    thread contending for the GIL with the workload it is measuring, so it is
    descheduled for as long as the workload holds the GIL. Measured on this
    project's own synthetic tasks: 20ms configured yields ~116ms actual during
    a 3s CPU-bound task (26 samples, not the ~150 the setting implies).

    Two consequences worth knowing before quoting these fields:

    * ``cpu_pct_peak`` is the max over however many samples landed. A short
      burst between two ticks is invisible to it.
    * ``cpu_pct_mean`` is time-weighted precisely because the interval is not
      dependable — see :attr:`SpanRecord.cpu_pct_mean`.

    Keep this loop cheap for the same reason: its cost is measurement error,
    and it grows the interval it is already failing to hit.
    """

    def __init__(
        self,
        registry: SpanRegistry,
        *,
        sample_interval_ms: float = 20.0,
    ) -> None:
        self._registry = registry
        self._interval_s = sample_interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Any = None
        # Wall clock of the previous tick. Each cpu_percent() reading is the
        # average since the previous call on the same Process object, so this is
        # the left edge of the window that reading describes.
        self._last_tick_ns: int = 0

    def process_cpu_time(self) -> float:
        """Cumulative CPU seconds (user+system) for the whole process.

        Span CPU time is process-scoped, not thread-scoped, for two reasons:

        1. **The work usually runs on a different thread than the span.**
           ``run_driver`` opens the span on the loop thread and submits the task
           to a ``ThreadPoolExecutor``, so a thread-scoped reading attributed
           ~0 CPU to a fully CPU-bound task.
        2. **Scope must match.** ``cpu_pct_mean`` above is already process-wide
           (``proc.cpu_percent()``). Pairing a process-wide percentage with a
           single-thread time in the same payload made the two disagree by the
           thread count, and ``cpu_time_s / duration_s`` is compared against a
           threshold calibrated on the percentage.

        Uses ``cpu_times()`` rather than summing ``threads()``: the latter sees
        only threads alive at the sample instant, so a worker that finishes
        before the span closes takes its CPU time with it.
        """
        proc = self._ensure_proc()
        try:
            t = proc.cpu_times()
            return float(t.user + t.system)
        except Exception:
            return 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ensure_proc()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="agentsysperf-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        self._thread = None

    def _ensure_proc(self) -> Any:
        if self._proc is None:
            import os
            import psutil
            self._proc = psutil.Process(os.getpid())
            # Priming call: cpu_percent(interval=None) returns 0.0 the first
            # time and measures from here on. Stamp the clock at the same
            # moment so the first real tick has a true left edge — otherwise
            # _last_tick_ns is 0, tick_start_ns falls back to now_ns, and dt_s
            # is 0 for spans already open, which routes them into the
            # zero-window path with a reading that spans pre-span CPU.
            self._proc.cpu_percent(interval=None)
            self._last_tick_ns = time.monotonic_ns()
        return self._proc

    def _run(self) -> None:
        """Poll psutil and fold each reading into every active span.

        Deliberately cheap. This loop runs on the machine being measured, so its
        own cost is measurement error: it previously called ``proc.threads()``
        every tick to fill a cache nothing ever read, and that call's cost scales
        with thread count. On a 220-thread process it stretched the nominal 20ms
        interval to ~2s, which collapsed a 2-second span to a single sample and
        made ``cpu_pct_mean`` report 0.0 for a fully saturated core. Add nothing
        per-tick whose cost grows with process size.
        """
        proc = self._ensure_proc()
        while not self._stop_event.is_set():
            try:
                proc_cpu = proc.cpu_percent(interval=None)
                rss = proc.memory_info().rss
                num_threads = proc.num_threads()
                now_ns = time.monotonic_ns()
                # Left edge of the window this reading describes: the previous
                # tick on this Process object, since that is what cpu_percent()
                # measures from. Seeded in _ensure_proc at the priming call, so
                # even the first tick here has a real left edge; the `or now_ns`
                # only guards a sampler driven without _ensure_proc.
                tick_start_ns = self._last_tick_ns or now_ns
                self._last_tick_ns = now_ns

                for span in self._registry.all_active():
                    span.sample_count += 1
                    # Weight this reading by how much of its window lies inside
                    # the span, and by nothing else. A tick that lands late
                    # covers proportionally more time and must count for more;
                    # but the first tick after a span opens describes a window
                    # that began BEFORE the span did, and crediting the span for
                    # that pre-span CPU is how a 220-thread process (whose
                    # thread creation is itself expensive) read 1.5x its true
                    # utilization.
                    overlap_start_ns = max(tick_start_ns, span.started_at_ns)
                    dt_s = max(now_ns - overlap_start_ns, 0) / 1e9
                    if dt_s > 0:
                        span._pct_weighted_sum += proc_cpu * dt_s
                        span._pct_weight_total += dt_s
                        span.cpu_pct_mean = (
                            span._pct_weighted_sum / span._pct_weight_total
                        )
                    elif span.sample_count == 1:
                        # Span opened and closed inside one tick: no overlap to
                        # weight by, so this reading is all there is.
                        span.cpu_pct_mean = proc_cpu
                    if proc_cpu > span.cpu_pct_peak:
                        span.cpu_pct_peak = proc_cpu
                    if rss > span.rss_kb_peak * 1024:
                        span.rss_kb_peak = rss // 1024
                    if num_threads > span.num_threads_peak:
                        span.num_threads_peak = num_threads
            except Exception:
                logger.debug("sampler tick failed", exc_info=True)

            self._stop_event.wait(self._interval_s)


# =============================================================================
# RunContext — scoped container for one benchmark run
# =============================================================================


class RunContext:
    """Scoped run container.  Holds measurements, collects records.

    Use as a context manager for automatic start/stop::

        with RunContext(measurements=[l3, l1]) as ctx:
            with track_span(ctx, "task-1", kind="cpu", node_id="linalg"):
                do_work()
        print(ctx.records)

    Or call start()/stop() manually for advanced control.
    """

    def __init__(
        self,
        *,
        measurements: Optional[Sequence[Measurement]] = None,
        output_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        sampler_interval_ms: float = 20.0,
        result_store: Optional[Any] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.output_dir = output_dir or Path(f"{_TMP}/agentsysperf_scratch/{self.run_id}")
        # Optional ResultStore: when provided, stop() atomically persists the
        # run (metadata + measurements + spans) to it IN ADDITION to the JSON
        # (dual-write). Default None means JSON-only — so callers that never opt
        # in (e.g. live_dashboard's per-click runs) don't churn the canonical
        # store. run_metadata is merged into what's persisted.
        self._result_store = result_store
        self._run_metadata: Dict[str, Any] = dict(run_metadata or {})
        self._measurements: List[Measurement] = list(measurements or [])
        self._records: List[MeasurementRecord] = []
        # Step-level execution trace (StepTrace rows). Populated by the agent
        # loop at task completion; persisted by the runner. Parallel to
        # _records (hardware) — together they're the full per-step picture.
        self._step_traces: List[Any] = []
        self._lock = threading.Lock()
        self._started = False

        self.registry = SpanRegistry()
        self._sampler: Optional[PsutilSampler] = None
        self._sampler_interval_ms = sampler_interval_ms

    @property
    def records(self) -> List[MeasurementRecord]:
        """All measurement records emitted during this run."""
        return list(self._records)

    @property
    def step_traces(self) -> List[Any]:
        """All StepTrace rows accumulated during this run (thread-safe copy)."""
        with self._lock:
            return list(self._step_traces)

    def add_step_traces(self, traces: Sequence[Any]) -> None:
        """Append StepTrace rows (called by the agent loop per task)."""
        with self._lock:
            self._step_traces.extend(traces)

    @property
    def measurements(self) -> List[Measurement]:
        return list(self._measurements)

    def _needs_sampler(self) -> bool:
        """Check if any registered measurement declares it needs the sampler."""
        for m in self._measurements:
            if getattr(m, "_needs_sampler", False):
                return True
            if getattr(m, "layer", "") == "l1":
                return True
        return False

    def start(self) -> None:
        """Start all measurements and the sampler (if needed)."""
        if self._started:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self._needs_sampler():
            self._sampler = PsutilSampler(
                self.registry, sample_interval_ms=self._sampler_interval_ms,
            )
            self._sampler.start()

        for m in self._measurements:
            if hasattr(m, "set_context"):
                m.set_context(self)
            try:
                m.start(run_id=self.run_id, output_dir=self.output_dir)
            except Exception:
                logger.warning(
                    "Measurement %r failed to start; continuing without it",
                    getattr(m, "name", type(m).__name__), exc_info=True,
                )
        self._started = True

    def stop(self) -> None:
        """Stop all measurements and the sampler; collect final records."""
        if not self._started:
            return
        for m in self._measurements:
            try:
                for rec in m.stop() or ():
                    self._records.append(rec)
            except Exception:
                logger.warning(
                    "Measurement %r failed in stop()",
                    getattr(m, "name", type(m).__name__), exc_info=True,
                )
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler = None
        self._started = False

        # Serialize records to JSON for analysis (kept during the transition —
        # dual-write; JSON-reading dashboards/exporter still work).
        self._serialize_records()

        # Dual-write: also persist atomically to the ResultStore if one was
        # provided. measurements + spans come from this context; task_results
        # and analyzer verdicts are caller-supplied (the runner doesn't own
        # them), so callers that have them should call persist_run directly with
        # the full set. Failure here must not crash the run — JSON already wrote.
        if self._result_store is not None:
            try:
                meta = {"start_time": 0, **self._run_metadata}
                self._result_store.persist_run(
                    run_id=self.run_id,
                    metadata=meta,
                    records=self._records,
                    spans=self._step_traces,
                    artifacts=self._collect_artifacts(),
                )
            except Exception:
                logger.warning(
                    "RunContext failed to persist run %s to the store (JSON is intact)",
                    self.run_id, exc_info=True,
                )

    def _collect_artifacts(self) -> List[Dict[str, Any]]:
        """Gather bulk on-disk files the measurements produced.

        A measurement that writes a file too large for the relational store (an
        EMON pyEDP CSV, a flame graph) declares it by implementing
        ``artifacts() -> Iterable[{"kind", "name", "path"}]``. Called after
        every ``m.stop()``, so post-processing has already produced the file.

        Registration is what makes the file addressable by run_id instead of
        rediscovered by globbing scratch dirs (the artifacts table's whole
        purpose). Paths that no longer exist are skipped — the store's readers
        filter on existence anyway, so a dead row buys nothing.
        """
        out: List[Dict[str, Any]] = []
        for m in self._measurements:
            if not hasattr(m, "artifacts"):
                continue
            try:
                declared = list(m.artifacts() or ())
            except Exception:
                logger.warning(
                    "Measurement %r failed in artifacts()",
                    getattr(m, "name", type(m).__name__), exc_info=True,
                )
                continue
            for art in declared:
                try:
                    kind = art["kind"]
                    name = art["name"]
                    path = Path(art["path"]).expanduser().resolve()
                except Exception:
                    logger.warning(
                        "Measurement %r returned invalid artifact declaration: %r",
                        getattr(m, "name", type(m).__name__), art, exc_info=True,
                    )
                    continue

                if not path.exists():
                    logger.warning(
                        "Measurement %r declared artifact %s that is not on disk; "
                        "skipping registration",
                        getattr(m, "name", type(m).__name__), path,
                    )
                    continue
                out.append({"kind": kind, "name": name, "path": str(path)})
        return out

    def _append_records(self, records: List[MeasurementRecord]) -> None:
        with self._lock:
            self._records.extend(records)

    def _serialize_records(self) -> None:
        """Serialize measurement records to JSON for later analysis."""
        import json

        records_file = self.output_dir / "measurement_records.json"
        try:
            with open(records_file, "w") as f:
                data = [
                    {
                        "span_id": r.span_id,
                        "layer": r.layer,
                        "payload": dict(r.payload),
                    }
                    for r in self._records
                ]
                json.dump(data, f, indent=2)
            logger.info("Serialized %d records to %s", len(self._records), records_file)
        except Exception as e:
            logger.warning("Failed to serialize records: %s", e)

    def __enter__(self) -> "RunContext":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


# =============================================================================
# track_span — the one context manager developers use around work units
# =============================================================================


@contextmanager
def track_span(
    ctx: RunContext,
    span_id: str,
    *,
    kind: str = "task",
    node_id: str = "",
    phase: str = "",
) -> Iterator[SpanRecord]:
    """Open a measurement span around a unit of work.

    All registered :class:`Measurement` plugins receive ``observe_span``
    on entry and ``finalize_span`` on exit — automatically.  Records are
    collected into ``ctx.records``.

    Usage::

        with track_span(ctx, task.id, kind="synthetic_cpu", node_id=task.id) as span:
            adapter.run_task(task, agent_invoker=invoker)
        # span.duration_us() now reflects wall time

    Nested spans are supported::

        with track_span(ctx, "outer", kind="task", node_id="main"):
            with track_span(ctx, "inner", kind="llm_call", node_id="planner"):
                call_llm()

    Plugin exceptions are caught and logged — a buggy probe never crashes
    the measured workload.
    """
    tid = threading.get_native_id()
    record = SpanRecord(
        span_id=span_id,
        kind=kind,
        node_id=node_id,
        tid=tid,
        started_at_ns=time.monotonic_ns(),
        phase=phase,
    )

    # Process-scoped, deliberately — see PsutilSampler.process_cpu_time. The
    # span-opening thread is frequently not the thread doing the work.
    cpu_time_at_entry = 0.0
    if ctx._sampler is not None:
        cpu_time_at_entry = ctx._sampler.process_cpu_time()

    ctx.registry.push(record)

    for m in ctx._measurements:
        try:
            m.observe_span(span_id=span_id, kind=kind, node_id=node_id)
        except Exception:
            logger.warning(
                "Measurement %r raised in observe_span; continuing",
                getattr(m, "name", type(m).__name__), exc_info=True,
            )

    try:
        yield record
    finally:
        record.closed_at_ns = time.monotonic_ns()
        if ctx._sampler is not None:
            cpu_time_at_exit = ctx._sampler.process_cpu_time()
            record.cpu_time_s = max(0.0, cpu_time_at_exit - cpu_time_at_entry)

        span_records: List[MeasurementRecord] = []
        for m in ctx._measurements:
            try:
                for rec in m.finalize_span(span_id) or ():
                    span_records.append(rec)
            except Exception:
                logger.warning(
                    "Measurement %r raised in finalize_span(%s); skipping",
                    getattr(m, "name", type(m).__name__),
                    span_id, exc_info=True,
                )

        ctx._append_records(span_records)
        ctx.registry.pop(span_id, tid)
