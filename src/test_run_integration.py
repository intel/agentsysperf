#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""End-to-end assertions on the numbers `agentsysperf run` actually produces.

Every other test in this suite either exercises a component in isolation with
synthetic inputs or, in the case of ``storage/test_run_driver.py``, drives
``run_benchmark()`` and then checks only *counts* — 3 tasks ran, N measurements
landed, status is "complete". That is why 156 tests passed while the flagship
path reported ``cpu_time_s = 0.0`` on every task and classified nine
deliberately CPU-saturating workloads as ``io_bound``: nothing asserted a
measured value or a verdict.

This module closes that gap. It runs the real thing through
``run_benchmark()`` — not through ``examples/``, which is what the broken path
was being demonstrated with — and asserts on what came out.

The invariant under test is **scope**. ``cpu_time_s`` must count every thread
of the process, because ``cpu_pct_mean`` (``proc.cpu_percent()``) already does
and ``CPUBoundAnalyzer`` compares ``cpu_time_s / duration_s`` against a
threshold calibrated on that percentage. Reading ``cpu_time_s`` off the
span-opening thread made the two disagree by the thread count — silently, and
always in the direction of "this workload is not using the CPU".

Scope is pinned directly (spawn N busy threads, check the reading sees all N)
rather than by comparing ``cpu_time_s`` to ``cpu_pct_mean``. ``cpu_pct_mean`` is
an *unweighted* mean of ``cpu_percent()`` readings taken at a nominal 20ms that
in practice jitters from 1ms to 60ms, over a numerator quantized to 10ms
(``SC_CLK_TCK`` = 100). Measured on the same 2s of single-core spinning it
returns anywhere from 54 to 162 for a true 100. That bias is real but it is a
separate defect from scope, and asserting the two agree would make this file
fail for the wrong reason.

Cost: ~9s (3 synthetic tasks x 3.0s each). Deliberate — the bug is only visible
in a workload long enough to accumulate CPU time.
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest

from src.run_driver import RunConfig, run_benchmark
from src.runner import RunContext, track_span
from src.storage.sqlite_store import SQLiteResultStore

# The synthetic_cpu workloads are CPU exercisers with no I/O and no sleeping, so
# a correct reading is ~1.0 cores' worth or more. Loose enough to survive a
# loaded CI box, tight enough that 0.0 (the thread-scope bug) and a
# divided-by-thread-count reading both fail.
_MIN_UTILIZATION = 0.5


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """One real 3-task run, shared by the assertions below."""
    store = SQLiteResultStore(tmp_path_factory.mktemp("store"))
    cfg = RunConfig(benchmark="synthetic_cpu", num_tasks=3, run_id="measured_values")
    summary = run_benchmark(cfg, store=store, stamp=1)
    assert summary.passed == 3, "synthetic tasks did not run; nothing to assert on"
    return store, summary


def _l1_payloads(store, run_id):
    rows = store.query_measurements(run_id, layer="l1")
    assert rows, "no L1 records — the sampler never attached, so cpu_time_s is untested"
    return [r["payload"] for r in rows]


def test_span_cpu_time_is_nonzero_for_cpu_bound_work(synthetic_run):
    """The regression itself: a saturated core must not report 0.0 CPU seconds.

    ``run_driver`` opens the span on the loop thread and submits the work to a
    ``ThreadPoolExecutor`` (which exists only to enforce the per-task timeout).
    A thread-scoped ``cpu_time_s`` therefore measured the thread that did
    nothing but wait on ``fut.result()``.
    """
    store, summary = synthetic_run
    for p in _l1_payloads(store, summary.run_id):
        duration_s = p["duration_us"] / 1e6
        assert duration_s > 0, f"{p['node_id']}: span has no duration"
        assert p["cpu_time_s"] > 0.0, (
            f"{p['node_id']}: cpu_time_s is 0.0 over {duration_s:.2f}s of "
            f"CPU-bound work — cpu_time_s is being read off the wrong thread"
        )
        utilization = p["cpu_time_s"] / duration_s
        assert utilization >= _MIN_UTILIZATION, (
            f"{p['node_id']}: utilization {utilization:.3f} < {_MIN_UTILIZATION} "
            f"(cpu_time_s={p['cpu_time_s']:.3f}s over {duration_s:.2f}s)"
        )


@pytest.mark.load_sensitive
def test_span_cpu_time_is_process_scoped_not_thread_scoped():
    """cpu_time_s must count every thread, matching cpu_pct_mean's scope.

    This is the property, rather than the symptom. Four threads spin for 1s on a
    span opened by a fifth thread that only sleeps. A process-scoped reading sees
    ~4 CPU seconds; a thread-scoped one sees ~0. Any future change that narrows
    the scope fails here even if it leaves cpu_time_s nonzero — which the test
    above, on its own, would not catch.

    Deliberately does NOT go through run_benchmark: the point is to control the
    thread count exactly, and 4 busy threads is a shape no synthetic task has.
    """
    n_threads, spin_s = 4, 1.0
    # hashlib releases the GIL, so these threads genuinely run in parallel. A
    # pure-Python loop would not: 4 GIL-bound spinners total ~1.0 core, which is
    # indistinguishable from the single-thread bug this test exists to catch.
    buf = b"\xa5" * (4 * 1024 * 1024)

    with RunContext(measurements=[]) as ctx:
        # RunContext only starts the sampler when a measurement declares it needs
        # one; with measurements=[] we start it by hand. cpu_time_s is captured by
        # track_span itself, not by a probe, so no measurement is required.
        from src.runner import PsutilSampler
        ctx._sampler = PsutilSampler(ctx.registry)
        ctx._sampler.start()

        def spin():
            end = time.monotonic() + spin_s
            while time.monotonic() < end:
                hashlib.sha256(buf).digest()

        with track_span(ctx, "scope", kind="test", node_id="scope") as span:
            workers = [threading.Thread(target=spin) for _ in range(n_threads)]
            for w in workers:
                w.start()
            for w in workers:
                w.join()

    duration_s = span.duration_us() / 1e6
    utilization = span.cpu_time_s / duration_s
    # Loose lower bound: 4 threads on a busy CI box may not each get a full core,
    # but they cannot collectively look like less than 2. Thread-scoped reads ~0.
    assert utilization >= 2.0, (
        f"{n_threads} spinning threads over {duration_s:.2f}s reported "
        f"{span.cpu_time_s:.3f} CPU seconds (utilization {utilization:.2f}) — "
        f"cpu_time_s is not counting all threads of the process"
    )


@pytest.mark.load_sensitive
def test_cpu_pct_mean_survives_a_high_thread_count_process():
    """cpu_pct_mean must not depend on how many threads the process happens to have.

    The sampler shares a process with the workload, so anything it does per tick
    that scales with process size feeds back into its own sampling interval. It
    used to call ``proc.threads()`` every tick (measured ~5ms against ~270us for
    every other call in the loop combined) to fill a cache no code read. At 220
    threads that stretched a nominal 20ms interval past 2s, collapsing a 2s span
    to one sample and reporting cpu_pct_mean = 0.0 for a saturated core.

    220 threads is not a shape today's synthetic tasks reach (they peak at 4) —
    it is where the agentic workloads this suite targets live, and it is where
    the field was silently wrong.
    """
    from src.runner import PsutilSampler

    idle = threading.Event()
    for _ in range(220):
        threading.Thread(target=idle.wait, args=(60,), daemon=True).start()
    try:
        with RunContext(measurements=[]) as ctx:
            ctx._sampler = PsutilSampler(ctx.registry)
            ctx._sampler.start()
            with track_span(ctx, "threads", kind="test", node_id="threads") as span:
                end = time.monotonic() + 1.0
                while time.monotonic() < end:
                    pass
    finally:
        idle.set()

    # Ground truth is one saturated core: a single GIL-bound Python loop.
    assert span.sample_count >= 5, (
        f"only {span.sample_count} samples in {span.duration_us() / 1e6:.2f}s — "
        f"the sampler's own per-tick cost is scaling with thread count"
    )
    assert 0.75 <= span.cpu_pct_mean / 100.0 <= 1.4, (
        f"cpu_pct_mean={span.cpu_pct_mean:.1f} for one saturated core in a "
        f"220-thread process (expected ~100)"
    )


@pytest.mark.load_sensitive
def test_cpu_bound_analyzer_calls_synthetic_workloads_core_bound(synthetic_run):
    """The verdict a user actually sees.

    Skipped when perf counters are unavailable: ``CPUBoundAnalyzer`` needs
    ``ipc`` and ``branch_miss_pct`` from L3 and yields nothing without them, so
    on a locked-down host (``kernel.perf_event_paranoid`` >= 2, no
    CAP_PERFMON) there is no verdict to assert on. The two tests above still
    run there — they need no privileges.
    """
    store, summary = synthetic_run
    if not store.query_measurements(summary.run_id, layer="l3"):
        pytest.skip("no L3 records (perf unavailable) — no verdict is produced")

    verdicts = store.query_verdicts(summary.run_id, "cpu_bound")
    assert verdicts, "L3 data present but CPUBoundAnalyzer emitted no verdict"
    for v in verdicts:
        assert v["verdict"] == "core_bound", (
            f"{v['task_id']}: a CPU exerciser was classified {v['verdict']!r} "
            f"(cpu_utilization={v['evidence'].get('cpu_utilization')})"
        )
