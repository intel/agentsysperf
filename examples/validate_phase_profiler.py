#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Validate the PhaseProfiler end-to-end with simulated Terminal-Bench spans.

Exercises the full pipeline:
  track_span(phase="reason") → L1 MeasurementRecord (with phase in payload)
  track_span(phase="act")    → L1 MeasurementRecord (with phase in payload)
  PhaseProfiler.analyze(records) → AnalysisResult

Usage:
    poetry run python examples/validate_phase_profiler.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzers.phase_profiler import PhaseProfiler
from src.measurements.l1_subspan.probe import L1SubSpanMeasurement
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def simulate_terminal_bench_run(num_turns: int = 8) -> RunContext:
    """Simulate a Terminal-Bench run with phase-tagged spans."""
    l1 = L1SubSpanMeasurement()
    ctx = RunContext(
        measurements=[l1],
        output_dir=Path(f"{_TMP}/agentsysperf_phase_validation"),
        run_id="phase_validation",
        sampler_interval_ms=10,
    )

    task_id = "tb2_setup_flask_server"
    print(f"Simulating {num_turns} turns of Terminal-Bench task: {task_id}")
    print()

    with ctx:
        for turn in range(num_turns):
            # Phase 03: Reason (LLM inference) — simulate with CPU-light sleep
            span_id = f"{task_id}/turn_{turn}_llm"
            with track_span(ctx, span_id, kind="inference",
                           node_id=f"llm_call_{turn}", phase="reason"):
                # Simulate LLM latency (network-bound, low CPU)
                time.sleep(0.05)

            # Phase 04: Act (command execution) — simulate with CPU work
            cmd_span_id = f"{task_id}/turn_{turn}_cmd"
            with track_span(ctx, cmd_span_id, kind="execution",
                           node_id=f"cmd_{turn}", phase="act"):
                # Simulate command execution (growing over turns)
                _burn_cpu(iterations=50_000 + turn * 20_000)

    return ctx


def _burn_cpu(iterations: int) -> None:
    """Burn CPU cycles to produce measurable work."""
    total = 0.0
    for i in range(iterations):
        total += i * 0.001
    return


def run_phase_profiler(ctx: RunContext) -> None:
    """Run PhaseProfiler on collected records and display results."""
    profiler = PhaseProfiler()
    records = ctx.records

    print(f"Collected {len(records)} measurement records")
    print()

    # Show raw records (first few)
    print("Sample records:")
    for r in records[:4]:
        phase = r.payload.get("phase", "(none)")
        dur = r.payload.get("duration_us", "?")
        print(f"  {r.layer} | {r.span_id} | phase={phase} | duration_us={dur}")
    print(f"  ... ({len(records)} total)")
    print()

    # Run analyzer
    results = list(profiler.analyze(records))

    if not results:
        print("ERROR: PhaseProfiler produced no results!")
        sys.exit(1)

    result = results[0]
    print("=" * 60)
    print("PHASE PROFILER RESULTS")
    print("=" * 60)
    print(f"Verdict:    {result.verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Phases:     {result.evidence['phases_detected']}")
    print(f"Iterations: {result.evidence['iteration_count']}")
    print(f"Total wall: {result.evidence['total_wall_ms']:.1f} ms")
    print()

    print("Per-Phase Breakdown:")
    print(f"{'Phase':<10} {'Wall%':>6} {'CPU%':>6} {'IPC':>6} {'Miss%':>6} {'Pattern':<20} {'Spans':>5}")
    print("-" * 65)
    for phase, data in result.evidence["phase_breakdown"].items():
        ipc = f"{data['avg_ipc']:.2f}" if data["avg_ipc"] is not None else "N/A"
        miss = f"{data['avg_cache_miss_pct']:.1f}" if data["avg_cache_miss_pct"] is not None else "N/A"
        print(f"{phase:<10} {data['wall_pct']:>5.1f}% {data['cpu_pct']:>5.1f}% {ipc:>6} {miss:>6} {data['pattern']:<20} {data['span_count']:>5}")
    print()

    inflection = result.evidence.get("inflection")
    if inflection:
        print(f"Inflection Point: iteration {inflection['iteration']}")
        print(f"  {inflection['reason']}")
        print(f"  Ratio: {inflection['ratio']:.2f}x")
    else:
        print("Inflection Point: not reached (inference still dominates)")
    print()

    print("Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print()

    # Show solutions mapping
    print("Per-Phase Solutions:")
    for phase, solutions in result.evidence["phase_solutions"].items():
        if solutions:
            print(f"  {phase}: {', '.join(solutions)}")
    print()
    print("VALIDATION PASSED")


def main() -> int:
    ctx = simulate_terminal_bench_run(num_turns=8)
    run_phase_profiler(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
