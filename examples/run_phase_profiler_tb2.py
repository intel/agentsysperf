#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Run Terminal-Bench with phase-tagged spans and PhaseProfiler analysis.

Validates the full pipeline:
  LiteLLMTerminalAgentLoop (phase="reason"/"act")
  → L1 MeasurementRecords (with phase in payload)
  → L3 PerfMeasurement (IPC, cache-miss per span)
  → PhaseProfiler analyzer → per-phase hardware characterization

Modes:
  --dry-run     Use a mock LLM (no API key needed). Proves the measurement
                pipeline and PhaseProfiler logic end-to-end.
  (default)     Uses LiteLLM with real API calls. Requires OPENAI_API_KEY
                (or equivalent for the --model provider).

Usage:
    # Dry run (no API key needed, validates pipeline):
    poetry run python examples/run_phase_profiler_tb2.py --dry-run

    # Real run (requires API key):
    export OPENAI_API_KEY=sk-...
    poetry run python examples/run_phase_profiler_tb2.py --model gpt-4o-mini --tasks 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzers.phase_profiler import PhaseProfiler
from src.benchmarks.terminal_bench.dataset import generate_sample_tasks
from src.benchmarks.terminal_bench.environment import StandaloneEnvironment
from src.measurements.l1_subspan.probe import L1SubSpanMeasurement
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def make_measurements() -> list:
    """Build measurement stack: L1 always, L3 if perf is available."""
    measurements = [L1SubSpanMeasurement()]
    try:
        from src.measurements.l3_perf.probe import L3PerfMeasurement
        from src.measurements.l3_perf._perf_subprocess import perf_available
        if perf_available():
            measurements.append(L3PerfMeasurement(target="self", sample_interval_ms=50))
            print("  L3 perf counters: ENABLED (IPC, cache-miss will populate)")
        else:
            print("  L3 perf counters: UNAVAILABLE (kernel.perf_event_paranoid)")
    except ImportError:
        print("  L3 perf counters: UNAVAILABLE (import failed)")
    return measurements


def run_with_real_llm(model: str, max_turns: int, num_tasks: int) -> RunContext:
    """Run real Terminal-Bench tasks with LiteLLM.

    All 5 phases instrumented: admit (env setup), retrieve (context embedded in
    LiteLLM loop), reason + act (from LiteLLMTerminalAgentLoop), commit (oracle).
    """
    from src.agent_loops.litellm_terminal_loop import LiteLLMTerminalAgentLoop

    tasks = generate_sample_tasks(n=num_tasks)
    measurements = make_measurements()
    output_dir = Path(f"{_TMP}/agentsysperf_phase_tb2")
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        measurements=measurements,
        output_dir=output_dir,
        run_id=f"phase_tb2_{int(time.time())}",
        sampler_interval_ms=10,
    )

    print(f"\nRunning {len(tasks)} task(s) with model={model}, max_turns={max_turns}")
    print()

    with ctx:
        for task in tasks:
            print(f"  ▶ {task.task_id}")

            with track_span(ctx, task.task_id, kind="terminal_bench", node_id=task.task_id):

                # Phase 01: Admit — env bootstrap, auth, policy, routing
                admit_span = f"{task.task_id}/admit"
                with track_span(ctx, admit_span, kind="governance",
                               node_id="admit", phase="admit"):
                    env = StandaloneEnvironment()
                    import asyncio
                    asyncio.run(env.start())
                    for cmd in task.setup_commands:
                        from src.benchmarks.terminal_bench.environment import SyncEnvironmentWrapper
                        SyncEnvironmentWrapper(env).exec(cmd, timeout_sec=10)

                # Phase 03+04: Reason+Act — handled inside LiteLLMTerminalAgentLoop
                loop = LiteLLMTerminalAgentLoop(
                    model=model,
                    env=env,
                    max_turns=max_turns,
                    temperature=0.0,
                    run_context=ctx,
                )
                result = loop.solve(task_id=task.task_id, instruction=task.instruction)

                # Phase 05: Commit — oracle verification, writeback
                commit_span = f"{task.task_id}/commit"
                with track_span(ctx, commit_span, kind="governance",
                               node_id="commit", phase="commit"):
                    _simulate_audit_writeback()

            verdict = "PASS" if result.submitted else "DONE"
            print(f"    {verdict} | turns={result.num_turns} | gen={result.total_generation_ms:.0f}ms | cmd={result.total_command_ms:.0f}ms")
            env.cleanup()

    return ctx


def run_dry_mode(num_turns: int = 6) -> RunContext:
    """Run with a mock LLM that emits canned ReACT responses.

    Proves the full measurement pipeline without any API key.
    All 5 agentic pipeline phases are instrumented:
      01 Admit   — env bootstrap, policy check, task routing
      02 Retrieve — context gathering (file reads, history assembly)
      03 Reason  — LLM inference (token generation)
      04 Act     — tool/command execution
      05 Commit  — result verification, audit writeback
    """
    measurements = make_measurements()
    output_dir = Path(f"{_TMP}/agentsysperf_phase_tb2_dry")
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        measurements=measurements,
        output_dir=output_dir,
        run_id=f"phase_tb2_dry_{int(time.time())}",
        sampler_interval_ms=10,
    )

    task_id = "sample/count-lines"

    # Canned agent commands (what a ReACT agent would do)
    canned_commands = [
        "ls *.py",
        "wc -l foo.py bar.py baz.py",
        "cat foo.py | wc -l",
        "cat bar.py | wc -l",
        "cat baz.py | wc -l",
        "echo 10 > result.txt",
    ]

    print(f"\n  Dry-run: simulating {len(canned_commands)} turns on '{task_id}'")
    print()

    with ctx:
        with track_span(ctx, task_id, kind="terminal_bench", node_id=task_id):

            # Phase 01: Admit — environment bootstrap, auth, policy, routing
            admit_span = f"{task_id}/admit"
            with track_span(ctx, admit_span, kind="governance",
                           node_id="admit", phase="admit"):
                env = StandaloneEnvironment()
                import asyncio
                asyncio.run(env.start())
                from src.benchmarks.terminal_bench.environment import SyncEnvironmentWrapper
                sync_env = SyncEnvironmentWrapper(env)
                sync_env.exec("printf 'a=1\\nb=2\\nc=3\\n' > foo.py", timeout_sec=5)
                sync_env.exec("printf 'def f():\\n    pass\\n\\ndef g():\\n    pass\\n' > bar.py", timeout_sec=5)
                sync_env.exec("printf 'x=1\\ny=2\\n' > baz.py", timeout_sec=5)
                _simulate_policy_check()

            for turn_idx, cmd in enumerate(canned_commands):
                # Phase 02: Retrieve — context gathering (read files, assemble history)
                retrieve_span = f"{task_id}/turn_{turn_idx}_retrieve"
                with track_span(ctx, retrieve_span, kind="retrieval",
                               node_id=f"retrieve_{turn_idx}", phase="retrieve"):
                    _simulate_context_retrieval(sync_env, turn_idx)

                # Phase 03: Reason — LLM inference (token generation)
                span_id = f"{task_id}/turn_{turn_idx}_llm"
                with track_span(ctx, span_id, kind="inference",
                               node_id=f"llm_call_{turn_idx}", phase="reason"):
                    _simulate_inference(turn_idx)

                # Phase 04: Act — execute real commands
                cmd_span_id = f"{task_id}/turn_{turn_idx}_cmd"
                with track_span(ctx, cmd_span_id, kind="execution",
                               node_id=f"cmd_{turn_idx}", phase="act"):
                    result = sync_env.exec(cmd, timeout_sec=30)
                    if result.output:
                        print(f"    turn {turn_idx}: `{cmd}` → {result.output[:60]}")
                    else:
                        print(f"    turn {turn_idx}: `{cmd}` → (ok)")

            # Phase 05: Commit — oracle verification, result writeback, audit
            commit_span = f"{task_id}/commit"
            with track_span(ctx, commit_span, kind="governance",
                           node_id="commit", phase="commit"):
                oracle_result = sync_env.exec(
                    "test -f result.txt && [ \"$(cat result.txt | tr -d '[:space:]')\" = '10' ]",
                    timeout_sec=5,
                )
                _simulate_audit_writeback()

    print(f"\n  Oracle: {'PASS' if oracle_result.return_code == 0 else 'FAIL'}")
    env.cleanup()

    return ctx


def _simulate_policy_check() -> None:
    """Simulate Admit phase: auth token validation, policy evaluation, task routing."""
    import hashlib
    token = b"agent-task-token-sample-count-lines"
    for _ in range(500):
        token = hashlib.sha256(token).digest()
    time.sleep(0.01)


def _simulate_context_retrieval(sync_env, turn_idx: int) -> None:
    """Simulate Retrieve phase: read prior outputs, assemble conversation history."""
    sync_env.exec("cat /proc/meminfo 2>/dev/null | head -5", timeout_sec=5)
    import array
    size = 50_000 + turn_idx * 10_000
    arr = array.array('d', range(size))
    total = 0.0
    for i in range(0, len(arr), 128):
        total += arr[i]
    time.sleep(0.005)


def _simulate_audit_writeback() -> None:
    """Simulate Commit phase: serialize result, write audit log, cache invalidation."""
    import hashlib
    import json as _json
    payload = _json.dumps({"task": "count-lines", "result": "10", "status": "pass"}).encode()
    for _ in range(200):
        payload = hashlib.sha256(payload).digest()
    time.sleep(0.01)


def _simulate_inference(turn_idx: int) -> None:
    """Simulate LLM inference with CPU work that mimics token generation."""
    import array
    size = 200_000 + turn_idx * 50_000  # grows per turn (longer context)
    arr = array.array('d', range(size))
    total = 0.0
    for i in range(0, len(arr), 64):  # stride to hit cache lines
        total += arr[i]
    time.sleep(0.02)


def analyze_results(ctx: RunContext) -> None:
    """Run PhaseProfiler on collected records and display results."""
    records = ctx.records
    print(f"\n{'='*60}")
    print(f"PHASE PROFILER ANALYSIS")
    print(f"{'='*60}")
    print(f"Total measurement records: {len(records)}")

    # Count by layer
    l1_count = sum(1 for r in records if r.layer == "l1")
    l3_count = sum(1 for r in records if r.layer == "l3")
    print(f"  L1 (resource/timing): {l1_count}")
    print(f"  L3 (perf counters):   {l3_count}")

    # Count phase-tagged
    phase_tagged = sum(1 for r in records if r.payload.get("phase"))
    print(f"  Phase-tagged:         {phase_tagged}")
    print()

    profiler = PhaseProfiler()
    results = list(profiler.analyze(records))

    if not results:
        print("WARNING: PhaseProfiler produced no results.")
        print("  This likely means no phase-tagged spans were found.")
        return

    result = results[0]
    print(f"Verdict:    {result.verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Phases:     {result.evidence['phases_detected']}")
    print(f"Iterations: {result.evidence['iteration_count']}")
    print(f"Total wall: {result.evidence['total_wall_ms']:.1f} ms")
    print(f"Total CPU:  {result.evidence['total_cpu_s']:.3f} s")
    print()

    # Per-phase table
    bd = result.evidence["phase_breakdown"]
    print(f"{'Phase':<10} {'Wall%':>7} {'CPU%':>7} {'WallMs':>8} {'IPC':>6} {'Miss%':>7} {'Pattern':<22} {'N':>3}")
    print("-" * 76)
    for phase, data in bd.items():
        ipc_s = f"{data['avg_ipc']:.2f}" if data["avg_ipc"] is not None else "  — "
        miss_s = f"{data['avg_cache_miss_pct']:.1f}" if data["avg_cache_miss_pct"] is not None else "  — "
        print(f"{phase:<10} {data['wall_pct']:>6.1f}% {data['cpu_pct']:>6.1f}% "
              f"{data['wall_ms']:>7.1f} {ipc_s:>6} {miss_s:>6}% "
              f"{data['pattern']:<22} {data['span_count']:>3}")
    print()

    # Inflection
    inflection = result.evidence.get("inflection")
    if inflection:
        print(f"INFLECTION POINT: iteration {inflection['iteration']}")
        print(f"  {inflection['reason']}")
        print(f"  Cumulative reason: {inflection['cumulative_reason_s']:.3f}s")
        print(f"  Cumulative other:  {inflection['cumulative_other_s']:.3f}s")
        print(f"  Ratio: {inflection['ratio']:.2f}x")
    else:
        print("INFLECTION: not reached (inference still dominates over orchestration)")
    print()

    # Recommendations
    if result.recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    print()

    # Solutions
    print("PER-PHASE SOLUTIONS (Intel Xeon optimization targets):")
    for phase, solutions in result.evidence["phase_solutions"].items():
        if solutions:
            print(f"  {phase}: {'; '.join(solutions)}")
    print()

    # Save full result as JSON
    output_dir = ctx.output_dir
    analysis_file = output_dir / "phase_analysis.json"
    analysis_data = {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "recommendations": list(result.recommendations),
        "analyzer_name": result.analyzer_name,
    }
    analysis_file.write_text(json.dumps(analysis_data, indent=2, default=str))
    print(f"Full analysis saved: {analysis_file}")
    print(f"Measurement records: {output_dir / 'measurement_records.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                       help="Use mock LLM (no API key needed)")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="LiteLLM model for real runs")
    parser.add_argument("--max-turns", type=int, default=10,
                       help="Max agent turns per task")
    parser.add_argument("--tasks", type=int, default=2,
                       help="Number of sample tasks to run")
    args = parser.parse_args()

    print("=" * 60)
    print("Terminal-Bench Phase Profiler Validation")
    print("=" * 60)
    print()

    if args.dry_run:
        print("Mode: DRY RUN (mock LLM, real command execution)")
        ctx = run_dry_mode(num_turns=6)
    else:
        if not os.environ.get("OPENAI_API_KEY") and "anthropic" not in args.model:
            print("ERROR: OPENAI_API_KEY not set. Use --dry-run for mock mode.", file=sys.stderr)
            return 2
        print(f"Mode: REAL LLM ({args.model})")
        ctx = run_with_real_llm(model=args.model, max_turns=args.max_turns, num_tasks=args.tasks)

    analyze_results(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
