#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Simplified TB2 benchmark runner - loads first N tasks from registry.

Usage:
    export OPENAI_API_KEY=sk-...
    poetry run python examples/run_tb2_simple.py --limit 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.benchmarks.terminal_bench.harbor_loader import load_tasks_from_harbor_registry
from src.protocols import discover_measurements, discover_analyzers
from src.runner import RunContext, track_span
from src.storage.sqlite_store import SQLiteResultStore
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3, help="Number of tasks to run")
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model")
    parser.add_argument("--max-turns", type=int, default=15, help="Max turns per task")
    parser.add_argument("--output-dir", default=f"{_TMP}/agentsysperf_tb2_simple")
    parser.add_argument("--run-id", help="Run identifier")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    run_id = args.run_id or f"tb2_simple_{int(time.time())}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== TB2 Simple Benchmark ===")
    print(f"Run ID: {run_id}")
    print(f"Limit: {args.limit} tasks")
    print(f"Model: {args.model}\n")

    # Load tasks from Harbor (first N from registry)
    print("1. Loading tasks from Harbor...")
    try:
        tasks = load_tasks_from_harbor_registry(limit=args.limit)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(f"   Loaded {len(list(tasks))} task(s)\n")

    # Re-load (tasks is iterator, consumed above)
    tasks = load_tasks_from_harbor_registry(limit=args.limit)

    # Initialize
    adapter = TerminalBenchAdapter(dataset_loader=lambda: tasks)
    invoker = LiteLLMAgentInvoker(model=args.model, max_turns=args.max_turns, temperature=0.0)
    measurements = discover_measurements()
    analyzers = discover_analyzers()

    # Storage
    store = SQLiteResultStore(output_dir=output_dir)
    store.store_run_metadata(
        run_id=run_id,
        metadata={
            "start_time": int(time.time()),
            "hardware_sku": "Intel Xeon Platinum 8592+",
            "model": args.model,
        }
    )

    # Run
    print("2. Running benchmark...\n")
    ctx = RunContext(measurements=list(measurements.values()), output_dir=output_dir)

    task_specs = list(adapter.list_tasks(limit=args.limit))
    passed = 0

    with ctx:
        for spec in task_specs:
            print(f"   ▶ {spec.id}")
            t0 = time.time()

            with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
                result = adapter.run_task(spec, agent_invoker=invoker, run_context=ctx)

            duration_s = time.time() - t0
            verdict = "✓" if result.passed else "✗"
            print(f"     {verdict} {duration_s:.1f}s")

            if result.passed:
                passed += 1

            store.store_task_result(
                run_id=run_id,
                task_id=result.task_id,
                result={
                    "passed": result.passed,
                    "duration_s": duration_s,
                    "workload_type": "unknown",
                }
            )

    # Store measurements, step traces & run analyzers
    print(f"\n3. Storing & analyzing ({len(ctx.records)} records)...")
    store.store_measurements(run_id=run_id, records=ctx.records)
    # Persist step-level execution trace (per-turn LLM + tool events with
    # tokens/cost/exit status) the agent loop accumulated on the RunContext.
    store.store_spans(run_id=run_id, spans=ctx.step_traces)

    analysis_results = []
    for name, analyzer in analyzers.items():
        try:
            for ar in analyzer.analyze(ctx.records):
                analysis_results.append(ar)
        except Exception as e:
            print(f"   WARNING: {name} failed: {e}")

    store.store_analysis_results(run_id=run_id, results=analysis_results)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Passed: {passed}/{len(task_specs)}")
    print(f"Measurements: {len(ctx.records)}")
    print(f"Verdicts: {len(analysis_results)}")
    print(f"Database: {output_dir}/agentsysperf_results.db")

    return 0


if __name__ == "__main__":
    sys.exit(main())
