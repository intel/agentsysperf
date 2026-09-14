#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Terminal-Bench TB2: Run 10-20 tasks from Harbor registry with breakdown analysis.

Demonstrates multi-dimensional analysis (inference/execution/orchestration) on
real Terminal-Bench 2 tasks loaded from Harbor.

Usage:
    export OPENAI_API_KEY=sk-...
    poetry run python examples/run_terminal_bench_tb2.py --limit 10
    poetry run python examples/run_terminal_bench_tb2.py --limit 20 --model gpt-4o

Requirements:
    - Harbor 0.8.0+ (in dependencies)
    - Docker running (for Harbor environments)
    - OPENAI_API_KEY in environment

Output:
    - Measurement records with L1+L3 metrics
    - Breakdown analysis: inference/execution/orchestration percentages
    - Per-task and run-wide analyzer verdicts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.benchmarks.terminal_bench.harbor_loader import (
    load_tasks_from_harbor_registry,
)
from src.protocols import discover_measurements
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of TB2 tasks to run (default: 10).",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*",
        help="Specific task IDs to run (e.g., terminal-bench/make-mips-interpreter).",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LiteLLM model identifier (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=15,
        help="Max agent turns per task.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{_TMP}/agentsysperf_scratch/terminal_bench_tb2",
        help="Where to write measurement records and trial results.",
    )
    args = parser.parse_args()

    print(f"=== Terminal-Bench TB2: {args.model} on {args.limit} task(s) ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2

    print("1. Loading TB2 tasks from Harbor registry...")
    try:
        tasks = asyncio.run(load_tasks_from_harbor_registry(limit=args.limit))
    except Exception as e:
        print(f"ERROR: Failed to load TB2 tasks from Harbor: {e}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("  - Is Docker running?", file=sys.stderr)
        print("  - Is Harbor registry accessible?", file=sys.stderr)
        return 1

    # Filter by specific task IDs if provided
    if args.tasks:
        tasks = [t for t in tasks if t.task_id in args.tasks]
        if not tasks:
            print(f"ERROR: No tasks found matching: {args.tasks}", file=sys.stderr)
            return 1

    adapter = TerminalBenchAdapter(dataset_loader=lambda: tasks)
    print(f"   {len(tasks)} task(s) loaded from Harbor TB2\n")

    print(f"2. Building LiteLLMAgentInvoker (model={args.model})...")
    invoker = LiteLLMAgentInvoker(
        model=args.model,
        max_turns=args.max_turns,
        temperature=0.0,
    )
    print("   Invoker ready\n")

    print("3. Discovering measurement plugins...")
    measurements = discover_measurements()
    print(f"   {len(measurements)} plugin(s): {', '.join(measurements.keys())}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"4. Output: {output_dir}\n")

    ctx = RunContext(
        measurements=list(measurements.values()),
        output_dir=output_dir,
    )

    task_specs = list(adapter.list_tasks(limit=args.limit))
    trial_results = []

    print("5. Running tasks...")
    with ctx:
        for spec in task_specs:
            print(f"\n   ▶ {spec.id}")
            with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
                result = adapter.run_task(spec, agent_invoker=invoker, run_context=ctx)

            extra = result.extra or {}
            verdict = "PASS" if result.passed else "FAIL"
            oracle_run = extra.get("oracle_run", False)
            print(
                f"     {verdict}  reward={result.reward:.2f}  "
                f"oracle={'Harbor' if oracle_run else 'no'}  "
                f"turns={extra.get('num_turns', '?')}  "
                f"cmds={extra.get('num_commands', '?')}  "
                f"submitted={extra.get('submitted', False)}"
            )
            if result.error:
                print(f"     error: {result.error[:200]}")

            trial_results.append({
                "task_id": result.task_id,
                "passed": result.passed,
                "reward": result.reward,
                "num_turns": extra.get("num_turns"),
                "num_commands": extra.get("num_commands"),
            })

    passed = sum(1 for r in trial_results if r["passed"])
    print(f"\n6. Summary")
    print(f"   {passed}/{len(trial_results)} task(s) passed oracle")
    print(f"   {len(ctx.records)} measurement record(s)")
    print(f"   Layers: {', '.join(sorted(set(r.layer for r in ctx.records)))}")

    trial_results_file = output_dir / "trial_results.json"
    with open(trial_results_file, "w") as f:
        json.dump(trial_results, f, indent=2)
    print(f"   Wrote: {trial_results_file}")

    print(f"\n7. Run analysis:")
    print(f"   poetry run agentsysperf analyze {output_dir}\n")
    print(f"Expected output:")
    print(f"  - Per-task breakdown: inference/execution/orchestration %")
    print(f"  - CPU bottleneck classification (io_bound for hosted LLM)")
    print(f"  - Cache behavior (L3 pressure vs resident)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
