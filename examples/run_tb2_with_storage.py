#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Run TB2 workload benchmark with result storage and reporting.

Executes 15 curated TB2 tasks (light/medium/heavy), stores results in SQLite,
and generates PowerPoint report highlighting Xeon EMR performance.

Usage:
    export OPENAI_API_KEY=sk-...
    poetry run python examples/run_tb2_with_storage.py --workload all
    poetry run python examples/run_tb2_with_storage.py --workload light
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

# Curated task selection (from tb2_workload_selection.txt)
TASK_SELECTION = {
    'light': [
        'terminal-bench/count-vowels',
        'terminal-bench/find-largest-file',
        'terminal-bench/extract-emails',
        'terminal-bench/list-permissions',
        'terminal-bench/count-http-codes',
    ],
    'medium': [
        'terminal-bench/json-transformation',
        'terminal-bench/csv-aggregate',
        'terminal-bench/log-analysis',
        'terminal-bench/markdown-to-html',
        'terminal-bench/data-validation',
    ],
    'heavy': [
        'terminal-bench/make-mips-interpreter',
        'terminal-bench/circuit-fibsqrt',
        'terminal-bench/compile-c-module',
        'terminal-bench/compress-decompress',
        'terminal-bench/hash-computation',
    ],
}


def load_curated_tasks(workload_types: list[str]) -> list:
    """Load tasks from Harbor registry filtered by curated selection."""
    # Build task selection list
    selected_ids = []
    for wl_type in workload_types:
        selected_ids.extend(TASK_SELECTION[wl_type])

    # Load specific tasks from Harbor by name
    # Extract task names from full IDs (remove "terminal-bench/" prefix)
    task_names = [tid.replace('terminal-bench/', '') for tid in selected_ids]

    all_tasks = load_tasks_from_harbor_registry(
        dataset_name="terminal-bench/terminal-bench-2",
        ref="latest",
        limit=None,
        task_names=task_names,
    )

    # Attach workload_type metadata
    for task in all_tasks:
        for wl_type, task_list in TASK_SELECTION.items():
            if task.task_id in task_list:
                if not task.extra:
                    task.extra = {}
                task.extra['workload_type'] = wl_type
                break

    return list(all_tasks)


def classify_workload(task_result: dict, duration_s: float) -> str:
    """Infer workload type if not already set."""
    # Try to get from task metadata
    if 'workload_type' in task_result.get('extra', {}):
        return task_result['extra']['workload_type']

    # Fallback heuristic
    execution_pct = task_result.get('extra', {}).get('execution_pct', 0)
    if duration_s < 10 and execution_pct < 20:
        return 'light'
    elif duration_s > 60 or execution_pct > 60:
        return 'heavy'
    else:
        return 'medium'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=['light', 'medium', 'heavy', 'all'],
        default='all',
        help="Which workload category to run (default: all 15 tasks).",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LiteLLM model identifier.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Max agent turns per task (heavy workloads need more).",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{_TMP}/agentsysperf_tb2_benchmark",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier (default: timestamp).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    run_id = args.run_id or f"tb2_{int(time.time())}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which workloads to run
    if args.workload == 'all':
        workload_types = ['light', 'medium', 'heavy']
    else:
        workload_types = [args.workload]

    print(f"=== TB2 Workload Benchmark ===")
    print(f"Run ID: {run_id}")
    print(f"Workloads: {', '.join(workload_types)}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}\n")

    # Load tasks from Harbor
    print("1. Loading tasks from Harbor...")
    try:
        tasks = load_curated_tasks(workload_types)
    except Exception as e:
        print(f"ERROR: Failed to load tasks: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(f"   Loaded {len(tasks)} task(s)\n")

    # Initialize components
    print("2. Initializing benchmark components...")
    adapter = TerminalBenchAdapter(dataset_loader=lambda: tasks)
    invoker = LiteLLMAgentInvoker(
        model=args.model,
        max_turns=args.max_turns,
        temperature=0.0,
    )
    measurements = discover_measurements()
    analyzers = discover_analyzers()

    print(f"   {len(measurements)} measurements: {', '.join(measurements.keys())}")
    print(f"   {len(analyzers)} analyzers: {', '.join(analyzers.keys())}\n")

    # Initialize storage
    store = SQLiteResultStore(output_dir=output_dir)
    # Source the SKU from the host, not a literal — a hardcoded SKU mislabels
    # results on any other CPU (the most credibility-damaging field in a
    # hardware benchmark). See STORAGE_IMPLEMENTATION_PLAN P2/P3.
    from src.platform.detect import detect_platform
    hardware_sku = detect_platform().model_name
    store.store_run_metadata(
        run_id=run_id,
        metadata={
            "start_time": int(time.time()),
            "hardware_sku": hardware_sku,
            "model": args.model,
            "workload_types": workload_types,
            "max_turns": args.max_turns,
        }
    )

    # Run benchmark
    print("3. Running benchmark...\n")
    # Thread the SAME run_id into RunContext (it would otherwise mint its own
    # run-<uuid>, diverging from the store/Langfuse/Prometheus run_id). One id
    # end-to-end is the cross-system join key — see STORAGE_IMPLEMENTATION_PLAN P2.
    ctx = RunContext(
        run_id=run_id,
        measurements=list(measurements.values()),
        output_dir=output_dir,
    )

    task_specs = list(adapter.list_tasks(limit=len(tasks)))
    trial_results = []
    passed_count = 0

    with ctx:
        for spec in task_specs:
            print(f"   ▶ {spec.id}")
            t0 = time.time()

            with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
                result = adapter.run_task(spec, agent_invoker=invoker, run_context=ctx)

            duration_s = time.time() - t0
            extra = result.extra or {}
            verdict = "✓ PASS" if result.passed else "✗ FAIL"

            print(f"     {verdict}  {duration_s:.1f}s  turns={extra.get('num_turns', '?')}  "
                  f"cmds={extra.get('num_commands', '?')}")

            if result.passed:
                passed_count += 1

            # Infer workload type
            workload_type = extra.get('workload_type') or classify_workload(extra, duration_s)

            # Store task result
            store.store_task_result(
                run_id=run_id,
                task_id=result.task_id,
                result={
                    "passed": result.passed,
                    "reward": result.reward,
                    "duration_s": duration_s,
                    "num_turns": extra.get('num_turns'),
                    "num_commands": extra.get('num_commands'),
                    "workload_type": workload_type,
                    "error": result.error,
                }
            )

            trial_results.append(result)

    # Store measurements
    print(f"\n4. Storing measurements ({len(ctx.records)} records)...")
    store.store_measurements(run_id=run_id, records=ctx.records)

    # Run analyzers
    print(f"5. Running analyzers...")
    analysis_results = []
    for analyzer_name, analyzer in analyzers.items():
        try:
            for analysis_result in analyzer.analyze(ctx.records):
                analysis_results.append(analysis_result)
                print(f"   {analysis_result.span_id}: {analyzer_name}={analysis_result.verdict}")
        except Exception as e:
            print(f"   WARNING: {analyzer_name} failed: {e}")

    # Store analyzer verdicts
    store.store_analysis_results(run_id=run_id, results=analysis_results)

    # Update run metadata
    store.store_run_metadata(
        run_id=run_id,
        metadata={
            "end_time": int(time.time()),
            "total_tasks": len(task_specs),
            "passed_tasks": passed_count,
        }
    )

    # Summary
    print(f"\n=== Summary ===")
    print(f"Run ID: {run_id}")
    print(f"Passed: {passed_count}/{len(task_specs)}")
    print(f"Measurements: {len(ctx.records)} records")
    print(f"Analyzer verdicts: {len(analysis_results)}")
    print(f"Database: {output_dir}/agentsysperf_results.db")
    print(f"\nNext: Generate report")
    print(f"  poetry run python examples/generate_report.py {run_id} --output tb2_report.pptx")

    return 0


if __name__ == "__main__":
    sys.exit(main())
