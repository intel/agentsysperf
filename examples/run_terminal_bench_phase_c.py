#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Terminal-Bench Phase C: real Harbor tasks with Docker + test oracles.

Demonstrates:
1. Loading real Terminal-Bench tasks from Harbor registry
2. Running gpt-4o-mini via LiteLLM against Harbor Docker containers
3. Scoring via Harbor's deterministic test scripts (tests/test.sh)
4. L1+L3 measurement capture per task

Requires:
- OPENAI_API_KEY in env
- Docker daemon accessible
- Harbor >=0.8.0 (Python 3.12+)

Usage:
    poetry run python examples/run_terminal_bench_phase_c.py --tasks 2
    poetry run python examples/run_terminal_bench_phase_c.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.benchmarks.terminal_bench.harbor_loader import load_tasks_from_harbor_registry
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
        "--limit", type=int, default=2,
        help="Number of tasks to load from Harbor registry.",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LiteLLM model identifier.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=15,
        help="Max agent turns per task.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{_TMP}/agentsysperf_scratch/terminal_bench_phase_c",
        help="Where to write measurement records and trial results.",
    )
    args = parser.parse_args()

    print(f"=== Terminal-Bench Phase C: Harbor + {args.model} ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2

    print(f"1. Loading {args.limit} task(s) from Harbor registry...")
    try:
        harbor_tasks = list(load_tasks_from_harbor_registry(
            dataset_name="terminal-bench/terminal-bench-2",
            ref="latest",
            limit=args.limit,
        ))
    except Exception as e:
        print(f"ERROR: Harbor task loading failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(f"   Loaded {len(harbor_tasks)} Harbor task(s)")
    for t in harbor_tasks[:3]:
        print(f"   - {t.task_id} ({t.category}, {t.difficulty})")
    print()

    print(f"2. Building LiteLLMAgentInvoker (model={args.model})...")
    invoker = LiteLLMAgentInvoker(
        model=args.model,
        max_turns=args.max_turns,
        temperature=0.0,
    )
    print("   Invoker ready\n")

    print("3. Creating TerminalBenchAdapter with Harbor loader...")
    adapter = TerminalBenchAdapter(dataset_loader=lambda: harbor_tasks)
    print("   Adapter ready\n")

    print("4. Discovering measurement plugins...")
    measurements = discover_measurements()
    print(f"   {len(measurements)} plugin(s): {', '.join(measurements.keys())}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"5. Output: {output_dir}\n")

    ctx = RunContext(
        measurements=list(measurements.values()),
        output_dir=output_dir,
    )

    task_specs = list(adapter.list_tasks(limit=args.limit))
    trial_results = []

    print("6. Running tasks (Harbor Docker + real test oracles)...\n")
    with ctx:
        for spec in task_specs:
            print(f"   ▶ {spec.id}")
            try:
                with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
                    result = adapter.run_task(spec, agent_invoker=invoker)

                extra = result.extra or {}
                verdict = "✓ PASS" if result.passed else "✗ FAIL"
                oracle_run = extra.get("oracle_run", False)
                print(
                    f"     {verdict}  reward={result.reward:.2f}  "
                    f"oracle={'Harbor' if oracle_run else 'none'}  "
                    f"turns={extra.get('num_turns', '?')}  "
                    f"cmds={extra.get('num_commands', '?')}"
                )
                if result.error:
                    print(f"     error: {result.error[:150]}")

                trial_results.append({
                    "task_id": result.task_id,
                    "passed": result.passed,
                    "reward": result.reward,
                    "oracle_run": oracle_run,
                    "submitted": extra.get("submitted"),
                    "num_turns": extra.get("num_turns"),
                    "num_commands": extra.get("num_commands"),
                    "elapsed_s": extra.get("elapsed_s"),
                    "error": result.error,
                })
            except Exception as e:
                print(f"     ✗ EXCEPTION: {type(e).__name__}: {e}")
                trial_results.append({
                    "task_id": spec.id,
                    "passed": False,
                    "reward": 0.0,
                    "error": str(e),
                })
            print()

    print("7. Summary")
    n_pass = sum(1 for r in trial_results if r["passed"])
    print(f"   {n_pass}/{len(trial_results)} task(s) passed Harbor oracle")
    print(f"   {len(ctx.records)} measurement record(s)")
    if ctx.records:
        layers = sorted({r.layer for r in ctx.records})
        print(f"   Layers: {', '.join(layers)}")

    # Persist artifacts
    (output_dir / "trial_results.json").write_text(
        json.dumps(trial_results, indent=2)
    )
    (output_dir / "measurement_records.json").write_text(
        json.dumps(
            [{"span_id": r.span_id, "layer": r.layer, "payload": r.payload}
             for r in ctx.records],
            indent=2,
        )
    )
    print(f"   Wrote: {output_dir}/trial_results.json")
    print(f"   Wrote: {output_dir}/measurement_records.json")

    print("\n=== Phase C complete ===")
    return 0 if n_pass == len(trial_results) else 1


if __name__ == "__main__":
    sys.exit(main())
