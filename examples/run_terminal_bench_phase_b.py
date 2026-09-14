#!/usr/bin/env python3.11
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Terminal-Bench Phase B smoke test: real LLM via LiteLLM.

Runs synthetic Terminal-Bench tasks against gpt-4o-mini through the
LiteLLM-backed agent loop, with per-task oracle scoring and L1
measurement capture.

Requires: OPENAI_API_KEY in env.

Usage:
    poetry run python examples/run_terminal_bench_phase_b.py
    poetry run python examples/run_terminal_bench_phase_b.py --tasks 1
    poetry run python examples/run_terminal_bench_phase_b.py --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.benchmarks.terminal_bench.dataset import generate_sample_tasks
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
        "--tasks", type=int, default=3,
        help="Number of synthetic tasks to run (1-3).",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LiteLLM model identifier (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10,
        help="Max agent turns per task.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{_TMP}/agentsysperf_scratch/terminal_bench_phase_b",
        help="Where to write measurement records and trial results.",
    )
    args = parser.parse_args()

    print(f"=== Terminal-Bench Phase B: {args.model} on {args.tasks} task(s) ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2

    print("1. Loading synthetic tasks...")
    sample_tasks = generate_sample_tasks(n=args.tasks)
    adapter = TerminalBenchAdapter(dataset_loader=lambda: sample_tasks)
    print(f"   {len(sample_tasks)} task(s) ready (with oracles)\n")

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

    task_specs = list(adapter.list_tasks(limit=args.tasks))
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
                f"oracle={'yes' if oracle_run else 'no'}  "
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
                "oracle_run": oracle_run,
                "agent_self_reported_passed": extra.get("agent_self_reported_passed"),
                "submitted": extra.get("submitted"),
                "num_turns": extra.get("num_turns"),
                "num_commands": extra.get("num_commands"),
                "elapsed_s": extra.get("elapsed_s"),
                "error": result.error,
            })

    print("\n6. Summary")
    n_pass = sum(1 for r in trial_results if r["passed"])
    print(f"   {n_pass}/{len(trial_results)} task(s) passed oracle")
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

    return 0 if n_pass == len(trial_results) else 1


if __name__ == "__main__":
    sys.exit(main())
