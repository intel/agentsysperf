#!/usr/bin/env python3.11
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Terminal-Bench Phase A smoke test.

Demonstrates:
1. Loading synthetic tasks (no Harbor)
2. Using NoOpAgentInvoker (no real LLM)
3. Running through TerminalBenchAdapter
4. Verifying measurements are captured

This is the minimal end-to-end test for Phase A: foundation without
Harbor/Docker or real agent integration.
"""

import json
from pathlib import Path
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.benchmarks.terminal_bench.dataset import generate_sample_tasks
from src.testing.noop_invoker import NoOpAgentInvoker
from src.runner import RunContext, track_span
from src.protocols import discover_measurements
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main():
    print("=== Terminal-Bench Phase A: Smoke Test ===\n")

    # 1. Create adapter with synthetic tasks
    print("1. Initializing TerminalBenchAdapter with synthetic tasks...")
    sample_tasks = generate_sample_tasks(n=3)
    adapter = TerminalBenchAdapter(
        dataset_loader=lambda: sample_tasks,
    )
    print(f"   Loaded {len(sample_tasks)} synthetic tasks\n")

    # 2. Create NoOp agent invoker
    print("2. Creating NoOpAgentInvoker...")
    invoker = NoOpAgentInvoker(simulate_failure=False, sleep_duration=0.05)
    print("   Invoker ready (will return canned success responses)\n")

    # 3. Discover measurements
    print("3. Discovering measurement plugins...")
    measurements = discover_measurements()
    print(f"   Found {len(measurements)} measurement plugins: {', '.join(measurements.keys())}\n")

    # 4. Set up run context
    print("4. Setting up RunContext...")
    output_dir = Path(f"{_TMP}/agentsysperf_scratch/terminal_bench_phase_a")
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        measurements=list(measurements.values()),
        output_dir=output_dir,
    )
    print(f"   Output directory: {output_dir}\n")

    # 5. Run tasks with measurement tracking
    print("5. Running tasks...")
    task_specs = list(adapter.list_tasks(limit=3))
    print(f"   Processing {len(task_specs)} tasks:")
    for spec in task_specs:
        print(f"   - {spec.id}")

    with ctx:
        for spec in task_specs:
            print(f"\n   Running {spec.id}...")
            with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
                result = adapter.run_task(spec, agent_invoker=invoker)
            print(f"     → passed={result.passed}, reward={result.reward:.2f}")

    print(f"\n6. Results written to: {output_dir}")
    print(f"   Invoker was called {invoker.call_count} times")

    # 7. Verify measurement records
    print(f"   Captured {len(ctx.records)} measurement records")
    if ctx.records:
        layers = {r.layer for r in ctx.records}
        print(f"   Layers: {', '.join(sorted(layers))}")

        # Serialize records to JSON for analysis
        records_file = output_dir / "measurement_records.json"
        with open(records_file, "w") as f:
            json.dump(
                [{"span_id": r.span_id, "layer": r.layer, "payload": r.payload} for r in ctx.records],
                f,
                indent=2,
            )
        print(f"   Saved to: {records_file}")
    else:
        print("   [WARNING] No measurement records captured")

    print("\n=== Phase A smoke test complete ===")


if __name__ == "__main__":
    main()
