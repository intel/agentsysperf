#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""End-to-end composition: synthetic_cpu + L3 perf via RunContext.

Demonstrates the developer experience — register measurements, wrap
work in track_span, collect records automatically::

    python examples/run_synthetic_with_l3.py
    python examples/run_synthetic_with_l3.py --tasks linalg compile
    python examples/run_synthetic_with_l3.py --duration 5

Requires perf access (``sudo sysctl kernel.perf_event_paranoid=-1``).
If perf is locked down L3 degrades to no-op and the benchmark still runs.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from src.benchmarks.synthetic_cpu import SyntheticCpuAdapter
from src.measurements.l3_perf import L3PerfMeasurement
from src.protocols import TaskSpec
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


class _NoopInvoker:
    """Synthetic CPU tasks ignore the agent invoker, but the Protocol needs one."""

    def invoke(self, instruction: str, **kwargs: Any) -> Any:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Task ids to run (default: all 9). E.g. --tasks linalg compile",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0,
        help="Per-task workload duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"{_TMP}/agentsysperf_scratch/synthetic_l3"),
        help="Where the L3 plugin writes its perf-stat CSV.",
    )
    parser.add_argument(
        "--target", default="self",
        help="L3 attribution scope: 'self', 'system', 'pid:N', 'cgroup:PATH'",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    adapter = SyntheticCpuAdapter()
    invoker = _NoopInvoker()

    ctx = RunContext(
        measurements=[L3PerfMeasurement(target=args.target)],
        output_dir=args.output_dir,
    )

    with ctx:
        # Give perf interval sampler a moment to start producing rows.
        time.sleep(0.3)

        print(f"\nRun {ctx.run_id} — duration {args.duration}s per task")
        print(
            f"{'Task':<12} {'Pass':<5} {'Elapsed':>9} "
            f"{'IPC':>6} {'Cache%':>7} {'BrMiss%':>8}"
        )
        print("-" * 60)

        selected = list(adapter.list_tasks(include=args.tasks))
        if not selected:
            print(f"No matching tasks for include={args.tasks}")
            return

        for spec in selected:
            spec = TaskSpec(
                **{**spec.__dict__, "extra": {**spec.extra, "duration_s": args.duration}},
            )

            with track_span(ctx, f"{ctx.run_id}::{spec.id}", kind="synthetic_cpu", node_id=spec.id):
                task_result = adapter.run_task(spec, agent_invoker=invoker)

            # Records for this span are the last ones appended.
            l3_records = [r for r in ctx.records if r.span_id == f"{ctx.run_id}::{spec.id}"]

            ipc = cache = br = float("nan")
            if l3_records:
                payload = l3_records[0].payload
                ipc = payload.get("ipc", float("nan"))
                cache = payload.get("cache_miss_pct", float("nan"))
                br = payload.get("branch_miss_pct", float("nan"))

            print(
                f"{spec.id:<12} {str(task_result.passed):<5} "
                f"{task_result.extra['elapsed_s']:>8.2f}s "
                f"{ipc:>6.2f} {cache:>6.2f}% {br:>7.2f}%"
            )

    adapter.teardown()
    print(f"\n{len(ctx.records)} L3 record(s) emitted across {len(selected)} task(s).")
    print(f"perf interval CSV: {args.output_dir / 'l3_perf_continuous.csv'}")


if __name__ == "__main__":
    main()
