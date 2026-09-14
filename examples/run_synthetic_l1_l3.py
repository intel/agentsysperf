#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Composed L1 + L3: per-span resource metrics AND hardware counters.

Demonstrates composing multiple measurement layers on the same run.
L1 gives you CPU time, wall duration, RSS, thread count per span.
L3 gives you IPC, cache-miss%, branch-miss% per span.
Both attach transparently via RunContext — zero extra wiring::

    python examples/run_synthetic_l1_l3.py
    python examples/run_synthetic_l1_l3.py --tasks linalg compile ml_train
    python examples/run_synthetic_l1_l3.py --duration 3
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from src.benchmarks.synthetic_cpu import SyntheticCpuAdapter
from src.measurements.l1_subspan import L1SubSpanMeasurement
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
    def invoke(self, instruction: str, **kwargs: Any) -> Any:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"{_TMP}/agentsysperf_scratch/synthetic_l1_l3"),
    )
    parser.add_argument("--target", default="self")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    adapter = SyntheticCpuAdapter()
    invoker = _NoopInvoker()

    ctx = RunContext(
        measurements=[
            L1SubSpanMeasurement(),
            L3PerfMeasurement(target=args.target),
        ],
        output_dir=args.output_dir,
    )

    with ctx:
        time.sleep(0.3)

        print(f"\nRun {ctx.run_id} — L1+L3 composed, {args.duration}s/task")
        print(
            f"{'Task':<12} {'Duration':>10} {'CPU-s':>7} "
            f"{'CPU%':>6} {'RSS-MB':>7} {'IPC':>6} {'Cache%':>7}"
        )
        print("-" * 68)

        selected = list(adapter.list_tasks(include=args.tasks))
        if not selected:
            print(f"No matching tasks for include={args.tasks}")
            return

        for spec in selected:
            spec = TaskSpec(
                **{**spec.__dict__, "extra": {**spec.extra, "duration_s": args.duration}},
            )
            span_id = f"{ctx.run_id}::{spec.id}"

            with track_span(ctx, span_id, kind="synthetic_cpu", node_id=spec.id):
                adapter.run_task(spec, agent_invoker=invoker)

            l1 = next((r for r in ctx.records if r.span_id == span_id and r.layer == "l1"), None)
            l3 = next((r for r in ctx.records if r.span_id == span_id and r.layer == "l3"), None)

            dur_ms = l1.payload["duration_us"] / 1000 if l1 else 0
            cpu_s = l1.payload.get("cpu_time_s", 0) if l1 else 0
            cpu_pct = l1.payload.get("cpu_pct_mean", 0) if l1 else 0
            rss_mb = l1.payload.get("rss_kb_peak", 0) / 1024 if l1 else 0
            ipc = l3.payload.get("ipc", float("nan")) if l3 else float("nan")
            cache = l3.payload.get("cache_miss_pct", float("nan")) if l3 else float("nan")

            print(
                f"{spec.id:<12} {dur_ms:>8.0f}ms {cpu_s:>6.2f}s "
                f"{cpu_pct:>5.1f}% {rss_mb:>6.0f}M {ipc:>6.2f} {cache:>6.2f}%"
            )

    adapter.teardown()

    l1_count = sum(1 for r in ctx.records if r.layer == "l1")
    l3_count = sum(1 for r in ctx.records if r.layer == "l3")
    print(f"\n{l1_count} L1 + {l3_count} L3 records across {len(selected)} tasks.")


if __name__ == "__main__":
    main()
