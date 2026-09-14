#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Smoke test: one 6-phase agent loop with L1+L3, per-phase breakdown.

Validates that the six phase spans each emit L1 (and L3 where available)
records before any concurrency sweep or cloud run::

    python examples/run_six_phase_smoke.py
    python examples/run_six_phase_smoke.py --scale 2 --loops 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.benchmarks.six_phase_agent import SixPhaseAgentAdapter
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

PHASE_ORDER = ["reason", "retrieve", "act", "admit", "context", "commit"]


class _NoopInvoker:
    def invoke(self, instruction: str, **kwargs: Any) -> Any:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--target", default="self")
    p.add_argument("--output-dir", type=Path,
                   default=Path(f"{_TMP}/agentsysperf_scratch/six_phase_smoke"))
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    ctx = RunContext(
        measurements=[L1SubSpanMeasurement(), L3PerfMeasurement(target=args.target)],
        output_dir=args.output_dir,
    )
    adapter = SixPhaseAgentAdapter(ctx, n_loops=args.loops)
    invoker = _NoopInvoker()

    with ctx:
        print(f"\nRun {ctx.run_id} — 6-phase loop, scale={args.scale}, loops={args.loops}")
        print(f"{'Phase':<10} {'Wall':>9} {'CPU-s':>7} {'CPU%':>6} {'RSS-MB':>7} {'IPC':>6} {'Cache%':>7}")
        print("-" * 62)

        specs = list(adapter.list_tasks())
        for spec in specs:
            spec = TaskSpec(**{**spec.__dict__, "extra": {**spec.extra, "scale": args.scale}})
            outer = f"{ctx.run_id}::{spec.id}"
            with track_span(ctx, outer, kind="loop", node_id=spec.id):
                result = adapter.run_task(spec, agent_invoker=invoker)

            for phase in PHASE_ORDER:
                span_id = f"{spec.id}::{phase}"
                l1 = next((r for r in ctx.records
                           if r.span_id == span_id and r.layer == "l1"), None)
                l3 = next((r for r in ctx.records
                           if r.span_id == span_id and r.layer == "l3"), None)
                dur_ms = l1.payload["duration_us"] / 1000 if l1 else 0
                cpu_s = l1.payload.get("cpu_time_s", 0) if l1 else 0
                cpu_pct = l1.payload.get("cpu_pct_mean", 0) if l1 else 0
                rss_mb = l1.payload.get("rss_kb_peak", 0) / 1024 if l1 else 0
                ipc = l3.payload.get("ipc", float("nan")) if l3 else float("nan")
                cache = l3.payload.get("cache_miss_pct", float("nan")) if l3 else float("nan")
                print(f"{phase:<10} {dur_ms:>7.1f}ms {cpu_s:>6.2f}s {cpu_pct:>5.1f}% "
                      f"{rss_mb:>6.0f}M {ipc:>6.2f} {cache:>6.2f}%")
            print(f"{'loop total':<10} {result.extra['loop_ms']:>7.1f}ms  passed={result.passed}")
            print("-" * 62)

    adapter.teardown()
    l1_n = sum(1 for r in ctx.records if r.layer == "l1")
    l3_n = sum(1 for r in ctx.records if r.layer == "l3")
    print(f"\n{l1_n} L1 + {l3_n} L3 records "
          f"(expect {6*args.loops} phase spans + {args.loops} loop spans per layer)")


if __name__ == "__main__":
    main()
