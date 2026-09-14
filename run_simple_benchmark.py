#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Simple benchmark run with L1 measurements only (no perf counters needed)."""

from pathlib import Path
import time
from typing import Any

from src.benchmarks.synthetic_cpu import SyntheticCpuAdapter
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.platform import detect_platform
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
    adapter = SyntheticCpuAdapter()
    invoker = _NoopInvoker()

    output_dir = Path(f"{_TMP}/agentsysperf_results")
    plat = detect_platform()

    ctx = RunContext(
        measurements=[L1SubSpanMeasurement()],
        output_dir=output_dir,
    )

    with ctx:
        time.sleep(0.3)

        print(f"\n=== AgentSysPerf Benchmark Run {ctx.run_id} ===")
        print(f"System: {plat.microarchitecture} ({plat.physical_cores} cores)")
        print(f"Output: {output_dir}\n")
        print(f"{'Task':<15} {'Duration':>10} {'CPU Time':>10} {'CPU%':>7} {'Peak RSS':>10} {'Threads':>8}")
        print("-" * 80)

        # Run all available synthetic tasks
        for spec in adapter.list_tasks():
            # Run each task for 2 seconds
            spec = TaskSpec(
                **{**spec.__dict__, "extra": {**spec.extra, "duration_s": 2.0}},
            )
            span_id = f"{ctx.run_id}::{spec.id}"

            with track_span(ctx, span_id, kind="synthetic_cpu", node_id=spec.id):
                adapter.run_task(spec, agent_invoker=invoker)

            # Extract L1 metrics
            l1 = next((r for r in ctx.records if r.span_id == span_id and r.layer == "l1"), None)

            if l1:
                dur_ms = l1.payload["duration_us"] / 1000
                cpu_s = l1.payload.get("cpu_time_s", 0)
                cpu_pct = l1.payload.get("cpu_pct_mean", 0)
                rss_mb = l1.payload.get("rss_kb_peak", 0) / 1024
                threads = l1.payload.get("thread_count_peak", 0)

                print(
                    f"{spec.id:<15} {dur_ms:>8.0f}ms {cpu_s:>8.2f}s "
                    f"{cpu_pct:>6.1f}% {rss_mb:>8.0f}MB {threads:>8}"
                )

    adapter.teardown()

    l1_count = sum(1 for r in ctx.records if r.layer == "l1")
    print(f"\n=== Completed: {l1_count} measurements collected ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
