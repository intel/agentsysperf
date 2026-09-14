#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for CPUBoundAnalyzer classification boundaries.

Pins the calibrated decision-tree thresholds (calibration/README.md) so a
transcription error like branch_miss_pct 0.11-vs-11 can't silently recur.
KEY: branch_miss_pct and cache_miss_pct are PERCENTAGES (0-100, from the L3
probe's 100*misses/total); cpu_utilization is cpu_time_s/duration_s, which is
cores' worth of CPU and is NOT capped at 1.0 — cpu_time_s is process-wide, so a
multi-threaded span legitimately exceeds its own wall duration.

These are unit tests on the thresholds. For the end-to-end property — that the
cpu_time_s a real run feeds in is scoped to match cpu_pct_mean — see
src/test_run_integration.py.
"""

from __future__ import annotations

from src.analyzers.cpu_bound import CPUBoundAnalyzer
from src.protocols import MeasurementRecord as MR


def _verdict(*, ipc, branch_miss_pct, cache_miss_pct, llc_miss_per_s,
             cpu_time_s, duration_s):
    """Run the analyzer on a single span built from the given features."""
    recs = [
        MR(span_id="s", layer="l3", payload={
            "ipc": ipc,
            "branch_miss_pct": branch_miss_pct,
            "cache_miss_pct": cache_miss_pct,
            "llc_miss_per_s": llc_miss_per_s,
            "duration_s": duration_s,
        }),
        MR(span_id="s", layer="l1", payload={"cpu_time_s": cpu_time_s}),
    ]
    out = list(CPUBoundAnalyzer().analyze(recs))
    return out[0].verdict if out else None


def test_memory_bound():
    # High LLC miss/s AND high cache-miss% → memory_bound (checked first).
    assert _verdict(ipc=1.5, branch_miss_pct=0.5, cache_miss_pct=30.0,
                    llc_miss_per_s=15_000_000, cpu_time_s=5, duration_s=10) == "memory_bound"


def test_core_bound():
    # High CPU utilization (>=0.94 ratio), not memory-bound → core_bound.
    assert _verdict(ipc=3.0, branch_miss_pct=0.2, cache_miss_pct=2.0,
                    llc_miss_per_s=100_000, cpu_time_s=9.6, duration_s=10) == "core_bound"


def test_utilization_above_one_is_not_an_error():
    # cpu_time_s is process-wide, so 4 busy threads over a 10s span is 40 CPU
    # seconds -> 4.0. The gate is >=0.94 ("at least ~one core saturated"), not a
    # 0-1 range check, and nothing downstream may clamp or reject this.
    assert _verdict(ipc=3.0, branch_miss_pct=0.2, cache_miss_pct=2.0,
                    llc_miss_per_s=100_000, cpu_time_s=40, duration_s=10) == "core_bound"


def test_io_bound_typical_low_branch_miss():
    # Low CPU util + realistic sub-1% branch miss → io_bound.
    # REGRESSION GUARD: with the old 0.11 bug, 0.5 > 0.11 wrongly gave
    # frontend_starved. Real agentic workloads sit here.
    assert _verdict(ipc=1.4, branch_miss_pct=0.5, cache_miss_pct=5.0,
                    llc_miss_per_s=200_000, cpu_time_s=1, duration_s=10) == "io_bound"


def test_frontend_starved_needs_high_branch_miss():
    # Only a genuinely branch-heavy workload (>11%) is frontend_starved.
    assert _verdict(ipc=0.8, branch_miss_pct=15.0, cache_miss_pct=3.0,
                    llc_miss_per_s=200_000, cpu_time_s=1, duration_s=10) == "frontend_starved"


def test_branch_miss_unit_boundary():
    # The threshold is 11 (percent), not 0.11. A 5% miss rate must NOT trip it.
    assert _verdict(ipc=1.0, branch_miss_pct=5.0, cache_miss_pct=3.0,
                    llc_miss_per_s=200_000, cpu_time_s=1, duration_s=10) == "io_bound"
    # 12% does.
    assert _verdict(ipc=1.0, branch_miss_pct=12.0, cache_miss_pct=3.0,
                    llc_miss_per_s=200_000, cpu_time_s=1, duration_s=10) == "frontend_starved"
