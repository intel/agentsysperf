#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""CPUBoundAnalyzer — classifies workload as core-bound, memory-bound, or frontend-starved.

Uses industry-standard Top-Down Microarchitecture Analysis (TMA) methodology
heuristics based on IPC and cache behavior.

References:
- Intel Top-Down Methodology: https://www.intel.com/content/www/us/en/developer/articles/technical/top-down-microarchitecture-analysis.html
- Yasin, "A Top-Down method for performance analysis and counters architecture", ISPASS 2014
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)


class CPUBoundAnalyzer:
    """Classifies CPU bottleneck as core-bound | memory-bound | frontend-starved.

    Consumes L3 perf counters (IPC, cache miss rates, branch miss rates) and L1 CPU
    utilization to classify the dominant bottleneck.

    Classification rules learned from calibration (decision tree; see
    calibration/README.md). Units: cache_miss_pct / branch_miss_pct are
    PERCENTAGES (0-100); cpu_utilization is ``cpu_time_s / duration_s``, which
    is process-wide CPU seconds per wall second — i.e. *cores' worth of CPU*,
    NOT a 0-1 ratio. A single saturated core is 1.0; four busy threads is ~4.0.
    Evaluated in order:
    1. llc_miss_per_s > 10M  AND  cache_miss_pct > 20%  → memory_bound
    2. else cpu_utilization ≥ 0.94                       → core_bound
    3. else branch_miss_pct > 11%                        → frontend_starved
    4. else                                              → io_bound

    NOTE: the 0.94 gate therefore reads as "at least ~one core saturated". It is
    not normalized by core count, so a 4-thread workload on a 288-core host
    scores 4.0 and lands in core_bound identically to a 1-thread one — both are
    "core_bound" while using 1.4% of the machine. Recalibrating against core
    count is a real open question, deliberately not answered here. What is fixed
    here is the *claim*: the docstring said 0-1 ratio, which was never true of a
    multi-threaded workload.
    """

    name: str = "cpu_bound"
    input_layers = frozenset(["l1", "l3"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Analyze CPU bottleneck for each span with L1+L3 data."""
        # Group records by span_id to pair L1+L3
        spans = {}
        for record in records:
            if record.span_id not in spans:
                spans[record.span_id] = {}
            spans[record.span_id][record.layer] = record

        for span_id, layers in spans.items():
            l3_record = layers.get("l3")
            l1_record = layers.get("l1")

            if l3_record is None:
                logger.debug(
                    "CPUBoundAnalyzer: span %s missing L3 data, skipping", span_id
                )
                continue

            l3 = l3_record.payload
            l1 = l1_record.payload if l1_record else {}

            # Extract features
            ipc = l3.get("ipc")
            branch_miss_pct = l3.get("branch_miss_pct")
            llc_miss_per_s = l3.get("llc_miss_per_s")
            cache_miss_pct = l3.get("cache_miss_pct")

            cpu_time_s = l1.get("cpu_time_s", 0)
            duration_s = l3.get("duration_s", 0)
            cpu_utilization = (cpu_time_s / duration_s) if duration_s > 0 else 0

            if ipc is None or branch_miss_pct is None:
                logger.debug(
                    "CPUBoundAnalyzer: span %s missing features, skipping", span_id
                )
                continue

            # Apply learned decision tree rules (from calibration)
            # Check memory-bound first (high LLC miss dominates)
            if llc_miss_per_s is not None and llc_miss_per_s > 10_000_000 and cache_miss_pct is not None and cache_miss_pct > 20.0:
                verdict = "memory_bound"
                confidence = 0.90
                recommendations = [
                    "High LLC miss rate + high cache miss % → memory bandwidth bottleneck",
                    "Enable hugepages (2M) to reduce TLB misses",
                    "Apply NUMA pinning to reduce remote memory access",
                    "Consider Xeon SKU with higher memory bandwidth (DDR5-4800+)",
                ]
            elif cpu_utilization >= 0.94:
                # High CPU utilization without high LLC miss → core-bound
                verdict = "core_bound"
                confidence = 0.92
                recommendations = [
                    "High CPU utilization → execution ports saturated",
                    "Higher core count or frequency will improve throughput",
                    "Consider Xeon P-core SKU for sustained compute workloads",
                ]
            else:
                # Low CPU utilization → I/O-bound or frontend-starved.
                # branch_miss_pct is a PERCENTAGE (0-100, = 100*misses/branches
                # from the L3 probe), so the calibrated 11% threshold is 11.0,
                # not 0.11. (The decision tree in calibration/ trained on the
                # 0-100 scale; see calibration/README.md.)
                if branch_miss_pct > 11.0:  # 11% threshold for frontend starvation
                    verdict = "frontend_starved"
                    confidence = 0.88
                    recommendations = [
                        "High branch misprediction → frontend bottleneck",
                        "Profile with VTune for micro-op queue stalls",
                        "Consider code refactoring to reduce branch mispredictions",
                    ]
                else:
                    verdict = "io_bound"
                    confidence = 0.85
                    recommendations = [
                        "Low CPU utilization → blocked on I/O or network",
                        "Profile network latency (hosted LLM calls, API requests)",
                        "Consider async I/O or batching to hide latency",
                    ]

            evidence = {
                "ipc": float(ipc),
                "branch_miss_pct": float(branch_miss_pct),
                "llc_miss_per_s": float(llc_miss_per_s) if llc_miss_per_s is not None else None,
                "cache_miss_pct": float(cache_miss_pct) if cache_miss_pct is not None else None,
                "cpu_utilization": float(cpu_utilization),
                "duration_s": float(duration_s),
            }

            # Add context-aware insights if baseline available
            if context and context.baseline_records:
                baseline_ipc = self._get_baseline_ipc(context.baseline_records, span_id)
                if baseline_ipc is not None:
                    ipc_delta = ((ipc - baseline_ipc) / baseline_ipc) * 100
                    evidence["ipc_vs_baseline_pct"] = ipc_delta
                    if ipc_delta < -10:
                        recommendations.insert(0, f"IPC is {abs(ipc_delta):.0f}% lower than baseline — investigate regression")

            yield AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                recommendations=recommendations,
                span_id=span_id,
                analyzer_name=self.name,
            )

    def _get_baseline_ipc(self, baseline_records: Sequence[MeasurementRecord], span_id: str) -> Optional[float]:
        """Extract IPC from baseline for the same span_id."""
        for rec in baseline_records:
            if rec.layer == "l3" and rec.span_id == span_id:
                return rec.payload.get("ipc")
        return None


__all__ = ["CPUBoundAnalyzer"]
