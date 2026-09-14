#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""CacheAnalyzer — determines dominant cache tier (L1/L2/L3/DRAM).

Classifies working set behavior based on cache miss rates at each level.
Helps determine whether adding more cores (if L1/L2-resident) or larger
L3 (if L3-resident) will improve performance.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)


class CacheAnalyzer:
    """Determines working set's dominant cache tier.

    Classification thresholds:
    - L3 miss rate < 2% → l3_resident (working set fits in L3)
    - L3 miss rate 2-15% → l3_pressure (some spilling to DRAM)
    - L3 miss rate > 15% → dram_bound (working set exceeds L3)

    Note: L1/L2 miss rates require separate counters not in default
    perf events. If available, refine classification further.
    """

    name: str = "cache"
    input_layers = frozenset(["l3"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Analyze cache behavior for each span with L3 data."""
        for record in records:
            if record.layer != "l3":
                continue

            payload = record.payload
            cache_miss_pct = payload.get("cache_miss_pct")
            llc_miss_per_s = payload.get("llc_miss_per_s")

            if cache_miss_pct is None:
                logger.debug(
                    "CacheAnalyzer: span %s missing cache_miss_pct, skipping",
                    record.span_id,
                )
                continue

            # Estimate working set size (heuristic)
            # Assume 64B cache line, estimate from miss rate and LLC misses
            working_set_mb = None
            if llc_miss_per_s is not None and payload.get("duration_s"):
                total_llc_misses = llc_miss_per_s * payload["duration_s"]
                # Rough estimate: working set = misses * line size / miss rate
                if cache_miss_pct > 0:
                    working_set_bytes = (total_llc_misses * 64) / (cache_miss_pct / 100.0)
                    working_set_mb = working_set_bytes / (1024 * 1024)

            # Classification
            if cache_miss_pct < 2.0:
                verdict = "l3_resident"
                confidence = 0.90
                recommendations = [
                    "Working set fits comfortably in L3 cache",
                    "More cores will scale well (no cache capacity bottleneck)",
                    "Larger L3 not needed for this workload",
                ]
            elif cache_miss_pct < 15.0:
                verdict = "l3_pressure"
                confidence = 0.80
                recommendations = [
                    "Working set partially spills to DRAM",
                    "Consider Xeon SKU with larger L3 if scaling core count",
                    f"Estimated working set: ~{int(working_set_mb)}MB" if working_set_mb else "Profile with VTune to estimate working set size",
                ]
            else:
                verdict = "dram_bound"
                confidence = 0.85
                recommendations = [
                    "Working set exceeds L3 capacity — bound by DRAM bandwidth",
                    "Larger L3 will help, but memory BW is the primary constraint",
                    "Apply NUMA pinning + hugepages before upgrading SKU",
                ]

            evidence = {
                "cache_miss_pct": float(cache_miss_pct),
                "llc_miss_per_s": float(llc_miss_per_s) if llc_miss_per_s is not None else None,
                "working_set_mb": int(working_set_mb) if working_set_mb else None,
                "duration_s": payload.get("duration_s", 0),
            }

            # Context-aware: compare against baseline
            if context and context.baseline_records:
                baseline_miss_pct = self._get_baseline_cache_miss(
                    context.baseline_records, record.span_id
                )
                if baseline_miss_pct is not None:
                    delta = cache_miss_pct - baseline_miss_pct
                    evidence["cache_miss_vs_baseline"] = delta
                    if delta > 5.0:
                        recommendations.insert(
                            0,
                            f"Cache miss rate increased by {delta:.1f}% vs baseline — possible regression",
                        )

            yield AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                recommendations=recommendations,
                span_id=record.span_id,
                analyzer_name=self.name,
            )

    def _get_baseline_cache_miss(
        self, baseline_records: Sequence[MeasurementRecord], span_id: str
    ) -> Optional[float]:
        """Extract cache miss % from baseline for the same span_id."""
        for rec in baseline_records:
            if rec.layer == "l3" and rec.span_id == span_id:
                return rec.payload.get("cache_miss_pct")
        return None


__all__ = ["CacheAnalyzer"]
