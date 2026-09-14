#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""MemoryLeakAnalyzer — detects memory leaks via RSS growth analysis.

Uses linear regression on RSS time-series to detect monotonic growth
patterns that indicate leaks, vs stable patterns (healthy) or sawtooth
patterns (GC pressure).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)


class MemoryLeakAnalyzer:
    """Detects memory leaks from L1 RSS time-series.

    Classification:
    - Monotonic growth (slope > threshold, high r²) → leak_suspected
    - Stable (low slope, high r²) → stable
    - Sawtooth (low r², varying slope) → gc_pressure

    Thresholds:
    - Leak: slope > 1 MB/min AND r² > 0.7
    - Stable: slope < 0.5 MB/min
    - GC pressure: r² < 0.5 (high variance)
    """

    name: str = "memory_leak"
    input_layers = frozenset(["l1"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Analyze memory growth across all L1 records in the run."""
        # Extract RSS time-series from L1 records
        rss_series = []
        for rec in records:
            if rec.layer != "l1":
                continue
            rss_kb = rec.payload.get("rss_kb_peak")
            duration_us = rec.payload.get("duration_us")
            if rss_kb is not None and duration_us is not None:
                # Approximate cumulative time (this is per-span, so rough)
                rss_series.append((duration_us / 1_000_000.0, rss_kb / 1024.0))  # (time_s, rss_mb)

        if len(rss_series) < 3:
            # Need at least 3 points for meaningful regression
            logger.debug(
                "MemoryLeakAnalyzer: only %d L1 records, skipping (need >= 3)",
                len(rss_series),
            )
            return

        # Fit linear regression
        slope, r_squared = self._fit_linear(rss_series)

        # Classify
        if slope > 1.0 and r_squared > 0.7:
            verdict = "leak_suspected"
            confidence = min(r_squared, 0.95)
            recommendations = [
                f"Memory growth detected: {slope:.1f} MB/min",
                "Profile with py-spy or valgrind to identify leak source",
                "Check for unbounded cache growth in agent framework",
            ]
        elif abs(slope) < 0.5:
            verdict = "stable"
            confidence = 0.90
            recommendations = [
                "RSS is stable after warmup — no leak detected",
            ]
        elif r_squared < 0.5:
            verdict = "gc_pressure"
            confidence = 0.75
            recommendations = [
                "High RSS variance suggests GC pressure",
                "Consider tuning GC parameters or reducing allocation rate",
            ]
        else:
            verdict = "growing"
            confidence = 0.70
            recommendations = [
                f"RSS growing at {slope:.1f} MB/min",
                "Monitor over longer run to confirm leak vs warmup",
            ]

        evidence = {
            "rss_growth_mb_per_min": float(slope),
            "r_squared": float(r_squared),
            "sample_count": len(rss_series),
            "initial_rss_mb": rss_series[0][1],
            "final_rss_mb": rss_series[-1][1],
        }

        yield AnalysisResult(
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            span_id=None,  # Run-wide analysis, not per-span
            analyzer_name=self.name,
        )

    def _fit_linear(self, series: list[tuple[float, float]]) -> tuple[float, float]:
        """Fit y = mx + b to time-series, return (slope, r²).

        Simple least-squares linear regression.
        """
        n = len(series)
        if n < 2:
            return 0.0, 0.0

        x = [t for t, _ in series]
        y = [rss for _, rss in series]

        # Normalize time to minutes for interpretable slope
        x_min = [(t - x[0]) / 60.0 for t in x]

        mean_x = sum(x_min) / n
        mean_y = sum(y) / n

        ss_xx = sum((xi - mean_x) ** 2 for xi in x_min)
        ss_yy = sum((yi - mean_y) ** 2 for yi in y)
        ss_xy = sum((x_min[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        if ss_xx == 0:
            return 0.0, 0.0

        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x

        # Compute r²
        y_pred = [slope * xi + intercept for xi in x_min]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_yy) if ss_yy > 0 else 0.0

        return slope, r_squared


__all__ = ["MemoryLeakAnalyzer"]
