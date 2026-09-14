#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Validate calibrated analyzer on archetype workloads.

Runs 3 archetypes (memory_bound, core_bound, io_bound) with L1+L3 measurements,
then invokes the updated CPUBoundAnalyzer to verify correct classification.

Expected verdicts:
- memory_bound_stream → memory_bound
- core_bound_fibonacci → core_bound
- io_bound_sleep → io_bound

Usage:
    poetry run python examples/validate_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.analyzers.cpu_bound import CPUBoundAnalyzer
from src.calibration.archetypes import ARCHETYPES
from src.protocols import AnalysisContext, discover_measurements
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main() -> int:
    print("=== Calibration Validation ===\n")

    # Select 3 representative archetypes
    test_archetypes = [
        next(a for a in ARCHETYPES if a.name == "memory_bound_stream"),
        next(a for a in ARCHETYPES if a.name == "core_bound_fibonacci"),
        next(a for a in ARCHETYPES if a.name == "io_bound_sleep"),
    ]

    measurements = discover_measurements()
    if "l1_subspan" not in measurements or "l3_perf" not in measurements:
        print("ERROR: L1 and L3 measurements required.", file=sys.stderr)
        return 1

    analyzer = CPUBoundAnalyzer()
    results = []

    for archetype in test_archetypes:
        print(f"\nRunning: {archetype.name} (expected: {archetype.true_bottleneck})")

        ctx = RunContext(
            measurements=list(measurements.values()),
            output_dir=Path(f"{_TMP}/agentsysperf_validation_{archetype.name}"),
        )

        span_id = archetype.name
        try:
            with ctx:
                with track_span(ctx, span_id, kind="validation", node_id=archetype.name):
                    archetype.run_fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            continue

        # Run analyzer
        analysis_results = list(analyzer.analyze(ctx.records, context=None))

        if not analysis_results:
            print("  FAIL: No analysis results produced")
            continue

        result = analysis_results[0]
        verdict = result.verdict
        confidence = result.confidence
        evidence = result.evidence

        match = verdict == archetype.true_bottleneck or (
            verdict == "memory_bound_likely" and archetype.true_bottleneck == "memory_bound"
        )

        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  {status}: {verdict} (confidence={confidence:.2f})")
        print(f"    Evidence: IPC={evidence.get('ipc', 0):.2f}, branch_miss={evidence.get('branch_miss_pct', 0):.2f}%, "
              f"llc_miss/s={evidence.get('llc_miss_per_s', 0):.0f}, cpu_util={evidence.get('cpu_utilization', 0):.2f}")

        results.append({
            "archetype": archetype.name,
            "expected": archetype.true_bottleneck,
            "predicted": verdict,
            "match": match,
            "confidence": confidence,
        })

    # Summary
    print("\n=== Summary ===\n")
    passed = sum(r["match"] for r in results)
    total = len(results)
    accuracy = (passed / total) * 100 if total > 0 else 0

    for r in results:
        status = "✓" if r["match"] else "✗"
        print(f"{status} {r['archetype']:30s} expected={r['expected']:20s} predicted={r['predicted']:20s}")

    print(f"\nAccuracy: {passed}/{total} ({accuracy:.0f}%)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
