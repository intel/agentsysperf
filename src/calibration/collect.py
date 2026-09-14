#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Archetype measurement collector for threshold calibration.

Runs each archetype workload multiple times with L1+L3 measurements,
extracts features, and saves to a CSV dataset for training.

Usage:
    poetry run python -m src.calibration.collect --iterations 5 --output calibration_data.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from pathlib import Path

from src.calibration.archetypes import ARCHETYPES
from src.protocols import discover_measurements
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of times to run each archetype.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV file for calibration dataset.",
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        help="Archetype names to skip (space-separated).",
    )
    args = parser.parse_args()

    print(f"=== Archetype Measurement Collection ===\n")
    print(f"Archetypes: {len(ARCHETYPES)}")
    print(f"Iterations per archetype: {args.iterations}")
    print(f"Output: {args.output}\n")

    # Discover measurements
    measurements = discover_measurements()
    print(f"Measurements: {', '.join(measurements.keys())}\n")

    if "l1_subspan" not in measurements or "l3_perf" not in measurements:
        print("ERROR: L1 and L3 measurements required for calibration.", file=sys.stderr)
        return 1

    # Prepare output CSV
    fieldnames = [
        "archetype_name",
        "iteration",
        "true_bottleneck",
        # L1 features
        "duration_s",
        "cpu_time_s",
        "cpu_pct_mean",
        "cpu_pct_peak",
        "rss_kb_peak",
        "num_threads_peak",
        # L3 features
        "ipc",
        "cache_miss_pct",
        "branch_miss_pct",
        "llc_miss_per_s",
        "context_switches",
        "cpu_migrations",
        "page_faults",
    ]

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for archetype in ARCHETYPES:
            if archetype.name in args.skip:
                print(f"Skipping {archetype.name}")
                continue

            print(f"\n▶ {archetype.name} ({archetype.true_bottleneck})")
            print(f"  {archetype.description}")

            for iteration in range(args.iterations):
                print(f"  Iteration {iteration + 1}/{args.iterations}...", end=" ", flush=True)

                # Run with measurements
                ctx = RunContext(
                    measurements=list(measurements.values()),
                    output_dir=Path(f"{_TMP}/agentsysperf_calibration_{uuid.uuid4().hex[:8]}"),
                )

                span_id = f"{archetype.name}_iter{iteration}"
                try:
                    with ctx:
                        with track_span(ctx, span_id, kind="calibration", node_id=archetype.name):
                            archetype.run_fn()
                except Exception as e:
                    print(f"FAIL ({e})")
                    continue

                # Extract features from records
                l1_record = None
                l3_record = None
                for r in ctx.records:
                    if r.span_id == span_id:
                        if r.layer == "l1":
                            l1_record = r
                        elif r.layer == "l3":
                            l3_record = r

                if l1_record is None or l3_record is None:
                    print("FAIL (no measurements)")
                    continue

                l1 = l1_record.payload
                l3 = l3_record.payload

                row = {
                    "archetype_name": archetype.name,
                    "iteration": iteration,
                    "true_bottleneck": archetype.true_bottleneck,
                    # L1
                    "duration_s": l1.get("duration_us", 0) / 1e6,
                    "cpu_time_s": l1.get("cpu_time_s", 0),
                    "cpu_pct_mean": l1.get("cpu_pct_mean", 0),
                    "cpu_pct_peak": l1.get("cpu_pct_peak", 0),
                    "rss_kb_peak": l1.get("rss_kb_peak", 0),
                    "num_threads_peak": l1.get("num_threads_peak", 0),
                    # L3
                    "ipc": l3.get("ipc", 0),
                    "cache_miss_pct": l3.get("cache_miss_pct", 0),
                    "branch_miss_pct": l3.get("branch_miss_pct", 0),
                    "llc_miss_per_s": l3.get("llc_miss_per_s", 0),
                    "context_switches": l3.get("events", {}).get("context-switches", 0),
                    "cpu_migrations": l3.get("events", {}).get("cpu-migrations", 0),
                    "page_faults": l3.get("events", {}).get("page-faults", 0),
                }

                writer.writerow(row)
                csvfile.flush()
                print(f"OK (IPC={row['ipc']:.2f}, cache_miss={row['cache_miss_pct']:.1f}%)")

    print(f"\n✓ Dataset saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
