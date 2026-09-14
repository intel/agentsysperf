#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Demonstrate PowerPoint report generation from SQLite benchmark results.

This example shows how to:
1. Generate a PowerPoint report from stored benchmark results
2. Query data from SQLiteResultStore
3. Produce publication-quality slides for stakeholders

Prerequisites:
    - Run a benchmark with Terminal-Bench adapter (or use demo_sqlite_store.py to create test data)
    - Ensure analyzers have been run (breakdown, cpu_bound, cache)

Usage:
    # Generate report from existing run
    python examples/demo_pptx_report.py --run-id run_20260528 --output-dir /tmp/reports

    # Or with test data
    python examples/demo_sqlite_store.py  # Create test data first
    python examples/demo_pptx_report.py --run-id demo_run_* --output-dir /tmp/reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.reporting import XeonPowerPointGenerator
from src.storage import SQLiteResultStore
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PowerPoint report from AgentSysPerf benchmark results"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID to generate report for (from SQLite database)",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path(f"{_TMP}/agentsysperf_scratch/sqlite_demo"),
        help="Path to directory containing agentsysperf_results.db (default: /tmp/agentsysperf_scratch/sqlite_demo)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"{_TMP}/agentsysperf_reports"),
        help="Directory to save generated PowerPoint report (default: /tmp/agentsysperf_reports)",
    )

    args = parser.parse_args()

    # Validate store path
    db_path = args.store_path / "agentsysperf_results.db"
    if not db_path.exists():
        print(f"Error: SQLite database not found at {db_path}")
        print("Run examples/demo_sqlite_store.py first to create test data.")
        sys.exit(1)

    # Initialize store
    print(f"Loading results from: {db_path}")
    store = SQLiteResultStore(output_dir=args.store_path)

    # Check if run exists
    tasks = store.query_tasks(args.run_id)
    if not tasks:
        print(f"Error: No tasks found for run_id '{args.run_id}'")
        print("\nAvailable runs:")
        # Try to list runs (query all tasks and extract unique run_ids)
        # Note: SQLiteResultStore doesn't have a list_runs() method, so we can't list here
        print("  (Cannot list runs - query the database directly or check your run_id)")
        sys.exit(1)

    print(f"Found {len(tasks)} tasks for run '{args.run_id}'")

    # Generate report
    generator = XeonPowerPointGenerator()
    output_path = args.output_dir / f"{args.run_id}_report.pptx"

    print(f"\nGenerating PowerPoint report...")
    result_path = generator.generate_report(
        run_id=args.run_id,
        store=store,
        output_path=output_path,
    )

    print(f"\n✓ PowerPoint report generated successfully!")
    print(f"  Location: {result_path}")
    print(f"  Size: {result_path.stat().st_size / 1024:.1f} KB")

    # Print summary
    verdicts = store.query_verdicts(args.run_id)
    print(f"\nReport includes:")
    print(f"  • {len(tasks)} task results")
    print(f"  • {len(verdicts)} analyzer verdicts")

    # Count slides (estimate)
    num_slides = 4  # Title, breakdown, CPU, cache
    num_slides += (len(tasks) + 4) // 5  # Task tables (5 per slide)
    num_slides += 1  # Recommendations
    print(f"  • ~{num_slides} slides")

    store.close()


if __name__ == "__main__":
    main()
