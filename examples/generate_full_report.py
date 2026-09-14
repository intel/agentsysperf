#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Generate comprehensive PowerPoint report using XeonPowerPointGenerator.

Usage:
    poetry run python examples/generate_full_report.py production_run \
      --db-path /tmp/agentsysperf_tb2_production/agentsysperf_results.db \
      --output tb2_production_report.pptx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.reporting.xeon_pptx import XeonPowerPointGenerator
from src.storage.sqlite_store import SQLiteResultStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run identifier from benchmark")
    parser.add_argument("--db-path", required=True, help="Path to agentsysperf_results.db")
    parser.add_argument("--output", default="tb2_report.pptx", help="Output PowerPoint file")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        return 1

    print(f"=== Generating Comprehensive Report ===")
    print(f"Run ID: {args.run_id}")
    print(f"Database: {db_path}")
    print(f"Output: {args.output}\n")

    # Initialize store (db_path is file, not directory)
    db_dir = db_path.parent
    store = SQLiteResultStore(output_dir=db_dir)

    # Verify run exists
    tasks = store.query_tasks(args.run_id)
    if not tasks:
        print(f"ERROR: No tasks found for run_id '{args.run_id}'", file=sys.stderr)
        print("Available data:", file=sys.stderr)
        # Try to list all tasks
        all_tasks = store.query_tasks("")  # Empty run_id might show all
        if all_tasks:
            print(f"  Found {len(all_tasks)} tasks in database", file=sys.stderr)
        return 1

    print(f"Found {len(tasks)} tasks in run '{args.run_id}'")

    # Generate report
    generator = XeonPowerPointGenerator()
    output_path = Path(args.output)

    try:
        result_path = generator.generate_report(
            run_id=args.run_id,
            store=store,
            output_path=output_path,
        )
        print(f"\n✓ Report generated: {result_path}")
        return 0
    except Exception as e:
        print(f"\nERROR: Report generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
