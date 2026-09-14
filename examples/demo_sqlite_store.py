#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Demonstrate SQLiteResultStore for persisting benchmark results.

This example shows how to:
1. Create a SQLite result store
2. Store run metadata and task results
3. Store analyzer verdicts
4. Query stored data for reporting

Usage:
    python examples/demo_sqlite_store.py
"""

from __future__ import annotations

import time
from pathlib import Path

from src.protocols import AnalysisResult
from src.storage import SQLiteResultStore
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def main() -> None:
    output_dir = Path(f"{_TMP}/agentsysperf_scratch/sqlite_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the store
    store = SQLiteResultStore(output_dir=output_dir)
    print(f"SQLite store created at: {store.db_path}")

    # Store run metadata
    run_id = f"demo_run_{int(time.time())}"
    store.store_run_metadata(
        run_id=run_id,
        metadata={
            "start_time": int(time.time()),
            "hardware_sku": "Intel Xeon Platinum 8592+",
            "model": "claude-3-opus-20240229",
            "total_tasks": 3,
            "passed_tasks": 2,
            "optimization_profile": "base",
        },
    )
    print(f"\n✓ Stored run metadata for {run_id}")

    # Store task results
    tasks = [
        {
            "task_id": "task_001",
            "workload_type": "linalg",
            "passed": True,
            "duration_s": 12.5,
            "num_turns": 3,
            "num_commands": 8,
        },
        {
            "task_id": "task_002",
            "workload_type": "compile",
            "passed": True,
            "duration_s": 18.3,
            "num_turns": 5,
            "num_commands": 12,
        },
        {
            "task_id": "task_003",
            "workload_type": "ml_train",
            "passed": False,
            "duration_s": 45.1,
            "num_turns": 8,
            "num_commands": 15,
        },
    ]

    for task in tasks:
        store.store_task_result(run_id=run_id, task_id=task["task_id"], result=task)
    print(f"✓ Stored {len(tasks)} task results")

    # Store analyzer verdicts
    analysis_results = [
        AnalysisResult(
            verdict="cpu-bound",
            confidence=0.85,
            evidence={"ipc": 2.3, "cache_miss_rate": 0.02, "cpu_pct": 95.0},
            recommendations=[
                "Consider AVX-512 optimizations",
                "Profile for vectorization opportunities",
            ],
            span_id="task_001",
            analyzer_name="cpu_bound",
        ),
        AnalysisResult(
            verdict="memory-bound",
            confidence=0.72,
            evidence={"ipc": 0.8, "cache_miss_rate": 0.15, "llc_miss_rate": 0.08},
            recommendations=[
                "Optimize data layout for cache locality",
                "Consider hugepages for large allocations",
            ],
            span_id="task_002",
            analyzer_name="cpu_bound",
        ),
        AnalysisResult(
            verdict="cache-resident",
            confidence=0.91,
            evidence={
                "l1d_miss_rate": 0.01,
                "l2_miss_rate": 0.003,
                "llc_miss_rate": 0.001,
            },
            recommendations=["Working set fits in L2 cache"],
            span_id="task_001",
            analyzer_name="cache",
        ),
    ]

    store.store_analysis_results(run_id=run_id, results=analysis_results)
    print(f"✓ Stored {len(analysis_results)} analyzer verdicts")

    # Query tasks
    print("\n=== Task Query Results ===")
    all_tasks = store.query_tasks(run_id)
    print(f"Total tasks: {len(all_tasks)}")
    for task in all_tasks:
        status = "PASS" if task["passed"] else "FAIL"
        print(
            f"  {task['task_id']:<12} {task['workload_type']:<12} "
            f"{status:<6} {task['duration_s']:>6.1f}s"
        )

    # Query by workload type
    linalg_tasks = store.query_tasks(run_id, workload_type="linalg")
    print(f"\nLinalg tasks: {len(linalg_tasks)}")

    # Query verdicts
    print("\n=== Analyzer Verdicts ===")
    all_verdicts = store.query_verdicts(run_id)
    print(f"Total verdicts: {len(all_verdicts)}")
    for verdict in all_verdicts:
        print(
            f"  {verdict['analyzer_name']:<12} {verdict['task_id']:<12} "
            f"{verdict['verdict']:<15} confidence={verdict['confidence']:.2f}"
        )
        print(f"    Evidence: {verdict['evidence']}")
        if verdict["recommendations"]:
            print(f"    Recommendations: {verdict['recommendations']}")

    # Query by analyzer
    cpu_bound_verdicts = store.query_verdicts(run_id, analyzer_name="cpu_bound")
    print(f"\nCPU-bound verdicts: {len(cpu_bound_verdicts)}")

    store.close()
    print(f"\n✓ Store closed. Database persisted at: {store.db_path}")


if __name__ == "__main__":
    main()
