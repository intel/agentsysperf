#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Complete workflow: synthetic data → SQLite storage → PowerPoint report.

This example demonstrates the full reporting pipeline:
1. Create synthetic benchmark results (tasks + analyzer verdicts)
2. Store results in SQLiteResultStore
3. Generate PowerPoint report with XeonPowerPointGenerator
4. Show report summary

This is useful for:
- Testing the reporting pipeline without running actual benchmarks
- Demonstrating report capabilities to stakeholders
- Validating report generator updates

Usage:
    python examples/demo_full_reporting_workflow.py
"""

from __future__ import annotations

import time
from pathlib import Path

from src.protocols import AnalysisResult
from src.reporting import XeonPowerPointGenerator
from src.storage import SQLiteResultStore
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def create_synthetic_data() -> tuple[str, list[dict], list[AnalysisResult]]:
    """Create synthetic benchmark data for demonstration.

    Returns:
        (run_id, tasks, analysis_results) tuple
    """
    run_id = f"demo_run_{int(time.time())}"

    # Synthetic tasks covering multiple workload types
    tasks = [
        {
            "task_id": "task_001_linalg",
            "workload_type": "linalg",
            "passed": True,
            "duration_s": 12.5,
            "num_turns": 3,
            "num_commands": 8,
        },
        {
            "task_id": "task_002_linalg",
            "workload_type": "linalg",
            "passed": True,
            "duration_s": 10.2,
            "num_turns": 2,
            "num_commands": 6,
        },
        {
            "task_id": "task_003_compile",
            "workload_type": "compile",
            "passed": True,
            "duration_s": 18.3,
            "num_turns": 5,
            "num_commands": 12,
        },
        {
            "task_id": "task_004_compile",
            "workload_type": "compile",
            "passed": False,
            "duration_s": 45.1,
            "num_turns": 8,
            "num_commands": 15,
        },
        {
            "task_id": "task_005_ml_train",
            "workload_type": "ml_train",
            "passed": True,
            "duration_s": 32.7,
            "num_turns": 6,
            "num_commands": 10,
        },
        {
            "task_id": "task_006_ml_train",
            "workload_type": "ml_train",
            "passed": True,
            "duration_s": 28.4,
            "num_turns": 5,
            "num_commands": 9,
        },
        {
            "task_id": "task_007_web_scrape",
            "workload_type": "web_scrape",
            "passed": True,
            "duration_s": 8.1,
            "num_turns": 4,
            "num_commands": 7,
        },
        {
            "task_id": "task_008_web_scrape",
            "workload_type": "web_scrape",
            "passed": False,
            "duration_s": 15.3,
            "num_turns": 6,
            "num_commands": 11,
        },
    ]

    # Synthetic analyzer verdicts
    analysis_results = []

    # Breakdown verdicts (inference/execution/orchestration)
    breakdown_data = [
        ("task_001_linalg", "inference_dominant", 0.90, 65.0, 25.0, 10.0),
        ("task_002_linalg", "inference_heavy", 0.85, 58.0, 30.0, 12.0),
        ("task_003_compile", "execution_dominant", 0.92, 20.0, 68.0, 12.0),
        ("task_004_compile", "execution_dominant", 0.88, 15.0, 75.0, 10.0),
        ("task_005_ml_train", "inference_dominant", 0.91, 62.0, 28.0, 10.0),
        ("task_006_ml_train", "inference_heavy", 0.87, 55.0, 35.0, 10.0),
        ("task_007_web_scrape", "orchestration_heavy", 0.75, 25.0, 35.0, 40.0),
        ("task_008_web_scrape", "execution_heavy", 0.80, 30.0, 50.0, 20.0),
    ]

    for task_id, verdict, confidence, inf_pct, exec_pct, orch_pct in breakdown_data:
        analysis_results.append(
            AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "inference_pct": inf_pct,
                    "execution_pct": exec_pct,
                    "orchestration_pct": orch_pct,
                    "task_duration_s": next(t["duration_s"] for t in tasks if t["task_id"] == task_id),
                    "inference_s": next(t["duration_s"] for t in tasks if t["task_id"] == task_id) * inf_pct / 100,
                    "execution_s": next(t["duration_s"] for t in tasks if t["task_id"] == task_id) * exec_pct / 100,
                    "orchestration_s": next(t["duration_s"] for t in tasks if t["task_id"] == task_id) * orch_pct / 100,
                },
                recommendations=[
                    f"{verdict.replace('_', ' ').title()} — optimize accordingly"
                ],
                span_id=task_id,
                analyzer_name="breakdown",
            )
        )

    # CPU bottleneck verdicts
    cpu_data = [
        ("task_001_linalg", "io_bound", 0.85, 1.8, 0.65, 120000, 2.5),
        ("task_002_linalg", "io_bound", 0.87, 1.9, 0.68, 110000, 2.3),
        ("task_003_compile", "core_bound", 0.92, 2.5, 0.96, 85000, 1.8),
        ("task_004_compile", "memory_bound", 0.88, 0.9, 0.92, 15000000, 18.5),
        ("task_005_ml_train", "io_bound", 0.90, 1.7, 0.62, 200000, 3.2),
        ("task_006_ml_train", "core_bound", 0.86, 2.3, 0.95, 95000, 2.1),
        ("task_007_web_scrape", "io_bound", 0.91, 1.5, 0.45, 80000, 1.5),
        ("task_008_web_scrape", "frontend_starved", 0.78, 1.2, 0.88, 150000, 15.2),
    ]

    for task_id, verdict, confidence, ipc, cpu_util, llc_miss, cache_miss_pct in cpu_data:
        analysis_results.append(
            AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "ipc": ipc,
                    "cpu_utilization": cpu_util,
                    "llc_miss_per_s": llc_miss,
                    "cache_miss_pct": cache_miss_pct,
                    "duration_s": next(t["duration_s"] for t in tasks if t["task_id"] == task_id),
                },
                recommendations=[
                    f"{verdict.replace('_', ' ').title()} workload — see CPU optimization guide"
                ],
                span_id=task_id,
                analyzer_name="cpu_bound",
            )
        )

    # Cache verdicts
    cache_data = [
        ("task_001_linalg", "l3_resident", 0.92, 2.5, 120000),
        ("task_002_linalg", "l3_resident", 0.91, 2.3, 110000),
        ("task_003_compile", "l3_resident", 0.89, 1.8, 85000),
        ("task_004_compile", "dram_bound", 0.85, 18.5, 15000000),
        ("task_005_ml_train", "l3_pressure", 0.82, 3.2, 200000),
        ("task_006_ml_train", "l3_resident", 0.88, 2.1, 95000),
        ("task_007_web_scrape", "l3_resident", 0.93, 1.5, 80000),
        ("task_008_web_scrape", "l3_pressure", 0.80, 15.2, 150000),
    ]

    for task_id, verdict, confidence, cache_miss_pct, llc_miss in cache_data:
        analysis_results.append(
            AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "cache_miss_pct": cache_miss_pct,
                    "llc_miss_per_s": llc_miss,
                    "duration_s": next(t["duration_s"] for t in tasks if t["task_id"] == task_id),
                },
                recommendations=[
                    f"{verdict.replace('_', ' ').title()} — cache sizing appropriate"
                ],
                span_id=task_id,
                analyzer_name="cache",
            )
        )

    return run_id, tasks, analysis_results


def main() -> None:
    print("AgentSysPerf Full Reporting Workflow Demo")
    print("=" * 60)

    # Step 1: Create synthetic data
    print("\n[1/4] Creating synthetic benchmark data...")
    run_id, tasks, analysis_results = create_synthetic_data()
    print(f"  ✓ Generated {len(tasks)} tasks")
    print(f"  ✓ Generated {len(analysis_results)} analyzer verdicts")
    print(f"  ✓ Run ID: {run_id}")

    # Step 2: Store in SQLite
    print("\n[2/4] Storing results in SQLite...")
    output_dir = Path(f"{_TMP}/agentsysperf_scratch/reporting_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteResultStore(output_dir=output_dir)
    print(f"  ✓ SQLite store created at: {store.db_path}")

    # Store run metadata
    store.store_run_metadata(
        run_id=run_id,
        metadata={
            "start_time": int(time.time()),
            "end_time": int(time.time()) + 300,
            "hardware_sku": "Intel Xeon Platinum 8592+ (EMR)",
            "model": "claude-3-sonnet-20240229",
            "total_tasks": len(tasks),
            "passed_tasks": sum(1 for t in tasks if t["passed"]),
            "optimization_profile": "base",
        },
    )
    print("  ✓ Stored run metadata")

    # Store tasks
    for task in tasks:
        store.store_task_result(run_id=run_id, task_id=task["task_id"], result=task)
    print(f"  ✓ Stored {len(tasks)} task results")

    # Store analyzer verdicts
    store.store_analysis_results(run_id=run_id, results=analysis_results)
    print(f"  ✓ Stored {len(analysis_results)} analyzer verdicts")

    # Step 3: Generate PowerPoint report
    print("\n[3/4] Generating PowerPoint report...")
    generator = XeonPowerPointGenerator()
    report_path = output_dir / f"{run_id}_report.pptx"

    result_path = generator.generate_report(
        run_id=run_id,
        store=store,
        output_path=report_path,
    )

    print(f"  ✓ PowerPoint report generated")
    print(f"  ✓ Location: {result_path}")
    print(f"  ✓ Size: {result_path.stat().st_size / 1024:.1f} KB")

    # Step 4: Display summary
    print("\n[4/4] Report Summary")
    print("=" * 60)

    # Workload distribution
    workload_counts = {}
    for task in tasks:
        wtype = task["workload_type"]
        workload_counts[wtype] = workload_counts.get(wtype, 0) + 1

    print("\nWorkload Distribution:")
    for wtype, count in sorted(workload_counts.items()):
        print(f"  • {wtype:<15} {count} tasks")

    # Pass rate
    passed = sum(1 for t in tasks if t["passed"])
    pass_rate = (passed / len(tasks)) * 100
    print(f"\nPass Rate: {passed}/{len(tasks)} ({pass_rate:.0f}%)")

    # Dominant verdicts
    breakdown_verdicts = [
        v["verdict"]
        for v in store.query_verdicts(run_id=run_id, analyzer_name="breakdown")
    ]
    cpu_verdicts = [
        v["verdict"]
        for v in store.query_verdicts(run_id=run_id, analyzer_name="cpu_bound")
    ]

    print("\nDominant Patterns:")
    print(f"  • Breakdown: {max(set(breakdown_verdicts), key=breakdown_verdicts.count)}")
    print(f"  • CPU Bottleneck: {max(set(cpu_verdicts), key=cpu_verdicts.count)}")

    # Estimate slide count
    num_slides = 4  # Title, breakdown, CPU, cache
    num_slides += (len(tasks) + 4) // 5  # Task tables (5 per slide)
    num_slides += 1  # Recommendations
    print(f"\nSlides Generated: ~{num_slides}")

    print("\n" + "=" * 60)
    print("✓ Full workflow completed successfully!")
    print(f"\nOpen the report: {result_path}")

    store.close()


if __name__ == "__main__":
    main()
