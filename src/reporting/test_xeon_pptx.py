#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Unit tests for XeonPowerPointGenerator.

Tests the PowerPoint report generator without requiring real benchmark data.
Uses mock ResultStore to verify slide generation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.reporting.xeon_pptx import XeonPowerPointGenerator


class MockResultStore:
    """Mock ResultStore for testing."""

    name: str = "mock"

    def __init__(self, tasks: List[Dict[str, Any]], verdicts: List[Dict[str, Any]]) -> None:
        self.tasks = tasks
        self.verdicts = verdicts

    def query_tasks(self, run_id: str, workload_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock tasks, optionally filtered by workload_type."""
        if workload_type:
            return [t for t in self.tasks if t.get("workload_type") == workload_type]
        return self.tasks

    def query_verdicts(self, run_id: str, analyzer_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock verdicts, optionally filtered by analyzer."""
        if analyzer_name:
            return [v for v in self.verdicts if v.get("analyzer_name") == analyzer_name]
        return self.verdicts


def test_generator_initialization():
    """Test that generator initializes with correct attributes."""
    generator = XeonPowerPointGenerator()
    assert generator.name == "xeon_pptx"
    assert generator.output_formats == frozenset(["pptx"])


def test_generate_report_empty_data(tmp_path: Path):
    """Test report generation with empty data (should not crash)."""
    generator = XeonPowerPointGenerator()
    store = MockResultStore(tasks=[], verdicts=[])
    output_path = tmp_path / "empty_report.pptx"

    result_path = generator.generate_report(
        run_id="test_run_empty",
        store=store,
        output_path=output_path,
    )

    assert result_path.exists()
    assert result_path.suffix == ".pptx"
    assert result_path.stat().st_size > 0  # Non-empty file


def test_generate_report_with_tasks(tmp_path: Path):
    """Test report generation with task data."""
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
            "passed": False,
            "duration_s": 18.3,
            "num_turns": 5,
            "num_commands": 12,
        },
    ]

    verdicts = [
        {
            "task_id": "task_001",
            "analyzer_name": "breakdown",
            "verdict": "inference_dominant",
            "confidence": 0.90,
            "evidence": {
                "inference_pct": 65.0,
                "execution_pct": 25.0,
                "orchestration_pct": 10.0,
            },
            "recommendations": ["Consider faster model for lower latency"],
        },
        {
            "task_id": "task_001",
            "analyzer_name": "cpu_bound",
            "verdict": "io_bound",
            "confidence": 0.85,
            "evidence": {
                "ipc": 1.8,
                "cpu_utilization": 0.65,
                "llc_miss_per_s": 120000,
            },
            "recommendations": ["Profile network latency for hosted LLM calls"],
        },
        {
            "task_id": "task_001",
            "analyzer_name": "cache",
            "verdict": "l3_resident",
            "confidence": 0.92,
            "evidence": {
                "cache_miss_pct": 1.5,
                "llc_miss_per_s": 100000,
            },
            "recommendations": ["Working set fits comfortably in L3 cache"],
        },
    ]

    generator = XeonPowerPointGenerator()
    store = MockResultStore(tasks=tasks, verdicts=verdicts)
    output_path = tmp_path / "report_with_data.pptx"

    result_path = generator.generate_report(
        run_id="test_run_001",
        store=store,
        output_path=output_path,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_categorize_recommendation():
    """Test recommendation categorization logic."""
    generator = XeonPowerPointGenerator()

    # Test various categories
    assert generator._categorize_recommendation("Consider faster model for lower latency") == "Model Selection"
    assert generator._categorize_recommendation("Enable hugepages to reduce TLB misses") == "System Configuration"
    assert generator._categorize_recommendation("Optimize for memory bandwidth") == "Memory Optimization"
    assert generator._categorize_recommendation("Large L3 cache will help") == "Cache Optimization"
    assert generator._categorize_recommendation("Higher core count SKU recommended") == "Xeon SKU Selection"
    assert generator._categorize_recommendation("Consider async execution") == "Execution Efficiency"
    assert generator._categorize_recommendation("Profile with VTune") == "General Optimization"


def test_hex_to_rgb():
    """Test hex color conversion."""
    generator = XeonPowerPointGenerator()

    assert generator._hex_to_rgb("#0071C5") == (0, 113, 197)
    assert generator._hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert generator._hex_to_rgb("#000000") == (0, 0, 0)
    assert generator._hex_to_rgb("0071C5") == (0, 113, 197)  # Without '#'


def test_query_breakdown_data():
    """Test breakdown data aggregation."""
    tasks = [
        {"task_id": "task_001", "workload_type": "linalg", "passed": True},
        {"task_id": "task_002", "workload_type": "compile", "passed": True},
    ]

    verdicts = [
        {
            "task_id": "task_001",
            "analyzer_name": "breakdown",
            "verdict": "inference_dominant",
            "evidence": {"inference_pct": 60.0, "execution_pct": 30.0, "orchestration_pct": 10.0},
        },
        {
            "task_id": "task_002",
            "analyzer_name": "breakdown",
            "verdict": "execution_heavy",
            "evidence": {"inference_pct": 20.0, "execution_pct": 70.0, "orchestration_pct": 10.0},
        },
    ]

    generator = XeonPowerPointGenerator()
    store = MockResultStore(tasks=tasks, verdicts=verdicts)

    df = generator._query_breakdown_data("test_run", store)

    assert len(df) == 2
    assert "task_id" in df.columns
    assert "workload_type" in df.columns
    assert "inference_pct" in df.columns
    assert df.loc[df["task_id"] == "task_001", "inference_pct"].values[0] == 60.0
    assert df.loc[df["task_id"] == "task_002", "execution_pct"].values[0] == 70.0
