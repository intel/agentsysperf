#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for BreakdownAnalyzer — time breakdown + per-phase hardware rollup."""

from __future__ import annotations

from src.analyzers.breakdown import BreakdownAnalyzer
from src.protocols import MeasurementRecord as MR


def _records():
    return [
        MR(span_id="t", layer="l1", payload={"duration_us": 10_000_000}),
        MR(span_id="t/turn_0_llm", layer="l1",
           payload={"kind": "inference", "duration_us": 800_000}),
        MR(span_id="t/turn_0_cmd", layer="l1",
           payload={"kind": "execution", "duration_us": 4_000_000}),
        MR(span_id="t/turn_1_cmd", layer="l1",
           payload={"kind": "execution", "duration_us": 3_000_000}),
        # L3 hardware for the execution sub-spans (span0 has 9x the instructions).
        MR(span_id="t/turn_0_cmd", layer="l3",
           payload={"kind": "execution", "instructions": 9e9, "cycles": 6e9,
                    "cache_miss_pct": 5.0, "branch_miss_pct": 0.1}),
        MR(span_id="t/turn_1_cmd", layer="l3",
           payload={"kind": "execution", "instructions": 1e9, "cycles": 1e9,
                    "cache_miss_pct": 15.0, "branch_miss_pct": 0.5}),
        # An inference L3 record — must be IGNORED (network wait, meaningless HW).
        MR(span_id="t/turn_0_llm", layer="l3",
           payload={"kind": "inference", "instructions": 1e6, "cycles": 1e9,
                    "cache_miss_pct": 99.0}),
    ]


def test_time_breakdown_and_verdict():
    res = list(BreakdownAnalyzer().analyze(_records()))
    assert len(res) == 1
    ev = res[0].evidence
    assert res[0].verdict == "execution_dominant"
    assert round(ev["execution_pct"]) == 70
    assert round(ev["orchestration_pct"]) == 22   # residual
    assert round(ev["inference_pct"]) == 8


def test_execution_hardware_instruction_weighted():
    ev = list(BreakdownAnalyzer().analyze(_records()))[0].evidence
    hw = ev["execution_hw"]
    # instruction-weighted IPC = 10e9 / 7e9 = 1.429 (NOT naive mean 1.25)
    assert abs(hw["ipc"] - 1.429) < 0.01
    # cache-miss weighted by instructions = (5*9 + 15*1)/10 = 6.0
    assert abs(hw["cache_miss_pct"] - 6.0) < 0.01
    assert hw["span_count"] == 2


def test_inference_hardware_suppressed():
    ev = list(BreakdownAnalyzer().analyze(_records()))[0].evidence
    # Inference is a remote network wait — no hardware rollup, by design.
    assert "inference_hw" not in ev
    assert "orchestration_hw" not in ev   # residual, no spans/counters


def test_no_l3_means_no_hw_key():
    # L1-only run (no perf counters) still produces the time breakdown.
    recs = [r for r in _records() if r.layer == "l1"]
    ev = list(BreakdownAnalyzer().analyze(recs))[0].evidence
    assert "execution_hw" not in ev
    assert ev["execution_pct"] > 0
