#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for ScalingAnalyzer knee detection + bottleneck classification.

Pins the curve shapes that must classify a given way so threshold edits can't
silently change verdicts:
  * a throughput curve that flattens has a detectable knee,
  * the bottleneck at the knee is read from node telemetry, not assumed,
  * a still-rising curve reports headroom, not a false knee.
"""
from __future__ import annotations

from src.analyzers.scaling import ScalingAnalyzer, _kneedle

LOGICAL = 256  # EMR


def _pt(density, concurrency, throughput, p95, cpu_avg, cpu_peak,
        runqueue_max, mem_avail=1_000_000.0, iowait=0.0):
    return {
        "density": density, "concurrency": concurrency,
        "throughput_per_min": throughput, "p95_trial_latency_s": p95,
        "cpu_avg": cpu_avg, "cpu_peak": cpu_peak, "runqueue_max": runqueue_max,
        "ctx_sw_per_s": 5000.0, "mem_avail_mb_min": mem_avail, "iowait_pct_avg": iowait,
    }


def test_kneedle_flattening_curve():
    # Throughput rises then flattens: knee near where slope drops.
    xs = [0.25, 0.5, 1.0, 2.0, 4.0]
    ys = [10.0, 20.0, 38.0, 42.0, 43.0]  # flattens after density ~1.0
    i = _kneedle(xs, ys)
    assert i is not None
    assert xs[i] in (0.5, 1.0)  # elbow sits at the bend


def test_kneedle_linear_no_knee():
    # Perfectly linear throughput → no elbow.
    xs = [0.25, 0.5, 1.0, 2.0, 4.0]
    ys = [10.0, 20.0, 40.0, 80.0, 160.0]
    assert _kneedle(xs, ys) is None


def test_cpu_bound_knee():
    # Flattening throughput + CPU pegged at the knee → cpu_bound.
    points = [
        _pt(0.25, 64, 10.0, 5.0, 25.0, 40.0, 10),
        _pt(0.5, 128, 20.0, 5.5, 50.0, 70.0, 20),
        _pt(1.0, 256, 38.0, 7.0, 92.0, 99.0, 60),   # knee: CPU pegged
        _pt(2.0, 512, 41.0, 12.0, 96.0, 100.0, 120),
        _pt(4.0, 1024, 42.0, 25.0, 98.0, 100.0, 200),
    ]
    out = ScalingAnalyzer().analyze_sweep(points, logical_cpus=LOGICAL)
    assert out is not None
    assert "cpu_bound" in out.verdict
    assert out.evidence["knee"] is not None
    assert out.evidence["knee"]["density"] in (0.5, 1.0)


def test_scheduler_oversubscription_knee():
    # Runqueue far exceeds logical CPUs at the knee → scheduler-bound,
    # even though CPU% is not pegged.
    points = [
        _pt(0.5, 128, 20.0, 5.0, 40.0, 60.0, 50),
        _pt(1.0, 256, 35.0, 6.0, 55.0, 75.0, 120),
        _pt(2.0, 512, 40.0, 9.0, 60.0, 80.0, 600),   # runqueue 600 >> 1.5*256
        _pt(4.0, 1024, 41.0, 18.0, 62.0, 82.0, 1200),
    ]
    out = ScalingAnalyzer().analyze_sweep(points, logical_cpus=LOGICAL)
    assert out is not None
    assert "scheduler_oversubscription" in out.verdict


def test_headroom_when_still_scaling():
    # Throughput still climbing linearly, CPU low → no knee, headroom.
    points = [
        _pt(0.1, 26, 10.0, 5.0, 5.0, 10.0, 5),
        _pt(0.2, 51, 20.0, 5.0, 10.0, 18.0, 8),
        _pt(0.4, 102, 40.0, 5.0, 20.0, 30.0, 12),
        _pt(0.8, 205, 80.0, 5.0, 38.0, 55.0, 20),
    ]
    out = ScalingAnalyzer().analyze_sweep(points, logical_cpus=LOGICAL)
    assert out is not None
    assert out.verdict in ("no_knee_within_swept_range", "saturating:headroom_remaining")
    assert out.evidence["bottleneck"] == "headroom_remaining"


def test_insufficient_data():
    out = ScalingAnalyzer().analyze_sweep([_pt(1.0, 256, 30.0, 5.0, 50.0, 70.0, 30)],
                                          logical_cpus=LOGICAL)
    assert out.verdict == "insufficient_data"


def test_per_task_breakdown():
    points = [
        _pt(0.5, 128, 20.0, 5.0, 50.0, 70.0, 20),
        _pt(1.0, 256, 38.0, 7.0, 92.0, 99.0, 60),
        _pt(2.0, 512, 41.0, 12.0, 96.0, 100.0, 120),
    ]
    per_task = {"mteb-retrieve": points}
    out = ScalingAnalyzer().analyze_sweep(points, logical_cpus=LOGICAL,
                                          per_task_points=per_task)
    assert "per_task" in out.evidence
    assert "mteb-retrieve" in out.evidence["per_task"]
    assert "bottleneck" in out.evidence["per_task"]["mteb-retrieve"]


def test_replicates_averaged():
    # Two replicates at each density should collapse to one point each.
    points = [
        _pt(0.5, 128, 20.0, 5.0, 50.0, 70.0, 20),
        _pt(0.5, 128, 22.0, 5.2, 52.0, 72.0, 22),
        _pt(1.0, 256, 38.0, 7.0, 92.0, 99.0, 60),
        _pt(1.0, 256, 40.0, 7.2, 90.0, 99.0, 62),
        _pt(2.0, 512, 41.0, 12.0, 96.0, 100.0, 120),
    ]
    out = ScalingAnalyzer().analyze_sweep(points, logical_cpus=LOGICAL)
    assert out.evidence["n_points"] == 3  # 3 distinct densities
