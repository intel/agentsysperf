#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the scaling-view figure builders — pure, no Streamlit/DB.

These cover the four current builders plus the helpers; the COMPARISON-OVERHAUL
section near the bottom is where new builders (side-by-side cpu/runqueue overlay,
delta table, per-task compare) get their tests as the UI overhaul lands.

Run: poetry run pytest src/dashboard/test_scaling_views.py -q
"""
from __future__ import annotations

import pytest

from src.dashboard.scaling_views import (
    BOTTLENECK_STYLE,
    LIMITING_FACTOR_STYLE,
    limiting_factor_band_figure,
    bottleneck_heatstrip_figure,
    throughput_knee_figure,
    throughput_compare_figure,
    efficiency_figure,
    efficiency_compare_figure,
    cpu_runqueue_figure,
    cpu_runqueue_compare_figure,
    saturation_signature_figure,
    per_task_profile_figure,
    _avg_by_density,
    _pressure_scores,
    _dominant_factor,
)


# ── fixtures ──────────────────────────────────────────────────────────────

def _pts(scale=1.0):
    """A small sweep_points-shaped list (3 densities), like query_sweep_points."""
    return [
        {"density": 0.5, "concurrency": 64, "throughput_per_min": 40 * scale,
         "p95_trial_latency_s": 5, "cpu_avg": 40, "cpu_peak": 50, "runqueue_max": 32,
         "ctx_sw_per_s": 5000, "iowait_pct_avg": 1.0, "mem_avail_mb_min": 900000},
        {"density": 1.0, "concurrency": 128, "throughput_per_min": 60 * scale,
         "p95_trial_latency_s": 8, "cpu_avg": 75, "cpu_peak": 83, "runqueue_max": 64,
         "ctx_sw_per_s": 8000, "iowait_pct_avg": 2.0, "mem_avail_mb_min": 850000},
        {"density": 2.0, "concurrency": 256, "throughput_per_min": 55 * scale,
         "p95_trial_latency_s": 14, "cpu_avg": 99, "cpu_peak": 100, "runqueue_max": 128,
         "ctx_sw_per_s": 12000, "iowait_pct_avg": 4.0, "mem_avail_mb_min": 800000},
    ]


def _curve():
    """A per-task curve, like evidence['per_task'][task]['curve']."""
    return [
        {"density": 0.5, "p95_latency": 4.0, "cpu_peak": 45},
        {"density": 1.0, "p95_latency": 9.0, "cpu_peak": 88},
        {"density": 2.0, "p95_latency": 18.0, "cpu_peak": 100},
    ]


# ── _avg_by_density helper ──────────────────────────────────────────────────

def test_avg_by_density_sorts_and_averages_replicates():
    pts = [
        {"density": 1.0, "throughput_per_min": 60},
        {"density": 0.5, "throughput_per_min": 40},
        {"density": 1.0, "throughput_per_min": 80},  # replicate at density 1.0
    ]
    out = _avg_by_density(pts, "throughput_per_min")
    assert out == [(0.5, 40.0), (1.0, 70.0)], "should sort by density and average replicates"


def test_avg_by_density_skips_missing_values():
    pts = [{"density": 1.0, "cpu_peak": None}, {"density": 0.5, "cpu_peak": 50}]
    assert _avg_by_density(pts, "cpu_peak") == [(0.5, 50.0)]


# ── throughput_knee_figure (Plot 1) ─────────────────────────────────────────

def test_knee_figure_has_throughput_and_p95_traces():
    fig = throughput_knee_figure(_pts(), knee={"density": 1.0, "concurrency": 128},
                                 bottleneck="cpu_bound")
    assert len(fig.data) == 2


def test_knee_figure_without_knee_still_builds():
    fig = throughput_knee_figure(_pts())  # no knee/bottleneck
    assert len(fig.data) == 2


def test_knee_figure_empty_points():
    fig = throughput_knee_figure([])
    assert len(fig.data) == 0


# ── cpu_runqueue_figure (Plot 2) ─────────────────────────────────────────────

def test_cpu_runqueue_figure_three_traces():
    fig = cpu_runqueue_figure(_pts())
    # cpu_peak, cpu_avg, runqueue_max
    assert len(fig.data) == 3
    names = {t.name for t in fig.data}
    assert {"CPU peak %", "CPU avg %", "Tasks waiting to run (peak)"} <= names


def test_cpu_runqueue_figure_empty():
    assert len(cpu_runqueue_figure([]).data) == 0


# ── per_task_profile_figure (Plot 3) ─────────────────────────────────────────

def test_per_task_profile_builds_with_knee():
    fig = per_task_profile_figure("terminal-bench/largest-eigenval", _curve(),
                                  knee_density=1.0, bottleneck="cpu_bound")
    # latency + cpu traces
    assert len(fig.data) == 2
    assert "largest-eigenval" in fig.layout.title.text


def test_per_task_profile_no_knee():
    fig = per_task_profile_figure("t", _curve())
    assert len(fig.data) == 2


# ── BOTTLENECK_STYLE ─────────────────────────────────────────────────────────

def test_bottleneck_style_covers_all_classes():
    for cls in ("cpu_bound", "scheduler_oversubscription", "memory_bound",
                "io_bound", "headroom_remaining"):
        color, label = BOTTLENECK_STYLE[cls]
        assert color.startswith("#") and label


# ── throughput_compare_figure (current overlay) ──────────────────────────────

def test_compare_one_trace_per_sweep():
    fig = throughput_compare_figure([
        {"label": "numa unpinned", "points": _pts(1.0), "knee": {"density": 1.0}},
        {"label": "numa pinned", "points": _pts(1.3), "knee": {"density": 1.5}},
    ])
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"numa unpinned", "numa pinned"}


def test_compare_skips_empty_sweep():
    fig = throughput_compare_figure([
        {"label": "good", "points": _pts()},
        {"label": "empty", "points": []},
    ])
    assert len(fig.data) == 1 and fig.data[0].name == "good"


def test_compare_handles_zero_sweeps():
    assert len(throughput_compare_figure([]).data) == 0


# ── limiting-factor pressure + band + heatstrip (P1 exec reframe) ────────────

def test_pressure_scores_compute_bound_cell():
    # cpu pegged (peak 100, avg 99), low runq/iowait, plenty of memory → compute dominant
    pt = {"density": 1.0, "cpu_peak": 100, "cpu_avg": 99, "runqueue_max": 10,
          "mem_avail_mb_min": 900000, "iowait_pct_avg": 0.0}
    s = _pressure_scores(pt, logical_cpus=256)
    assert s["compute"] >= 0.99
    assert s["compute"] > s["memory"] and s["compute"] > s["scheduler"] and s["compute"] > s["io"]
    assert _dominant_factor(pt, 256) == "compute"


def test_pressure_scores_scheduler_bound_cell():
    # runqueue way over 1.5× logical_cpus → scheduler dominant (precedence)
    pt = {"density": 3.0, "cpu_peak": 100, "cpu_avg": 99, "runqueue_max": 600,
          "mem_avail_mb_min": 900000, "iowait_pct_avg": 0.0}
    assert _dominant_factor(pt, logical_cpus=256) == "scheduler"


def test_pressure_scores_headroom_cell():
    # nothing saturated → headroom
    pt = {"density": 0.25, "cpu_peak": 30, "cpu_avg": 20, "runqueue_max": 5,
          "mem_avail_mb_min": 900000, "iowait_pct_avg": 0.0}
    s = _pressure_scores(pt, logical_cpus=256)
    assert s["headroom"] >= 0.5
    assert _dominant_factor(pt, 256) == "headroom"


def test_limiting_factor_band_normalizes_to_100():
    fig = limiting_factor_band_figure(_pts(), logical_cpus=256, knee={"density": 1.0})
    # one stacked trace per limiting factor (5)
    assert len(fig.data) == 5
    assert {t.name for t in fig.data} == {"Compute", "Memory", "Scheduler", "I/O", "Headroom"}


def test_bottleneck_heatstrip_one_cell_per_density():
    fig = bottleneck_heatstrip_figure(_pts(), logical_cpus=256)
    assert len(fig.data) == 1  # single heatmap row
    z = fig.data[0].z[0]
    assert len(z) == 3  # 3 densities in _pts()


def test_limiting_factor_band_empty():
    fig = limiting_factor_band_figure([], logical_cpus=128)
    # builds; stacked traces exist but with no x points
    assert all(len(t.x) == 0 for t in fig.data)


# ── efficiency (throughput per core) — P3 ────────────────────────────────────

def test_efficiency_figure_normalizes_by_basis():
    fig = efficiency_figure(_pts(), vcpu_basis=128, knee={"density": 1.0},
                            bottleneck="cpu_bound")
    assert len(fig.data) == 1
    # at density 1.0 throughput=60 → per-core = 60/128
    ys = dict(zip(fig.data[0].x, fig.data[0].y))
    assert abs(ys[1.0] - 60 / 128) < 1e-9


def test_efficiency_figure_zero_basis_safe():
    # basis 0 must not divide-by-zero (clamped to 1)
    fig = efficiency_figure(_pts(), vcpu_basis=0)
    assert len(fig.data) == 1


def test_efficiency_compare_one_line_per_sweep():
    fig = efficiency_compare_figure([
        {"label": "A", "points": _pts(1.0), "vcpu_basis": 128},
        {"label": "B", "points": _pts(1.5), "vcpu_basis": 256},
    ])
    assert {t.name for t in fig.data} == {"A", "B"}
    # B has 2× the basis, so at equal throughput-scale its per-core is lower
    a = dict(zip(fig.data[0].x, fig.data[0].y))
    b = dict(zip(fig.data[1].x, fig.data[1].y))
    assert b[1.0] < a[1.0] * 1.5  # 1.5×throughput / 2×cores < 1.5× per-core


# ── cpu_runqueue_compare — P5 ────────────────────────────────────────────────

def test_cpu_runqueue_compare_overlays_two_sweeps():
    fig = cpu_runqueue_compare_figure([
        {"label": "A", "points": _pts(1.0)},
        {"label": "B", "points": _pts(1.2)},
    ])
    # one CPU-peak + one runqueue trace per sweep => 4
    assert len(fig.data) == 4
    names = {t.name for t in fig.data}
    assert "A · CPU peak %" in names and "B · tasks waiting" in names


# ── saturation_signature panel — P4 ──────────────────────────────────────────

def test_saturation_signature_plots_proof_signals():
    fig = saturation_signature_figure(_pts(), logical_cpus=256,
                                      bottleneck="cpu_bound", knee={"density": 1.0})
    names = {t.name for t in fig.data}
    # tasks-waiting + ctx switches + iowait + mem avail. The waiting-tasks trace
    # is named "(whole machine)" only when the cell records
    # runqueue_is_host_wide; this fixture does not, so accept either form rather
    # than pinning the test to one scope.
    assert {"ctx switches/s", "iowait %", "mem avail (MB)"} <= names
    assert any(n.startswith("Tasks waiting to run") for n in names)


def test_saturation_signature_empty_points():
    fig = saturation_signature_figure([], logical_cpus=128)
    assert len(fig.data) == 0  # no traces, but builds without error
