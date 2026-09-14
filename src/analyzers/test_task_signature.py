#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the task-signature resource spectrum.

The load-bearing behaviours, in order of what would cause a wrong published
claim if broken:

  * a missing component is None, NEVER 0.0 — zero-filling is the defect this
    module exists to avoid (scaling.py returns 0.0 for a missing average and inf
    for missing memory, which makes an unmeasured sweep emit a confident verdict),
  * retention is measured against the PEAK per-agent cell, not the lowest
    concurrency, because low-density cells can be startup-dominated,
  * explain() covers the whole quota_fill range with no silent gap,
  * caveats (multiplexing, scope, replicate count) survive aggregation.

Run: poetry run pytest src/analyzers/test_task_signature.py -q
"""
from __future__ import annotations

import pytest

from src.analyzers.task_signature import (
    COMPONENTS,
    PROVENANCE,
    explain,
    signature_for_row,
    signature_for_sweep,
)


def _row(n, *, cpus=1.0, cores=16, cpu_avg=50.0, elapsed=100.0, completed=None,
         trial_wall=70.0, thr=None, bw=512.0, disk=10.0, net=0.0, **extra):
    """A sweep_points row shaped like import_points writes it."""
    completed = n if completed is None else completed
    meta = {
        "task": "t", "task_cpus": cpus, "cores_in_cpuset": cores,
        "quota_demand": round(n * cpus / cores, 4),
        "quota_fill": round((cpu_avg / 100.0 * cores) / (n * cpus), 4),
        "p50_trial_total_s": trial_wall,
        "mem_bw_mib_s_socket": bw,
        "disk_write_mb_s_host": disk,
        "net_total_mb_s_host": net,
        "completed_trials": completed,
        "ipc": 1.9,
    }
    meta.update(extra)
    return {
        "concurrency": n, "density": round(n / cores, 4),
        "elapsed_s": elapsed, "cpu_avg": cpu_avg,
        "throughput_per_min": thr if thr is not None else 0.6 * n,
        "p95_trial_latency_s": 1.0,
        "metadata": meta,
    }


def test_missing_component_is_none_not_zero():
    """A fabricated 0.0 reads as 'measured, and it was zero'."""
    r = _row(8)
    r["metadata"].pop("mem_bw_mib_s_socket")
    r["metadata"].pop("disk_write_mb_s_host")
    p = signature_for_row(r)
    assert p.components["mem_bw_gb_s"] is None
    assert p.components["disk_write_mb_s"] is None
    assert "mem_bw_gb_s" in p.missing and "disk_write_mb_s" in p.missing


def test_measured_zero_is_distinguishable_from_unmeasured():
    """net was 0.0 in every real cell (images pre-cached) — that is a finding,
    not an absence, and the two must not collapse."""
    measured = signature_for_row(_row(8, net=0.0))
    absent_row = _row(8)
    absent_row["metadata"].pop("net_total_mb_s_host")
    absent = signature_for_row(absent_row)
    assert measured.components["net_mb_s"] == 0.0
    assert "net_mb_s" not in measured.missing
    assert absent.components["net_mb_s"] is None
    assert "net_mb_s" in absent.missing


def test_every_component_has_a_provenance_tag():
    """Scope is load-bearing: socket/host numbers are not per-cell attributable."""
    assert set(PROVENANCE) == set(COMPONENTS)
    p = signature_for_row(_row(4))
    assert set(p.as_dict()["provenance"]) == set(p.components)
    assert PROVENANCE["mem_bw_gb_s"].endswith("socket")
    assert PROVENANCE["disk_write_mb_s"].endswith("host")


def test_serialization_is_off_cpu_share():
    """cpu_avg 50% of 16 cores over 100s = 800 CPU-s; 8 trials -> 100 CPU-s each
    against a 200s wall = 0.5 off-CPU."""
    p = signature_for_row(
        _row(8, cpu_avg=50.0, cores=16, elapsed=100.0, completed=8, trial_wall=200.0))
    assert p.components["serialization"] == pytest.approx(0.5, abs=1e-3)


def test_serialization_clamped_not_negative():
    """CPU-seconds can exceed a trial's wall when the cpuset ran other work; a
    negative 'share of wall time' is nonsense."""
    p = signature_for_row(
        _row(2, cpu_avg=100.0, cores=16, elapsed=100.0, completed=1, trial_wall=10.0))
    assert p.components["serialization"] == 0.0


def test_serialization_none_when_inputs_missing():
    r = _row(8)
    r["metadata"].pop("p50_trial_total_s")
    r["metadata"].pop("p50_agent_exec_s", None)
    assert signature_for_row(r).components["serialization"] is None


def test_retention_is_relative_to_peak_not_lowest_n():
    """Measured on real hardware, the lowest-density cell can be worse per-agent
    than the peak (startup-dominated), so baselining on it overstates scaling."""
    rows = [
        _row(2, thr=1.0),     # 0.50/agent  <- lowest n, NOT the peak
        _row(4, thr=3.2),     # 0.80/agent  <- peak
        _row(8, thr=4.8),     # 0.60/agent
    ]
    pts = {p.concurrency: p for p in signature_for_sweep(rows)}
    assert pts[4].retention == pytest.approx(1.0)
    assert pts[2].retention == pytest.approx(0.625, abs=1e-3)
    assert pts[8].retention == pytest.approx(0.75, abs=1e-3)


def test_replicates_are_averaged_and_counted():
    rows = [_row(8, cpu_avg=40.0), _row(8, cpu_avg=50.0), _row(8, cpu_avg=60.0)]
    pts = signature_for_sweep(rows)
    assert len(pts) == 1
    assert pts[0].replicates == 3


def test_single_replicate_is_flagged():
    """hbox shipped with 1 replicate and non-monotonic retention; a reader must
    be told the point has no confidence interval."""
    pts = signature_for_sweep([_row(8)])
    assert any("replicate" in c for c in pts[0].caveats)


def test_multiplexing_caveat_survives_aggregation():
    rows = [_row(8, counters_multiplexed=True, counter_enabled_pct_min=47.0)
            for _ in range(3)]
    pts = signature_for_sweep(rows)
    assert any("multiplexed" in c for c in pts[0].caveats)
    # ...and is not duplicated once per replicate.
    assert sum("multiplexed" in c for c in pts[0].caveats) == 1


def test_container_undercount_is_flagged():
    """The container sampler misses containers that start and finish between
    ticks; averaging over a subset silently would be worse than saying so."""
    p = signature_for_row(_row(24, ctr_containers_seen=9))
    assert any("9 of 24" in c for c in p.caveats)


def test_explain_covers_the_whole_quota_fill_range():
    """A gap between the CPU-saturation and off-CPU bands silently dropped the
    explanation for the one cell that mattered (fib at n=24, fill 0.53)."""
    for fill_cpu_avg, label in ((80.0, "high"), (53.0, "middle"), (10.0, "low")):
        rows = [
            _row(2, cpu_avg=fill_cpu_avg, thr=1.6),
            _row(24, cpu_avg=fill_cpu_avg, thr=6.0, trial_wall=300.0),
        ]
        lines = explain(signature_for_sweep(rows))
        assert any(l.startswith("**Why") for l in lines), (
            f"no cause explanation for the {label} band")


def test_explain_reports_no_knee_when_nothing_degrades():
    """Flat per-agent throughput: the peak is the last point, so degradation is
    not observable rather than absent — the message must say which."""
    rows = [_row(n, thr=0.8 * n) for n in (2, 4, 8)]
    lines = explain(signature_for_sweep(rows))
    assert any("No saturation found" in l or "Cannot tell where it saturates" in l
               for l in lines)


def test_degradation_is_only_looked_for_past_the_peak():
    """Regression: explain() scanned the WHOLE sweep, so hbox's n=2 (retention
    0.50, below its n=8 peak) was reported as the "first >20% loss" — naming a
    point BEFORE the peak as the onset of decline. Points below the peak at lower
    concurrency are the ramp region, not saturation."""
    rows = [
        _row(2, thr=1.0),      # 0.500/agent -> 0.50 retention, ramp
        _row(4, thr=1.4),      # 0.350/agent -> 0.35 retention, ramp
        _row(8, thr=8.0),      # 1.000/agent -> peak
        _row(16, thr=8.0),     # 0.500/agent -> 0.50, real degradation
    ]
    lines = explain(signature_for_sweep(rows))
    onset = [l for l in lines if "Saturates at" in l]
    assert onset, "no degradation onset reported"
    assert "n=16" in onset[0], f"onset should be past the peak, got: {onset[0]}"
    assert "n=2" not in onset[0]
    # ...and the sub-peak points are explicitly labelled as ramp, not a knee.
    assert any("ramp region" in l for l in lines)


def test_explain_names_unmeasured_components_at_the_knee():
    rows = [_row(2, thr=1.6), _row(24, thr=4.0)]
    for r in rows:
        r["metadata"].pop("mem_bw_mib_s_socket")
    lines = explain(signature_for_sweep(rows))
    assert any("Not measured here" in l and "mem_bw_gb_s" in l for l in lines)


def test_explain_handles_empty_input():
    assert "nothing to explain" in explain([])[0]


def test_quota_demand_separates_tasks_with_different_cpus():
    """The originating finding: two tasks knee at the same quota demand while
    appearing to knee at 2x different agent counts."""
    fib = signature_for_sweep([_row(16, cpus=1.0, thr=11.8)])
    hbox = signature_for_sweep([_row(8, cpus=2.0, thr=5.8)])
    assert fib[0].components["quota_demand"] == pytest.approx(1.0)
    assert hbox[0].components["quota_demand"] == pytest.approx(1.0)
    assert fib[0].concurrency != hbox[0].concurrency
