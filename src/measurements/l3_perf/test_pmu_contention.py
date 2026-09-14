#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for L3's physical-plausibility guard on perf counter sums.

Motivating observation (reproduced on a 288-core Clearwater Forest host): while
EMON holds the PMU through the SEP driver, a concurrent system-wide `perf stat`
exits 0 and writes well-formed CSV in which every event reads back a
near-identical ~1.7e18 count. The derived metrics come out as IPC ~= 1.0 and
cache/branch miss rates ~= 100%, which the analyzers then convert into
confident `memory_bound` / `dram_bound` verdicts.

Measured on that host, same 5s window:
  PMU free:      cycles 4.24e10, IPC 0.26, cache-miss  0.46%, branch-miss 5.58%
  emon running:  cycles 1.72e18, IPC 1.02, cache-miss 99.79%, branch-miss 99.43%
"""

from __future__ import annotations

import os

from src.measurements.l3_perf.probe import _counts_are_physical


def test_healthy_counters_accepted():
    """Real 5s system-wide sample with the PMU free."""
    sums = {
        "cycles": 42_426_279_458.0,
        "instructions": 11_059_603_645.0,
        "cache-references": 1_214_887_556.0,
        "cache-misses": 5_618_377.0,
        "branch-instructions": 3_275_786_549.0,
        "branch-misses": 182_674_555.0,
    }
    assert _counts_are_physical(sums, 5.035) is True


def test_pmu_contention_counters_rejected():
    """Real 5s system-wide sample taken while emon held the PMU."""
    sums = {
        "cycles": 1.724945541861901e18,
        "instructions": 1.754319817156429e18,
        "cache-references": 1.734034557738039e18,
        "cache-misses": 1.730415875007039e18,
        "branch-instructions": 1.706729638320561e18,
        "branch-misses": 1.697051867288366e18,
    }
    assert _counts_are_physical(sums, 5.039) is False


def test_more_misses_than_references_rejected():
    """>100% cache miss rate is impossible regardless of cycle count."""
    sums = {"cache-references": 1000.0, "cache-misses": 1001.0}
    assert _counts_are_physical(sums, 1.0) is False


def test_more_branch_misses_than_branches_rejected():
    sums = {"branch-instructions": 1000.0, "branch-misses": 1001.0}
    assert _counts_are_physical(sums, 1.0) is False


def test_all_misses_is_allowed():
    """A 100% miss rate is extreme but physically reachable — don't reject it."""
    sums = {"cache-references": 1000.0, "cache-misses": 1000.0}
    assert _counts_are_physical(sums, 1.0) is True


def test_cycle_ceiling_scales_with_cpu_count():
    """Just under the per-host ceiling passes; an order of magnitude over fails."""
    cpus = os.cpu_count() or 1
    just_under = {"cycles": cpus * 5e9}          # 5 GHz/CPU for one second
    way_over = {"cycles": cpus * 100e9}          # 100 GHz/CPU
    assert _counts_are_physical(just_under, 1.0) is True
    assert _counts_are_physical(way_over, 1.0) is False


def test_missing_cycles_not_rejected():
    """Absent counters are a coverage gap, not corruption."""
    assert _counts_are_physical({}, 1.0) is True


# ─── per-interval salvage ────────────────────────────────────────────

def _finalize(monkeypatch, tmp_path, rows, *, window=(0.0, 1.0)):
    """Drive _emit_for_window over a synthetic interval list."""
    import src.measurements.l3_perf.probe as mod

    # Make the physical ceilings deterministic across CI runners.
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(mod, "parse_perf_csv_interval", lambda text: rows)
    csv = tmp_path / "l3.csv"
    csv.write_text("stub")

    m = mod.L3PerfMeasurement()
    m._csv_path = csv
    m._sample_interval_ms = 100
    return m._emit_for_window(
        span_id="span", kind="test", node_id="n",
        window_start_s=window[0], window_end_s=window[1],
    )


def test_bursty_corruption_salvages_the_good_intervals(monkeypatch, tmp_path):
    """The real pattern: corruption arrives in bursts, not uniformly.

    Summing first would let two ~1e17 intervals poison a span that is otherwise
    entirely valid. Six good intervals must survive two bad ones.
    """
    rows = []
    for ts in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        rows += [(ts, "cycles", 1.5e9), (ts, "instructions", 1.0e9)]
    for ts in (0.7, 0.8):
        rows += [(ts, "cycles", 3.9e16), (ts, "instructions", 3.8e16)]

    rec = _finalize(monkeypatch, tmp_path, rows)
    assert rec is not None, "a mostly-good span must not be thrown away"
    assert rec.payload["sample_counts"]["cycles"] == 6
    # 6.0e9 instructions / 9.0e9 cycles — the corrupt intervals contributed
    # nothing, so IPC reflects the good data only.
    assert abs(rec.payload["ipc"] - (6.0e9 / 9.0e9)) < 1e-9


def test_corrupt_interval_drops_its_sibling_events(monkeypatch, tmp_path):
    """Events at a bad timestamp are corrupted together, so the whole
    interval goes — not just the cycles counter."""
    rows = [
        (0.1, "cycles", 1.5e9), (0.1, "cache-references", 1.0e8),
        (0.2, "cycles", 3.9e16), (0.2, "cache-references", 3.8e16),
    ]
    rec = _finalize(monkeypatch, tmp_path, rows)
    assert rec.payload["events"]["cache-references"] == 1.0e8
    assert rec.payload["sample_counts"]["cache-references"] == 1


def test_fully_corrupt_span_still_rejected(monkeypatch, tmp_path):
    """Nothing salvageable -> no record, rather than a fabricated one."""
    rows = [(ts, "cycles", 3.9e16) for ts in (0.1, 0.2, 0.3)]
    assert _finalize(monkeypatch, tmp_path, rows) is None
