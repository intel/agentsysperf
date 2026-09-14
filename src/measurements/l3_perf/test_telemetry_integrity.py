#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests that PerfStatTelemetry never turns a non-reading into a reading.

`PerfStatTelemetry` feeds `OptimizationProfilePlugin.verify_engaged()`, which
compares each counter against a threshold. That makes the difference between
"unmeasured" and "measured zero" load-bearing: a fabricated 0.0 clears no
threshold, so it is reported as *the optimization did not engage* when the
truth is *this host could not measure it*. One is a finding about the SUT, the
other is a finding about the measurement rig.

Two ways perf hands back a non-reading while exiting 0:
  - `<not counted>` / `<not supported>` for an event the SKU or paranoid level
    won't give up.
  - Well-formed CSV in which every event reads back a near-identical ~1e18
    count, observed while EMON holds the PMU through the SEP driver (see
    test_pmu_contention.py for the measured numbers).
"""

from __future__ import annotations

import src.measurements.l3_perf.telemetry as tmod
from src.measurements.l3_perf._perf_subprocess import parse_perf_csv_summary
from src.optimization_profiles._base import evaluate_thresholds

# perf stat -x , writes `<value>,<unit>,<event>,<run>,<pct>` to stderr.
HEALTHY_CSV = (
    "42426279458,,cycles,5035000000,100.00\n"
    "11059603645,,instructions,5035000000,100.00\n"
)
PARTIAL_CSV = (
    "42426279458,,cycles,5035000000,100.00\n"
    "<not counted>,,instructions,0,0.00\n"
    "<not supported>,,cache-misses,0,0.00\n"
)
CONTENDED_CSV = (
    "1724945541861901000,,cycles,5039000000,100.00\n"
    "1754319817156429000,,instructions,5039000000,100.00\n"
)


# ─── parser keeps "unmeasured" out of the values map ──────────────────

def test_healthy_summary_parses_all_values():
    values, unmeasured = parse_perf_csv_summary(HEALTHY_CSV)
    assert values == {"cycles": 42426279458.0, "instructions": 11059603645.0}
    assert unmeasured == []


def test_not_counted_is_unmeasured_not_zero():
    """The regression: these used to land in values as 0.0."""
    values, unmeasured = parse_perf_csv_summary(PARTIAL_CSV)
    assert values == {"cycles": 42426279458.0}
    assert "instructions" not in values
    assert "cache-misses" not in values
    assert sorted(unmeasured) == ["cache-misses", "instructions"]


# ─── telemetry propagates the distinction ─────────────────────────────

def _telemetry(monkeypatch, csv_text):
    """A PerfStatTelemetry whose perf window returns `csv_text`."""
    monkeypatch.setattr(tmod, "perf_available", lambda: True)
    monkeypatch.setattr(tmod, "detect_cpu_model", lambda: "test-cpu")
    monkeypatch.setattr(
        tmod, "run_perf_window",
        lambda events, window_s, target=None: csv_text,
    )
    return tmod.PerfStatTelemetry()


def test_read_counters_omits_uncounted_events(monkeypatch):
    t = _telemetry(monkeypatch, PARTIAL_CSV)
    reading = t.read_counters(["cycles", "instructions", "cache-misses"])
    assert reading.values == {"cycles": 42426279458.0}
    assert "not counted" in reading.notes
    assert "instructions" in reading.notes


def test_read_counters_discards_pmu_contention_window(monkeypatch):
    """Physically impossible counts are no reading at all, not a low one."""
    t = _telemetry(monkeypatch, CONTENDED_CSV)
    reading = t.read_counters(["cycles", "instructions"], window_s=5.039)
    assert reading.values == {}
    assert "physically impossible" in reading.notes


def test_read_counters_passes_healthy_window_through(monkeypatch):
    t = _telemetry(monkeypatch, HEALTHY_CSV)
    reading = t.read_counters(["cycles", "instructions"], window_s=5.035)
    assert reading.values["cycles"] == 42426279458.0
    assert reading.notes == ""


# ─── the verdict downstream is the reason any of this matters ─────────

def test_uncounted_counter_reads_as_unmeasured_not_disengaged(monkeypatch):
    """An event perf refused to count must not read as "did not engage".

    Before the fix `instructions` arrived as 0.0, cleared no threshold, and
    produced "below min ... (claimed optimization did not engage)" — a
    confident claim about the SUT derived from a missing measurement.
    """
    t = _telemetry(monkeypatch, PARTIAL_CSV)
    reading = t.read_counters(["instructions"])

    verdict = evaluate_thresholds(
        verify_counters={"instructions": 1000.0},
        measured=reading.values,
        available_events=t.available_events,
    )

    assert verdict["engaged"] is False
    assert len(verdict["failures"]) == 1
    failure = verdict["failures"][0]
    assert "did not return a reading" in failure
    assert "did not engage" not in failure
