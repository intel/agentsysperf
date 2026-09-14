#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the PerfSpect measurement plugin.

Focus: the command actually handed to the binary. PerfSpect writes NO metric
files at all when privilege elevation fails, so whether `--noroot` is present
is the difference between 5-of-5 and 4-of-5 measurement layers on a host
without passwordless sudo.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.measurements.perfspect import measurement as ps
from src.measurements.perfspect.measurement import PerfSpectMeasurement


@pytest.fixture
def probe(monkeypatch):
    """A measurement instance that never touches the real host."""
    def _make(**kwargs):
        monkeypatch.setattr(PerfSpectMeasurement, "_check_available", lambda self: True)
        return PerfSpectMeasurement(perfspect_path="/fake/perfspect", **kwargs)
    return _make


# ─── --noroot selection ──────────────────────────────────────────────

def test_noroot_added_when_sudo_needs_password(probe, monkeypatch):
    monkeypatch.setattr(ps.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(ps, "_needs_noroot", lambda: True)
    cmd = probe()._build_command(Path("/tmp/out"))
    assert "--noroot" in cmd


def test_noroot_omitted_when_elevation_works(probe, monkeypatch):
    monkeypatch.setattr(ps, "_needs_noroot", lambda: False)
    cmd = probe()._build_command(Path("/tmp/out"))
    assert "--noroot" not in cmd


def test_noroot_explicit_override_wins(probe, monkeypatch):
    monkeypatch.setattr(ps, "_needs_noroot", lambda: True)
    assert "--noroot" not in probe(noroot=False)._build_command(Path("/tmp/out"))
    monkeypatch.setattr(ps, "_needs_noroot", lambda: False)
    assert "--noroot" in probe(noroot=True)._build_command(Path("/tmp/out"))


# ─── _needs_noroot itself ────────────────────────────────────────────

def test_needs_noroot_false_as_root(monkeypatch):
    monkeypatch.setattr(ps.os, "geteuid", lambda: 0)
    assert ps._needs_noroot() is False


def test_needs_noroot_false_with_passwordless_sudo(monkeypatch):
    monkeypatch.setattr(ps.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        ps.subprocess, "run",
        lambda *a, **k: type("R", (), {"returncode": 0})(),
    )
    assert ps._needs_noroot() is False


def test_needs_noroot_true_when_sudo_rejects(monkeypatch):
    monkeypatch.setattr(ps.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        ps.subprocess, "run",
        lambda *a, **k: type("R", (), {"returncode": 1})(),
    )
    assert ps._needs_noroot() is True


def test_needs_noroot_true_when_sudo_missing(monkeypatch):
    """No sudo binary at all -> don't try to elevate."""
    monkeypatch.setattr(ps.os, "geteuid", lambda: 1000)
    def _boom(*a, **k):
        raise FileNotFoundError("sudo")
    monkeypatch.setattr(ps.subprocess, "run", _boom)
    assert ps._needs_noroot() is True


# ─── PMU-contention detection ────────────────────────────────────────

def test_pmu_corruption_fingerprint_detected():
    """Reproduced on hardware: perfspect during an emon collection emits
    IPC=CPI=1.000000 with stddev 0 and no error from either tool."""
    assert ps._pmu_corrupted({"ipc": 1.0, "cpi": 1.0}) is True


def test_impossible_cpu_utilization_detected():
    """Real value from a contended collection: 56,874,283% utilization."""
    assert ps._pmu_corrupted({"cpu_utilization_pct": 56874283.5}) is True


def test_healthy_metrics_not_flagged():
    """Standalone perfspect on the same host: IPC 0.42, CPI 2.40."""
    assert ps._pmu_corrupted({"ipc": 0.421208, "cpi": 2.395828}) is False


def test_genuine_ipc_of_one_not_flagged():
    """A real IPC of 1.0 has CPI 1.0 too, so the pair alone is ambiguous —
    but a workload at exactly 1.000000/1.000000 is not physically plausible.
    Guard the near-miss case that must survive."""
    assert ps._pmu_corrupted({"ipc": 1.0, "cpi": 1.02}) is False


def test_missing_ipc_not_flagged():
    """--noroot collections omit some metrics; absence is not corruption."""
    assert ps._pmu_corrupted({}) is False


# ─── span-kind normalization (W1.5) ──────────────────────────────────

@pytest.mark.parametrize("kind", ["terminal-bench", "terminal_bench", "swe_bench"])
def test_collect_kinds_match_either_separator(probe, kind):
    assert _norm(kind) in probe(noroot=True)._collect_kinds


def test_synthetic_cpu_not_instrumented_by_default(probe):
    """Its tasks target 3s, far under the floor, so collecting on them is waste."""
    assert _norm("synthetic_cpu") not in probe(noroot=True)._collect_kinds


def test_synthetic_cpu_can_still_be_opted_into(probe):
    """Excluded by default, not forbidden — a long-running driver may want it."""
    p = probe(noroot=True, collect_kinds=["synthetic_cpu"])
    assert _norm("synthetic_cpu") in p._collect_kinds


def _norm(kind: str) -> str:
    return ps._norm_kind(kind)


# ─── spans too short to measure ──────────────────────────────────────

class _FakeProc:
    """A perfspect process that exits on request and records that it was asked."""

    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0


class _Clock:
    """A fake clock so the poll loop's deadline expires without real waiting."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _finalize(probe, monkeypatch, tmp_path, elapsed_s):
    """Finalize a span that ran ``elapsed_s`` seconds with no metric files."""
    p = probe(noroot=True)
    proc = _FakeProc()
    p._active_spans["compile"] = {
        "proc": proc, "dir": tmp_path, "kind": "terminal_bench",
        "node_id": None, "start_time": 1000.0,
    }
    clock = _Clock(1000.0 + elapsed_s)
    monkeypatch.setattr(ps.time, "time", clock.time)
    monkeypatch.setattr(ps.time, "sleep", clock.sleep)
    return p, proc, p.finalize_span("compile")


def test_span_with_no_metric_files_emits_no_record(probe, monkeypatch, tmp_path):
    """Zero records, not zeroed metrics.

    The perfspect layer must be *absent* rather than present-and-zero, so
    downstream analyzers see a missing layer instead of a fabricated reading.
    """
    _, proc, records = _finalize(probe, monkeypatch, tmp_path, 3.3)
    assert records == []
    # The collector must still be stopped, or it runs for its full --duration.
    assert proc.terminated is True


def test_short_span_warns_once_naming_the_span_as_unmeasured(
    probe, monkeypatch, tmp_path, caplog
):
    with caplog.at_level("WARNING"):
        _finalize(probe, monkeypatch, tmp_path, 3.3)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, "one cause should produce one warning"
    msg = warnings[0].getMessage()
    assert "UNMEASURED" in msg
    assert "compile" in msg
    assert "3.3s" in msg
    assert "Expected" in msg


def test_long_span_with_no_metrics_warns_that_it_is_unexpected(
    probe, monkeypatch, tmp_path, caplog
):
    """A span that clears the floor and still yields nothing is a real problem.

    Wording it the same as the short-span case would hide a privilege or
    PMU-contention failure behind an "expected" message.
    """
    with caplog.at_level("WARNING"):
        _finalize(probe, monkeypatch, tmp_path, 45.0)
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "unexpected" in msg
    assert "Expected for a span this short" not in msg


def test_duration_never_discards_a_span_before_parsing(probe, monkeypatch, tmp_path):
    """A short span whose metrics DID land must still be kept.

    The floor is an estimate that shifted by >1.5s between code paths on one
    host, so it must not veto data that actually exists.
    """
    (tmp_path / "host_metrics_summary.csv").write_text(
        "metric,mean,min,max,stddev\nIPC,1.5,1.5,1.5,0\n"
    )
    _, _, records = _finalize(probe, monkeypatch, tmp_path, 3.3)
    assert len(records) == 1, "metrics existed for a short span and were discarded"
    assert records[0].payload["ipc"] == 1.5


def test_min_span_is_derived_from_init_and_interval(probe):
    """The floor must track its causes, not be a hardcoded constant.

    Measured on perfspect 3.17.0: floor is 12s at interval=5 and 8s at
    interval=1, i.e. a fixed ~7s init plus one interval.
    """
    assert probe(noroot=True)._min_span_s == ps._INIT_OVERHEAD_S + 5
    assert probe(noroot=True, interval_s=1)._min_span_s == ps._INIT_OVERHEAD_S + 1


def test_measured_floor_matches_observed_boundaries(probe):
    """Guards the constant against drifting back to the old, too-low value of 10.

    A 11s span produced no metrics at interval=5 on the reference host, so the
    gate must reject it; the old `< 10` threshold let it through to a futile poll.
    """
    assert probe(noroot=True)._min_span_s == pytest.approx(12.0)
    assert probe(noroot=True, interval_s=1)._min_span_s == pytest.approx(8.0)


# ─── metric parsing: platform-conditional TMA, and NaN ───────────────

_P_CORE_CSV = """metric,mean,min,max,stddev
IPC,1.147424,1.147424,1.147424,0.000000
TMA_Backend_Bound(%),58.747156,58.747156,58.747156,0.000000
TMA_..Memory_Bound(%),16.285242,16.285242,16.285242,0.000000
TMA_......MEM_Bandwidth(%),NaN,NaN,NaN,NaN
memory bandwidth total (MB/sec),689.737600,689.737600,689.737600,0.000000
"""

# E-core parts emit no TMA rows at all (verified on Clearwater Forest).
_E_CORE_CSV = """metric,mean,min,max,stddev
IPC,1.147424,1.147424,1.147424,0.000000
memory bandwidth total (MB/sec),689.737600,689.737600,689.737600,0.000000
"""


def _parse(probe, tmp_path, csv_text):
    (tmp_path / "host_metrics_summary.csv").write_text(csv_text)
    return probe(noroot=True)._parse_metrics(tmp_path)


def test_memory_bound_tma_bucket_is_read_when_the_part_reports_it(probe, tmp_path):
    """The level-2 bucket was never mapped, so this signal was dead everywhere."""
    m = _parse(probe, tmp_path, _P_CORE_CSV)
    assert m["tma_memory_bound"] == pytest.approx(0.16285242)


def test_memory_bound_absent_on_parts_without_tma(probe, tmp_path):
    """Absence must come from the data, not from a hardcoded None.

    The key is simply missing, which reads downstream as "not measured" rather
    than as a measured value.
    """
    m = _parse(probe, tmp_path, _E_CORE_CSV)
    assert "tma_memory_bound" not in m
    assert m["ipc"] == pytest.approx(1.147424)  # the rest still parses


def test_nan_metrics_are_dropped_not_stored(probe, tmp_path):
    """float("NaN") does not raise, and a stored NaN fails every > silently.

    That is indistinguishable downstream from "measured and below threshold",
    which is the same fabrication as recording an uncounted event as 0.0.
    """
    m = _parse(probe, tmp_path, _P_CORE_CSV)
    assert not any(
        isinstance(v, float) and math.isnan(v) for v in m.values()
    ), "a NaN reached the payload"
    assert "tma_......mem_bandwidth(pct)" not in m


def test_interval_is_passed_explicitly(probe):
    """The interval sets the span floor, so it must not be left to the tool default."""
    cmd = probe(noroot=True, interval_s=1)._build_command(Path("/tmp/x"))
    assert "--interval" in cmd
    assert cmd[cmd.index("--interval") + 1] == "1"
