#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the shell-runner -> store sweep import.

The load-bearing behaviours, in the order they matter:
  * the density basis is the CPUSET size, not the machine core count,
  * the agent name survives (ADR 0001: unrecorded agent = uncomparable sweep),
  * failed cells are refused, not persisted as zero-throughput points,
  * absent node telemetry stays NULL rather than becoming a fabricated 0.0.

Run: poetry run pytest src/sweep/test_import_points.py -q
"""
from __future__ import annotations

import json

import pytest

from src.storage.sqlite_store import SQLiteResultStore
from src.sweep.import_points import import_sweep

PROVENANCE = """\
date: 2026-08-01T00:00:00+00:00
host: testhost
model_name: Intel(R) Processor
cores_total: 288
node0_cpulist: 0-95
cpuset_used: 0-15  (16 cores)  [SUB-RANGE OVERRIDE]
governor: powersave
sep_driver_loaded: yes
agent: oracle (solve.sh, no LLM)
task: overfull-hbox
"""


def _cell(d, n, rep=1, *, status="ok", telemetry=False, cores=16, extra=None):
    """Write one n{N}_r{R}/point.json shaped like the runner emits."""
    p = d / f"n{n}_r{rep}"
    p.mkdir(parents=True, exist_ok=True)
    point = {
        "concurrency_requested": n,
        "concurrency_reachable": n,
        "replicate": rep,
        "numa_node": 0,
        "cpuset": "0-15",
        "cores_in_cpuset": cores,
        "agents_per_core": round(n / cores, 4),
        "elapsed_s": 100.0,
        "harbor_returncode": 0,
        "cell_status": status,
        "expected_trials": n,
        "completed_trials": n,
        "trials_found": n,
        "mean_reward": 1.0,
        "throughput_per_min": 0.6 * n,
        "p50_agent_exec_s": 50.0,
        "p95_agent_exec_s": 55.0,
        "ipc": 2.5,
        "cache_miss_pct": 20.0,
        "instructions_per_trial": 3.7e11,
        "counter_enabled_pct_min": 47.0,
        "counters_multiplexed": True,
        "trials": [{"task": "t", "reward": 1.0}] * n,
        "counters": {"cycles": 1e11},
    }
    if telemetry:
        point.update({
            "cpu_avg": 88.5, "cpu_p95": 96.0, "cpu_peak": 99.1,
            "iowait_pct_avg": 0.02, "ctx_sw_per_s": 50000.0,
            "runqueue_max": 40.0, "mem_avail_mb_min": 700000.0,
            "cpu_scope": "cpuset",
        })
    if extra:
        point.update(extra)
    (p / "point.json").write_text(json.dumps(point))
    return p


@pytest.fixture
def sweep_dir(tmp_path):
    d = tmp_path / "density_run"
    d.mkdir()
    (d / "provenance.txt").write_text(PROVENANCE)
    return d


def _store(tmp_path):
    return SQLiteResultStore(tmp_path / "store")


def test_basis_is_the_cpuset_not_the_machine(sweep_dir, tmp_path):
    """16 agents on a 16-core cpuset is density 1.0, not 16/288 = 0.056."""
    for n in (1, 8, 16):
        _cell(sweep_dir, n)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)

    assert r["vcpu_basis"] == 16
    sw = s.query_sweeps()[0]
    assert sw["vcpu_basis"] == 16
    assert sw["vcpu_basis_kind"] == "cpuset_cores"
    pts = {p["concurrency"]: p["density"] for p in s.query_sweep_points(r["sweep_id"])}
    assert pts[16] == pytest.approx(1.0)
    assert pts[8] == pytest.approx(0.5)


def test_agent_is_recorded(sweep_dir, tmp_path):
    """ADR 0001: a sweep whose agent is unknown cannot be compared to any other."""
    _cell(sweep_dir, 4)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)

    assert r["agent"] == "oracle"           # parsed out of "oracle (solve.sh, no LLM)"
    sw = s.query_sweeps()[0]
    assert sw["model"] == "oracle"
    assert sw["metadata"]["agent"] == "oracle"
    assert s.query_sweep_points(r["sweep_id"])[0]["metadata"]["agent"] == "oracle"


def test_missing_provenance_yields_unknown_agent_not_a_crash(sweep_dir, tmp_path):
    (sweep_dir / "provenance.txt").unlink()
    _cell(sweep_dir, 4)
    r = import_sweep(sweep_dir, store=_store(tmp_path), stamp=1)
    assert r["agent"] == "unknown"


def test_failed_cells_are_refused(sweep_dir, tmp_path):
    """A failed cell has a wrong throughput denominator — not an operating point."""
    _cell(sweep_dir, 4)
    _cell(sweep_dir, 8, status="failed")
    _cell(sweep_dir, 16, status="trial_count_mismatch")
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)

    assert r["imported"] == 1
    assert {x["cell"] for x in r["skipped"]} == {"n8_r1", "n16_r1"}
    assert [p["concurrency"] for p in s.query_sweep_points(r["sweep_id"])] == [4]


def test_incomplete_trials_cell_is_refused(sweep_dir, tmp_path):
    """A cell where every result.json landed but a trial RAISED is not usable.

    Regression: the runner's gate tested only `len(trials) == n`, so a cell with
    24 result.json files of which one carried exception_info read as `ok`.
    Measured at n=24: one trial hung 750 s (vs ~50 s), pushing elapsed_s to 774 s
    and collapsing throughput 10.4 -> 1.78/min. Throughput is completed/elapsed,
    so the failed trial's wall time is in the denominator while it contributes
    nothing to the numerator — a fabricated operating point.
    """
    _cell(sweep_dir, 4)
    _cell(sweep_dir, 24, status="incomplete_trials",
          extra={"completed_trials": 23, "throughput_per_min": 1.782,
                 "elapsed_s": 774.58, "mean_reward": 0.9583})
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)

    assert r["imported"] == 1
    assert [x["cell"] for x in r["skipped"]] == ["n24_r1"]
    assert [p["concurrency"] for p in s.query_sweep_points(r["sweep_id"])] == [4]


def test_include_failed_overrides(sweep_dir, tmp_path):
    _cell(sweep_dir, 4)
    _cell(sweep_dir, 8, status="failed")
    r = import_sweep(sweep_dir, store=_store(tmp_path), stamp=1, include_failed=True)
    assert r["imported"] == 2


def test_all_cells_failed_raises(sweep_dir, tmp_path):
    _cell(sweep_dir, 4, status="failed")
    with pytest.raises(ValueError, match="no usable cells"):
        import_sweep(sweep_dir, store=_store(tmp_path), stamp=1)


def test_absent_telemetry_stays_null(sweep_dir, tmp_path):
    """A fabricated 0.0 would read as 'no CPU pressure' to the analyzer."""
    _cell(sweep_dir, 8, telemetry=False)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    p = s.query_sweep_points(r["sweep_id"])[0]
    for col in ("cpu_avg", "cpu_p95", "cpu_peak", "runqueue_max",
                "ctx_sw_per_s", "mem_avail_mb_min", "iowait_pct_avg"):
        assert p[col] is None, f"{col} should be NULL, got {p[col]!r}"


def test_telemetry_is_passed_through_when_present(sweep_dir, tmp_path):
    _cell(sweep_dir, 8, telemetry=True)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    p = s.query_sweep_points(r["sweep_id"])[0]
    assert p["cpu_avg"] == pytest.approx(88.5)
    assert p["cpu_peak"] == pytest.approx(99.1)
    assert p["runqueue_max"] == pytest.approx(40.0)


def test_mixed_cpuset_sizes_rejected(sweep_dir, tmp_path):
    """One sweep must share one density basis, or the x-axis is meaningless."""
    _cell(sweep_dir, 4, cores=16)
    _cell(sweep_dir, 8, cores=96)
    with pytest.raises(ValueError, match="cores_in_cpuset"):
        import_sweep(sweep_dir, store=_store(tmp_path), stamp=1)


def test_bulky_keys_are_dropped_from_metadata(sweep_dir, tmp_path):
    _cell(sweep_dir, 4)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    meta = s.query_sweep_points(r["sweep_id"])[0]["metadata"]
    assert "trials" not in meta and "counters" not in meta
    # ...but the derived hardware summary survives.
    assert meta["ipc"] == pytest.approx(2.5)
    assert meta["counters_multiplexed"] is True


def test_run_rows_exist_for_fk_targets(sweep_dir, tmp_path):
    """sweep_points.run_id and the verdict both reference runs(run_id)."""
    _cell(sweep_dir, 4)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    assert s.get_run(r["sweep_id"]) is not None
    for p in s.query_sweep_points(r["sweep_id"]):
        assert s.get_run(p["run_id"]) is not None


def test_empty_dir_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        import_sweep(tmp_path / "empty", store=_store(tmp_path), stamp=1)


def test_scaling_analyzer_consumes_the_imported_rows(sweep_dir, tmp_path):
    """End-to-end: the point of the import is that the analyzer can read it."""
    from src.analyzers.scaling import ScalingAnalyzer

    for n in (1, 2, 4, 8, 16, 24):
        # Throughput that flattens, so a knee exists to find.
        _cell(sweep_dir, n, telemetry=True,
              extra={"throughput_per_min": 0.65 * n * (1.0 if n <= 8 else 8.0 / n * 1.3)})
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    res = ScalingAnalyzer().analyze_sweep(
        s.query_sweep_points(r["sweep_id"]), logical_cpus=16,
    )
    assert res is not None
    assert "knee" in res.evidence


def test_import_stores_a_scaling_verdict(sweep_dir, tmp_path):
    """The import must leave a verdict behind, not just points.

    The Python runner (sweep/harbor_sweep.py) calls analyze_sweep itself; this
    shell-runner bridge did not, so imported sweeps landed with zero verdicts and
    the dashboard drew no knee marker and no bottleneck label. No CLI backfills
    it: `agentsysperf analyze` reads a directory of MeasurementRecords, prints,
    and never touches sweep_points.
    """
    for n in (1, 2, 4, 8, 16, 24):
        _cell(sweep_dir, n, telemetry=True,
              extra={"throughput_per_min": 0.65 * n * (1.0 if n <= 8 else 8.0 / n * 1.3)})
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)

    assert r["verdict"], "import returned no verdict"
    verdicts = s.query_verdicts(r["sweep_id"], analyzer_name="scaling")
    assert len(verdicts) == 1, "the verdict must be persisted under the sweep_id"
    assert "knee" in verdicts[0]["evidence"]


def test_verdict_uses_the_host_basis_when_runqueue_is_host_wide(sweep_dir, tmp_path):
    """runqueue_max must be divided by the scope it was sampled at.

    The runner samples a host-wide runqueue. Dividing a 288-CPU box's runqueue by
    a 16-core cpuset declares scheduler_oversubscription on a machine that is
    ~94% idle, so the basis follows runqueue_is_host_wide.
    """
    for n in (1, 2, 4, 8, 16, 24):
        _cell(sweep_dir, n, telemetry=True, extra={
            "throughput_per_min": 0.65 * n * (1.0 if n <= 8 else 8.0 / n * 1.3),
            # 40 > 1.5x16 (cpuset) but far below 1.5x288 (host).
            "runqueue_max": 40.0,
            "runqueue_is_host_wide": True,
            "logical_cpus_host": 288,
        })
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)

    ev = s.query_verdicts(r["sweep_id"], analyzer_name="scaling")[0]["evidence"]
    assert ev.get("bottleneck") != "scheduler_oversubscription", (
        "a host-wide runqueue of 40 on 288 CPUs is not oversubscription; "
        "the cpuset size was used as the basis"
    )


def test_import_survives_an_unanalyzable_sweep(sweep_dir, tmp_path):
    """A sweep too small to analyze must still import its points."""
    _cell(sweep_dir, 1, telemetry=True)  # one density -> insufficient_data
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    assert r["imported"] == 1, "analysis trouble must not lose the import"


def test_import_registers_cell_evidence_as_artifacts(sweep_dir, tmp_path):
    """The raw perf/vmstat/telemetry files must be addressable by run_id.

    The artifacts table was empty for every run ever stored, so
    dashboard_data.get_artifact_path() always fell through to hardcoded /tmp
    globs — 22 of which were dead paths, and all of which vanish on reboot.
    """
    cell = _cell(sweep_dir, 4, telemetry=True)
    (cell / "perf.csv").write_text("# counters\n")
    (cell / "vmstat.txt").write_text("procs\n 1 0 0\n")
    (cell / "telemetry.json").write_text("{}")

    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    run_id = f"{r['sweep_id']}::n4_r1"

    assert s.get_artifact_path(run_id, kind="perf_csv") is not None
    assert s.get_artifact_path(run_id, kind="vmstat_txt") is not None
    # point.json itself is registered, so a rollup can be traced to its source.
    assert s.get_artifact_path(run_id, kind="sweep_point_json") is not None


def test_missing_evidence_files_are_skipped_not_fatal(sweep_dir, tmp_path):
    """A sweep predating a collector simply has fewer files; that is not an error."""
    _cell(sweep_dir, 2, telemetry=True)  # point.json only, no perf/vmstat
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    assert r["imported"] == 1
    run_id = f"{r['sweep_id']}::n2_r1"
    assert s.get_artifact_path(run_id, kind="perf_csv") is None
    assert s.get_artifact_path(run_id, kind="sweep_point_json") is not None


def test_cell_dir_marker_does_not_leak_into_the_point(sweep_dir, tmp_path):
    """__cell_dir__ is importer bookkeeping and must not become a stored field."""
    _cell(sweep_dir, 2, telemetry=True)
    s = _store(tmp_path)
    r = import_sweep(sweep_dir, store=s, stamp=1)
    point = s.query_sweep_points(r["sweep_id"])[0]
    assert "__cell_dir__" not in point
    import json as _json
    md = point.get("metadata")
    md = _json.loads(md) if isinstance(md, str) else (md or {})
    assert "__cell_dir__" not in md, "must not survive into metadata either"
