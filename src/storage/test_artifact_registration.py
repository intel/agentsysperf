#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Artifact registration: measurements declare bulk files, persist_run stores them.

Covers the gap that made the dashboard's EMON tab fall back to globbing
hardcoded /tmp dirs — `store_artifact` existed but had no callers, so the
`artifacts` table was permanently empty and the store-first lookup in
`dashboard_data.get_artifact_path` always missed.

Run: poetry run pytest src/storage/test_artifact_registration.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.protocols import MeasurementRecord, NullMeasurement
from src.storage.sqlite_store import SQLiteResultStore


class _ArtifactMeasurement(NullMeasurement):
    """A measurement that declares one on-disk file."""

    name = "fake_emon"
    layer = "emon"

    def __init__(self, path: Path | None, *, kind: str = "emon_csv") -> None:
        self._path = path
        self._kind = kind

    def artifacts(self):
        if self._path is None:
            return ()
        return [{"kind": self._kind, "name": self._path.name,
                 "path": str(self._path)}]


def test_persist_run_stores_artifacts(tmp_path):
    csv = tmp_path / "emon_run_system_view_details.csv"
    csv.write_text("#sample,timestamp\n1,0.0\n")
    s = SQLiteResultStore(tmp_path / "store")
    s.persist_run(
        run_id="r", metadata={"start_time": 1, "benchmark_id": "terminal-bench"},
        artifacts=[{"kind": "emon_csv", "name": csv.name, "path": str(csv)}],
    )
    assert s.get_artifact_path("r", kind="emon_csv") == csv


def test_artifacts_roll_back_with_the_rest_of_the_run(tmp_path):
    """Artifacts are inside persist_run's transaction, not committed separately."""
    csv = tmp_path / "a.csv"
    csv.write_text("x")
    s = SQLiteResultStore(tmp_path / "store")
    bad = [MeasurementRecord(span_id="r::t", layer="l1", payload={"x": object()})]
    with pytest.raises(Exception):
        s.persist_run(
            run_id="r", metadata={"start_time": 1}, records=bad,
            artifacts=[{"kind": "emon_csv", "name": csv.name, "path": str(csv)}],
        )
    conn = s._get_connection()
    assert conn.execute("SELECT count(*) FROM artifacts WHERE run_id='r'").fetchone()[0] == 0


def test_runcontext_registers_measurement_artifacts(tmp_path):
    """The end-to-end path: a measurement declares, RunContext persists."""
    from src.runner import RunContext

    csv = tmp_path / "metrics.csv"
    csv.write_text("#sample\n1\n")
    store = SQLiteResultStore(tmp_path / "store")
    ctx = RunContext(
        output_dir=tmp_path / "run", run_id="ctxrun", result_store=store,
        measurements=[_ArtifactMeasurement(csv)],
        run_metadata={"benchmark_id": "synthetic", "start_time": 7},
    )
    with ctx:
        pass
    assert store.get_artifact_path("ctxrun", kind="emon_csv") == csv


def test_missing_file_is_not_registered(tmp_path):
    """A declared path that isn't on disk buys nothing — readers filter on
    existence, so a dead row would just be noise."""
    from src.runner import RunContext

    store = SQLiteResultStore(tmp_path / "store")
    ctx = RunContext(
        output_dir=tmp_path / "run", run_id="ctxrun", result_store=store,
        measurements=[_ArtifactMeasurement(tmp_path / "never_written.csv")],
        run_metadata={"start_time": 7},
    )
    with ctx:
        pass
    conn = store._get_connection()
    assert conn.execute(
        "SELECT count(*) FROM artifacts WHERE run_id='ctxrun'"
    ).fetchone()[0] == 0


def test_measurement_without_artifacts_method_is_fine(tmp_path):
    """Declaring artifacts is opt-in; the other four layers don't implement it."""
    from src.runner import RunContext

    store = SQLiteResultStore(tmp_path / "store")
    ctx = RunContext(
        output_dir=tmp_path / "run", run_id="ctxrun", result_store=store,
        measurements=[NullMeasurement()], run_metadata={"start_time": 7},
    )
    with ctx:
        pass
    assert store.get_run("ctxrun") is not None


def test_artifacts_failure_does_not_kill_the_run(tmp_path):
    """A plugin raising in artifacts() must not lose the run's measurements."""
    from src.runner import RunContext

    class _Exploding(NullMeasurement):
        name = "boom"

        def artifacts(self):
            raise RuntimeError("post-processing never ran")

    store = SQLiteResultStore(tmp_path / "store")
    ctx = RunContext(
        output_dir=tmp_path / "run", run_id="ctxrun", result_store=store,
        measurements=[_Exploding()], run_metadata={"start_time": 7},
    )
    with ctx:
        pass
    assert store.get_run("ctxrun") is not None


def test_reregistering_same_artifact_upserts(tmp_path):
    """UNIQUE(run_id, kind, name) — a re-run updates the path, not duplicates."""
    s = SQLiteResultStore(tmp_path / "store")
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    for p in (first, second):
        p.write_text("x")
    s.persist_run(run_id="r", metadata={"start_time": 1},
                  artifacts=[{"kind": "emon_csv", "name": "m.csv", "path": str(first)}])
    s.persist_run(run_id="r", metadata={"start_time": 1},
                  artifacts=[{"kind": "emon_csv", "name": "m.csv", "path": str(second)}])
    conn = s._get_connection()
    assert conn.execute("SELECT count(*) FROM artifacts WHERE run_id='r'").fetchone()[0] == 1
    assert s.get_artifact_path("r", kind="emon_csv") == second
