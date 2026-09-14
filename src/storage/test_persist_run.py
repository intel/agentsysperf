#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P5 verification: atomic persist_run, rollback-on-failure, dual-write via
RunContext, and the no-churn default (no store -> no DB write).

Run: poetry run pytest src/storage/test_persist_run.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.protocols import AnalysisResult, MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore


def _records():
    return [
        MeasurementRecord(span_id="r::t1", layer="l1", payload={"ipc": 1.0, "duration_us": 5}),
        MeasurementRecord(span_id="r::t1", layer="l3", payload={"cache_miss_pct": 30.0}),
    ]


def test_persist_run_writes_everything_atomically(tmp_path):
    s = SQLiteResultStore(tmp_path)
    s.persist_run(
        run_id="r",
        metadata={"start_time": 1, "benchmark_id": "terminal-bench"},
        task_results=[("t1", {"passed": True, "duration_s": 2.0, "elapsed_s": 1.5})],
        records=_records(),
        verdicts=[AnalysisResult(analyzer_name="cpu_bound", verdict="core_bound",
                                 confidence=0.9, evidence={}, recommendations=[], span_id="t1")],
    )
    conn = s._get_connection()
    assert conn.execute("SELECT count(*) FROM runs WHERE run_id='r'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM task_results WHERE run_id='r'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM measurements WHERE run_id='r'").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM analyzer_verdicts WHERE run_id='r'").fetchone()[0] == 1
    # logical_task_key resolved from the run's benchmark_id within the same txn
    key = conn.execute("SELECT logical_task_key FROM task_results WHERE run_id='r'").fetchone()[0]
    assert key == "terminal-bench::t1"
    elapsed = conn.execute(
        "SELECT elapsed_s FROM task_results WHERE run_id='r' AND task_id='t1'"
    ).fetchone()[0]
    assert elapsed == 1.5


def test_persist_run_rolls_back_on_failure(tmp_path):
    s = SQLiteResultStore(tmp_path)
    # A measurement whose payload can't be JSON-serialized fails mid-flush AFTER
    # run metadata was written within the transaction -> must roll back fully.
    bad = [MeasurementRecord(span_id="r::t", layer="l1", payload={"x": object()})]
    with pytest.raises(Exception):
        s.persist_run(run_id="r", metadata={"start_time": 1},
                     task_results=[("t", {"passed": True})], records=bad)
    conn = s._get_connection()
    # nothing partially committed: not even the run row
    assert conn.execute("SELECT count(*) FROM runs WHERE run_id='r'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM task_results WHERE run_id='r'").fetchone()[0] == 0


def test_in_transaction_flag_reset_after_failure(tmp_path):
    s = SQLiteResultStore(tmp_path)
    with pytest.raises(Exception):
        s.persist_run(run_id="r", metadata={"start_time": 1},
                     records=[MeasurementRecord(span_id="x", layer="l1", payload={"x": object()})])
    assert s._in_transaction is False, "transaction flag leaked after failure"
    # store still usable afterward (standalone write commits normally)
    s.store_run_metadata(run_id="ok", metadata={"start_time": 2})
    assert s.get_run("ok") is not None


def test_runcontext_dualwrite(tmp_path):
    from src.runner import RunContext
    store = SQLiteResultStore(tmp_path / "store")
    out = tmp_path / "run"
    ctx = RunContext(output_dir=out, run_id="ctxrun", result_store=store,
                     run_metadata={"benchmark_id": "synthetic", "start_time": 7})
    with ctx:
        # emit a record directly (no measurement plugin needed for the dual-write test)
        ctx._records.append(MeasurementRecord(span_id="ctxrun::a", layer="l1", payload={"ipc": 2.0}))
    # JSON written
    assert (out / "measurement_records.json").exists()
    data = json.loads((out / "measurement_records.json").read_text())
    assert any(r["span_id"] == "ctxrun::a" for r in data)
    # AND persisted to the store
    got = store.query_measurements("ctxrun")
    assert len(got) == 1 and got[0]["payload"] == {"ipc": 2.0}
    assert store.get_run("ctxrun")["benchmark_id"] == "synthetic"


def test_runcontext_no_store_does_not_persist(tmp_path):
    """live_dashboard builds RunContext WITHOUT result_store — must not touch
    any canonical store (no churn)."""
    from src.runner import RunContext
    out = tmp_path / "run"
    ctx = RunContext(output_dir=out, run_id="noStore")
    with ctx:
        ctx._records.append(MeasurementRecord(span_id="noStore::a", layer="l1", payload={"ipc": 1.0}))
    # JSON still written (unchanged behavior)...
    assert (out / "measurement_records.json").exists()
    # ...but NO db created anywhere under the run dir
    assert not list(out.glob("*.db")), "a store DB was created despite result_store=None"
