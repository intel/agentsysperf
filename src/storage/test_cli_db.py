#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P9 verification: `agentsysperf db ls/show/rm` against the canonical store.

Run: poetry run pytest src/storage/test_cli_db.py -q
"""
from __future__ import annotations

from typer.testing import CliRunner

from src.cli import app
from src.protocols import MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore

runner = CliRunner()


def _seed(home):
    store = SQLiteResultStore.open()
    store.store_run_metadata(run_id="run_a", metadata={
        "start_time": 100, "benchmark_id": "terminal-bench", "model": "gpt-4o-mini",
        "total_tasks": 2, "passed_tasks": 1, "owner_id": "testuser"})
    store.store_task_result(run_id="run_a", task_id="t1", result={"passed": True})
    store.store_measurements(run_id="run_a", records=[
        MeasurementRecord(span_id="run_a::t1", layer="l1", payload={"ipc": 1.0})])
    store._get_connection().commit()
    store.close()


def test_db_ls(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    _seed(tmp_path)
    res = runner.invoke(app, ["db", "ls"])
    assert res.exit_code == 0, res.output
    assert "run_a" in res.output and "terminal-bench" in res.output


def test_db_ls_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "empty"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    res = runner.invoke(app, ["db", "ls"])
    assert res.exit_code == 0
    assert "No runs" in res.output


def test_db_show(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    _seed(tmp_path)
    res = runner.invoke(app, ["db", "show", "run_a"])
    assert res.exit_code == 0, res.output
    assert "terminal-bench" in res.output
    assert "'l1': 1" in res.output or "l1" in res.output


def test_db_show_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    _seed(tmp_path)
    res = runner.invoke(app, ["db", "show", "nope"])
    assert res.exit_code == 1


def test_db_rm_with_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    _seed(tmp_path)
    res = runner.invoke(app, ["db", "rm", "run_a", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Deleted 1" in res.output
    # gone + cascaded
    store = SQLiteResultStore.open()
    assert store.get_run("run_a") is None
    assert store.query_measurements("run_a") == []


def test_db_rm_aborts_without_confirm(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    _seed(tmp_path)
    res = runner.invoke(app, ["db", "rm", "run_a"], input="n\n")
    assert res.exit_code != 0  # aborted
    store = SQLiteResultStore.open()
    assert store.get_run("run_a") is not None, "run deleted despite abort"
