#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P3 verification: composite task PK, benchmarks, provenance, discovery,
deletion cascades, and verdict-upsert idempotency.

Run: poetry run pytest src/storage/test_p3_schema.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.protocols import AnalysisResult, MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore


def _store(tmp_path) -> SQLiteResultStore:
    return SQLiteResultStore(tmp_path)


# ── composite task_id PK (the bug fix) ────────────────────────────────────

def test_same_task_id_survives_across_runs(tmp_path):
    s = _store(tmp_path)
    for rid in ("runA", "runB"):
        s.store_run_metadata(run_id=rid, metadata={"start_time": 0})
        s.store_task_result(run_id=rid, task_id="shared-task",
                            result={"passed": True, "duration_s": 1.0})
    rows = s._get_connection().execute(
        "SELECT run_id FROM task_results WHERE task_id='shared-task'").fetchall()
    assert {r[0] for r in rows} == {"runA", "runB"}, "composite PK didn't keep both runs"


def test_logical_task_key_built_from_benchmark(tmp_path):
    s = _store(tmp_path)
    s.store_benchmark(benchmark_id="terminal-bench", metadata={"display_name": "TB"})
    s.store_run_metadata(run_id="r1", metadata={"start_time": 0, "benchmark_id": "terminal-bench"})
    s.store_task_result(run_id="r1", task_id="t1", result={"passed": True})
    key = s._get_connection().execute(
        "SELECT logical_task_key FROM task_results WHERE run_id='r1' AND task_id='t1'"
    ).fetchone()[0]
    assert key == "terminal-bench::t1"


# ── benchmarks + provenance ───────────────────────────────────────────────

def test_benchmarks_and_list(tmp_path):
    s = _store(tmp_path)
    s.store_benchmark(benchmark_id="tau-bench", metadata={"display_name": "Tau"})
    s.store_run_metadata(run_id="r", metadata={"start_time": 1, "benchmark_id": "tau-bench"})
    names = {b["benchmark_id"] for b in s.list_benchmarks()}
    assert "tau-bench" in names


def test_provenance_columns_persist(tmp_path):
    s = _store(tmp_path)
    s.store_run_metadata(run_id="r", metadata={
        "start_time": 1, "owner_id": "testuser", "owner_kind": "os_user",
        "host_id": "gnr-node", "optimization_profile": "amx_bf16",
        "hardware_sku": "Granite Rapids test-sku",
    })
    run = s.get_run("r")
    assert run["owner_id"] == "testuser"
    assert run["host_id"] == "gnr-node"
    assert run["optimization_profile"] == "amx_bf16"
    assert run["hardware_sku"] == "Granite Rapids test-sku"


def test_agentsysperf_version_persists(tmp_path):
    """Regression: migration 0004 shipped the column as ``agentperf_version``
    but store_run_metadata writes ``agentsysperf_version``, so every
    persist_run rolled back with "table runs has no column named
    agentsysperf_version". Migration 0007 renames it; this pins the name.
    """
    s = _store(tmp_path)
    s.store_run_metadata(run_id="r", metadata={
        "start_time": 1, "agentsysperf_version": "0.1.0"})
    assert s.get_run("r")["agentsysperf_version"] == "0.1.0"


def test_list_runs_newest_first_and_owner_scope(tmp_path):
    s = _store(tmp_path)
    s.store_run_metadata(run_id="old", metadata={"start_time": 100, "owner_id": "me"})
    s.store_run_metadata(run_id="new", metadata={"start_time": 200, "owner_id": "you"})
    ids = [r["run_id"] for r in s.list_runs()]
    assert ids[0] == "new", "list_runs not newest-first"
    mine = [r["run_id"] for r in s.list_runs(owner_id="me")]
    assert mine == ["old"], "owner scoping wrong"


def test_latest_run(tmp_path):
    s = _store(tmp_path)
    s.store_run_metadata(run_id="r1", metadata={"start_time": 10, "benchmark_id": "b"})
    s.store_run_metadata(run_id="r2", metadata={"start_time": 20, "benchmark_id": "b"})
    assert s.latest_run(benchmark_id="b") == "r2"


# ── deletion cascades ─────────────────────────────────────────────────────

def test_delete_run_cascades_all_children(tmp_path):
    s = _store(tmp_path)
    s.store_run_metadata(run_id="r", metadata={"start_time": 0})
    s.store_task_result(run_id="r", task_id="t", result={"passed": True})
    s.store_measurements(run_id="r", records=[
        MeasurementRecord(span_id="r::t", layer="l1", payload={"ipc": 1.0})])
    s.store_analysis_results(run_id="r", results=[
        AnalysisResult(analyzer_name="cpu_bound", verdict="core_bound",
                       confidence=0.9, evidence={}, recommendations=[], span_id="t")])
    assert s.delete_run("r") == 1
    conn = s._get_connection()
    for tbl in ("task_results", "measurements", "analyzer_verdicts"):
        n = conn.execute(f"SELECT count(*) FROM {tbl} WHERE run_id='r'").fetchone()[0]
        assert n == 0, f"{tbl} not cascaded on delete_run"


def test_delete_benchmark_removes_its_runs(tmp_path):
    s = _store(tmp_path)
    s.store_benchmark(benchmark_id="b", metadata={})
    s.store_run_metadata(run_id="r", metadata={"start_time": 0, "benchmark_id": "b"})
    s.store_task_result(run_id="r", task_id="t", result={"passed": True})
    assert s.delete_benchmark("b") == 1
    assert s.get_run("r") is None


# ── verdict upsert idempotency (UNIQUE constraint) ────────────────────────

def test_verdict_upsert_not_duplicated(tmp_path):
    s = _store(tmp_path)
    s.store_run_metadata(run_id="r", metadata={"start_time": 0})
    v = AnalysisResult(analyzer_name="cache", verdict="l3_resident",
                       confidence=0.8, evidence={}, recommendations=[], span_id="t")
    s.store_analysis_results(run_id="r", results=[v])
    s.store_analysis_results(run_id="r", results=[v])  # re-run analyzer
    n = s._get_connection().execute(
        "SELECT count(*) FROM analyzer_verdicts WHERE run_id='r'").fetchone()[0]
    assert n == 1, "verdict duplicated on re-store"
