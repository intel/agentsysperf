#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P0 + P1 verification: migration runner, connection hardening, and
measurement round-trip fidelity against REAL docs/* payloads.

Run: poetry run pytest src/storage/test_measurements_roundtrip.py -q
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from src.protocols import MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_RECORDS = sorted(glob.glob(str(REPO_ROOT / "docs" / "*" / "measurement_records.json")))


def _seed_run(store, run_id):
    """measurements.run_id FKs to runs.run_id (ON DELETE CASCADE), and P0 turns
    foreign_keys ON — so a run row must exist before its measurements. Mirrors
    real runners, which always store_run_metadata first."""
    store.store_run_metadata(run_id=run_id, metadata={"start_time": 0})


# ── P0: schema versioning + connection hardening ──────────────────────────

def test_p0_pragmas_and_version(tmp_path):
    store = SQLiteResultStore(tmp_path)
    conn = store._get_connection()
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 1, "migrations didn't set user_version"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1, "FK enforcement off"
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal", "not in WAL mode"


def test_p0_measurements_table_created(tmp_path):
    store = SQLiteResultStore(tmp_path)
    conn = store._get_connection()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(measurements)").fetchall()}
    assert {"run_id", "span_id", "layer", "task_id", "payload", "seq"} <= cols


def test_p0_task_results_include_elapsed_s(tmp_path):
    store = SQLiteResultStore(tmp_path)
    cols = {row[1] for row in store._get_connection().execute(
        "PRAGMA table_info(task_results)"
    )}
    assert "elapsed_s" in cols


def test_p0_migration_adds_elapsed_s_to_v8_database(tmp_path):
    db_path = tmp_path / "legacy.db"
    store = SQLiteResultStore(db_path=db_path)
    connection = store._get_connection()
    connection.executescript(
        """
        CREATE TABLE task_results_v8 (
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            logical_task_key TEXT,
            workload_type TEXT,
            passed BOOLEAN NOT NULL,
            duration_s REAL,
            num_turns INTEGER,
            num_commands INTEGER,
            PRIMARY KEY (run_id, task_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        INSERT INTO task_results_v8
            (run_id, task_id, logical_task_key, workload_type, passed,
             duration_s, num_turns, num_commands)
        SELECT run_id, task_id, logical_task_key, workload_type, passed,
               duration_s, num_turns, num_commands
        FROM task_results;
        DROP TABLE task_results;
        ALTER TABLE task_results_v8 RENAME TO task_results;
        PRAGMA user_version = 8;
        """
    )
    connection.commit()
    store.close()

    upgraded = SQLiteResultStore(db_path=db_path)
    cols = {row[1] for row in upgraded._get_connection().execute(
        "PRAGMA table_info(task_results)"
    )}
    assert "elapsed_s" in cols
    assert upgraded._get_connection().execute(
        "PRAGMA user_version"
    ).fetchone()[0] == 9


def test_p0_read_only_does_not_create_or_migrate(tmp_path):
    # Opening a non-existent DB read-only must NOT create the file or any tables.
    store = SQLiteResultStore(tmp_path, read_only=True)
    assert not (tmp_path / "agentsysperf_results.db").exists() or \
        store._get_connection().execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0] == 0
    # A read-only store must refuse to find a missing table (no DDL ran).
    with pytest.raises(Exception):
        store._get_connection().execute("SELECT * FROM measurements").fetchall()


def test_p0_migrations_idempotent(tmp_path):
    SQLiteResultStore(tmp_path).close()
    v1 = SQLiteResultStore(tmp_path)
    conn = v1._get_connection()
    # Re-opening an already-migrated DB must not error and must hold the version.
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 2  # 0002 applied


# ── P1: store_measurements / query_measurements round-trip ────────────────

def test_p1_roundtrip_synthetic(tmp_path):
    store = SQLiteResultStore(tmp_path)
    _seed_run(store, "run")
    recs = [
        MeasurementRecord(span_id="run::taskA", layer="l1",
                          payload={"kind": "task", "duration_us": 1234, "cpu_time_s": 0.5,
                                   "cpu_pct_mean": 88.0, "rss_kb_peak": 4096, "node_id": "n0"}),
        MeasurementRecord(span_id="run::taskA", layer="l3",
                          payload={"ipc": 1.65, "cache_miss_pct": 97.0, "events": {"x": 1}}),
        MeasurementRecord(span_id="bench/t/turn_0", layer="perfspect",
                          payload={"frontend_bound": 0.1, "backend_bound": 0.5, "scope": "system"}),
    ]
    store.store_measurements(run_id="run", records=recs)
    got = store.query_measurements("run")
    assert len(got) == 3, "row count mismatch"
    # payload byte-equal (dict equality) and ordering preserved by seq
    assert [g["payload"] for g in got] == [r.payload for r in recs]
    # task_id derivation matches the exporter rule
    assert got[0]["task_id"] == "taskA"            # had '::'
    assert got[2]["task_id"] == "bench/t/turn_0"   # no '::', whole string


def test_p1_null_not_zero(tmp_path):
    # A record WITHOUT ipc must store NULL in the promoted column, not 0.
    store = SQLiteResultStore(tmp_path)
    _seed_run(store, "r")
    store.store_measurements(run_id="r", records=[
        MeasurementRecord(span_id="r::a", layer="l1", payload={"duration_us": 10})])
    val = store._get_connection().execute(
        "SELECT ipc FROM measurements WHERE run_id='r'").fetchone()[0]
    assert val is None, "missing key must be NULL, not 0"


def test_p1_upsert_idempotent(tmp_path):
    store = SQLiteResultStore(tmp_path)
    _seed_run(store, "r")
    rec = MeasurementRecord(span_id="r::a", layer="l1", payload={"ipc": 1.0})
    store.store_measurements(run_id="r", records=[rec])
    store.store_measurements(run_id="r", records=[rec])  # re-store same span/layer
    assert len(store.query_measurements("r")) == 1, "upsert duplicated a row"


@pytest.mark.skipif(not DOCS_RECORDS, reason="no docs/* measurement_records.json present")
def test_p1_roundtrip_real_docs_payloads(tmp_path):
    """The make-or-break check: real L1/L3/perfspect (100+ key) payloads must
    round-trip byte-equal through the store."""
    store = SQLiteResultStore(tmp_path)
    for f in DOCS_RECORDS:
        raw = json.loads(Path(f).read_text())
        run_id = Path(f).parent.name
        _seed_run(store, run_id)
        recs = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"])
                for r in raw]
        store.store_measurements(run_id=run_id, records=recs)
        got = store.query_measurements(run_id)
        assert len(got) == len(recs), f"{f}: count {len(got)} != {len(recs)}"
        # Compare as (span_id, layer) -> payload maps to be order-independent of
        # any same-key collisions (verified none exist in real data).
        want = {(r["span_id"], r["layer"]): r["payload"] for r in raw}
        have = {(g["span_id"], g["layer"]): g["payload"] for g in got}
        assert have == want, f"{f}: payload not byte-equal after round-trip"
        # perfspect (100+ keys) specifically survives
        ps = [g for g in got if g["layer"] == "perfspect"]
        for g in ps:
            assert len(g["payload"]) > 50, "perfspect payload lost keys"
