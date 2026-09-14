#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P10 verification: migrate_to_unified_store is correct, idempotent, and never
mutates originals.

Run: poetry run pytest scripts/test_migrate.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# load the script module
_spec = importlib.util.spec_from_file_location(
    "migrate_mod", REPO / "scripts" / "migrate_to_unified_store.py")
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)

from src.storage.sqlite_store import SQLiteResultStore


def _make_source(tmp_path: Path) -> Path:
    """A fake source dir: a sidecar db with one run + a measurements JSON."""
    d = tmp_path / "tb2_fake"
    d.mkdir()
    db = d / "agentsysperf_results.db"
    s = SQLiteResultStore(d)  # creates + migrates the schema
    s.store_run_metadata(run_id="fake", metadata={"start_time": 1, "benchmark_id": "terminal-bench"})
    s.store_task_result(run_id="fake", task_id="t1", result={"passed": True, "duration_s": 1.0})
    s._get_connection().commit()
    s.close()
    (d / "measurement_records.json").write_text(json.dumps([
        {"span_id": "terminal-bench/t1/turn_0", "layer": "l1", "payload": {"ipc": 1.0, "duration_us": 5}},
        {"span_id": "terminal-bench/t1/turn_0", "layer": "l3", "payload": {"ipc": 1.0, "cache_miss_pct": 9.0}},
    ]))
    return d


def test_run_id_convention():
    assert mig._run_id_for_dir(Path("/x/tb2_dashboard_demo")) == "dashboard_demo"
    assert mig._run_id_for_dir(Path("/x/something")) == "something"


def test_discover_finds_source(tmp_path):
    _make_source(tmp_path)
    found = mig.discover([tmp_path])
    assert len(found) == 1
    assert found[0]["run_id"] == "fake"
    assert found[0]["benchmark"] == "terminal-bench"


def test_forbidden_roots_skipped(tmp_path):
    # a path containing a forbidden segment must never be scanned
    assert mig._is_forbidden(Path("/repo/harness/results/x"))
    assert mig._is_forbidden(Path("/repo/wss_orchestration_example"))
    assert not mig._is_forbidden(tmp_path / "docs" / "tb2_x")


def test_apply_migrates_and_is_idempotent(tmp_path, monkeypatch):
    src_root = tmp_path / "src"; src_root.mkdir()
    _make_source(src_root)
    home = tmp_path / "home"
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(home))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)

    src_db = src_root / "tb2_fake" / "agentsysperf_results.db"
    src_json = src_root / "tb2_fake" / "measurement_records.json"
    md5_before = (hashlib.md5(src_db.read_bytes()).hexdigest(),
                  hashlib.md5(src_json.read_bytes()).hexdigest())

    # apply
    rc = mig.main.__wrapped__ if hasattr(mig.main, "__wrapped__") else mig.main
    monkeypatch.setattr(sys, "argv", ["mig", "--source-root", str(src_root), "--apply"])
    assert mig.main() == 0

    store = SQLiteResultStore.open(read_only=True)
    c = store._get_connection()
    runs1 = c.execute("SELECT count(*) FROM runs").fetchone()[0]
    meas1 = c.execute("SELECT count(*) FROM measurements").fetchone()[0]
    assert runs1 == 1 and meas1 == 2, f"expected 1 run / 2 measurements, got {runs1}/{meas1}"

    # originals untouched
    md5_after = (hashlib.md5(src_db.read_bytes()).hexdigest(),
                 hashlib.md5(src_json.read_bytes()).hexdigest())
    assert md5_after == md5_before, "migration mutated a source file!"

    # idempotent: re-apply, counts unchanged
    store.close()
    assert mig.main() == 0
    store2 = SQLiteResultStore.open(read_only=True)
    c2 = store2._get_connection()
    assert c2.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert c2.execute("SELECT count(*) FROM measurements").fetchone()[0] == 2


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    src_root = tmp_path / "src"; src_root.mkdir()
    _make_source(src_root)
    home = tmp_path / "home2"
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(home))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["mig", "--source-root", str(src_root)])  # no --apply
    assert mig.main() == 0
    # dry-run must not create the canonical store
    assert not (home / "results.db").exists(), "dry-run wrote to the store!"
