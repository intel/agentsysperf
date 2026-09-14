#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P4 verification: ResultStore.open() seam, $AGENTSYSPERF_HOME resolution,
discovery fix, read-only safety, and Protocol conformance.

Run: poetry run pytest src/storage/test_open_seam.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.protocols import ResultStore, discover_result_stores
from src.storage.sqlite_store import SQLiteResultStore


def test_open_resolves_agentsysperf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open()
    assert store.db_path == tmp_path / "results.db"
    assert store.db_path.exists()


def test_open_honors_dsn_env(tmp_path, monkeypatch):
    target = tmp_path / "explicit.db"
    monkeypatch.setenv("AGENTSYSPERF_STORE_DSN", f"sqlite:///{target}")
    store = SQLiteResultStore.open()
    # sqlite:///<abs> resolves to the absolute path
    assert store.db_path == target


def test_open_explicit_dsn_arg_wins(tmp_path):
    target = tmp_path / "arg.db"
    store = SQLiteResultStore.open(dsn=f"sqlite:///{target}")
    assert store.db_path == target


def test_from_dsn_rejects_unknown_scheme():
    with pytest.raises(NotImplementedError) as ei:
        SQLiteResultStore.from_dsn("postgresql://host/agentsysperf")
    assert "postgres" in str(ei.value)


def test_back_compat_output_dir_still_works(tmp_path):
    store = SQLiteResultStore(output_dir=tmp_path)
    assert store.db_path == tmp_path / "agentsysperf_results.db"


def test_output_dir_warns_when_it_shadows_a_canonical_store(tmp_path, caplog):
    """Two filenames, one directory: warn instead of silently reading nothing.

    ``open()`` writes ``results.db``; this constructor opens
    ``agentsysperf_results.db``. Aiming the constructor at a directory the CLI
    wrote creates an empty second file, and every query then returns zero rows
    with no error — which reads as "the run recorded nothing" rather than
    "you opened the wrong file".
    """
    SQLiteResultStore(db_path=tmp_path / "results.db").close()

    with caplog.at_level("WARNING"):
        store = SQLiteResultStore(output_dir=tmp_path)

    # Behavior is unchanged (the dashboards rely on the legacy name resolving).
    assert store.db_path == tmp_path / "agentsysperf_results.db"
    assert "results.db" in caplog.text
    assert store.query_measurements("anything") == []


def test_output_dir_is_silent_when_nothing_is_shadowed(tmp_path, caplog):
    """No canonical store present → the ordinary path must not warn."""
    with caplog.at_level("WARNING"):
        SQLiteResultStore(output_dir=tmp_path)
    assert "results.db (written by" not in caplog.text


def test_discovery_no_longer_drops_sqlite(monkeypatch, tmp_path):
    # Previously _discover did cls() which raised TypeError for a store needing
    # output_dir, silently dropping it. open() fixes that.
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    stores = discover_result_stores()
    assert "sqlite" in stores, "sqlite store was dropped from discovery"
    assert isinstance(stores["sqlite"], SQLiteResultStore)


def test_runtime_checkable_protocol_conformance(tmp_path):
    store = SQLiteResultStore(output_dir=tmp_path)
    # The widened Protocol must be satisfied by the concrete store — guards
    # against a future backend implementing only a subset.
    assert isinstance(store, ResultStore)


def test_readonly_open_does_not_create_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "ro_home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open(read_only=True)
    # read-only must not run DDL: querying a table on a never-created DB raises.
    with pytest.raises(Exception):
        store._get_connection().execute("SELECT * FROM runs").fetchall()
