#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P8 verification: DuckDB analytics accelerator + Parquet export.

Skips the live-DuckDB checks if the optional extra isn't installed *or* if
DuckDB's sqlite extension can't be loaded, but ALWAYS verifies the no-extra
contract: the module imports cleanly and core never needs it.
Run: pytest src/storage/test_duckdb_analytics.py -q
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.protocols import MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore

try:
    import duckdb  # noqa: F401
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False


def _have_sqlite_extension() -> bool:
    """Whether DuckDB can load its sqlite extension.

    Separate from HAVE_DUCKDB: the extension is not part of the pip package. It
    is fetched from DuckDB's registry on first use and cached under
    ~/.duckdb/extensions, so on a host with no egress these tests cannot run even
    though `import duckdb` succeeds. Skipping is what this file already promises
    for an unavailable dependency; without this guard it reported a hard failure
    that looked like a store bug.
    """
    if not HAVE_DUCKDB:
        return False
    import src.storage.duckdb_analytics as dk
    try:
        con = dk._connect()
    except ImportError:
        return False
    try:
        con.execute("INSTALL sqlite; LOAD sqlite;")
        return True
    except duckdb.Error:
        return False
    finally:
        con.close()


HAVE_SQLITE_EXT = _have_sqlite_extension()
NEEDS_EXT = pytest.mark.skipif(
    not HAVE_SQLITE_EXT, reason="duckdb sqlite extension unavailable (no egress?)"
)


def test_module_imports_without_using_duckdb():
    """The module must import even if duckdb is absent (lazy dependency)."""
    mod = importlib.import_module("src.storage.duckdb_analytics")
    assert hasattr(mod, "attach") and hasattr(mod, "export_parquet")


def test_missing_extra_raises_clear_hint(monkeypatch):
    """If duckdb can't be imported, _duckdb() raises a clear install hint, not a
    raw ModuleNotFoundError surfacing from deep in the call. Simulate the
    missing module by making `import duckdb` fail."""
    import builtins
    import src.storage.duckdb_analytics as dk

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "duckdb":
            raise ImportError("No module named 'duckdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError) as ei:
        dk._duckdb()
    assert "agentsysperf[analytics]" in str(ei.value)


def test_core_does_not_import_duckdb_analytics():
    """Importing the storage package / store must NOT pull in duckdb_analytics
    (it's read-side-only; core and dashboards never touch it)."""
    import sys
    # Fresh-ish check: the store module itself must not reference the analytics module.
    import src.storage.sqlite_store as store_mod
    src = Path(store_mod.__file__).read_text()
    assert "duckdb_analytics" not in src
    assert "import duckdb" not in src


@NEEDS_EXT
def test_attach_reads_same_counts_as_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open()
    store.store_run_metadata(run_id="r", metadata={"start_time": 1, "benchmark_id": "terminal-bench"})
    store.store_measurements(run_id="r", records=[
        MeasurementRecord(span_id="r::a", layer="l1", payload={"ipc": 2.0, "cache_miss_pct": 10.0}),
        MeasurementRecord(span_id="r::b", layer="l3", payload={"ipc": 1.5, "cache_miss_pct": 40.0}),
    ])
    store._get_connection().commit()
    sqlite_n = store._get_connection().execute("SELECT count(*) FROM measurements").fetchone()[0]

    import src.storage.duckdb_analytics as dk
    assert dk.measurement_count() == sqlite_n, "DuckDB attach count != SQLite (duplication?)"
    agg = dk.ipc_by_benchmark()
    assert agg and agg[0][0] == "terminal-bench"


@NEEDS_EXT
def test_parquet_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open()
    store.store_run_metadata(run_id="r", metadata={"start_time": 1})
    store.store_measurements(run_id="r", records=[
        MeasurementRecord(span_id="r::a", layer="l1", payload={"ipc": 2.0})])
    store._get_connection().commit()

    import src.storage.duckdb_analytics as dk
    out = tmp_path / "m.parquet"
    dk.export_parquet(out)
    assert out.exists()
    back = dk.read_parquet(out)
    assert len(back) == 1
