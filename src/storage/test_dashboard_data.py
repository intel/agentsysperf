#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P6 verification: dashboard_data DAL — store-first with verbatim JSON
fallback, artifact resolution, and parity with the legacy direct read.

Run: poetry run pytest src/storage/test_dashboard_data.py -q
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

import src.dashboard_data as dal
from src.protocols import MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(glob.glob(str(REPO_ROOT / "docs" / "*" / "measurement_records.json")))


def test_fallback_reads_legacy_json_verbatim(tmp_path, monkeypatch):
    # Empty store -> DAL must fall back to the legacy JSON path, byte-identical.
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "empty_home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    payload = [
        {"span_id": "run::t", "layer": "l1", "payload": {"duration_us": 5, "ipc": 1.2}},
        {"span_id": "run::t", "layer": "l3", "payload": {"cache_miss_pct": 30.0}},
    ]
    (legacy / "measurement_records.json").write_text(json.dumps(payload))

    got = dal.get_records("terminal-bench", fallback_paths=[legacy])
    assert got == payload, "fallback did not return the legacy JSON verbatim"


def test_store_takes_precedence_over_fallback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(home))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    # Seed the canonical store with a benchmark + run + measurements.
    store = SQLiteResultStore.open()
    store.store_run_metadata(run_id="r1", metadata={"start_time": 1, "benchmark_id": "terminal-bench"})
    store.store_measurements(run_id="r1", records=[
        MeasurementRecord(span_id="r1::a", layer="l1", payload={"ipc": 9.9})])
    store._get_connection().commit()
    store.close()

    # A legacy path also exists but must be IGNORED (store wins).
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "measurement_records.json").write_text(
        json.dumps([{"span_id": "x", "layer": "l1", "payload": {"ipc": 0.0}}]))

    got = dal.get_records("terminal-bench", fallback_paths=[legacy])
    assert len(got) == 1 and got[0]["payload"]["ipc"] == 9.9, "store did not take precedence"


def test_layer_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "h"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "measurement_records.json").write_text(json.dumps([
        {"span_id": "s", "layer": "l1", "payload": {}},
        {"span_id": "s", "layer": "l3", "payload": {}},
    ]))
    l3 = dal.get_records("b", fallback_paths=[legacy], layer="l3")
    assert len(l3) == 1 and l3[0]["layer"] == "l3"


def test_artifact_fallback_glob(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "h"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    emon_dir = tmp_path / "emon"; emon_dir.mkdir()
    csv = emon_dir / "x_system_view_summary.csv"
    csv.write_text("metric,value\n")
    got = dal.get_artifact_path("tb", kind="emon_csv",
                               fallback_dirs=[emon_dir],
                               fallback_patterns=["*_system_view_summary.csv"])
    assert got == csv


@pytest.mark.skipif(not DOCS, reason="no docs/* data")
def test_parity_with_legacy_loader_on_real_data(tmp_path, monkeypatch):
    """DAL fallback must reproduce EXACTLY what demo_app's load_benchmarks built
    from the same JSON (the span->layer rollup), on real docs/* data."""
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "empty"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    src = Path(DOCS[0]).parent

    # Legacy loader logic (copied from demo_app.load_benchmarks).
    raw = json.loads((src / "measurement_records.json").read_text())
    spans: dict = {}
    for r in raw:
        spans.setdefault(r["span_id"], {})[r["layer"]] = r["payload"]
    legacy_rows = []
    for sid, layers in spans.items():
        l1, l3 = layers.get("l1", {}), layers.get("l3", {})
        legacy_rows.append((sid.split("::")[-1], l1.get("ipc"), l3.get("ipc")))

    # Same transform fed by the DAL fallback.
    recs = dal.get_records("terminal-bench", fallback_paths=[src])
    spans2: dict = {}
    for r in recs:
        spans2.setdefault(r["span_id"], {})[r["layer"]] = r["payload"]
    dal_rows = []
    for sid, layers in spans2.items():
        l1, l3 = layers.get("l1", {}), layers.get("l3", {})
        dal_rows.append((sid.split("::")[-1], l1.get("ipc"), l3.get("ipc")))

    assert sorted(map(str, dal_rows)) == sorted(map(str, legacy_rows)), \
        "DAL fallback diverged from the legacy loader on real data"
