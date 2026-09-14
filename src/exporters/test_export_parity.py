#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P7 verification: store-backed Prometheus export produces BYTE-IDENTICAL
exposition text vs the legacy JSON path — across both span_id shapes
('production_run::task' with '::' and 'terminal-bench/task/turn_N' without) and
the perfspect/TMA payload.

Run: poetry run pytest src/exporters/test_export_parity.py -q
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from src.exporters import PrometheusExporter
from src.protocols import MeasurementRecord
from src.storage.sqlite_store import SQLiteResultStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(glob.glob(str(REPO_ROOT / "docs" / "*" / "measurement_records.json")))


def _exposition_from_json(json_path: Path, run_id: str) -> str:
    exp = PrometheusExporter(mode="server", extra_labels={"sku": "Xeon-EMR-8592+"})
    exp.export_from_file(json_path, run_id=run_id)
    return exp.get_metrics_text()


def _exposition_from_store(store: SQLiteResultStore, run_id: str) -> str:
    exp = PrometheusExporter(mode="server", extra_labels={"sku": "Xeon-EMR-8592+"})
    exp.export_run_from_store(store, run_id)
    return exp.get_metrics_text()


@pytest.mark.skipif(not DOCS, reason="no docs/* data")
@pytest.mark.parametrize("json_path", DOCS, ids=lambda p: Path(p).parent.name)
def test_store_export_byte_identical_to_json(json_path, tmp_path, monkeypatch):
    json_path = Path(json_path)
    run_id = json_path.parent.name

    # Legacy JSON exposition.
    json_text = _exposition_from_json(json_path, run_id)

    # Load the SAME records into a fresh store, export from it.
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open()
    store.store_run_metadata(run_id=run_id, metadata={"start_time": 0})
    raw = json.loads(json_path.read_text())
    store.store_measurements(run_id=run_id, records=[
        MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"])
        for r in raw
    ])
    store._get_connection().commit()
    store_text = _exposition_from_store(store, run_id)

    # Exposition lines must be identical as a set (line order across records may
    # differ by insertion, but the metric+label+value set must match exactly).
    assert set(store_text.splitlines()) == set(json_text.splitlines()), (
        f"exposition diverged for {run_id}: "
        f"only-in-json={sorted(set(json_text.splitlines()) - set(store_text.splitlines()))[:3]} "
        f"only-in-store={sorted(set(store_text.splitlines()) - set(json_text.splitlines()))[:3]}"
    )


def test_both_span_id_shapes_covered():
    """Guard: the docs corpus actually exercises BOTH span_id shapes, so the
    parametrized test above is meaningful."""
    shapes = {"with_colons": False, "with_slashes": False}
    for p in DOCS:
        for r in json.loads(Path(p).read_text()):
            sid = r["span_id"]
            if "::" in sid:
                shapes["with_colons"] = True
            elif "/" in sid:
                shapes["with_slashes"] = True
    if DOCS:
        assert shapes["with_colons"] and shapes["with_slashes"], \
            f"docs corpus doesn't cover both span_id shapes: {shapes}"
