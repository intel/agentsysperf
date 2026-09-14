#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Dashboard data-access layer (P6).

The single data source for the Streamlit dashboards (demo_app, live_dashboard).
Each accessor tries the canonical ResultStore first and, when the store has no
data for a benchmark yet, FALLS BACK to the legacy filesystem path verbatim — so
rendering is byte-identical before and during the store migration (P10 removes
the fallback once parity is proven).

Design rules (from the storage plan):
- Open the store read-only, per call. Never cache a live connection; callers
  (Streamlit @st.cache_data) may cache the returned plain dicts/lists.
- Return raw measurement-record dicts ({span_id, layer, payload}) with payload
  intact, so the dashboards' existing transforms/analyzers run unchanged.
- Resolve file-addressed artifacts (EMON CSV, scaling PNG) to a real on-disk
  path, falling back to the legacy glob when the store has no registration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.storage.sqlite_store import SQLiteResultStore

log = logging.getLogger(__name__)


def _store() -> Optional[SQLiteResultStore]:
    """Open the canonical store read-only. None if it can't be opened (no DB
    file yet) — callers then use their legacy fallback.

    A read-only sqlite connection cannot create a missing file and raises only
    on first use (connect() is lazy), so we force the connection here and treat
    any failure as 'no store'.
    """
    try:
        store = SQLiteResultStore.open(read_only=True)
        store._get_connection()  # force open now so a missing DB fails here
        return store
    except Exception:
        return None


def _records_from_json(path: Path) -> List[Dict[str, Any]]:
    """Legacy fallback: read a measurement_records.json verbatim."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (ValueError, OSError):
        pass
    return []


def get_records(
    benchmark_id: str,
    *,
    fallback_paths: Sequence[Path] = (),
    layer: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Measurement records for a benchmark's latest run.

    Tries the store (latest run for ``benchmark_id``); if it has none, returns
    the first existing legacy ``measurement_records.json`` from ``fallback_paths``
    (the dashboards' current hardcoded probe, lifted verbatim). Records are
    ``{span_id, layer, payload}`` dicts — the shape both the store and the JSON
    already use, so dashboard transforms are unchanged.
    """
    store = _store()
    if store is not None:
        run_id = store.latest_run(benchmark_id=benchmark_id)
        if run_id:
            rows = store.query_measurements(run_id, layer=layer)
            if rows:
                return [
                    {"span_id": r["span_id"], "layer": r["layer"], "payload": r["payload"]}
                    for r in rows
                ]
    # Fallback: legacy JSON probe (first existing path wins — same as today).
    for cand in fallback_paths:
        recs = _records_from_json(Path(cand) / "measurement_records.json"
                                  if Path(cand).is_dir() else Path(cand))
        if recs:
            return recs if layer is None else [r for r in recs if r.get("layer") == layer]
    return []


def latest_run_id(benchmark_id: str) -> Optional[str]:
    """The most-recent run_id for a benchmark in the store, or None."""
    store = _store()
    return store.latest_run(benchmark_id=benchmark_id) if store else None


def get_artifact_path(
    benchmark_id: str,
    *,
    kind: str,
    fallback_dirs: Sequence[Path] = (),
    fallback_patterns: Sequence[str] = (),
) -> Optional[Path]:
    """Resolve a file-addressed artifact (EMON CSV, scaling plot) to a real path.

    Store first (artifacts registered for the benchmark's latest run); else the
    legacy dir+glob probe verbatim (first existing match wins). Always returns a
    path that exists on disk, or None — EmonAnalyzer/st.image need a real file.

    Use :func:`artifact_source` when the caller needs to tell the user WHICH of
    those two answered, because the difference matters: the store path is
    addressed by run_id, while a /tmp glob is whatever survived the last reboot
    and may belong to an entirely different run.
    """
    return artifact_source(
        benchmark_id, kind=kind,
        fallback_dirs=fallback_dirs, fallback_patterns=fallback_patterns,
    )[0]


def artifact_source(
    benchmark_id: str,
    *,
    kind: str,
    fallback_dirs: Sequence[Path] = (),
    fallback_patterns: Sequence[str] = (),
) -> tuple[Optional[Path], str]:
    """Like :func:`get_artifact_path`, but also report where the path came from.

    Returns ``(path, source)`` where source is one of:

    * ``"store"``    — registered against the benchmark's latest run. Trustworthy:
      the file belongs to that run.
    * ``"scratch"``  — found by globbing a hardcoded scratch dir. The file exists
      but nothing ties it to the current run; it is whatever survived the last
      reboot. Callers should say so rather than presenting it as current.
    * ``"missing"``  — neither. ``path`` is None.

    This distinction is why the Hardware Health panel could show five-day-old
    EMON data beside a fresh run with no indication they were unrelated.
    """
    store = _store()
    if store is not None:
        run_id = store.latest_run(benchmark_id=benchmark_id)
        if run_id:
            p = store.get_artifact_path(run_id, kind=kind)
            if p is not None:
                return p, "store"
    for d in fallback_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pat in fallback_patterns:
            matches = sorted(d.glob(pat))
            if matches:
                log.debug(
                    "artifact %s/%s resolved from scratch dir %s, not the store; "
                    "it is not tied to any run",
                    benchmark_id, kind, d,
                )
                return matches[0], "scratch"
    return None, "missing"


__all__ = ["get_records", "latest_run_id", "get_artifact_path",
           "artifact_source"]
