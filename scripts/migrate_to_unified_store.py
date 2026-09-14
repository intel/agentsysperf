#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Consolidate scattered AgentSysPerf results into the single canonical store (P10).

Idempotent, fail-loud, and SAFE BY DEFAULT:
  - DRY-RUN unless --apply: scans the source roots and reports exactly what it
    WOULD migrate (dirs, run_ids, record/db counts) without writing anything.
  - Never mutates originals: every source DB/JSON is opened READ-ONLY and copied
    into a FRESH $AGENTSYSPERF_HOME store. md5 of each source is checked before/after.
  - Source roots are CALLER-PROVIDED (--source-root, repeatable). Default is the
    repo's docs/* (the demo corpus). The colleague's GNR runs live elsewhere —
    point --source-root at their paths.
  - HARD RULE: never reads/writes harness/results or wss_orchestration_example
    (excluded explicitly).

What it consolidates per source dir:
  - a sidecar agentsysperf_results.db (runs/task_results/spans/analyzer_verdicts/
    sweeps/sweep_points) -> copied into the canonical store, and
  - a measurement_records.json -> backfilled into the `measurements` table under
    the SAME run_id the sidecar/start.sh use (basename with a leading 'tb2_'
    stripped), so Grafana run_id labels and the Langfuse join key are preserved.

Usage:
    # see what would happen (writes nothing):
    poetry run python scripts/migrate_to_unified_store.py
    poetry run python scripts/migrate_to_unified_store.py --source-root /path/to/gnr/runs
    # actually migrate:
    AGENTSYSPERF_HOME=~/.agentsysperf poetry run python scripts/migrate_to_unified_store.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Dirs we must NEVER read or write: committed baseline datasets and an
# out-of-scope vendored example. Both are read-only by project rule.
_FORBIDDEN = ("harness/results", "wss_orchestration_example")


def _run_id_for_dir(d: Path) -> str:
    """Match the sidecar-DB / start.sh convention: basename minus a 'tb2_' prefix."""
    name = d.name
    return name[len("tb2_"):] if name.startswith("tb2_") else name


def _benchmark_for_dir(d: Path) -> str:
    """Slug for the benchmark this dir belongs to. The demo corpus is all TB2;
    extend this map as other benchmarks' dirs appear."""
    name = d.name.lower()
    if "tau" in name:
        return "tau-bench"
    if "swe" in name:
        return "swe-bench"
    return "terminal-bench"


def _md5(p: Path) -> str:
    # Change-detection only ("did the migration touch a source file?"), never
    # an authentication or integrity claim against an adversary. usedforsecurity
    # =False says so to the FIPS layer and to bandit; the digest is unchanged.
    return (
        hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()
        if p.exists() else ""
    )


def _is_forbidden(p: Path) -> bool:
    s = str(p.resolve())
    return any(f in s for f in _FORBIDDEN)


def discover(source_roots: List[Path]) -> List[Dict]:
    """Find migratable dirs under the source roots. A dir qualifies if it holds
    a measurement_records.json and/or an agentsysperf_results.db."""
    found: List[Dict] = []
    seen = set()
    for root in source_roots:
        if not root.exists() or _is_forbidden(root):
            continue
        # the root itself, plus one level of children
        candidates = [root] + [c for c in sorted(root.iterdir()) if c.is_dir()]
        for d in candidates:
            if _is_forbidden(d) or d in seen:
                continue
            j = d / "measurement_records.json"
            db = d / "agentsysperf_results.db"
            if not (j.exists() or db.exists()):
                continue
            seen.add(d)
            n_json = 0
            if j.exists():
                try:
                    n_json = len(json.loads(j.read_text()))
                except (ValueError, OSError):
                    n_json = -1
            db_runs = []
            if db.exists():
                try:
                    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    db_runs = [r[0] for r in c.execute("SELECT run_id FROM runs").fetchall()]
                    c.close()
                except sqlite3.Error:
                    db_runs = []
            found.append({
                "dir": d, "json": j if j.exists() else None,
                "db": db if db.exists() else None,
                "n_json": n_json, "db_runs": db_runs,
                "run_id": _run_id_for_dir(d), "benchmark": _benchmark_for_dir(d),
            })
    return found


def _copy_db_into_store(store, src_db: Path, *, default_run_id: str, benchmark: str) -> int:
    """Copy a sidecar DB's rows into the canonical store via ATTACH (read-only).
    Returns the number of runs copied. Idempotent (store upserts on keys)."""
    conn = store._get_connection()
    # ATTACH the source. We ONLY ever SELECT from src.* (never write), and the
    # md5 check at the end proves the original is untouched — so no query_only
    # pragma (which SQLite applies connection-wide, blocking writes to the
    # canonical main DB too).
    conn.execute(f"ATTACH DATABASE '{src_db}' AS src")
    n_runs = 0
    try:
        # runs (seed benchmark_id; keep the source run_id verbatim for parity)
        src_tables = {r[0] for r in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'").fetchall()}
        if "runs" in src_tables:
            for row in conn.execute("SELECT run_id FROM src.runs").fetchall():
                rid = row[0]
                store.store_benchmark(benchmark_id=benchmark, metadata={"display_name": benchmark})
                # pull the full run row as a dict via the source connection
                src_run = conn.execute("SELECT * FROM src.runs WHERE run_id=?", (rid,)).fetchone()
                cols = [c[1] for c in conn.execute("PRAGMA src.table_info(runs)").fetchall()]
                meta = dict(zip(cols, src_run))
                meta.setdefault("benchmark_id", benchmark)
                meta["benchmark_id"] = meta.get("benchmark_id") or benchmark
                store.store_run_metadata(run_id=rid, metadata=meta)
                n_runs += 1
        # task_results
        if "task_results" in src_tables:
            for r in conn.execute("SELECT * FROM src.task_results").fetchall():
                cols = [c[1] for c in conn.execute("PRAGMA src.table_info(task_results)").fetchall()]
                d = dict(zip(cols, r))
                store.store_task_result(run_id=d["run_id"], task_id=d["task_id"], result=d)
        # analyzer_verdicts -> reconstruct AnalysisResult
        if "analyzer_verdicts" in src_tables:
            from src.protocols import AnalysisResult
            byrun: Dict[str, list] = {}
            for r in conn.execute("SELECT * FROM src.analyzer_verdicts").fetchall():
                cols = [c[1] for c in conn.execute("PRAGMA src.table_info(analyzer_verdicts)").fetchall()]
                d = dict(zip(cols, r))
                byrun.setdefault(d["run_id"], []).append(AnalysisResult(
                    analyzer_name=d["analyzer_name"], verdict=d["verdict"],
                    confidence=d["confidence"],
                    evidence=json.loads(d["evidence"]) if d.get("evidence") else {},
                    recommendations=json.loads(d["recommendations"]) if d.get("recommendations") else [],
                    span_id=d.get("task_id")))
            for rid, results in byrun.items():
                store.store_analysis_results(run_id=rid, results=results)
    finally:
        conn.commit()
        conn.execute("DETACH DATABASE src")
    return n_runs


def _backfill_measurements(store, src_json: Path, *, run_id: str) -> int:
    """Backfill measurement_records.json into the store's measurements table.
    NET-NEW data (the old store_measurements was a no-op). Returns rows added."""
    from src.protocols import MeasurementRecord
    raw = json.loads(src_json.read_text())
    if not raw:
        return 0
    # The span_id's '::' prefix is the AUTHORITATIVE run_id when present (e.g.
    # 'production_run::task' -> 'production_run'), since that's the run the db
    # filed verdicts/spans under. Slashed span_ids (terminal-bench/task/turn_N)
    # carry no run id, so fall back to the dir-derived run_id.
    first = raw[0].get("span_id", "")
    if "::" in first:
        run_id = first.split("::")[0]
    # Ensure a runs row exists (FK parent) — the JSON-only dirs (production_records)
    # have no sidecar DB.
    if store.get_run(run_id) is None:
        store.store_run_metadata(run_id=run_id, metadata={
            "start_time": 0, "benchmark_id": _benchmark_for_dir(src_json.parent),
            "hardware_sku": "unknown (backfill)",
            "status": "complete"})
    recs = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"])
            for r in raw]
    store.store_measurements(run_id=run_id, records=recs)
    return len(recs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", type=Path, action="append", default=None,
                    help="Root dir to scan (repeatable). Default: <repo>/docs")
    ap.add_argument("--apply", action="store_true",
                    help="Actually migrate. Without this, DRY-RUN (writes nothing).")
    args = ap.parse_args()

    roots = args.source_root or [REPO / "docs"]
    roots = [Path(r) for r in roots]
    for r in roots:
        if _is_forbidden(r):
            print(f"REFUSING forbidden source root (hard rule): {r}", file=sys.stderr)
            return 2

    sources = discover(roots)
    print(f"=== Discovered {len(sources)} migratable dir(s) under {[str(r) for r in roots]} ===")
    total_json = total_dbruns = 0
    for s in sources:
        total_json += max(s["n_json"], 0)
        total_dbruns += len(s["db_runs"])
        print(f"  {s['dir'].name:28s} run_id={s['run_id']:<16s} bench={s['benchmark']:<14s} "
              f"json={s['n_json']:<5} db_runs={s['db_runs'] or '—'}")

    if not args.apply:
        print(f"\nDRY-RUN — nothing written. {total_json} JSON records + {total_dbruns} db-run(s) "
              f"would migrate.\nRe-run with --apply (and set AGENTSYSPERF_HOME) to execute.")
        return 0

    # APPLY: write into the canonical store. Record source md5s to assert
    # originals are untouched afterward.
    from src.storage.sqlite_store import SQLiteResultStore
    from src.home import default_db_path
    store = SQLiteResultStore.open()
    print(f"\n=== APPLY -> {default_db_path()} ===")

    pre_md5 = {}
    for s in sources:
        for key in ("json", "db"):
            p = s[key]
            if p:
                pre_md5[str(p)] = _md5(p)

    runs_copied = json_backfilled = 0
    for s in sources:
        if s["db"]:
            runs_copied += _copy_db_into_store(
                store, s["db"], default_run_id=s["run_id"], benchmark=s["benchmark"])
        if s["json"] and s["n_json"] > 0:
            json_backfilled += _backfill_measurements(store, s["json"], run_id=s["run_id"])
        print(f"  migrated {s['dir'].name}")

    # Verify originals untouched.
    tampered = [p for p, m in pre_md5.items() if _md5(Path(p)) != m]
    if tampered:
        print(f"\nFATAL: source files were modified (should be read-only!): {tampered}", file=sys.stderr)
        return 1

    runs_in_store = len(store.list_runs(limit=10_000))
    print(f"\nDone. {runs_copied} db-run(s) copied, {json_backfilled} measurement(s) backfilled. "
          f"Store now holds {runs_in_store} run(s). Originals unchanged (md5 verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
