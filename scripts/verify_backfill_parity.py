#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Parity gate for the store-only flip (P10).

Before removing the legacy JSON/`/tmp` fallbacks and flipping start.sh +
dashboards to read ONLY from the canonical store, this script verifies the
migrated store holds at least what the legacy sources held — counts, not vibes.
Run it after `migrate_to_unified_store.py --apply`; a non-zero exit BLOCKS the
flip.

It does NOT flip anything. It reports per-benchmark/run coverage and flags two
known hazards:
  - latest_run ambiguity: migrated runs share start_time=0, so "newest" is
    arbitrary — the demo dashboards' curated priority is NOT reproduced by a
    bare latest_run(). Surface which run each benchmark would resolve to.
  - runs whose JSON measurements did NOT make it into the store.

Usage:
    AGENTSYSPERF_HOME=~/.agentsysperf poetry run python scripts/verify_backfill_parity.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _legacy_counts() -> dict:
    """Records per run_id from docs/* JSON (the pre-migration source of truth)."""
    out = {}
    for j in sorted(glob.glob(str(REPO / "docs" / "*" / "measurement_records.json"))):
        raw = json.loads(Path(j).read_text())
        if not raw:
            continue
        first = raw[0].get("span_id", "")
        rid = first.split("::")[0] if "::" in first else Path(j).parent.name[len("tb2_"):] \
            if Path(j).parent.name.startswith("tb2_") else Path(j).parent.name
        out[rid] = out.get(rid, 0) + len(raw)
    return out


def main() -> int:
    from src.storage.sqlite_store import SQLiteResultStore
    import src.dashboard_data as dal

    store = SQLiteResultStore.open(read_only=True)
    legacy = _legacy_counts()
    runs = store.list_runs(limit=10_000)
    store_by_run = {r["run_id"]: r for r in runs}

    print(f"=== Parity: {len(legacy)} legacy run(s) vs {len(runs)} store run(s) ===")
    failures = []
    for rid, n_legacy in sorted(legacy.items()):
        n_store = len(store.query_measurements(rid))
        ok = n_store >= n_legacy
        if not ok:
            failures.append((rid, n_legacy, n_store))
        print(f"  {rid:20s} legacy={n_legacy:<5} store={n_store:<5} {'OK' if ok else 'MISSING'}")

    # latest_run hazard: which run does each benchmark resolve to, and is it the
    # richest? (a proxy for the demo's curated priority).
    print("\n=== latest_run resolution (store-only flip would render THIS run) ===")
    for b in {r.get("benchmark_id") for r in runs if r.get("benchmark_id")}:
        lr = dal.latest_run_id(b)
        n = len(store.query_measurements(lr)) if lr else 0
        richest = max((r["run_id"] for r in runs if r.get("benchmark_id") == b),
                      key=lambda x: len(store.query_measurements(x)), default=None)
        flag = "" if lr == richest else f"  <-- richest is {richest} (curated-priority hazard)"
        print(f"  {b}: latest_run={lr} ({n} records){flag}")

    if failures:
        print(f"\nPARITY FAIL: {len(failures)} run(s) lost measurements — DO NOT flip to store-only.")
        return 1
    print("\nPARITY OK: store holds >= legacy for every run. Safe to flip (mind the "
          "latest_run hazard above — flip needs a curated/pinned run selector).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
