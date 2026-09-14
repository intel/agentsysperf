#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P2 verification: a sweep must leave NO dangling run_id FKs, so that P3's
ON DELETE CASCADE + foreign_keys=ON do not reject sweep_point / verdict rows.

Run: poetry run pytest src/sweep/test_sweep_run_rows.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.sqlite_store import SQLiteResultStore
from src.sweep.harbor_sweep import HarborSweep
from src.sweep.spec import SweepSpec


def _dry_sweep(tmp_path: Path):
    spec = SweepSpec(
        output_dir=tmp_path,
        densities=[0.25, 0.5, 1.0],
        replicates=1,
        benchmark="terminal-bench",
        tasks=["terminal-bench/hello-world"],
        attempts=1,
        llm_mode="off",  # dry_run synthesizes cells; no fixture/proxy needed
    )
    store = SQLiteResultStore(tmp_path)
    sweep = HarborSweep(spec, store=store)
    sweep_id = sweep.run(dry_run=True)
    return store, sweep_id


def test_sweep_creates_run_rows_for_sweep_id_and_cells(tmp_path):
    store, sweep_id = _dry_sweep(tmp_path)
    conn = store._get_connection()
    run_ids = {r[0] for r in conn.execute("SELECT run_id FROM runs").fetchall()}
    # sweep_id itself has a run row (the scaling verdict is filed under it)
    assert sweep_id in run_ids, "no runs row for sweep_id — verdict FK would dangle"
    # every sweep_point cell has a run row
    cell_ids = {r[0] for r in conn.execute("SELECT run_id FROM sweep_points").fetchall()}
    assert cell_ids, "no sweep_points stored"
    assert cell_ids <= run_ids, f"cell run_ids without a runs row: {cell_ids - run_ids}"


def test_no_dangling_run_id_fks(tmp_path):
    store, _ = _dry_sweep(tmp_path)
    conn = store._get_connection()
    # The two tables that reference runs(run_id) via run_id today.
    dangling_points = conn.execute(
        "SELECT count(*) FROM sweep_points sp "
        "WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = sp.run_id)"
    ).fetchone()[0]
    dangling_verdicts = conn.execute(
        "SELECT count(*) FROM analyzer_verdicts v "
        "WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = v.run_id)"
    ).fetchone()[0]
    assert dangling_points == 0, "sweep_points reference a missing runs row"
    assert dangling_verdicts == 0, "scaling verdict references a missing runs row"


def test_sweep_hardware_sku_not_literal(tmp_path):
    store, sweep_id = _dry_sweep(tmp_path)
    conn = store._get_connection()
    sku = conn.execute(
        "SELECT hardware_sku FROM runs WHERE run_id = ?", (sweep_id,)
    ).fetchone()[0]
    # must come from detect_platform(), never the old '8592+' literal
    assert sku and "8592+ (Emerald Rapids)" not in (sku or ""), \
        f"hardware_sku looks like a hardcoded literal: {sku!r}"
