--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0008 — drop the hardcoded runs.hardware_sku DEFAULT.
--
-- BUG fixed: schema.sql shipped `hardware_sku TEXT DEFAULT 'Intel Xeon Platinum
-- 8592+'` — the SKU of the host the schema was written on. Any run that did not
-- explicitly supply a SKU was therefore silently labelled Emerald Rapids,
-- regardless of the silicon it actually ran on. For a cross-platform benchmark
-- whose entire purpose is comparing hardware, a fabricated hardware label is
-- the worst possible default: it is wrong in a way that looks like data.
-- Release item positioning-4.
--
-- Fix: no DEFAULT. An unknown SKU is NULL, which reads as "not recorded"
-- instead of asserting a machine that was never benchmarked. run_driver now
-- populates it from platform detection.
--
-- SQLite cannot ALTER a column DEFAULT, so this is the documented table-rebuild
-- (new table -> copy -> drop -> rename), matching migration 0003. Runs with
-- foreign_keys OFF (set by the migration runner) inside its own transaction;
-- the runner's post-migration foreign_key_check is the integrity gate.
--
-- Data preservation: every existing column and row is copied verbatim. Rows
-- that were written with a real SKU keep it. Rows that merely inherited the
-- old DEFAULT are indistinguishable from rows where the SKU was genuinely
-- 8592+, so they are preserved as-is rather than guessed at — this migration
-- removes the mislabelling mechanism, it does not retro-edit history.

BEGIN;

-- 1. New-shape table: identical to the current shape, minus the SKU DEFAULT.
CREATE TABLE runs_new (
    run_id TEXT PRIMARY KEY,
    start_time INTEGER NOT NULL,
    end_time INTEGER,
    hardware_sku TEXT,
    model TEXT,
    total_tasks INTEGER DEFAULT 0,
    passed_tasks INTEGER DEFAULT 0,
    metadata JSON,
    benchmark_id TEXT,
    optimization_profile TEXT,
    numa_policy TEXT,
    agentsysperf_version TEXT,
    result_digest TEXT,
    owner_id TEXT,
    owner_kind TEXT,
    host_id TEXT,
    status TEXT
);

-- 2. Copy every row, column for column.
INSERT INTO runs_new (
    run_id, start_time, end_time, hardware_sku, model,
    total_tasks, passed_tasks, metadata, benchmark_id, optimization_profile,
    numa_policy, agentsysperf_version, result_digest,
    owner_id, owner_kind, host_id, status
)
SELECT
    run_id, start_time, end_time, hardware_sku, model,
    total_tasks, passed_tasks, metadata, benchmark_id, optimization_profile,
    numa_policy, agentsysperf_version, result_digest,
    owner_id, owner_kind, host_id, status
FROM runs;

-- 3. Swap.
DROP TABLE runs;
ALTER TABLE runs_new RENAME TO runs;

-- 4. Recreate indexes (the rebuild drops them with the old table).
CREATE INDEX IF NOT EXISTS idx_runs_benchmark ON runs(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(owner_id);

COMMIT;
