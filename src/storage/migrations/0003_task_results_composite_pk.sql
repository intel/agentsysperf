--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0003 — task_results composite PK + logical_task_key (P3).
--
-- BUG fixed: task_results.task_id was a GLOBAL primary key, so the same logical
-- task run twice (two run_ids) COLLIDED — the second run's row overwrote the
-- first. Cross-run aggregation ("task X across N runs") was impossible.
--
-- Fix: composite PK (run_id, task_id) + a stored logical_task_key
-- ("{benchmark}::{task_id}", benchmark filled in by the store at write time when
-- known) as the cross-run grouping key. Adds ON DELETE CASCADE so delete_run
-- removes a run's task rows.
--
-- SQLite cannot ALTER a PK/constraint, so this is the documented table-rebuild:
-- new table -> copy -> drop -> rename. Runs with foreign_keys OFF (set by the
-- migration runner) and is wrapped in its own transaction for atomicity. The
-- runner's post-migration foreign_key_check is the integrity gate.
--
-- Data preservation: legacy task_results rows whose run_id has NO runs row would
-- violate the new cascade FK once FKs are re-enabled. Backfill a placeholder
-- runs row for any such orphan FIRST, so no task rows are lost.

BEGIN;

-- 1. Backfill placeholder runs rows for any dangling task_results.run_id.
INSERT OR IGNORE INTO runs (run_id, start_time, metadata)
SELECT DISTINCT tr.run_id, 0, json('{"backfilled_orphan": true}')
FROM task_results tr
WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = tr.run_id);

-- 2. New-shape table.
CREATE TABLE task_results_new (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    logical_task_key TEXT,        -- "{benchmark}::{task_id}" — cross-run agg key
    workload_type TEXT,
    passed BOOLEAN NOT NULL,
    duration_s REAL,
    num_turns INTEGER,
    num_commands INTEGER,
    PRIMARY KEY (run_id, task_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- 3. Copy existing rows. logical_task_key left NULL here (no benchmark known at
--    migration time); the store backfills it going forward from run metadata.
INSERT INTO task_results_new
    (run_id, task_id, logical_task_key, workload_type, passed, duration_s, num_turns, num_commands)
SELECT run_id, task_id, NULL, workload_type, passed, duration_s, num_turns, num_commands
FROM task_results;

-- 4. Swap.
DROP TABLE task_results;
ALTER TABLE task_results_new RENAME TO task_results;

-- 5. Recreate indexes.
CREATE INDEX IF NOT EXISTS idx_task_workload ON task_results(workload_type);
CREATE INDEX IF NOT EXISTS idx_task_logical  ON task_results(logical_task_key);

COMMIT;
