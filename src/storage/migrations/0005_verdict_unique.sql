--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0005 — analyzer_verdicts idempotency (P3).
--
-- store_analysis_results did a plain INSERT, so re-running analyzers or
-- re-importing a run MULTIPLIED verdict rows. Add UNIQUE(run_id, task_id,
-- analyzer_name) so the store can upsert instead.
--
-- Table rebuild (UNIQUE cannot be added via ALTER). Dedup on copy: keep the
-- highest verdict_id per (run_id, task_id, analyzer_name) = the most recent.
-- Backfill placeholder runs rows for any dangling verdict.run_id first (legacy
-- sweep verdicts were filed under run_id=sweep_id without a runs row — P2 fixes
-- this going forward, but old DBs still have them). Runs FK-OFF (set by runner),
-- wrapped in its own transaction; runner's foreign_key_check is the gate.

BEGIN;

-- Backfill placeholder runs rows for dangling verdict run_ids (e.g. legacy
-- sweep_id verdicts) so the cascade FK does not orphan them.
INSERT OR IGNORE INTO runs (run_id, start_time, metadata)
SELECT DISTINCT v.run_id, 0, json('{"backfilled_orphan": true}')
FROM analyzer_verdicts v
WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = v.run_id);

CREATE TABLE analyzer_verdicts_new (
    verdict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    analyzer_name TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence JSON NOT NULL,
    recommendations JSON,
    UNIQUE(run_id, task_id, analyzer_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Keep the most recent row per identity (max verdict_id).
INSERT INTO analyzer_verdicts_new
    (run_id, task_id, analyzer_name, verdict, confidence, evidence, recommendations)
SELECT run_id, task_id, analyzer_name, verdict, confidence, evidence, recommendations
FROM analyzer_verdicts
WHERE verdict_id IN (
    SELECT MAX(verdict_id) FROM analyzer_verdicts
    GROUP BY run_id, task_id, analyzer_name
);

DROP TABLE analyzer_verdicts;
ALTER TABLE analyzer_verdicts_new RENAME TO analyzer_verdicts;

CREATE INDEX IF NOT EXISTS idx_verdict_analyzer ON analyzer_verdicts(analyzer_name, verdict);

COMMIT;
