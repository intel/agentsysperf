--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0002 — measurements table (P1).
--
-- Fixes the split-brain: until now store_measurements() was a no-op and L1/L3/
-- EMON/perfspect records lived only in per-dir measurement_records.json. This
-- table makes the DB a real system of record for measurements.
--
-- Design: promote the 6 hot keys the dashboards/exporter plot as NULLable
-- columns (NULL != 0 — a missing key stays NULL, never 0), keep the full
-- free-form payload as JSON (perfspect carries 100+ keys; new layers add keys
-- with zero migration). `seq` preserves original insertion order for the
-- PhaseProfiler, which parses turn order from span_id.
--
-- UNIQUE(run_id, span_id, layer): verified distinct across all real data
-- (l1/l3 one record per span+layer; perfspect uses per-task system scope with
-- distinct span_ids), so the upsert never silently collapses records.
-- Idempotent (IF NOT EXISTS) — safe to re-run after a partial failure.

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    span_id  TEXT NOT NULL,
    layer    TEXT NOT NULL,          -- l1 | l3 | emon | perfspect | ...
    task_id  TEXT,                   -- span_id.split('::')[-1] if '::' else span_id
    -- promoted hot metrics (NULL when the layer does not produce them)
    duration_us  INTEGER,
    cpu_time_s   REAL,
    cpu_pct_mean REAL,
    rss_kb_peak  INTEGER,
    ipc          REAL,
    cache_miss_pct REAL,
    -- the long tail (perfspect TMA, raw events, future keys) stays here
    payload  JSON NOT NULL,
    seq      INTEGER,                 -- insertion order within the run
    schema_version TEXT DEFAULT '0.4',
    UNIQUE(run_id, span_id, layer),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meas_run_layer ON measurements(run_id, layer);
CREATE INDEX IF NOT EXISTS idx_meas_span      ON measurements(run_id, span_id);
CREATE INDEX IF NOT EXISTS idx_meas_run_task  ON measurements(run_id, task_id);
