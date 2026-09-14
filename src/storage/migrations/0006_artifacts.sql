--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0006 — artifacts table (P6).
--
-- Bulk per-run files (EMON pyEDP CSVs, scaling PNGs, all_results.json,
-- analysis_report.json) do NOT belong in the relational store as blobs — they
-- stay on disk and are addressed BY run_id, not discovered by globbing scattered
-- /tmp dirs. This table registers each artifact so get_artifact_path can resolve
-- a run's EMON CSV / scaling plot back to a real filesystem path (EmonAnalyzer
-- and st.image both need an on-disk file).
--
-- path is stored as given (absolute for legacy/registered files); a future
-- shared-server tier swaps in an ArtifactStore (local|s3) shim without schema
-- change. Additive DDL — no rebuild.

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    kind     TEXT NOT NULL,        -- emon_csv | scaling_plot | scaling_results | analysis_report | ...
    name     TEXT NOT NULL,        -- file basename (or logical name)
    path     TEXT NOT NULL,        -- resolvable filesystem path
    created_at INTEGER,
    metadata JSON,
    UNIQUE(run_id, kind, name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind ON artifacts(run_id, kind);
