--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0004 — benchmarks entity + provenance/owner columns (P3).
--
-- Adds the missing top-level `benchmarks` entity and promotes provenance from
-- free-form metadata JSON into first-class columns, so a published number is
-- defensible and multi-user attribution works.
--
-- These are additive ALTERs (no table rebuild). benchmark_id is a plain indexed
-- column, NOT a FK: a run may be written before its benchmark row exists, and we
-- never want a run insert to fail on a missing benchmark. delete_benchmark is
-- handled app-level in the store. Pure additive DDL — no BEGIN/COMMIT needed
-- (the runner versions it atomically); ALTER ... ADD COLUMN is safe on existing
-- rows (new columns are NULL).

CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id TEXT PRIMARY KEY,     -- adapter slug: "terminal-bench", "tau-bench"
    display_name TEXT,
    version TEXT,
    created_at INTEGER,
    metadata JSON
);

-- Provenance + ownership on runs. (SQLite has no ADD COLUMN IF NOT EXISTS; these
-- run exactly once at this migration version, so plain ADD COLUMN is correct.)
ALTER TABLE runs ADD COLUMN benchmark_id TEXT;
ALTER TABLE runs ADD COLUMN optimization_profile TEXT;
ALTER TABLE runs ADD COLUMN numa_policy TEXT;
ALTER TABLE runs ADD COLUMN agentperf_version TEXT;
ALTER TABLE runs ADD COLUMN result_digest TEXT;
ALTER TABLE runs ADD COLUMN owner_id TEXT;     -- $USER locally; auth principal on a server
ALTER TABLE runs ADD COLUMN owner_kind TEXT;   -- 'os_user' | 'oidc' | 'token'
ALTER TABLE runs ADD COLUMN host_id TEXT;      -- hostname, for multi-host attribution
ALTER TABLE runs ADD COLUMN status TEXT;       -- running | complete | aborted | deleted

CREATE INDEX IF NOT EXISTS idx_runs_benchmark ON runs(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(owner_id);
