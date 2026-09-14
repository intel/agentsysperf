--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0007 — rename runs.agentperf_version -> runs.agentsysperf_version.
--
-- The agentperf -> agentsysperf rename updated the column name in the store's
-- INSERT/UPSERT code (sqlite_store.store_run_metadata) and in markdown_report,
-- but migration 0004 shipped the column as `agentperf_version`. Every DB —
-- fresh (0004 DDL) or already-migrated (stuck at v6) — therefore has the OLD
-- name, so persist_run failed with "table runs has no column named
-- agentsysperf_version" and rolled back the entire run write. This forward
-- migration realigns the schema with the code.
--
-- Single ALTER ... RENAME COLUMN (SQLite >= 3.25.0). No table rebuild, no FK
-- churn — the runner versions this atomically. Column data (if any) is
-- preserved; provenance NULLs on pre-existing rows are unaffected.

ALTER TABLE runs RENAME COLUMN agentperf_version TO agentsysperf_version;
