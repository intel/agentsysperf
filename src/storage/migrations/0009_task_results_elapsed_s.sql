--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- Migration 0009 — retain agent invocation latency alongside task duration.
--
-- ``duration_s`` remains the full measured task latency. ``elapsed_s`` is the
-- narrower agent invocation interval reported by TerminalBenchAdapter.

ALTER TABLE task_results ADD COLUMN elapsed_s REAL;
