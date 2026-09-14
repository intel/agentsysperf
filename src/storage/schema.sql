--=========================== begin_copyright_notice ============================
--
-- Copyright (C) 2026 Intel Corporation
--
-- SPDX-License-Identifier: Apache-2.0
--
--============================ end_copyright_notice =============================

-- AgentSysPerf SQLite schema for benchmark result storage
--
-- Design principles:
-- - Normalized schema for efficient querying
-- - JSON columns for extensibility (metadata, evidence)
-- - Indexes on common query patterns (workload type, analyzer name)
-- - Foreign keys for referential integrity

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    start_time INTEGER NOT NULL,
    end_time INTEGER,
    -- No DEFAULT: a baked-in SKU silently mislabels runs from every other
    -- host as the machine it was written on. Populated from platform detection.
    hardware_sku TEXT,
    model TEXT,
    total_tasks INTEGER DEFAULT 0,
    passed_tasks INTEGER DEFAULT 0,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS task_results (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    workload_type TEXT,
    passed BOOLEAN NOT NULL,
    duration_s REAL,
    num_turns INTEGER,
    num_commands INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS analyzer_verdicts (
    verdict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    analyzer_name TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence JSON NOT NULL,
    recommendations JSON,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Step-level execution trace (StepTrace v0.3): one row per agent step
-- (LLM call, tool call, agent-step boundary). Powers trace/step analytics —
-- per-step tokens, cost, latency, tool exit status — alongside the hardware
-- measurements. Mirrors src/trace/schema.py::StepTrace.
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    parent_span_id TEXT,
    task_id TEXT,
    span_kind TEXT NOT NULL,         -- llm_call | tool_call | agent_step | ...
    node_id TEXT,
    start_ts_us INTEGER DEFAULT 0,
    end_ts_us INTEGER DEFAULT 0,
    duration_us INTEGER DEFAULT 0,
    -- LLM-call columns
    model_id TEXT,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    -- tool-call columns
    tool_name TEXT,
    resource_tier TEXT,
    -- outcome
    status TEXT DEFAULT 'ok',
    error TEXT,
    extra JSON,
    schema_version TEXT DEFAULT '0.3',
    PRIMARY KEY (run_id, span_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Concurrency-scaling sweep: one sweep groups many runs (one per density
-- point × replicate). A scaling analysis spans MULTIPLE runs, which the
-- per-run analyzer model can't express — so sweeps sit ABOVE runs, additively.
-- Each density cell is still a normal `runs` row (tagged with sweep_id in its
-- metadata); `sweep_points` is the cell-level rollup the ScalingAnalyzer reads.
CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    hardware_sku TEXT,
    vcpu_basis INTEGER,              -- denominator for density (physical cores on EMR)
    vcpu_basis_kind TEXT,           -- 'physical_cores' | 'logical_cpus'
    numa_policy TEXT,               -- 'unpinned' | 'socket_pinned' | 'interleaved'
    model TEXT,
    replay_fixture TEXT,            -- fixture path (provenance: which trajectory)
    benchmark TEXT,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS sweep_points (
    sweep_id TEXT NOT NULL,
    run_id TEXT NOT NULL,           -- the per-cell run this point summarizes
    density REAL,                   -- concurrency / vcpu_basis ("agents per vCPU")
    concurrency INTEGER,            -- N agents run in parallel for this cell
    replicate INTEGER,
    elapsed_s REAL,                 -- cell wall time
    throughput_per_min REAL,        -- trials / (elapsed/60)
    completed_trials INTEGER,
    p95_trial_latency_s REAL,
    -- node telemetry rollup (from l1_system over the cell window)
    cpu_avg REAL,
    cpu_p95 REAL,
    cpu_peak REAL,
    runqueue_max REAL,
    ctx_sw_per_s REAL,
    mem_avail_mb_min REAL,
    iowait_pct_avg REAL,
    metadata JSON,
    PRIMARY KEY (sweep_id, run_id),
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE INDEX IF NOT EXISTS idx_task_workload ON task_results(workload_type);
CREATE INDEX IF NOT EXISTS idx_verdict_analyzer ON analyzer_verdicts(analyzer_name, verdict);
CREATE INDEX IF NOT EXISTS idx_spans_task ON spans(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(run_id, span_kind);
CREATE INDEX IF NOT EXISTS idx_sweep_points_density ON sweep_points(sweep_id, density);
