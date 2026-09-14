# Grafana phase/verdict/cost panels (enrichment)

**Date:** 2026-06-05 · Direction chosen: "just enrich Grafana for now" (keep
Langfuse for traces; PPTX/Streamlit left untouched).

## What was added
The Prometheus exporter now also reads the run's **SQLite store** (not just
`measurement_records.json`) and emits a per-phase metric family. Grafana gains a
"Phase Breakdown" row.

### New Prometheus metrics (labels: run_id, task_id, phase)
- `agentsysperf_phase_time_percent{phase}` — wall-clock share per phase
  (orchestration/execution/inference), from breakdown verdict evidence.
- `agentsysperf_phase_execution_ipc{phase="execution"}` — instruction-weighted IPC.
- `agentsysperf_phase_execution_cache_miss_percent` / `_branch_miss_percent`.
  (Execution only — inference hardware is a remote network wait and is
  deliberately NOT emitted.)
- `agentsysperf_phase_tokens_total{phase="inference",direction=in|out}` (counter).
- `agentsysperf_phase_cost_usd{phase="inference"}` (counter).

### New Grafana panels (row "Phase Breakdown")
- Time Share by Phase (%)
- Execution-Phase IPC (instruction-weighted)
- Inference Tokens (in/out)
- Inference Cost (USD)

## How it flows
`scripts/push_to_prometheus.py` auto-detects `agentsysperf_results.db` next to the
records file, queries `query_verdicts(run_id,"breakdown")` + `query_spans(run_id)`,
and calls `PrometheusExporter.export_phase_metrics(...)` which appends to the
hardware metric text and re-pushes. Best-effort: if the DB is absent/old, it
prints "phase metrics skipped" and the hardware push still succeeds.

## Source of truth note
Per the dashboard assessment, phase rollups are a per-run categorical artifact;
SQLite remains authoritative for cost. Prometheus carries a copy purely so
Grafana can render the panels. Prompt/completion text and the interactive
per-step waterfall stay in Langfuse (link out from Grafana). The per-task Gantt
will move into Grafana when Tempo (Phase 3, task #53) lands.

## Files touched
- `src/exporters/prometheus_exporter.py` — `export_phase_metrics()`
- `src/exporters/grafana_dashboard.py` — `_phase_panels()` + row
- `scripts/push_to_prometheus.py` — auto-read SQLite, call export_phase_metrics
- `monitoring/grafana/dashboards/agentsysperf_dashboard.json` — regenerated
