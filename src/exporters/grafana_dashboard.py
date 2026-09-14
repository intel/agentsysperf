#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Grafana Dashboard Generator for AgentSysPerf
===========================================

Generates a Grafana dashboard JSON that visualizes AgentSysPerf benchmark
metrics from Prometheus. The dashboard includes:

- Task duration and CPU time panels
- IPC and cache miss rate panels
- TMA breakdown (Frontend/Backend/Retiring/Bad Speculation)
- Memory usage and bandwidth panels
- Per-workload comparison views

Usage:
    generator = GrafanaDashboardGenerator()
    dashboard_json = generator.generate()
    generator.save("agentsysperf_dashboard.json")

Import into Grafana via:
    - UI: Dashboards → Import → Upload JSON
    - API: POST /api/dashboards/db
    - Provisioning: Copy to /etc/grafana/provisioning/dashboards/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GrafanaDashboardGenerator:
    """Generate Grafana dashboard JSON for AgentSysPerf metrics.

    Parameters
    ----------
    datasource : str, default="Prometheus"
        Grafana datasource name for Prometheus.
    title : str, default="AgentSysPerf Benchmark Dashboard"
        Dashboard title.
    refresh_interval : str, default="10s"
        Auto-refresh interval.
    """

    def __init__(
        self,
        *,
        datasource: str = "Prometheus",
        title: str = "AgentSysPerf Benchmark Dashboard",
        refresh_interval: str = "10s",
    ) -> None:
        self._datasource = datasource
        self._title = title
        self._refresh_interval = refresh_interval

    def generate(self) -> Dict[str, Any]:
        """Generate complete Grafana dashboard JSON."""
        dashboard = {
            "annotations": {"list": []},
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 1,
            "id": None,
            "links": [],
            "panels": self._build_panels(),
            "refresh": self._refresh_interval,
            "schemaVersion": 39,
            "tags": ["agentsysperf", "benchmark", "cpu", "intel"],
            "templating": {"list": self._build_variables()},
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
            "timezone": "browser",
            "title": self._title,
            "uid": "agentsysperf-main",
            "version": 1,
        }
        return dashboard

    def save(self, output_path: Path) -> None:
        """Save dashboard JSON to file."""
        dashboard = self.generate()
        # Wrap in Grafana import format
        export = {
            "dashboard": dashboard,
            "overwrite": True,
            "folderId": 0,
        }
        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)
        logger.info(f"Grafana dashboard saved to {output_path}")

    # ─── Variables (Dropdowns) ───────────────────────────────────────

    def _build_variables(self) -> List[Dict[str, Any]]:
        """Build template variables for dashboard dropdowns."""
        return [
            {
                "name": "run_id",
                "label": "Run ID",
                "type": "query",
                "datasource": self._datasource,
                "query": 'label_values(agentsysperf_task_duration_seconds, run_id)',
                "refresh": 2,
                "sort": 2,
            },
            {
                "name": "workload",
                "label": "Workload",
                "type": "query",
                "datasource": self._datasource,
                "query": 'label_values(agentsysperf_task_duration_seconds{run_id="$run_id"}, workload)',
                "refresh": 2,
                "sort": 1,
                "multi": True,
                "includeAll": True,
            },
        ]

    # ─── Panel Building ──────────────────────────────────────────────

    def _build_panels(self) -> List[Dict[str, Any]]:
        """Build all dashboard panels."""
        panels = []
        y = 0

        # Row: Overview
        panels.append(self._row_panel("Overview", y))
        y += 1
        panels.extend(self._overview_panels(y))
        y += 9

        # Row: CPU Performance
        panels.append(self._row_panel("CPU Performance", y))
        y += 1
        panels.extend(self._cpu_panels(y))
        y += 9

        # Row: Memory & Cache
        panels.append(self._row_panel("Memory & Cache", y))
        y += 1
        panels.extend(self._memory_panels(y))
        y += 9

        # Row: TMA Analysis
        panels.append(self._row_panel("Top-Down Microarchitecture Analysis (TMA)", y))
        y += 1
        panels.extend(self._tma_panels(y))
        y += 9

        # Row: Per-Workload Comparison
        panels.append(self._row_panel("Per-Workload Comparison", y))
        y += 1
        panels.extend(self._comparison_panels(y))
        y += 9

        # Row: Phase Breakdown (agentic — orchestration/execution/inference)
        panels.append(self._row_panel(
            "Phase Breakdown — Orchestration / Execution / Inference", y))
        y += 1
        panels.extend(self._phase_panels(y))
        y += 9

        return panels

    def _phase_panels(self, y: int) -> List[Dict[str, Any]]:
        """Per-phase metrics: time split, execution hardware, tokens, cost.

        Sourced from the SQLite analyzer_verdicts + spans (exported by
        push_to_prometheus.py's phase metrics), NOT from raw hardware records.
        Inference hardware is intentionally absent — it's a remote network wait.
        """
        rid = '{run_id="$run_id"}'
        return [
            # Time share by phase, per task (stacked).
            self._bar_panel(
                title="Time Share by Phase (%)",
                query=f'agentsysperf_phase_time_percent{rid}',
                x=0, y=y, w=12, h=8,
                legend="{{phase}} — {{task_id}}", unit="percent",
            ),
            # Execution-phase IPC per task (instruction-weighted).
            self._bar_panel(
                title="Execution-Phase IPC (instruction-weighted)",
                query=f'agentsysperf_phase_execution_ipc{rid}',
                x=12, y=y, w=12, h=8,
                legend="{{task_id}}", unit="short",
            ),
            # Per-task token totals (in/out).
            self._bar_panel(
                title="Inference Tokens (in/out)",
                query=f'agentsysperf_phase_tokens_total{rid}',
                x=0, y=y + 8, w=12, h=8,
                legend="{{direction}} — {{task_id}}", unit="short",
            ),
            # Per-task inference cost.
            self._bar_panel(
                title="Inference Cost (USD)",
                query=f'agentsysperf_phase_cost_usd{rid}',
                x=12, y=y + 8, w=12, h=8,
                legend="{{task_id}}", unit="currencyUSD",
            ),
        ]

    def _overview_panels(self, y: int) -> List[Dict[str, Any]]:
        """Build overview stat panels."""
        return [
            self._stat_panel(
                title="Total Tasks",
                query='count(agentsysperf_task_duration_seconds{run_id="$run_id"})',
                x=0, y=y, w=4, h=4,
                unit="short",
            ),
            self._stat_panel(
                title="Avg Duration",
                query='avg(agentsysperf_task_duration_seconds{run_id="$run_id"})',
                x=4, y=y, w=4, h=4,
                unit="s",
            ),
            self._stat_panel(
                title="Avg IPC",
                query='avg(agentsysperf_task_ipc{run_id="$run_id"})',
                x=8, y=y, w=4, h=4,
                unit="short",
            ),
            self._stat_panel(
                title="Avg Cache Miss %",
                query='avg(agentsysperf_task_cache_miss_percent{run_id="$run_id"})',
                x=12, y=y, w=4, h=4,
                unit="percent",
            ),
            self._stat_panel(
                title="Avg CPU Utilization",
                query='avg(agentsysperf_task_cpu_utilization_percent{run_id="$run_id"})',
                x=16, y=y, w=4, h=4,
                unit="percent",
            ),
            self._stat_panel(
                title="Peak Memory (MB)",
                query='max(agentsysperf_task_memory_rss_bytes{run_id="$run_id"}) / 1048576',
                x=20, y=y, w=4, h=4,
                unit="decmbytes",
            ),
        ]

    def _cpu_panels(self, y: int) -> List[Dict[str, Any]]:
        """Build CPU performance panels."""
        return [
            self._bar_panel(
                title="Task Duration (seconds)",
                query='agentsysperf_task_duration_seconds{run_id="$run_id", workload=~"$workload"}',
                x=0, y=y, w=12, h=8,
                legend="{{workload}}",
                unit="s",
            ),
            self._bar_panel(
                title="CPU Time (seconds)",
                query='agentsysperf_task_cpu_time_seconds{run_id="$run_id", workload=~"$workload"}',
                x=12, y=y, w=12, h=8,
                legend="{{workload}}",
                unit="s",
            ),
        ]

    def _memory_panels(self, y: int) -> List[Dict[str, Any]]:
        """Build memory and cache panels."""
        return [
            self._bar_panel(
                title="Peak RSS (MB)",
                query='agentsysperf_task_memory_rss_bytes{run_id="$run_id", workload=~"$workload"} / 1048576',
                x=0, y=y, w=12, h=8,
                legend="{{workload}}",
                unit="decmbytes",
            ),
            self._bar_panel(
                title="Cache Miss Rate (%)",
                query='agentsysperf_task_cache_miss_percent{run_id="$run_id", workload=~"$workload"}',
                x=12, y=y, w=12, h=8,
                legend="{{workload}}",
                unit="percent",
            ),
        ]

    def _tma_panels(self, y: int) -> List[Dict[str, Any]]:
        """Build TMA analysis panels."""
        return [
            self._stacked_bar_panel(
                title="TMA L1 Breakdown",
                queries=[
                    ('agentsysperf_tma_frontend_bound_ratio{run_id="$run_id", workload=~"$workload"}', "Frontend Bound"),
                    ('agentsysperf_tma_bad_speculation_ratio{run_id="$run_id", workload=~"$workload"}', "Bad Speculation"),
                    ('agentsysperf_tma_backend_bound_ratio{run_id="$run_id", workload=~"$workload"}', "Backend Bound"),
                    ('agentsysperf_tma_retiring_ratio{run_id="$run_id", workload=~"$workload"}', "Retiring"),
                ],
                x=0, y=y, w=16, h=8,
            ),
            self._bar_panel(
                title="Instructions Per Cycle (IPC)",
                query='agentsysperf_task_ipc{run_id="$run_id", workload=~"$workload"}',
                x=16, y=y, w=8, h=8,
                legend="{{workload}}",
                unit="short",
            ),
        ]

    def _comparison_panels(self, y: int) -> List[Dict[str, Any]]:
        """Build per-workload comparison panels."""
        return [
            self._table_panel(
                title="Workload Summary Table",
                queries=[
                    ('agentsysperf_task_duration_seconds{run_id="$run_id"}', "Duration (s)"),
                    ('agentsysperf_task_cpu_time_seconds{run_id="$run_id"}', "CPU Time (s)"),
                    ('agentsysperf_task_ipc{run_id="$run_id"}', "IPC"),
                    ('agentsysperf_task_cache_miss_percent{run_id="$run_id"}', "Cache Miss %"),
                    ('agentsysperf_task_memory_rss_bytes{run_id="$run_id"} / 1048576', "RSS (MB)"),
                ],
                x=0, y=y, w=24, h=8,
            ),
        ]

    # ─── Panel Templates ─────────────────────────────────────────────

    def _row_panel(self, title: str, y: int) -> Dict[str, Any]:
        return {
            "type": "row",
            "title": title,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "collapsed": False,
        }

    def _stat_panel(self, title: str, query: str, x: int, y: int,
                    w: int, h: int, unit: str = "short") -> Dict[str, Any]:
        return {
            "type": "stat",
            "title": title,
            "datasource": self._datasource,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [{"expr": query, "refId": "A"}],
            "fieldConfig": {
                "defaults": {"unit": unit, "thresholds": {"steps": [
                    {"color": "green", "value": None},
                ]}},
            },
        }

    def _bar_panel(self, title: str, query: str, x: int, y: int,
                   w: int, h: int, legend: str = "{{workload}}",
                   unit: str = "short") -> Dict[str, Any]:
        return {
            "type": "barchart",
            "title": title,
            "datasource": self._datasource,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [{"expr": query, "legendFormat": legend, "refId": "A"}],
            "fieldConfig": {"defaults": {"unit": unit}},
            "options": {"orientation": "horizontal", "showValue": "always"},
        }

    def _stacked_bar_panel(self, title: str, queries: List[tuple],
                           x: int, y: int, w: int, h: int) -> Dict[str, Any]:
        targets = []
        for i, (expr, legend) in enumerate(queries):
            targets.append({
                "expr": expr,
                "legendFormat": legend,
                "refId": chr(65 + i),
            })

        return {
            "type": "barchart",
            "title": title,
            "datasource": self._datasource,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": targets,
            "fieldConfig": {"defaults": {"unit": "percentunit"}},
            "options": {
                "stacking": {"mode": "normal"},
                "orientation": "horizontal",
                "showValue": "auto",
            },
        }

    def _table_panel(self, title: str, queries: List[tuple],
                     x: int, y: int, w: int, h: int) -> Dict[str, Any]:
        targets = []
        for i, (expr, legend) in enumerate(queries):
            targets.append({
                "expr": expr,
                "legendFormat": legend,
                "refId": chr(65 + i),
                "format": "table",
                "instant": True,
            })

        return {
            "type": "table",
            "title": title,
            "datasource": self._datasource,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": targets,
            "transformations": [
                {"id": "merge", "options": {}},
            ],
        }


__all__ = ["GrafanaDashboardGenerator"]
