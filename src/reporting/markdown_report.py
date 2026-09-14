#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Markdown report generator (P11).

A lightweight, dependency-free :class:`ReportGenerator` — useful for quick text
reports, PR comments, and CI artifacts where a .pptx is overkill. Consumes the
same ResultStore API as the PowerPoint generator (get_run / query_tasks /
query_verdicts / query_measurements), so it works against any backend.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.protocols import ResultStore

logger = logging.getLogger(__name__)


class MarkdownReportGenerator:
    """Render a benchmark run as a single Markdown file."""

    name: str = "markdown"
    output_formats = frozenset(["md"])

    def generate_report(self, *, run_id: str, store: ResultStore, output_path: Path) -> Path:
        logger.info("Generating Markdown report for run %s", run_id)
        run = (store.get_run(run_id) if hasattr(store, "get_run") else None) or {}
        tasks = store.query_tasks(run_id)
        measurements = (
            store.query_measurements(run_id) if hasattr(store, "query_measurements") else []
        )

        lines: List[str] = []
        lines += self._header(run_id, run, tasks)
        lines += self._provenance(run)
        lines += self._task_table(tasks, measurements)
        lines += self._verdicts(run_id, store)
        lines += self._layer_coverage(measurements)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n")
        logger.info("Markdown report saved to %s", output_path)
        return output_path

    # ── sections ───────────────────────────────────────────────────────

    @staticmethod
    def _header(run_id: str, run: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[str]:
        total = len(tasks)
        passed = sum(1 for t in tasks if t.get("passed"))
        rate = f"{passed / total * 100:.0f}%" if total else "—"
        bench = run.get("benchmark_id") or "—"
        out = [
            f"# AgentSysPerf Report — `{run_id}`",
            "",
            f"- **Benchmark:** {bench}",
            f"- **Tasks:** {passed}/{total} passed ({rate})",
        ]
        # Omit Model entirely rather than printing "—" for a run that made no
        # LLM call: an em-dash reads as "unknown", which invites the reader to
        # assume a model was used and the field simply was not captured.
        if run.get("model"):
            out.append(f"- **Model:** {run['model']}")
        out += [
            f"- **Hardware:** {run.get('hardware_sku') or '—'}",
            "",
        ]
        return out

    @staticmethod
    def _provenance(run: Dict[str, Any]) -> List[str]:
        meta = run.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = None
        meta = meta if isinstance(meta, dict) else {}
        plat = meta.get("platform") or {}
        env = meta.get("environment") or {}

        uarch = plat.get("microarchitecture")
        if uarch and plat.get("uarch_source"):
            uarch = f"{uarch} (via {plat['uarch_source']})"

        governor = env.get("cpu_governor")
        if governor and governor != "performance":
            # Flagged inline: a run collected under a scaling governor is not
            # comparable to one collected under 'performance'.
            governor = f"{governor} ⚠ not 'performance'"

        # Only emit rows that are actually present — keeps the report honest.
        rows = [
            ("Optimization profile", run.get("optimization_profile")),
            ("NUMA policy", run.get("numa_policy")),
            ("Microarchitecture", uarch),
            ("CPU governor", governor),
            ("perf_event_paranoid", env.get("perf_event_paranoid")),
            ("Owner", run.get("owner_id")),
            ("Host", run.get("host_id")),
            ("AgentSysPerf version", run.get("agentsysperf_version")),
            ("Status", run.get("status")),
        ]
        present = [(k, v) for k, v in rows if v not in (None, "")]
        if not present:
            return []
        out = ["## Provenance", "", "| Field | Value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in present]
        out.append("")
        return out

    @staticmethod
    def _task_table(tasks: List[Dict[str, Any]], measurements: List[Dict[str, Any]]) -> List[str]:
        if not tasks:
            return ["## Tasks", "", "_No task results recorded._", ""]
        # Index L1/L3 hot metrics per task_id from the measurement payloads.
        by_task: Dict[str, Dict[str, Any]] = {}
        for m in measurements:
            tid = m.get("task_id")
            if tid is None:
                continue
            slot = by_task.setdefault(tid, {})
            p = m.get("payload", {})
            if m.get("layer") == "l3":
                # A task can have several l3 spans (per turn); keep the last
                # non-null value, rounded for display.
                if p.get("ipc") is not None:
                    slot["ipc"] = round(float(p["ipc"]), 2)
                if p.get("cache_miss_pct") is not None:
                    slot["cache_miss_pct"] = round(float(p["cache_miss_pct"]), 1)
        out = [
            "## Tasks", "",
            "| Task | Passed | Duration (s) | IPC | Cache miss % |",
            "|---|---|---|---|---|",
        ]
        for t in tasks:
            tid = t.get("task_id", "—")
            hw = by_task.get(tid, {})
            dur = t.get("duration_s")
            dur_s = f"{dur:.1f}" if dur is not None else "—"
            out.append(
                f"| {tid} | {'✓' if t.get('passed') else '✗'} | "
                f"{dur_s} | {hw.get('ipc', '—')} | {hw.get('cache_miss_pct', '—')} |"
            )
        out.append("")
        return out

    @staticmethod
    def _verdicts(run_id: str, store: ResultStore) -> List[str]:
        try:
            verdicts = store.query_verdicts(run_id)
        except Exception:
            verdicts = []
        if not verdicts:
            return []
        # Include the task the verdict is about. Without it a per-task analyzer
        # on an N-task run renders as N byte-identical rows, which reads as a
        # duplication bug rather than as per-task results. query_verdicts
        # already returns task_id; only the rendering was dropping it.
        out = [
            "## Analyzer verdicts", "",
            "| Analyzer | Task | Verdict | Confidence |", "|---|---|---|---|",
        ]
        for v in verdicts:
            conf = v.get("confidence")
            conf_s = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "—"
            # A run-scoped verdict (no task) is labelled as such rather than
            # borrowing a task id it does not belong to. The store writes the
            # literal "unknown" for a verdict with no span_id
            # (sqlite_store.store_analysis_results), so both it and a NULL mean
            # run-wide here.
            task = v.get("task_id")
            task = "_run-wide_" if task in (None, "", "unknown") else task
            out.append(
                f"| {v.get('analyzer_name', '—')} | {task} | "
                f"{v.get('verdict', '—')} | {conf_s} |"
            )
        out.append("")
        return out

    @staticmethod
    def _layer_coverage(measurements: List[Dict[str, Any]]) -> List[str]:
        if not measurements:
            return []
        counts: Dict[str, int] = {}
        for m in measurements:
            counts[m.get("layer", "?")] = counts.get(m.get("layer", "?"), 0) + 1
        cov = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        return [f"_Measurement coverage — {cov}_", ""]


__all__ = ["MarkdownReportGenerator"]
