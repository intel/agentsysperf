#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""PowerPoint report generator for AgentSysPerf Terminal-Bench results.

Generates an 8-12 slide executive presentation highlighting Xeon EMR performance
on Terminal-Bench workloads with analyzer insights and recommendations.

Usage:
    from src.reporting.xeon_pptx import XeonPowerPointGenerator
    from src.storage import SQLiteResultStore

    generator = XeonPowerPointGenerator()
    store = SQLiteResultStore(output_dir=Path(f"{_TMP}/results"))
    output_path = generator.generate_report(
        run_id="run_20260528",
        store=store,
        output_path=Path(f"{_TMP}/report.pptx")
    )
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src.protocols import ResultStore
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

logger = logging.getLogger(__name__)

# Intel brand colors
INTEL_BLUE = "#0071C5"
INTEL_CYAN = "#00C7FD"
INTEL_ORANGE = "#F0AB00"
INTEL_GRAY = "#5B6770"
INTEL_LIGHT_GRAY = "#D1D3D4"

# Chart color palette (Intel-aligned)
CHART_COLORS = [INTEL_BLUE, INTEL_CYAN, INTEL_ORANGE, INTEL_GRAY]


class XeonPowerPointGenerator:
    """Generate PowerPoint reports from Terminal-Bench benchmark results.

    Produces an 8-12 slide presentation with:
    - Executive summary
    - Workload breakdown analysis
    - CPU bottleneck classification
    - Cache behavior analysis
    - Task-level detail tables
    - Actionable recommendations
    """

    name: str = "xeon_pptx"
    output_formats = frozenset(["pptx"])

    def generate_report(
        self,
        *,
        run_id: str,
        store: ResultStore,
        output_path: Path,
    ) -> Path:
        """Generate PowerPoint report from stored results.

        Args:
            run_id: Unique identifier for the benchmark run.
            store: ResultStore instance to query data from.
            output_path: Target path for the generated .pptx file.

        Returns:
            Path to the generated PowerPoint file.
        """
        logger.info("Generating PowerPoint report for run %s", run_id)

        # Create presentation with 16:9 aspect ratio
        prs = Presentation()
        prs.slide_width = Inches(10)
        prs.slide_height = Inches(5.625)

        # Fetch metadata for context
        metadata = self._fetch_run_metadata(run_id, store)

        # Slide 1: Title + Summary
        self._add_title_slide(prs, run_id, store, metadata)

        # Slide 2: Breakdown by Workload Type
        self._add_breakdown_chart(prs, run_id, store)

        # Slide 3: CPU Classification
        self._add_cpu_classification_chart(prs, run_id, store)

        # Slide 4: Cache Behavior
        self._add_cache_chart(prs, run_id, store)

        # Slide 4b: Per-task phase timeline (Gantt) — needs StepTrace spans
        self._add_phase_timeline(prs, run_id, store)

        # Slides 5-7: Task Detail Tables (5 tasks per slide)
        self._add_task_tables(prs, run_id, store)

        # Slide 8: Recommendations
        self._add_recommendations_slide(prs, run_id, store)

        # Save presentation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(output_path))
        logger.info("PowerPoint report saved to %s", output_path)

        return output_path

    def _fetch_run_metadata(self, run_id: str, store: ResultStore) -> Dict[str, Any]:
        """Fetch run metadata (hardware SKU, model, timestamps)."""
        # Query tasks to extract metadata indirectly
        # SQLiteResultStore doesn't expose query_run_metadata, so we reconstruct
        tasks = store.query_tasks(run_id)

        # Default metadata
        metadata = {
            "hardware_sku": "Intel Xeon EMR",
            "model": "claude-sonnet-3.5",
            "start_time": datetime.now(),
            "total_tasks": len(tasks),
            "passed_tasks": sum(1 for t in tasks if t.get("passed", False)),
        }

        return metadata

    def _add_title_slide(
        self, prs: Presentation, run_id: str, store: ResultStore, metadata: Dict[str, Any]
    ) -> None:
        """Slide 1: Title page with executive summary."""
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(1))
        title_frame = title_box.text_frame
        title_frame.text = "Terminal-Bench 2 Performance Analysis"
        title_para = title_frame.paragraphs[0]
        title_para.font.size = Pt(32)
        title_para.font.bold = True
        title_para.font.color.rgb = self._hex_to_rgb(INTEL_BLUE)
        title_para.alignment = PP_ALIGN.CENTER

        # Subtitle
        subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(0.6))
        subtitle_frame = subtitle_box.text_frame
        subtitle_frame.text = f"Intel Xeon EMR — Agentic AI Workload Characterization"
        subtitle_para = subtitle_frame.paragraphs[0]
        subtitle_para.font.size = Pt(20)
        subtitle_para.font.color.rgb = self._hex_to_rgb(INTEL_GRAY)
        subtitle_para.alignment = PP_ALIGN.CENTER

        # Summary table
        tasks = store.query_tasks(run_id)
        total_tasks = len(tasks)
        passed_tasks = sum(1 for t in tasks if t.get("passed", False))
        pass_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Aggregate workload types
        workload_counts = defaultdict(int)
        for task in tasks:
            wtype = task.get("workload_type", "unknown")
            workload_counts[wtype] += 1

        summary_text = f"""Run ID: {run_id}
Date: {metadata['start_time'].strftime('%Y-%m-%d')}
Hardware: {metadata['hardware_sku']}
Model: {metadata['model']}

Total Tasks: {total_tasks}
Passed: {passed_tasks} ({pass_rate:.0f}%)
Failed: {total_tasks - passed_tasks}

Workload Distribution:
{self._format_workload_distribution(workload_counts)}
"""

        summary_box = slide.shapes.add_textbox(Inches(2), Inches(3), Inches(6), Inches(2))
        summary_frame = summary_box.text_frame
        summary_frame.text = summary_text
        summary_frame.word_wrap = True
        for para in summary_frame.paragraphs:
            para.font.size = Pt(12)
            para.font.name = "Segoe UI"

    def _format_workload_distribution(self, workload_counts: Dict[str, int]) -> str:
        """Format workload distribution as text."""
        lines = []
        for wtype, count in sorted(workload_counts.items()):
            lines.append(f"  • {wtype}: {count}")
        return "\n".join(lines) if lines else "  • No workloads"

    def _add_breakdown_chart(self, prs: Presentation, run_id: str, store: ResultStore) -> None:
        """Slide 2: Breakdown by workload type (grouped bar chart)."""
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        # Title
        self._add_slide_title(slide, "Time Breakdown by Workload Type")

        # Query breakdown analyzer verdicts
        df = self._query_breakdown_data(run_id, store)

        if df.empty:
            self._add_no_data_message(slide, "No breakdown analysis data available")
            return

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

        # Prepare data for grouped bar chart
        workload_types = df["workload_type"].unique()
        x = range(len(workload_types))
        width = 0.25

        # Extract metrics
        inference_pcts = [df[df["workload_type"] == wt]["inference_pct"].mean() for wt in workload_types]
        execution_pcts = [df[df["workload_type"] == wt]["execution_pct"].mean() for wt in workload_types]
        orchestration_pcts = [df[df["workload_type"] == wt]["orchestration_pct"].mean() for wt in workload_types]

        # Plot bars
        ax.bar([i - width for i in x], inference_pcts, width, label="Inference", color=INTEL_BLUE)
        ax.bar(x, execution_pcts, width, label="Execution", color=INTEL_CYAN)
        ax.bar([i + width for i in x], orchestration_pcts, width, label="Orchestration", color=INTEL_ORANGE)

        ax.set_xlabel("Workload Type", fontsize=12)
        ax.set_ylabel("Time (%)", fontsize=12)
        ax.set_title("Time Allocation by Phase", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(workload_types, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Add chart to slide
        self._add_matplotlib_chart(slide, fig, left=Inches(1), top=Inches(1.5), width=Inches(8))

    def _query_breakdown_data(self, run_id: str, store: ResultStore) -> pd.DataFrame:
        """Query breakdown verdicts and join with task workload types."""
        verdicts = store.query_verdicts(run_id=run_id, analyzer_name="breakdown")
        tasks = store.query_tasks(run_id=run_id)

        # Build task_id → workload_type mapping
        task_map = {t["task_id"]: t.get("workload_type", "unknown") for t in tasks}

        # Extract evidence data
        rows = []
        for v in verdicts:
            task_id = v.get("task_id", "unknown")
            evidence = v.get("evidence", {})
            rows.append({
                "task_id": task_id,
                "workload_type": task_map.get(task_id, "unknown"),
                "inference_pct": evidence.get("inference_pct", 0),
                "execution_pct": evidence.get("execution_pct", 0),
                "orchestration_pct": evidence.get("orchestration_pct", 0),
            })

        return pd.DataFrame(rows)

    def _add_cpu_classification_chart(
        self, prs: Presentation, run_id: str, store: ResultStore
    ) -> None:
        """Slide 3: CPU bottleneck classification (stacked bar chart)."""
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        # Title
        self._add_slide_title(slide, "CPU Bottleneck Classification")

        # Query cpu_bound analyzer verdicts
        verdicts = store.query_verdicts(run_id=run_id, analyzer_name="cpu_bound")
        tasks = store.query_tasks(run_id=run_id)

        if not verdicts:
            self._add_no_data_message(slide, "No CPU bottleneck analysis data available")
            return

        # Build task_id → workload_type mapping
        task_map = {t["task_id"]: t.get("workload_type", "unknown") for t in tasks}

        # Aggregate verdicts by workload type
        workload_verdicts = defaultdict(lambda: defaultdict(int))
        for v in verdicts:
            task_id = v.get("task_id", "unknown")
            workload_type = task_map.get(task_id, "unknown")
            verdict = v.get("verdict", "unknown")
            workload_verdicts[workload_type][verdict] += 1

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

        workload_types = list(workload_verdicts.keys())
        verdict_types = ["io_bound", "core_bound", "memory_bound", "frontend_starved"]

        # Build matrix for stacked bars
        data_matrix = []
        for verdict_type in verdict_types:
            row = [workload_verdicts[wt].get(verdict_type, 0) for wt in workload_types]
            data_matrix.append(row)

        # Plot stacked bars
        bottom = [0] * len(workload_types)
        colors = [INTEL_BLUE, INTEL_ORANGE, "#E03C31", INTEL_GRAY]

        for i, (verdict_type, row) in enumerate(zip(verdict_types, data_matrix)):
            ax.bar(workload_types, row, bottom=bottom, label=verdict_type.replace("_", " ").title(), color=colors[i])
            bottom = [b + r for b, r in zip(bottom, row)]

        ax.set_xlabel("Workload Type", fontsize=12)
        ax.set_ylabel("Number of Tasks", fontsize=12)
        ax.set_title("CPU Bottleneck Distribution", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Add chart to slide
        self._add_matplotlib_chart(slide, fig, left=Inches(1), top=Inches(1.5), width=Inches(8))

    def _add_cache_chart(self, prs: Presentation, run_id: str, store: ResultStore) -> None:
        """Slide 4: Cache behavior (grouped bar + line overlay)."""
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        # Title
        self._add_slide_title(slide, "Cache Behavior Analysis")

        # Query cache analyzer verdicts
        verdicts = store.query_verdicts(run_id=run_id, analyzer_name="cache")
        tasks = store.query_tasks(run_id=run_id)

        if not verdicts:
            self._add_no_data_message(slide, "No cache analysis data available")
            return

        # Build task_id → workload_type mapping
        task_map = {t["task_id"]: t.get("workload_type", "unknown") for t in tasks}

        # Aggregate by workload type
        workload_cache = defaultdict(lambda: {"cache_miss_pct": [], "llc_miss_per_s": []})
        for v in verdicts:
            task_id = v.get("task_id", "unknown")
            workload_type = task_map.get(task_id, "unknown")
            evidence = v.get("evidence", {})

            cache_miss_pct = evidence.get("cache_miss_pct")
            llc_miss_per_s = evidence.get("llc_miss_per_s")

            if cache_miss_pct is not None:
                workload_cache[workload_type]["cache_miss_pct"].append(cache_miss_pct)
            if llc_miss_per_s is not None:
                workload_cache[workload_type]["llc_miss_per_s"].append(llc_miss_per_s)

        # Calculate averages
        workload_types = list(workload_cache.keys())
        avg_cache_miss = [
            sum(workload_cache[wt]["cache_miss_pct"]) / len(workload_cache[wt]["cache_miss_pct"])
            if workload_cache[wt]["cache_miss_pct"] else 0
            for wt in workload_types
        ]
        avg_llc_miss_rate = [
            sum(workload_cache[wt]["llc_miss_per_s"]) / len(workload_cache[wt]["llc_miss_per_s"])
            if workload_cache[wt]["llc_miss_per_s"] else 0
            for wt in workload_types
        ]

        # Create dual-axis chart
        fig, ax1 = plt.subplots(figsize=(8, 4), dpi=150)

        x = range(len(workload_types))
        ax1.bar(x, avg_cache_miss, color=INTEL_BLUE, alpha=0.7, label="Cache Miss %")
        ax1.set_xlabel("Workload Type", fontsize=12)
        ax1.set_ylabel("Cache Miss (%)", fontsize=12, color=INTEL_BLUE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(workload_types, rotation=45, ha="right")
        ax1.tick_params(axis="y", labelcolor=INTEL_BLUE)
        ax1.grid(axis="y", alpha=0.3)

        # Secondary axis for LLC miss rate
        ax2 = ax1.twinx()
        ax2.plot(x, avg_llc_miss_rate, color=INTEL_ORANGE, marker="o", linewidth=2, label="LLC Miss/s")
        ax2.set_ylabel("LLC Miss/s", fontsize=12, color=INTEL_ORANGE)
        ax2.tick_params(axis="y", labelcolor=INTEL_ORANGE)

        # Title
        ax1.set_title("Cache Performance by Workload", fontsize=14, fontweight="bold")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()

        # Add chart to slide
        self._add_matplotlib_chart(slide, fig, left=Inches(1), top=Inches(1.5), width=Inches(8))

    def _add_phase_timeline(self, prs: Presentation, run_id: str, store: ResultStore) -> None:
        """Slide: per-task phase timeline (Gantt) from StepTrace spans.

        Picks the task with the most turns, draws each turn's inference (LLM)
        and execution (tool) spans as horizontal bars on a wall-clock axis, and
        renders orchestration as the inter-span gaps (the residual where the
        agent loop itself is working). Requires absolute start_ts_us timestamps.
        """
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        self._add_slide_title(slide, "Agent Phase Timeline (per turn)")

        query_spans = getattr(store, "query_spans", None)
        if query_spans is None:
            self._add_no_data_message(slide, "Store does not support span queries")
            return

        # Choose the most illustrative task: most child spans with real timestamps.
        all_spans = [s for s in query_spans(run_id) if s.get("start_ts_us")]
        if not all_spans:
            self._add_no_data_message(
                slide, "No step-trace timestamps available (run with StepTrace enabled)"
            )
            return

        by_task: Dict[str, list] = {}
        for s in all_spans:
            if s.get("span_kind") in ("llm_call", "tool_call"):
                by_task.setdefault(s.get("task_id", "?"), []).append(s)
        if not by_task:
            self._add_no_data_message(slide, "No per-turn spans to plot")
            return

        task_id = max(by_task, key=lambda t: len(by_task[t]))
        spans = sorted(by_task[task_id], key=lambda s: s["start_ts_us"])
        t0 = spans[0]["start_ts_us"]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        # One row per span, newest at bottom; color by phase.
        phase_color = {"llm_call": INTEL_BLUE, "tool_call": INTEL_ORANGE}
        phase_label = {"llm_call": "Inference (LLM)", "tool_call": "Execution (tool)"}
        seen = set()
        for i, s in enumerate(spans):
            start_s = (s["start_ts_us"] - t0) / 1e6
            dur_s = max((s.get("end_ts_us", 0) - s["start_ts_us"]) / 1e6, 0.01)
            kind = s.get("span_kind")
            color = phase_color.get(kind, INTEL_GRAY)
            lbl = phase_label.get(kind, kind) if kind not in seen else None
            seen.add(kind)
            ax.barh(i, dur_s, left=start_s, height=0.6, color=color, label=lbl)

        # Orchestration = gaps between consecutive spans (agent-loop residual).
        gap_lbl = "Orchestration (gap)"
        for i in range(1, len(spans)):
            prev_end = (spans[i - 1].get("end_ts_us", 0) - t0) / 1e6
            cur_start = (spans[i]["start_ts_us"] - t0) / 1e6
            if cur_start - prev_end > 0.05:  # >50ms gap worth showing
                ax.barh(i - 0.5, cur_start - prev_end, left=prev_end, height=0.15,
                        color=INTEL_GRAY, alpha=0.6,
                        label=gap_lbl if gap_lbl not in seen else None)
                seen.add(gap_lbl)

        ax.set_xlabel("Seconds since task start", fontsize=11)
        ax.set_ylabel("Turn sequence", fontsize=11)
        ax.set_title(f"Phase timeline — {task_id.split('/')[-1]}", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._add_matplotlib_chart(slide, fig, left=Inches(1), top=Inches(1.5), width=Inches(8))

    def _add_task_tables(self, prs: Presentation, run_id: str, store: ResultStore) -> None:
        """Slides 5-7: Task detail tables (5 tasks per slide)."""
        tasks = store.query_tasks(run_id=run_id)

        if not tasks:
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            self._add_slide_title(slide, "Task Details")
            self._add_no_data_message(slide, "No task data available")
            return

        # Fetch all verdicts for annotation
        breakdown_verdicts = {
            v["task_id"]: v["verdict"]
            for v in store.query_verdicts(run_id=run_id, analyzer_name="breakdown")
        }
        cpu_verdicts = {
            v["task_id"]: v["verdict"]
            for v in store.query_verdicts(run_id=run_id, analyzer_name="cpu_bound")
        }

        # Fetch IPC from cpu_bound evidence
        cpu_evidence = {
            v["task_id"]: v["evidence"]
            for v in store.query_verdicts(run_id=run_id, analyzer_name="cpu_bound")
        }

        # Paginate tasks (5 per slide)
        tasks_per_slide = 5
        for i in range(0, len(tasks), tasks_per_slide):
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            page_num = i // tasks_per_slide + 1
            self._add_slide_title(slide, f"Task Details (Page {page_num})")

            # Create table
            rows = min(tasks_per_slide + 1, len(tasks) - i + 1)  # +1 for header
            cols = 7

            left = Inches(0.5)
            top = Inches(1.5)
            width = Inches(9)
            height = Inches(3.5)

            table = slide.shapes.add_table(rows, cols, left, top, width, height).table

            # Set column widths
            col_widths = [Inches(1.2), Inches(1.2), Inches(0.7), Inches(1), Inches(1.5), Inches(1.5), Inches(0.9)]
            for col_idx, col_width in enumerate(col_widths):
                table.columns[col_idx].width = col_width

            # Header row
            headers = ["Task ID", "Workload", "Status", "Duration(s)", "Breakdown", "CPU Verdict", "IPC"]
            for col_idx, header in enumerate(headers):
                cell = table.cell(0, col_idx)
                cell.text = header
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.size = Pt(10)
                cell.fill.solid()
                cell.fill.fore_color.rgb = self._hex_to_rgb(INTEL_BLUE)
                cell.text_frame.paragraphs[0].font.color.rgb = self._hex_to_rgb("#FFFFFF")

            # Data rows
            for row_idx, task in enumerate(tasks[i:i+tasks_per_slide], start=1):
                task_id = task["task_id"]
                workload = task.get("workload_type", "N/A")
                passed = "PASS" if task.get("passed", False) else "FAIL"
                duration = f"{task.get('duration_s', 0):.1f}"
                breakdown = breakdown_verdicts.get(task_id, "N/A")
                cpu = cpu_verdicts.get(task_id, "N/A")
                ipc = f"{cpu_evidence.get(task_id, {}).get('ipc', 0):.2f}" if task_id in cpu_evidence else "N/A"

                row_data = [task_id, workload, passed, duration, breakdown, cpu, ipc]

                for col_idx, value in enumerate(row_data):
                    cell = table.cell(row_idx, col_idx)
                    cell.text = str(value)
                    cell.text_frame.paragraphs[0].font.size = Pt(9)

                    # Color-code status
                    if col_idx == 2:  # Status column
                        if passed == "PASS":
                            cell.text_frame.paragraphs[0].font.color.rgb = self._hex_to_rgb("#006B3C")
                        else:
                            cell.text_frame.paragraphs[0].font.color.rgb = self._hex_to_rgb("#E03C31")

    def _add_recommendations_slide(
        self, prs: Presentation, run_id: str, store: ResultStore
    ) -> None:
        """Slide 8: Aggregated recommendations from all analyzers."""
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        # Title
        self._add_slide_title(slide, "Recommendations for Xeon EMR Optimization")

        # Query all verdicts
        all_verdicts = store.query_verdicts(run_id=run_id)

        if not all_verdicts:
            self._add_no_data_message(slide, "No recommendations available")
            return

        # Aggregate recommendations by theme
        recommendation_themes = defaultdict(lambda: {"count": 0, "confidence": 0.0, "examples": []})

        for verdict in all_verdicts:
            recommendations = verdict.get("recommendations", [])
            confidence = verdict.get("confidence", 0.0)

            for rec in recommendations:
                # Categorize recommendation (heuristic based on keywords)
                theme = self._categorize_recommendation(rec)
                recommendation_themes[theme]["count"] += 1
                recommendation_themes[theme]["confidence"] += confidence
                if len(recommendation_themes[theme]["examples"]) < 2:
                    recommendation_themes[theme]["examples"].append(rec)

        # Calculate average confidence per theme
        for theme in recommendation_themes:
            count = recommendation_themes[theme]["count"]
            recommendation_themes[theme]["avg_confidence"] = (
                recommendation_themes[theme]["confidence"] / count if count > 0 else 0.0
            )

        # Sort by count (most common first)
        sorted_themes = sorted(
            recommendation_themes.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )

        # Build recommendation text
        rec_text = ""
        for theme, data in sorted_themes[:6]:  # Top 6 themes
            rec_text += f"{theme} (n={data['count']}, confidence={data['avg_confidence']:.1f})\n"
            for example in data["examples"]:
                rec_text += f"  • {example}\n"
            rec_text += "\n"

        # Add text box
        text_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(3.8))
        text_frame = text_box.text_frame
        text_frame.text = rec_text.strip()
        text_frame.word_wrap = True

        for para in text_frame.paragraphs:
            para.font.size = Pt(11)
            para.font.name = "Segoe UI"

    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize a recommendation into a theme based on keywords."""
        rec_lower = recommendation.lower()

        if any(kw in rec_lower for kw in ["model", "inference", "llm", "latency"]):
            return "Model Selection"
        elif any(kw in rec_lower for kw in ["memory", "bandwidth", "dram", "numa"]):
            return "Memory Optimization"
        elif any(kw in rec_lower for kw in ["cache", "l3", "l2", "l1"]):
            return "Cache Optimization"
        elif any(kw in rec_lower for kw in ["cpu", "core", "frequency", "sku"]):
            return "Xeon SKU Selection"
        elif any(kw in rec_lower for kw in ["hugepage", "isolcpus", "pinning"]):
            return "System Configuration"
        elif any(kw in rec_lower for kw in ["execution", "command", "async", "parallel"]):
            return "Execution Efficiency"
        else:
            return "General Optimization"

    def _add_matplotlib_chart(
        self,
        slide,
        fig,
        left: Inches = Inches(1),
        top: Inches = Inches(1.5),
        width: Inches = Inches(8),
    ) -> None:
        """Convert matplotlib figure to PNG and add to slide."""
        img_stream = BytesIO()
        fig.savefig(img_stream, format="png", dpi=150, bbox_inches="tight")
        img_stream.seek(0)
        slide.shapes.add_picture(img_stream, left, top, width=width)
        plt.close(fig)

    def _add_slide_title(self, slide, title: str) -> None:
        """Add a title to a slide."""
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
        title_frame = title_box.text_frame
        title_frame.text = title
        title_para = title_frame.paragraphs[0]
        title_para.font.size = Pt(24)
        title_para.font.bold = True
        title_para.font.color.rgb = self._hex_to_rgb(INTEL_BLUE)

    def _add_no_data_message(self, slide, message: str) -> None:
        """Add a centered 'no data' message to a slide."""
        msg_box = slide.shapes.add_textbox(Inches(2), Inches(2.5), Inches(6), Inches(1))
        msg_frame = msg_box.text_frame
        msg_frame.text = message
        msg_para = msg_frame.paragraphs[0]
        msg_para.font.size = Pt(16)
        msg_para.font.italic = True
        msg_para.font.color.rgb = self._hex_to_rgb(INTEL_GRAY)
        msg_para.alignment = PP_ALIGN.CENTER

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> RGBColor:
        """Convert a hex color string to a python-pptx RGBColor.

        font.color.rgb / fill.fore_color.rgb require an RGBColor instance, not
        a plain tuple — returning a tuple raised "assigned value must be type
        RGBColor" and broke report generation.
        """
        hex_color = hex_color.lstrip("#")
        return RGBColor(*(int(hex_color[i:i + 2], 16) for i in (0, 2, 4)))


__all__ = ["XeonPowerPointGenerator"]
