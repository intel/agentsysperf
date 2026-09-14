#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""BreakdownAnalyzer — Multi-dimensional time/resource breakdown by span kind.

Aggregates sub-spans by kind (inference, execution, orchestration) to show
where time is spent in agentic AI workloads.

Usage:
    poetry run agentsysperf analyze /tmp/agentsysperf_terminal_bench_*/

Output:
    breakdown: inference_dominant (confidence: 0.90)
      Evidence: inference=45.2s (60%), execution=20.1s (27%), orchestration=9.7s (13%)
      Recommendations:
        • Inference latency dominates — consider faster model or batching
        • Execution overhead is acceptable
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)


class BreakdownAnalyzer:
    """Aggregate sub-spans by kind to show inference/execution/orchestration breakdown.

    Consumes L1 measurements (duration) from sub-spans emitted by agent loop.
    Produces a single breakdown verdict per task span.
    """

    name: str = "breakdown"
    # L1 drives the time breakdown; L3 adds per-phase hardware character.
    input_layers = frozenset(["l1", "l3"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Aggregate sub-spans by kind for each task span."""
        # Group records by task span (span_id without "/" is task-level)
        task_spans = {}
        sub_spans_by_task = defaultdict(list)
        # L3 sub-span records by parent task, for per-phase hardware rollup.
        l3_sub_by_task = defaultdict(list)

        for record in records:
            if record.layer == "l3":
                if "/turn_" in record.span_id:
                    l3_sub_by_task[record.span_id.split("/turn_")[0]].append(record)
                continue
            if record.layer != "l1":
                continue

            # Hierarchical span IDs: "task_id/turn_0_llm" or "task_id" (task-level)
            # Sub-spans have "/turn_" pattern
            if "/turn_" in record.span_id:
                # Sub-span: extract parent task_id (everything before "/turn_")
                parent_task_id = record.span_id.split("/turn_")[0]
                sub_spans_by_task[parent_task_id].append(record)
            else:
                # Task-level span
                task_spans[record.span_id] = record

        # For each task span, aggregate its sub-spans by kind
        for task_span_id, task_record in task_spans.items():
            sub_spans = sub_spans_by_task.get(task_span_id, [])

            # Aggregate by kind
            breakdown = defaultdict(lambda: {"duration_us": 0, "count": 0})
            task_duration_us = task_record.payload.get("duration_us", 0)

            for sub_rec in sub_spans:
                kind = sub_rec.payload.get("kind", "orchestration")
                duration_us = sub_rec.payload.get("duration_us", 0)
                breakdown[kind]["duration_us"] += duration_us
                breakdown[kind]["count"] += 1

            if task_duration_us == 0:
                logger.debug("BreakdownAnalyzer: task %s has zero duration", task_span_id)
                continue

            # Calculate orchestration as residual
            inference_us = breakdown.get("inference", {}).get("duration_us", 0)
            execution_us = breakdown.get("execution", {}).get("duration_us", 0)
            measured_us = inference_us + execution_us
            orchestration_us = max(0, task_duration_us - measured_us)

            # Convert to seconds
            task_duration_s = task_duration_us / 1e6
            inference_s = inference_us / 1e6
            execution_s = execution_us / 1e6
            orchestration_s = orchestration_us / 1e6

            # Calculate percentages
            inference_pct = (inference_s / task_duration_s) * 100 if task_duration_s > 0 else 0
            execution_pct = (execution_s / task_duration_s) * 100 if task_duration_s > 0 else 0
            orchestration_pct = (orchestration_s / task_duration_s) * 100 if task_duration_s > 0 else 0

            # Classify dominant category
            dominant_kind = max(
                [("inference", inference_pct), ("execution", execution_pct), ("orchestration", orchestration_pct)],
                key=lambda x: x[1],
            )[0]

            if dominant_kind == "inference":
                if inference_pct > 60:
                    verdict = "inference_dominant"
                    confidence = 0.90
                    recommendations = [
                        f"Inference latency dominates ({inference_pct:.0f}% of time)",
                        "Consider faster model (smaller or distilled) for lower latency",
                        "Explore batching or speculative decoding if multiple calls",
                        "Profile network latency if using hosted LLM",
                    ]
                else:
                    verdict = "inference_heavy"
                    confidence = 0.80
                    recommendations = [
                        f"Inference takes {inference_pct:.0f}% of time",
                        "Optimize model selection or caching strategy",
                    ]
            elif dominant_kind == "execution":
                if execution_pct > 60:
                    verdict = "execution_dominant"
                    confidence = 0.90
                    recommendations = [
                        f"Command execution dominates ({execution_pct:.0f}% of time)",
                        "Profile slow commands (disk I/O, network calls, compilation)",
                        "Consider async execution or parallelization",
                    ]
                else:
                    verdict = "execution_heavy"
                    confidence = 0.80
                    recommendations = [
                        f"Execution takes {execution_pct:.0f}% of time",
                        "Review command efficiency",
                    ]
            else:  # orchestration dominant
                verdict = "orchestration_heavy"
                confidence = 0.75
                recommendations = [
                    f"Orchestration overhead is {orchestration_pct:.0f}% of time",
                    "Check for excessive parsing, context switching, or framework overhead",
                    "Profile agent loop for bottlenecks",
                ]

            evidence = {
                "task_duration_s": round(task_duration_s, 2),
                "inference_s": round(inference_s, 2),
                "inference_pct": round(inference_pct, 1),
                "execution_s": round(execution_s, 2),
                "execution_pct": round(execution_pct, 1),
                "orchestration_s": round(orchestration_s, 2),
                "orchestration_pct": round(orchestration_pct, 1),
                "inference_calls": breakdown.get("inference", {}).get("count", 0),
                "execution_calls": breakdown.get("execution", {}).get("count", 0),
            }

            # Per-phase HARDWARE rollup — execution only. Inference is a remote
            # network wait (the LLM runs elsewhere), so its IPC/cache would
            # measure the Python process idling on a socket — meaningless, so we
            # deliberately omit it. Orchestration is a time residual with no
            # spans, hence no counters. See docs/phase_metrics_and_L2_plan.md.
            exec_hw = self._phase_hardware(
                l3_sub_by_task.get(task_span_id, []), phase_kind="execution"
            )
            if exec_hw:
                evidence["execution_hw"] = exec_hw

            yield AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                recommendations=recommendations,
                span_id=task_span_id,
                analyzer_name=self.name,
            )


    @staticmethod
    def _phase_hardware(l3_records, *, phase_kind: str) -> Optional[dict]:
        """Instruction-weighted hardware rollup over L3 sub-spans of one kind.

        IPC and cache-miss% are averaged weighted by instructions (not naive
        mean — a 10x-longer span should dominate). Returns None if no L3 data
        for this phase.
        """
        total_instr = 0.0
        total_cycles = 0.0
        weighted_cache_miss = 0.0
        weighted_branch_miss = 0.0
        span_count = 0

        for rec in l3_records:
            p = rec.payload
            if p.get("kind") != phase_kind:
                continue
            instr = float(p.get("instructions", 0) or 0)
            cycles = float(p.get("cycles", 0) or 0)
            if instr <= 0 or cycles <= 0:
                continue
            total_instr += instr
            total_cycles += cycles
            weighted_cache_miss += float(p.get("cache_miss_pct", 0) or 0) * instr
            weighted_branch_miss += float(p.get("branch_miss_pct", 0) or 0) * instr
            span_count += 1

        if span_count == 0 or total_instr <= 0:
            return None

        return {
            "ipc": round(total_instr / total_cycles, 3) if total_cycles else 0.0,
            "cache_miss_pct": round(weighted_cache_miss / total_instr, 3),
            "branch_miss_pct": round(weighted_branch_miss / total_instr, 3),
            "instructions": int(total_instr),
            "span_count": span_count,
        }


__all__ = ["BreakdownAnalyzer"]
