#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""PhaseProfiler — per-phase hardware characterization across iterations.

Consumes phase-tagged MeasurementRecords (from track_span(phase="reason") etc.)
and produces:

1. Per-phase time/hardware breakdown (wall-clock %, CPU %, avg IPC, avg cache miss)
2. Per-iteration trends (does Retrieve grow? does Act spike?)
3. Inflection detection: which iteration does orchestration (non-Reason) > inference?
4. Per-phase solution mapping via hardware pattern classification

The 5 pipeline phases:
  01 Admit    — auth, policy, route          (Governance / PRE-PLAN)
  02 Retrieve — vector, rerank, fetch        (Context Enrichment / OBSERVE)
  03 Reason   — LLM generates answer         (Reasoning / PLAN & INFER)
  04 Act      — tool call, API, code exec    (Execution / ACT)
  05 Commit   — write-back, audit, cache     (Governance / POST-ACT)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.protocols import AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)

PHASE_ORDER = ["admit", "retrieve", "reason", "act", "commit"]

PHASE_LABELS = {
    "admit": "01 Admit (Governance)",
    "retrieve": "02 Retrieve (Context Enrichment)",
    "reason": "03 Reason (Inference)",
    "act": "04 Act (Execution)",
    "commit": "05 Commit (Governance)",
}

PHASE_SOLUTIONS = {
    "admit": {
        "compute_efficient": [],
        "default": ["Connection pooling for auth service"],
    },
    "retrieve": {
        "working_set_overflow": ["NUMA-aware index partitioning", "Quantized embeddings (int8 HNSW)"],
        "bandwidth_saturation": ["Memory interleaving", "Quantized embeddings"],
        "cross_numa_traffic": ["Pin index shards to local NUMA node"],
        "default": ["Passage-level RAG (reduce retrieved doc size)"],
    },
    "reason": {
        "weight_streaming": ["Speculative decoding", "Quantization (INT8/INT4)", "Model sharding"],
        "kv_cache_pressure": ["KV-cache compression", "Context pruning", "RAG"],
        "bandwidth_saturation": ["Model sharding across NUMA nodes"],
        "default": ["Speculative decoding"],
    },
    "act": {
        "io_dominant": ["Async parallel tool execution", "Container pooling"],
        "compute_efficient": ["Workload isolation", "CPU pinning"],
        "default": ["Async parallel execution", "Warm container starts"],
    },
    "commit": {
        "io_dominant": ["Async batched writes", "Write-behind cache"],
        "default": ["Binary protocols (protobuf vs JSON)"],
    },
}


@dataclass
class PhaseMetrics:
    """Aggregated metrics for one phase across all iterations."""
    phase: str
    span_count: int = 0
    total_duration_us: int = 0
    total_cpu_time_s: float = 0.0
    ipc_values: List[float] = field(default_factory=list)
    cache_miss_values: List[float] = field(default_factory=list)
    rss_kb_values: List[int] = field(default_factory=list)

    @property
    def avg_ipc(self) -> Optional[float]:
        return sum(self.ipc_values) / len(self.ipc_values) if self.ipc_values else None

    @property
    def avg_cache_miss(self) -> Optional[float]:
        return sum(self.cache_miss_values) / len(self.cache_miss_values) if self.cache_miss_values else None

    @property
    def wall_ms(self) -> float:
        return self.total_duration_us / 1000.0


@dataclass
class IterationPhasePoint:
    """One phase measurement within one iteration (turn)."""
    iteration: int
    phase: str
    duration_us: int = 0
    cpu_time_s: float = 0.0
    ipc: Optional[float] = None
    cache_miss_pct: Optional[float] = None


def _extract_turn_index(span_id: str) -> Optional[int]:
    """Extract turn number from span_id like 'task_id/turn_3_llm'."""
    m = re.search(r"/turn_(\d+)_", span_id)
    if m:
        return int(m.group(1))
    return None


def _extract_phase_from_span_id(span_id: str) -> Optional[str]:
    """Infer phase from span_id naming convention when phase tag is missing."""
    if "_llm" in span_id:
        return "reason"
    if "_cmd" in span_id:
        return "act"
    return None


def _classify_pattern(ipc: Optional[float], cache_miss: Optional[float],
                      wall_ms: float, cpu_time_s: float) -> str:
    """Classify the hardware pattern for a phase."""
    if ipc is None or cache_miss is None:
        if wall_ms > 0 and cpu_time_s > 0:
            ratio = (wall_ms / 1000.0) / cpu_time_s
            if ratio > 5.0:
                return "io_dominant"
        return "unknown"

    if ipc < 1.5 and cache_miss > 70:
        return "weight_streaming"
    if cache_miss > 40 and ipc < 2.5:
        return "working_set_overflow"
    if cache_miss > 60 and ipc > 1.5:
        return "bandwidth_saturation"
    if ipc > 3.5 and cache_miss < 15:
        return "compute_efficient"

    if wall_ms > 0 and cpu_time_s > 0:
        ratio = (wall_ms / 1000.0) / cpu_time_s
        if ratio > 5.0:
            return "io_dominant"

    return "mixed"


class PhaseProfiler:
    """Aggregates per-phase hardware metrics across iterations.

    Produces:
    - Per-phase time breakdown (wall-clock and CPU-time)
    - Per-phase hardware signature (IPC, cache miss averages)
    - Phase-over-iteration trends
    - Inflection detection: which iteration does orchestration > inference?
    - Per-phase solution mapping based on detected hardware pattern
    """

    name = "phase_profiler"
    input_layers = frozenset(["l1", "l3"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Any = None,
    ) -> Iterable[AnalysisResult]:
        phase_data = self._group_by_phase(records)
        if not phase_data:
            return

        iteration_points = self._build_iteration_points(records)
        total_wall_us = sum(pm.total_duration_us for pm in phase_data.values())
        total_cpu_s = sum(pm.total_cpu_time_s for pm in phase_data.values())

        if total_wall_us == 0:
            return

        # Build per-phase breakdown
        breakdown = {}
        phase_patterns = {}
        phase_solutions = {}

        for phase in PHASE_ORDER:
            pm = phase_data.get(phase)
            if pm is None or pm.span_count == 0:
                continue

            wall_pct = (pm.total_duration_us / total_wall_us * 100) if total_wall_us else 0
            cpu_pct = (pm.total_cpu_time_s / total_cpu_s * 100) if total_cpu_s else 0

            pattern = _classify_pattern(pm.avg_ipc, pm.avg_cache_miss,
                                        pm.wall_ms, pm.total_cpu_time_s)
            phase_patterns[phase] = pattern

            solutions = PHASE_SOLUTIONS.get(phase, {}).get(
                pattern, PHASE_SOLUTIONS.get(phase, {}).get("default", [])
            )
            phase_solutions[phase] = solutions

            breakdown[phase] = {
                "wall_pct": round(wall_pct, 1),
                "cpu_pct": round(cpu_pct, 1),
                "wall_ms": round(pm.wall_ms, 1),
                "cpu_time_s": round(pm.total_cpu_time_s, 3),
                "span_count": pm.span_count,
                "avg_ipc": round(pm.avg_ipc, 2) if pm.avg_ipc is not None else None,
                "avg_cache_miss_pct": round(pm.avg_cache_miss, 1) if pm.avg_cache_miss is not None else None,
                "pattern": pattern,
                "solutions": solutions,
            }

        # Detect inflection point
        inflection = self._detect_inflection(iteration_points)

        # Determine dominant phase
        dominant_phase = max(breakdown.keys(), key=lambda p: breakdown[p]["wall_pct"]) if breakdown else None

        yield AnalysisResult(
            verdict=f"phase_profile_{dominant_phase}" if dominant_phase else "phase_profile",
            confidence=0.9 if len(breakdown) >= 2 else 0.6,
            evidence={
                "phase_breakdown": breakdown,
                "phase_patterns": phase_patterns,
                "phase_solutions": phase_solutions,
                "inflection": inflection,
                "total_wall_ms": round(total_wall_us / 1000.0, 1),
                "total_cpu_s": round(total_cpu_s, 3),
                "phases_detected": list(breakdown.keys()),
                "iteration_count": max(
                    (p.iteration for p in iteration_points), default=0
                ) + 1 if iteration_points else 0,
            },
            recommendations=self._build_recommendations(breakdown, inflection),
            analyzer_name=self.name,
        )

    def _group_by_phase(self, records: Sequence[MeasurementRecord]) -> Dict[str, PhaseMetrics]:
        """Group records by phase, merging L1 and L3 data."""
        # First pass: build span_id -> phase mapping from L1 records (which have duration)
        span_phases: Dict[str, str] = {}
        span_l1: Dict[str, Dict[str, Any]] = {}
        span_l3: Dict[str, Dict[str, Any]] = {}

        for rec in records:
            sid = rec.span_id
            # Try to determine phase from payload or span_id convention
            phase = rec.payload.get("phase") if hasattr(rec.payload, "get") else None
            if phase is None:
                phase = _extract_phase_from_span_id(sid)
            if phase:
                span_phases[sid] = phase

            if rec.layer == "l1":
                span_l1[sid] = dict(rec.payload) if hasattr(rec.payload, "items") else {}
            elif rec.layer == "l3":
                span_l3[sid] = dict(rec.payload) if hasattr(rec.payload, "items") else {}

        # Second pass: build PhaseMetrics
        result: Dict[str, PhaseMetrics] = {}
        for sid, phase in span_phases.items():
            if phase not in result:
                result[phase] = PhaseMetrics(phase=phase)
            pm = result[phase]
            pm.span_count += 1

            l1 = span_l1.get(sid, {})
            l3 = span_l3.get(sid, {})

            duration_us = l1.get("duration_us", 0)
            pm.total_duration_us += duration_us
            pm.total_cpu_time_s += l1.get("cpu_time_s", 0)

            if l1.get("rss_kb_peak"):
                pm.rss_kb_values.append(l1["rss_kb_peak"])

            ipc = l3.get("ipc")
            if ipc is not None and ipc > 0:
                pm.ipc_values.append(ipc)

            cache_miss = l3.get("cache_miss_pct")
            if cache_miss is not None:
                pm.cache_miss_values.append(cache_miss)

        return result

    def _build_iteration_points(self, records: Sequence[MeasurementRecord]) -> List[IterationPhasePoint]:
        """Build per-iteration per-phase data points for trend analysis."""
        points: List[IterationPhasePoint] = []
        span_data: Dict[str, Dict[str, Any]] = {}

        for rec in records:
            sid = rec.span_id
            if sid not in span_data:
                span_data[sid] = {"span_id": sid}
            if rec.layer == "l1":
                span_data[sid].update(rec.payload)
            elif rec.layer == "l3":
                span_data[sid].update(rec.payload)

        for sid, data in span_data.items():
            turn = _extract_turn_index(sid)
            if turn is None:
                continue
            phase = data.get("phase") or _extract_phase_from_span_id(sid)
            if phase is None:
                continue

            points.append(IterationPhasePoint(
                iteration=turn,
                phase=phase,
                duration_us=data.get("duration_us", 0),
                cpu_time_s=data.get("cpu_time_s", 0.0),
                ipc=data.get("ipc"),
                cache_miss_pct=data.get("cache_miss_pct"),
            ))

        return sorted(points, key=lambda p: (p.iteration, PHASE_ORDER.index(p.phase) if p.phase in PHASE_ORDER else 99))

    def _detect_inflection(self, points: List[IterationPhasePoint]) -> Optional[Dict[str, Any]]:
        """Find the iteration where non-Reason CPU time exceeds Reason CPU time cumulatively."""
        if not points:
            return None

        max_iter = max(p.iteration for p in points)
        if max_iter < 2:
            return None

        cumulative_reason = 0.0
        cumulative_other = 0.0

        by_iteration: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for p in points:
            by_iteration[p.iteration][p.phase] += p.duration_us / 1_000_000.0  # seconds

        for iteration in range(max_iter + 1):
            phases = by_iteration.get(iteration, {})
            reason_s = phases.get("reason", 0.0)
            other_s = sum(v for k, v in phases.items() if k != "reason")

            cumulative_reason += reason_s
            cumulative_other += other_s

            if cumulative_other > cumulative_reason and cumulative_reason > 0:
                return {
                    "iteration": iteration,
                    "reason": "Cumulative orchestration time (Retrieve+Act+Commit) exceeds Reason (inference)",
                    "cumulative_reason_s": round(cumulative_reason, 3),
                    "cumulative_other_s": round(cumulative_other, 3),
                    "ratio": round(cumulative_other / cumulative_reason, 2),
                }

        return None

    def _build_recommendations(self, breakdown: Dict[str, Any],
                               inflection: Optional[Dict[str, Any]]) -> List[str]:
        """Generate prioritized recommendations based on phase analysis."""
        recs = []

        # Find the phase consuming most wall-clock
        if not breakdown:
            return recs

        dominant = max(breakdown.keys(), key=lambda p: breakdown[p]["wall_pct"])
        dominant_pct = breakdown[dominant]["wall_pct"]
        dominant_pattern = breakdown[dominant].get("pattern", "unknown")
        dominant_solutions = breakdown[dominant].get("solutions", [])

        if dominant == "reason" and dominant_pct > 60:
            recs.append(
                f"Inference dominates ({dominant_pct:.0f}% wall-clock). "
                f"Pattern: {dominant_pattern}. "
                f"Top solution: {dominant_solutions[0] if dominant_solutions else 'N/A'}."
            )
        elif dominant == "act":
            recs.append(
                f"Execution dominates ({dominant_pct:.0f}% wall-clock). "
                f"Pattern: {dominant_pattern}. "
                f"Top solution: {dominant_solutions[0] if dominant_solutions else 'N/A'}."
            )
        elif dominant == "retrieve":
            recs.append(
                f"Retrieval dominates ({dominant_pct:.0f}% wall-clock). "
                f"Pattern: {dominant_pattern}. "
                f"Top solution: {dominant_solutions[0] if dominant_solutions else 'N/A'}."
            )

        if inflection:
            recs.append(
                f"Orchestration inflection at iteration {inflection['iteration']}: "
                f"non-inference phases exceed inference cumulatively "
                f"(ratio {inflection['ratio']:.1f}x). "
                f"CPU optimization in Retrieve+Act has more impact than shaving inference latency."
            )

        # Add secondary recommendations for phases > 15% wall-clock
        for phase, data in breakdown.items():
            if phase == dominant:
                continue
            if data["wall_pct"] > 15:
                solutions = data.get("solutions", [])
                if solutions:
                    recs.append(
                        f"{PHASE_LABELS.get(phase, phase)}: {data['wall_pct']:.0f}% wall-clock. "
                        f"Consider: {solutions[0]}."
                    )

        return recs


__all__ = ["PhaseProfiler"]
