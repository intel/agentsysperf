#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""MemoryBandwidthAnalyzer — classifies memory subsystem bottlenecks and maps to solutions.

Analyzes cache behavior, DRAM bandwidth utilization, and NUMA topology
effects to identify memory subsystem bottlenecks. Then maps each
bottleneck pattern to applicable solutions:

Bottleneck Patterns → Solution Mapping:
─────────────────────────────────────────────────────────────────────
Pattern: "weight_streaming"
  Signal: IPC < 1.5, cache miss > 70%, large RSS, sequential access
  Solutions: speculative decoding, quantization (INT8/INT4), model sharding

Pattern: "working_set_overflow"
  Signal: cache miss 40-70%, RSS > L3 budget, moderate IPC
  Solutions: NUMA-aware scheduling, data tiling, L3 partitioning (RDT)

Pattern: "cross_numa_traffic"
  Signal: remote DRAM access > 20%, hop penalty visible in latency
  Solutions: NUMA pinning, memory-local scheduling, SNC mode tuning

Pattern: "bandwidth_saturation"
  Signal: DRAM BW > 70% of MLC peak, queuing delays rising
  Solutions: memory interleaving (STRIPE), workload spreading, DDR5 channels

Pattern: "kv_cache_pressure"
  Signal: RSS grows with context length, cache miss spikes mid-inference
  Solutions: KV-cache compression, paged attention, context pruning, RAG (reduce context)

Pattern: "capacity_thrashing"
  Signal: cache miss rate oscillates, IPC unstable, LLC MPKI very high
  Solutions: cache partitioning (Intel RDT/CAT), workload isolation, fewer co-tenants

References:
- Intel Top-Down Methodology (TMA) — Backend-Memory decomposition
- EMAT framework (emat-benchmark) — tier selection by working set size
- Intel Memory Latency Checker (MLC) — measured peak bandwidth baseline
- Intel PerfSpect (https://github.com/intel/PerfSpect) — TMA and uncore metric collection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from src.platform import PlatformInfo, detect_platform
from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)


def _as_pct(value: Optional[float]) -> Optional[float]:
    """Normalize a TMA bucket to percent (0-100).

    The PerfSpect plugin stores TMA buckets as 0-1 ratios while the thresholds
    in this module are written in percent, so a raw ratio would read as ~100x
    too small and never trip. Values <= 1.0 are treated as ratios; anything
    larger is assumed to already be a percentage.
    """
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


# ─── Solution Registry ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Solution:
    """A potential fix for a memory bandwidth bottleneck."""
    name: str
    category: str           # scheduling | algorithmic | hardware | architecture
    applicability: str      # When this solution applies
    expected_impact: str    # Estimated improvement range
    prerequisites: str      # What's needed to apply


SOLUTIONS = {
    "speculative_decoding": Solution(
        name="Speculative Decoding",
        category="algorithmic",
        applicability="Autoregressive decode with model weights >> L3 and batch=1",
        expected_impact="2-4x decode throughput (depends on draft model acceptance rate)",
        prerequisites="Draft model that fits in L3 (size depends on platform — see platform.l3_budget_mb)",
    ),
    "quantization": Solution(
        name="Weight Quantization (INT8/INT4)",
        category="algorithmic",
        applicability="Model weights dominate memory traffic; accuracy budget allows",
        expected_impact="2-4x memory reduction, proportional BW savings",
        prerequisites="Quantization-aware training or post-training calibration",
    ),
    "numa_pinning": Solution(
        name="NUMA-Aware Scheduling",
        category="scheduling",
        applicability="Cross-NUMA traffic detected; workload not pinned to local node",
        expected_impact="15-40% latency reduction (eliminates 25ns/hop penalty)",
        prerequisites="numactl or VLLM_CPU_OMP_THREADS_BIND; identify optimal node",
    ),
    "memory_local_scheduling": Solution(
        name="Memory-Local Task Scheduling",
        category="scheduling",
        applicability="Multiple tasks competing for same NUMA node memory",
        expected_impact="20-50% throughput improvement under contention",
        prerequisites="Task classifier (WSS-aware routing) + NUMA topology map",
    ),
    "rag_context_reduction": Solution(
        name="RAG (Retrieval-Augmented Generation)",
        category="architecture",
        applicability="KV-cache growing with long context; most context is retrievable",
        expected_impact="Reduce context 50-90%, proportional cache pressure relief",
        prerequisites="Document index + retrieval pipeline; acceptable retrieval latency",
    ),
    "kv_cache_compression": Solution(
        name="KV-Cache Compression / Paged Attention",
        category="algorithmic",
        applicability="KV-cache grows beyond L3; context length > 4K tokens",
        expected_impact="2-4x KV memory reduction; keeps more cache entries resident",
        prerequisites="vLLM PagedAttention or equivalent; may need model support",
    ),
    "data_tiling": Solution(
        name="Data Tiling / Blocking",
        category="algorithmic",
        applicability="Working set slightly exceeds L3; access pattern is tileable",
        expected_impact="1.5-3x from improved cache reuse",
        prerequisites="Restructure data access to tile-sized blocks fitting L3",
    ),
    "cache_partitioning": Solution(
        name="Cache Partitioning (Intel RDT/CAT)",
        category="hardware",
        applicability="Noisy neighbor evicting hot data from shared L3",
        expected_impact="Guarantee L3 allocation; eliminate inter-workload thrashing",
        prerequisites="Intel RDT support (Xeon SP/AP); resctrl filesystem",
    ),
    "memory_interleaving": Solution(
        name="Memory Interleaving (STRIPE mode)",
        category="hardware",
        applicability="Sequential scans saturating single-node DRAM bandwidth",
        expected_impact="Up to Nx bandwidth where N=NUMA nodes (aggregate all channels)",
        prerequisites="numactl --interleave=all or BIOS UMA/interleave setting",
    ),
    "workload_isolation": Solution(
        name="Workload Isolation (Dedicated NUMA Node)",
        category="scheduling",
        applicability="Latency-sensitive workload co-located with bandwidth hogs",
        expected_impact="Eliminate interference; predictable P99 latency",
        prerequisites="Separate NUMA nodes for SLM inference vs tool execution",
    ),
    "model_sharding": Solution(
        name="Tensor Parallel / Model Sharding",
        category="architecture",
        applicability="Single-node BW insufficient; model too large for one NUMA node",
        expected_impact="Linear BW scaling with shard count (2-4 nodes)",
        prerequisites="Framework TP support (vLLM, DeepSpeed); inter-node communication",
    ),
}


# ─── Bottleneck Patterns ─────────────────────────────────────────────────

@dataclass(frozen=True)
class BottleneckPattern:
    """A recognized memory subsystem bottleneck pattern."""
    name: str
    description: str
    solutions: List[str]  # Keys into SOLUTIONS dict


PATTERNS = {
    "weight_streaming": BottleneckPattern(
        name="Weight Streaming Bottleneck",
        description=(
            "Model weights are streamed from DRAM every forward pass. "
            "Arithmetic units idle waiting for data. Classic LLM decode "
            "at small batch sizes."
        ),
        solutions=["speculative_decoding", "quantization", "model_sharding"],
    ),
    "working_set_overflow": BottleneckPattern(
        name="Working Set Exceeds L3",
        description=(
            "Active data set is larger than L3 cache but not drastically so. "
            "Some cache reuse exists but miss rate is elevated. "
            "Tiling or NUMA-aware placement can recover locality."
        ),
        solutions=["data_tiling", "numa_pinning", "cache_partitioning"],
    ),
    "cross_numa_traffic": BottleneckPattern(
        name="Cross-NUMA Memory Traffic",
        description=(
            "Significant fraction of memory accesses go to remote NUMA nodes. "
            "Each hop adds 25ns latency penalty (Granite Rapids). "
            "Workload is not pinned or data is allocated on wrong node."
        ),
        solutions=["numa_pinning", "memory_local_scheduling", "workload_isolation"],
    ),
    "bandwidth_saturation": BottleneckPattern(
        name="DRAM Bandwidth Saturation",
        description=(
            "Memory controller read queues filling up. Access latency grows "
            "non-linearly as utilization approaches 100%. Single-node BW "
            "is the ceiling."
        ),
        solutions=["memory_interleaving", "model_sharding", "quantization"],
    ),
    "kv_cache_pressure": BottleneckPattern(
        name="KV-Cache Memory Pressure",
        description=(
            "KV-cache grows with context length and evicts other useful data "
            "from cache. Manifests as cache miss spikes during long-context "
            "inference. RSS grows linearly with sequence length."
        ),
        solutions=["kv_cache_compression", "rag_context_reduction", "cache_partitioning"],
    ),
    "capacity_thrashing": BottleneckPattern(
        name="Cache Capacity Thrashing",
        description=(
            "Multiple workloads compete for shared L3 cache. Hit rate oscillates "
            "as workloads evict each other's data. IPC is unstable across time. "
            "Common in multi-tenant / multi-agent scenarios."
        ),
        solutions=["cache_partitioning", "workload_isolation", "numa_pinning"],
    ),
}


# ─── Analyzer ────────────────────────────────────────────────────────────

class MemoryBandwidthAnalyzer:
    """Classifies memory subsystem bottlenecks and maps them to solutions.

    Consumes L1 (resource usage), L3 (hardware counters), and PerfSpect
    (TMA) measurements. For each span, identifies which bottleneck
    pattern(s) match the observed data, then lists applicable solutions
    ranked by expected impact.

    The analyzer does NOT prescribe a single fix — it presents the
    bottleneck evidence and lets the engineer choose based on their
    constraints (latency budget, accuracy requirements, deployment model).
    """

    name: str = "memory_bandwidth"
    input_layers = frozenset(["l1", "l3", "perfspect"])

    def __init__(self, *, platform: Optional[PlatformInfo] = None) -> None:
        self._platform = platform or detect_platform()
        # The unknown-peak notice is a property of the host, not of any one
        # span, but _match_patterns runs per span. Latch it so a run with N
        # spans logs once instead of N times.
        self._warned_unknown_peak = False
        self._warned_no_numa_signal = False

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Analyze memory subsystem bottlenecks for each span."""
        spans: dict = {}
        for record in records:
            if record.span_id not in spans:
                spans[record.span_id] = {}
            spans[record.span_id][record.layer] = record

        for span_id, layers in spans.items():
            result = self._analyze_span(span_id, layers, context)
            if result is not None:
                yield result

    def _analyze_span(
        self,
        span_id: str,
        layers: dict,
        context: Optional[AnalysisContext],
    ) -> Optional[AnalysisResult]:
        """Analyze a single span for memory bottleneck patterns."""
        l3_record = layers.get("l3")
        l1_record = layers.get("l1")
        perfspect_record = layers.get("perfspect")

        if l3_record is None and perfspect_record is None:
            return None

        l3 = l3_record.payload if l3_record else {}
        l1 = l1_record.payload if l1_record else {}
        ps = perfspect_record.payload if perfspect_record else {}

        # Extract signals
        signals = {
            "ipc": l3.get("ipc"),
            "cache_miss_pct": l3.get("cache_miss_pct"),
            "branch_miss_pct": l3.get("branch_miss_pct"),
            "cpu_pct_mean": l1.get("cpu_pct_mean"),
            "rss_kb_peak": l1.get("rss_kb_peak"),
            "duration_us": l1.get("duration_us"),
            "tma_backend_bound": _as_pct(ps.get("backend_bound")),
            # Read it from the payload rather than hardcoding None. This was
            # pinned to None because the PerfSpect plugin's label_map mapped
            # only the four top-level TMA buckets, so no memory-bound key
            # existed — a gap in that map, NOT a property of the platform. The
            # plugin now maps TMA_..Memory_Bound(%), and its `if label in raw`
            # guard keeps the result honest per host: E-core parts emit no TMA
            # rows, so the key is absent and this stays None because it was not
            # measured. Never re-pin this to a constant — a platform benchmark
            # that hardcodes one platform's capability is not measuring.
            "tma_memory_bound": _as_pct(ps.get("tma_memory_bound")),
            "tma_retiring": _as_pct(ps.get("retiring")),
            "memory_bandwidth_gbs": ps.get("memory_bandwidth_gbs"),
            "cpi": ps.get("cpi", 1.0 / l3["ipc"] if l3.get("ipc") else None),
            # No installed telemetry plugin advertises this yet, so it is None
            # today. It is extracted anyway because cross_numa_traffic now
            # requires it: naming the missing instrument is what stops that
            # pattern from asserting a NUMA diagnosis it cannot support.
            "numa_remote_access_ratio": ps.get("numa_remote_access_ratio"),
        }

        # Which inputs were never measured for this span. Recorded explicitly
        # because the evidence dict below filters None out, so without this a
        # reader cannot tell "measured and healthy" from "never measured".
        unmeasured = sorted(k for k, v in signals.items() if v is None)

        # Detect which patterns match
        matched_patterns = self._match_patterns(signals)

        if not matched_patterns:
            # No memory bottleneck detected. A clean bill of health is only as
            # strong as its coverage: without the perfspect layer there is no
            # DRAM bandwidth and no TMA for this span, so the negative rests on
            # cache-miss and IPC alone. Say so and price the confidence
            # accordingly rather than asserting 0.8 over unmeasured hardware —
            # this verdict actively tells the reader to stop looking at memory.
            measured_memory = perfspect_record is not None
            recommendations = [
                "No significant memory subsystem bottleneck detected.",
                "Workload appears compute-bound or balanced.",
                "Consider compute-focused optimizations (AMX, vectorization, batching).",
            ]
            if not measured_memory:
                recommendations.insert(0, (
                    "Coverage caveat: the perfspect layer is absent for this "
                    "span, so DRAM bandwidth and TMA were never measured. This "
                    "verdict rests on cache-miss rate and IPC only; treat it as "
                    "'no evidence of a bottleneck', not 'no bottleneck'."
                ))
            return AnalysisResult(
                verdict="no_memory_bottleneck",
                confidence=0.8 if measured_memory else 0.4,
                evidence={
                    "signals": {k: v for k, v in signals.items() if v is not None},
                    "unmeasured_signals": unmeasured,
                    "memory_subsystem_measured": measured_memory,
                    "patterns_matched": [],
                },
                recommendations=recommendations,
                span_id=span_id,
                analyzer_name=self.name,
            )

        # Collect solutions from all matched patterns
        all_solutions = self._collect_solutions(matched_patterns)

        # Build result
        pattern_details = []
        for pattern_name, confidence in matched_patterns:
            pattern = PATTERNS[pattern_name]
            pattern_details.append({
                "pattern": pattern_name,
                "description": pattern.description,
                "confidence": confidence,
                "solutions": pattern.solutions,
            })

        # Determine overall verdict
        max_confidence = max(c for _, c in matched_patterns)
        if max_confidence >= 0.7:
            verdict = "memory_bottleneck_severe"
        elif max_confidence >= 0.4:
            verdict = "memory_bottleneck_moderate"
        else:
            verdict = "memory_bottleneck_mild"

        recommendation = self._build_recommendation(
            matched_patterns, all_solutions, signals
        )

        return AnalysisResult(
            verdict=verdict,
            confidence=max_confidence,
            evidence={
                "signals": {k: v for k, v in signals.items() if v is not None},
                "unmeasured_signals": unmeasured,
                "memory_subsystem_measured": perfspect_record is not None,
                "patterns_matched": pattern_details,
                "solutions": all_solutions,
            },
            recommendations=[recommendation],
            span_id=span_id,
            analyzer_name=self.name,
        )

    def _match_patterns(self, signals: dict) -> List[tuple]:
        """Match observed signals against known bottleneck patterns.

        Returns list of (pattern_name, confidence) tuples, sorted by
        confidence descending.
        """
        matched = []

        ipc = signals.get("ipc")
        cache_miss = signals.get("cache_miss_pct")
        rss_kb = signals.get("rss_kb_peak")
        tma_memory = signals.get("tma_memory_bound")
        tma_backend = signals.get("tma_backend_bound")
        mem_bw = signals.get("memory_bandwidth_gbs")

        # Pattern: weight_streaming
        # IPC < 1.5, cache miss > 70%, large RSS
        weight_score = 0.0
        if ipc is not None and ipc < 1.5:
            weight_score += 0.4
        elif ipc is not None and ipc < 2.0:
            weight_score += 0.2
        if cache_miss is not None and cache_miss > 70:
            weight_score += 0.35
        elif cache_miss is not None and cache_miss > 50:
            weight_score += 0.15
        if rss_kb is not None and rss_kb > 1_000_000:  # > 1GB
            weight_score += 0.25
        if weight_score >= 0.4:
            matched.append(("weight_streaming", min(1.0, weight_score)))

        # Pattern: working_set_overflow
        # Cache miss 30-70%, moderate IPC
        overflow_score = 0.0
        if cache_miss is not None and 30 < cache_miss <= 70:
            overflow_score += 0.4
        if ipc is not None and 1.5 <= ipc <= 3.5:
            overflow_score += 0.3
        l3_budget_kb = self._platform.l3_usable_per_node // 1024
        if rss_kb is not None and l3_budget_kb and rss_kb > l3_budget_kb:
            overflow_score += 0.3  # exceeds L3 node budget
        if overflow_score >= 0.4:
            matched.append(("working_set_overflow", min(1.0, overflow_score)))

        # Pattern: cross_numa_traffic
        # Requires a direct remote-access measurement, and therefore does not
        # fire on any currently installed telemetry plugin.
        #
        # This pattern names a specific physical mechanism — traffic crossing a
        # NUMA hop — and its recommendations (numa_pinning, SNC tuning, node
        # isolation) are only actionable if that mechanism is what is happening.
        # It used to infer that from an IPC-vs-cache-miss model alone, but an
        # IPC deficit is consistent with a dozen unrelated causes (store
        # stalls, frequency capping, port contention, a bad branch mix), so the
        # inference asserted a diagnosis the data could not distinguish. On
        # short spans, where perfspect is absent and cache_miss/ipc are the only
        # signals present, it fired anyway and produced NUMA remediation advice
        # for a span whose memory subsystem was never observed.
        numa_remote = signals.get("numa_remote_access_ratio")
        if numa_remote is None:
            if not self._warned_no_numa_signal:
                self._warned_no_numa_signal = True
                logger.info(
                    "Skipping cross_numa_traffic for this run: no installed "
                    "telemetry plugin advertises numa_remote_access_ratio, and "
                    "this pattern's recommendations are only valid if remote "
                    "access is actually measured."
                )
        else:
            numa_score = 0.0
            if numa_remote > 20:
                numa_score += 0.5
            if tma_memory is not None and tma_memory > 30:
                numa_score += 0.3
            if numa_score >= 0.4:
                matched.append(("cross_numa_traffic", min(1.0, numa_score)))

        # Pattern: bandwidth_saturation
        # DRAM BW > 70% of detected peak.
        #
        # Two scope/provenance traps make this verdict easy to get wrong, and
        # both errors run the same direction (inflated utilization -> false
        # "saturated"), so each is gated rather than papered over with a default:
        #   1. mem_bw comes from PerfSpect at --scope system, i.e. a machine-wide
        #      total. It must be compared against the machine-wide peak, not the
        #      per-node peak — dividing a system total by a per-node peak
        #      overstates utilization by the NUMA node count (3x under SNC3).
        #   2. The peak itself may be an unvalidated estimate on an unrecognized
        #      platform, or unknown entirely.
        bw_score = 0.0
        peak_bw = self._platform.dram_bw_total_gbs
        if mem_bw is not None and peak_bw > 0:
            utilization = mem_bw / peak_bw
            if utilization > 0.7:
                bw_score += 0.7
            elif utilization > 0.5:
                bw_score += 0.4
        elif mem_bw is not None and not self._warned_unknown_peak:
            self._warned_unknown_peak = True
            logger.info(
                "Skipping bandwidth_saturation for this run: DRAM peak unknown "
                "for %s (source=%s). Supply an MLC baseline for a utilization "
                "verdict.",
                self._platform.microarchitecture,
                self._platform.dram_bw_source,
            )
        if tma_memory is not None and tma_memory > 40:
            bw_score += 0.3
        if bw_score >= 0.4:
            matched.append(("bandwidth_saturation", min(1.0, bw_score)))

        # Pattern: kv_cache_pressure
        # Heuristic: high RSS + high cache miss + moderate IPC (not pure streaming)
        kv_score = 0.0
        if rss_kb is not None and rss_kb > 500_000:  # > 500MB
            kv_score += 0.3
        if cache_miss is not None and cache_miss > 40:
            kv_score += 0.3
        if ipc is not None and 1.0 < ipc < 3.0:
            kv_score += 0.2
        # KV cache pattern is hard to distinguish from weight streaming
        # without phase-level data; keep confidence lower
        if kv_score >= 0.5:
            matched.append(("kv_cache_pressure", min(0.8, kv_score)))

        # Pattern: capacity_thrashing
        # Best detected with time-series (IPC variance), but approximate with
        # very high cache miss + moderate IPC (implying intermittent hits)
        thrash_score = 0.0
        if cache_miss is not None and cache_miss > 60:
            thrash_score += 0.3
        if ipc is not None and 1.5 < ipc < 3.0:
            # Not fully streaming (that would be IPC < 1.5)
            # Not compute-bound (that would be IPC > 3.5)
            # This middle zone + high miss = thrashing
            thrash_score += 0.3
        if thrash_score >= 0.4:
            matched.append(("capacity_thrashing", min(0.7, thrash_score)))

        # Sort by confidence
        matched.sort(key=lambda x: x[1], reverse=True)
        return matched

    def _collect_solutions(self, matched_patterns: List[tuple]) -> List[dict]:
        """Collect and deduplicate solutions from matched patterns."""
        seen = set()
        solutions = []

        for pattern_name, confidence in matched_patterns:
            pattern = PATTERNS[pattern_name]
            for solution_key in pattern.solutions:
                if solution_key in seen:
                    continue
                seen.add(solution_key)
                sol = SOLUTIONS[solution_key]
                solutions.append({
                    "name": sol.name,
                    "key": solution_key,
                    "category": sol.category,
                    "applicability": sol.applicability,
                    "expected_impact": sol.expected_impact,
                    "prerequisites": sol.prerequisites,
                    "triggered_by": pattern_name,
                })

        return solutions

    def _build_recommendation(
        self,
        matched_patterns: List[tuple],
        solutions: List[dict],
        signals: dict,
    ) -> str:
        """Build human-readable recommendation summary."""
        if not matched_patterns:
            return "No memory bottleneck detected."

        top_pattern_name, top_confidence = matched_patterns[0]
        top_pattern = PATTERNS[top_pattern_name]

        parts = [
            f"Primary bottleneck: {top_pattern.name} "
            f"(confidence: {top_confidence:.0%}).",
            top_pattern.description,
            "",
            f"Applicable solutions ({len(solutions)} identified):",
        ]

        for i, sol in enumerate(solutions[:5], 1):
            parts.append(
                f"  {i}. {sol['name']} [{sol['category']}] — {sol['expected_impact']}"
            )

        if len(matched_patterns) > 1:
            secondary = [PATTERNS[p].name for p, _ in matched_patterns[1:3]]
            parts.append(f"\nSecondary patterns: {', '.join(secondary)}")

        return "\n".join(parts)


__all__ = ["MemoryBandwidthAnalyzer"]
