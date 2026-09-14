#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Post-processing: aggregate results, compute scaling curves, detect inflection points.

Reads the `all_results.json` produced by `run_experiment.py` and derives:
  - Per-density scaling curves (normalized to solo baseline)
  - Contention inflection points (LLC MPKI, IPC drops)
  - Phase breakdown (Reason vs Act vs overhead)
  - NUMA placement comparison
  - TMA data integration (when EMON CSVs available)

Usage:
    python -m experiments.scaling.analysis /tmp/agentsysperf_scaling/run_<ts>
    python -m experiments.scaling.analysis /tmp/agentsysperf_scaling/run_<ts> --output report.json
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentMetrics:
    """Derived metrics for a single agent within one experiment config."""
    agent_id: int
    phase_mix: str
    throughput: float
    mean_turn_ms: float
    reason_pct: float
    act_pct: float
    ipc: float
    cache_miss_pct: float
    llc_mpki: float
    context_switches: float


@dataclass
class ConfigSummary:
    """Aggregated summary for one experiment configuration."""
    density: int
    placement: str
    phase_mix: str
    aggregate_throughput: float
    mean_per_agent_throughput: float
    p50_turn_ms: float
    p95_turn_ms: float
    max_turn_ms: float
    mean_ipc: float
    mean_cache_miss_pct: float
    mean_llc_mpki: float
    mean_reason_pct: float
    mean_act_pct: float
    agents_ok: int
    agents_total: int
    emon_csv: Optional[str] = None


@dataclass
class ScalingAnalysis:
    """Full analysis output."""
    baselines: Dict[str, ConfigSummary]
    configs: List[ConfigSummary]
    scaling_curves: Dict[str, List[Dict[str, Any]]]
    inflection_points: List[Dict[str, Any]]
    numa_comparison: List[Dict[str, Any]]
    phase_breakdown: List[Dict[str, Any]]
    tma_heatmap: Optional[List[Dict[str, Any]]] = None


def _derive_agent_metrics(agent: Dict[str, Any]) -> AgentMetrics:
    """Compute derived metrics from a single agent result dict."""
    perf = agent.get("perf_events", {})
    turns = agent.get("turn_durations", [])

    # Phase split
    total_reason = sum(t.get("reason_s", 0) for t in turns)
    total_act = sum(t.get("act_s", 0) for t in turns)
    total_time = total_reason + total_act
    reason_pct = (total_reason / total_time * 100) if total_time > 0 else 0
    act_pct = (total_act / total_time * 100) if total_time > 0 else 0

    # IPC
    cycles = perf.get("cycles", 0)
    instructions = perf.get("instructions", 0)
    ipc = instructions / cycles if cycles > 0 else 0.0

    # Cache miss %
    cache_refs = perf.get("cache-references", 0)
    cache_misses = perf.get("cache-misses", 0)
    cache_miss_pct = (cache_misses / cache_refs * 100) if cache_refs > 0 else 0.0

    # LLC MPKI (misses per kilo instructions)
    llc_misses = perf.get("LLC-load-misses", cache_misses)
    llc_mpki = (llc_misses / (instructions / 1000)) if instructions > 0 else 0.0

    ctx = perf.get("context-switches", 0)

    return AgentMetrics(
        agent_id=agent.get("agent_id", 0),
        phase_mix=agent.get("phase_mix", ""),
        throughput=agent.get("throughput_turns_per_s", 0),
        mean_turn_ms=agent.get("mean_turn_ms", 0),
        reason_pct=reason_pct,
        act_pct=act_pct,
        ipc=ipc,
        cache_miss_pct=cache_miss_pct,
        llc_mpki=llc_mpki,
        context_switches=ctx,
    )


def _summarize_config(result: Dict[str, Any]) -> ConfigSummary:
    """Aggregate all agent results for one experiment configuration."""
    config = result.get("config", {})
    agents = result.get("agents", [])
    ok_agents = [a for a in agents if not a.get("error")]

    if not ok_agents:
        return ConfigSummary(
            density=config.get("density", 0),
            placement=config.get("placement", ""),
            phase_mix=config.get("phase_mix", ""),
            aggregate_throughput=0, mean_per_agent_throughput=0,
            p50_turn_ms=0, p95_turn_ms=0, max_turn_ms=0,
            mean_ipc=0, mean_cache_miss_pct=0, mean_llc_mpki=0,
            mean_reason_pct=0, mean_act_pct=0,
            agents_ok=0, agents_total=config.get("density", 0),
            emon_csv=result.get("emon_csv"),
        )

    metrics = [_derive_agent_metrics(a) for a in ok_agents]

    # Turn latency distribution (all turns from all agents)
    all_turns_ms = []
    for a in ok_agents:
        for t in a.get("turn_durations", []):
            all_turns_ms.append(t.get("total_s", 0) * 1000)
    all_turns_ms.sort()

    p50 = all_turns_ms[len(all_turns_ms) // 2] if all_turns_ms else 0
    p95_idx = int(len(all_turns_ms) * 0.95)
    p95 = all_turns_ms[min(p95_idx, len(all_turns_ms) - 1)] if all_turns_ms else 0
    max_turn = all_turns_ms[-1] if all_turns_ms else 0

    throughputs = [m.throughput for m in metrics]
    agg_tp = sum(throughputs)
    mean_tp = statistics.mean(throughputs) if throughputs else 0

    return ConfigSummary(
        density=config.get("density", 0),
        placement=config.get("placement", ""),
        phase_mix=config.get("phase_mix", ""),
        aggregate_throughput=agg_tp,
        mean_per_agent_throughput=mean_tp,
        p50_turn_ms=p50,
        p95_turn_ms=p95,
        max_turn_ms=max_turn,
        mean_ipc=statistics.mean(m.ipc for m in metrics) if metrics else 0,
        mean_cache_miss_pct=statistics.mean(m.cache_miss_pct for m in metrics) if metrics else 0,
        mean_llc_mpki=statistics.mean(m.llc_mpki for m in metrics) if metrics else 0,
        mean_reason_pct=statistics.mean(m.reason_pct for m in metrics) if metrics else 0,
        mean_act_pct=statistics.mean(m.act_pct for m in metrics) if metrics else 0,
        agents_ok=len(ok_agents),
        agents_total=config.get("density", 0),
        emon_csv=result.get("emon_csv"),
    )


def _compute_scaling_curves(
    summaries: List[ConfigSummary],
    baselines: Dict[str, ConfigSummary],
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute normalized scaling curves grouped by (placement, mix).

    Returns dict keyed by "{placement}_{mix}" with list of
    {density, normalized_throughput, raw_throughput, ipc, llc_mpki, p50_ms, p95_ms}.
    """
    curves: Dict[str, List[Dict[str, Any]]] = {}

    for s in summaries:
        key = f"{s.placement}_{s.phase_mix}"
        baseline_key = f"{s.placement}_{s.phase_mix}"
        baseline = baselines.get(baseline_key)
        baseline_tp = baseline.mean_per_agent_throughput if baseline else s.mean_per_agent_throughput

        normalized = (s.mean_per_agent_throughput / baseline_tp) if baseline_tp > 0 else 0

        if key not in curves:
            curves[key] = []
        curves[key].append({
            "density": s.density,
            "normalized_throughput": round(normalized, 4),
            "raw_throughput_per_agent": round(s.mean_per_agent_throughput, 4),
            "aggregate_throughput": round(s.aggregate_throughput, 4),
            "ipc": round(s.mean_ipc, 3),
            "llc_mpki": round(s.mean_llc_mpki, 2),
            "cache_miss_pct": round(s.mean_cache_miss_pct, 1),
            "p50_ms": round(s.p50_turn_ms, 1),
            "p95_ms": round(s.p95_turn_ms, 1),
        })

    # Sort each curve by density
    for key in curves:
        curves[key].sort(key=lambda x: x["density"])

    return curves


def _detect_inflection_points(
    curves: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Detect contention inflection points.

    An inflection is where:
      - LLC MPKI jumps >2x vs the density=1 baseline, OR
      - Normalized throughput drops below 0.7 (30% degradation), OR
      - IPC drops >40% from baseline
    """
    inflections = []

    for curve_key, points in curves.items():
        if len(points) < 2:
            continue

        baseline = points[0]
        base_mpki = baseline.get("llc_mpki", 0)
        base_ipc = baseline.get("ipc", 0)

        for pt in points[1:]:
            reasons = []
            density = pt["density"]

            # LLC MPKI >2x
            if base_mpki > 0 and pt["llc_mpki"] > base_mpki * 2:
                reasons.append(f"LLC_MPKI {pt['llc_mpki']:.1f} > 2x baseline ({base_mpki:.1f})")

            # Throughput < 70%
            if pt["normalized_throughput"] < 0.7:
                reasons.append(f"throughput {pt['normalized_throughput']:.0%} < 70% of baseline")

            # IPC drop > 40%
            if base_ipc > 0 and pt["ipc"] < base_ipc * 0.6:
                reasons.append(f"IPC {pt['ipc']:.2f} < 60% of baseline ({base_ipc:.2f})")

            if reasons:
                inflections.append({
                    "curve": curve_key,
                    "density": density,
                    "reasons": reasons,
                    "normalized_throughput": pt["normalized_throughput"],
                    "llc_mpki": pt["llc_mpki"],
                    "ipc": pt["ipc"],
                })
                break  # report first inflection per curve

    return inflections


def _numa_comparison(summaries: List[ConfigSummary]) -> List[Dict[str, Any]]:
    """Compare placements at each density level.

    Groups by (density, mix) and compares intra_node vs cross_node vs spread.
    """
    from collections import defaultdict
    grouped: Dict[Tuple[int, str], List[ConfigSummary]] = defaultdict(list)

    for s in summaries:
        grouped[(s.density, s.phase_mix)].append(s)

    comparisons = []
    for (density, mix), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        entry: Dict[str, Any] = {"density": density, "phase_mix": mix}
        for s in group:
            entry[s.placement] = {
                "throughput": round(s.mean_per_agent_throughput, 4),
                "ipc": round(s.mean_ipc, 3),
                "p50_ms": round(s.p50_turn_ms, 1),
            }

        # NUMA penalty: (cross_node - intra_node) / intra_node
        intra = entry.get("intra_node", {}).get("throughput", 0)
        cross = entry.get("cross_node", {}).get("throughput", 0)
        if intra > 0 and cross > 0:
            entry["numa_penalty_pct"] = round((intra - cross) / intra * 100, 1)
        comparisons.append(entry)

    return comparisons


def _phase_breakdown(summaries: List[ConfigSummary]) -> List[Dict[str, Any]]:
    """Phase split (Reason vs Act) across density levels."""
    breakdown = []
    for s in summaries:
        overhead = max(0, 100 - s.mean_reason_pct - s.mean_act_pct)
        breakdown.append({
            "density": s.density,
            "placement": s.placement,
            "phase_mix": s.phase_mix,
            "reason_pct": round(s.mean_reason_pct, 1),
            "act_pct": round(s.mean_act_pct, 1),
            "overhead_pct": round(overhead, 1),
        })
    return breakdown


def _load_tma_from_emon(summaries: List[ConfigSummary]) -> Optional[List[Dict[str, Any]]]:
    """Load TMA L1 metrics from EMON CSVs (if available)."""
    tma_data = []
    tma_keys = ["Frontend_Bound", "Backend_Bound", "Bad_Speculation", "Retiring",
                "Memory_Bound", "Core_Bound"]

    for s in summaries:
        if not s.emon_csv:
            continue
        csv_path = Path(s.emon_csv)
        if not csv_path.exists():
            continue

        try:
            import csv
            with open(csv_path) as f:
                reader = csv.reader(f)
                metrics = {}
                for row in reader:
                    if len(row) >= 2:
                        name = row[0].strip()
                        for key in tma_keys:
                            if key.lower() in name.lower():
                                try:
                                    metrics[key] = float(row[1])
                                except ValueError:
                                    pass
                if metrics:
                    tma_data.append({
                        "density": s.density,
                        "placement": s.placement,
                        "phase_mix": s.phase_mix,
                        **metrics,
                    })
        except Exception:
            continue

    return tma_data if tma_data else None


def analyze(results_dir: Path) -> ScalingAnalysis:
    """Run full analysis on experiment results.

    Args:
        results_dir: Directory containing all_results.json and per-config subdirs.

    Returns:
        ScalingAnalysis with all derived data.
    """
    results_file = results_dir / "all_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No all_results.json in {results_dir}")

    raw = json.loads(results_file.read_text())
    summaries = [_summarize_config(r) for r in raw]

    # Identify baselines (density=1 for each placement+mix combo)
    baselines = {}
    for s in summaries:
        if s.density == 1:
            baselines[f"{s.placement}_{s.phase_mix}"] = s

    curves = _compute_scaling_curves(summaries, baselines)
    inflections = _detect_inflection_points(curves)
    numa = _numa_comparison(summaries)
    phases = _phase_breakdown(summaries)
    tma = _load_tma_from_emon(summaries)

    return ScalingAnalysis(
        baselines={k: v for k, v in baselines.items()},
        configs=summaries,
        scaling_curves=curves,
        inflection_points=inflections,
        numa_comparison=numa,
        phase_breakdown=phases,
        tma_heatmap=tma,
    )


def generate_report(analysis: ScalingAnalysis) -> Dict[str, Any]:
    """Generate a JSON-serializable report from the analysis."""
    report: Dict[str, Any] = {
        "summary": {
            "total_configs": len(analysis.configs),
            "baselines": len(analysis.baselines),
            "inflection_points_found": len(analysis.inflection_points),
            "tma_data_available": analysis.tma_heatmap is not None,
        },
        "baselines": {},
        "scaling_curves": analysis.scaling_curves,
        "inflection_points": analysis.inflection_points,
        "numa_comparison": analysis.numa_comparison,
        "phase_breakdown": analysis.phase_breakdown,
    }

    for key, b in analysis.baselines.items():
        report["baselines"][key] = {
            "throughput": round(b.mean_per_agent_throughput, 4),
            "ipc": round(b.mean_ipc, 3),
            "llc_mpki": round(b.mean_llc_mpki, 2),
            "p50_ms": round(b.p50_turn_ms, 1),
            "reason_pct": round(b.mean_reason_pct, 1),
        }

    if analysis.tma_heatmap:
        report["tma_heatmap"] = analysis.tma_heatmap

    # Key findings
    findings = []
    for inf in analysis.inflection_points:
        findings.append(
            f"[{inf['curve']}] Contention inflection at density={inf['density']}: "
            + "; ".join(inf["reasons"])
        )

    # Best aggregate throughput
    if analysis.configs:
        best = max(analysis.configs, key=lambda s: s.aggregate_throughput)
        findings.append(
            f"Peak aggregate throughput: {best.aggregate_throughput:.2f} turns/s "
            f"at density={best.density} ({best.placement}, {best.phase_mix})"
        )

    # Optimal density (highest aggregate where per-agent > 50% of baseline)
    for curve_key, points in analysis.scaling_curves.items():
        viable = [p for p in points if p["normalized_throughput"] >= 0.5]
        if viable:
            optimal = max(viable, key=lambda p: p["density"])
            findings.append(
                f"[{curve_key}] Optimal density: {optimal['density']} "
                f"(per-agent at {optimal['normalized_throughput']:.0%} of baseline)"
            )

    report["findings"] = findings
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze scaling experiment results")
    parser.add_argument("results_dir", type=Path, help="Path to experiment results directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output JSON report path (default: <results_dir>/analysis_report.json)")
    parser.add_argument("--print-findings", action="store_true",
                       help="Print key findings to stdout")
    args = parser.parse_args()

    analysis = analyze(args.results_dir)
    report = generate_report(analysis)

    output_path = args.output or (args.results_dir / "analysis_report.json")
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Analysis report: {output_path}")
    print(f"  Configs analyzed: {report['summary']['total_configs']}")
    print(f"  Inflection points: {report['summary']['inflection_points_found']}")

    if args.print_findings or True:
        print("\n  Key Findings:")
        for f in report.get("findings", []):
            print(f"    - {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
