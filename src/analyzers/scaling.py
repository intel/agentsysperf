#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""ScalingAnalyzer — concurrency-scaling knee + bottleneck classification.

Turns the colleague's *manual* saturation interpretation (agentic-benchmark-4:
"CPU peak hits 100% while avg stays ~70%, wall time floors, p95 climbs") into an
algorithm. Consumes the per-cell rollups of a density sweep (one point per
``concurrency / vcpu_basis``) and answers two questions:

  1. **Where is the knee?** The density at which throughput stops scaling —
     adding agents stops buying throughput. Detected with the Kneedle
     max-distance-from-chord method on the throughput(density) curve
     (pure-Python, no SciPy), cross-validated against the p95-latency rise.

  2. **What is the bottleneck at saturation?** Classified from the node
     telemetry at/after the knee:
       runqueue_max >> logical_cpus  → scheduler_oversubscription
       cpu_peak ~100% & cpu_avg high → cpu_bound
       mem_avail_mb_min → low        → memory_bound
       iowait high                   → io_bound
       else                          → headroom_remaining (knee not reached)

Multi-run modeling: a scaling analysis spans MANY runs (one per density point),
which the per-span :class:`Analyzer` contract can't express. So the real entry
point is :meth:`analyze_sweep` (called by the sweep runner with the sweep's
points). :meth:`analyze` is implemented for Protocol conformance / discovery and
returns nothing on ordinary per-run records. Precedent: ``PhaseProfiler`` is a
registered analyzer that emits one aggregate verdict rather than per-span ones.

Thresholds below are EMR-calibration ESTIMATES, not validated constants — they
are starting points to be tuned against real sweeps (flagged in
``recommendations`` and in the verdict confidence).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.protocols import AnalysisContext, AnalysisResult, MeasurementRecord

logger = logging.getLogger(__name__)

# ── Bottleneck classification thresholds (EMR estimates — calibrate) ──
RUNQUEUE_OVERSUB_RATIO = 1.5     # runqueue_max > ratio × logical_cpus → scheduler-bound
CPU_SATURATION_PEAK = 95.0       # cpu_peak >= this (%) → CPU pegged
CPU_SATURATION_AVG = 80.0        # cpu_avg >= this (%) → sustained, not bursty
MEM_LOW_MB = 2048.0              # mem_avail_mb_min < this → memory pressure
IOWAIT_HIGH_PCT = 15.0          # iowait_pct_avg > this → I/O-bound
KNEE_MIN_POINTS = 3              # need at least this many densities to fit a curve


@dataclass
class _Point:
    """One density operating point, averaged over replicates."""

    density: float
    concurrency: int
    throughput: float
    p95_latency: float
    cpu_avg: float
    cpu_peak: float
    # None means "not measurable at this scope", not zero. A cpuset-confined
    # sweep cannot read run-queue depth or context switches (node-wide counters).
    runqueue_max: Optional[float]
    ctx_sw_per_s: Optional[float]
    mem_avail_mb_min: float
    iowait_pct_avg: float


def _aggregate_replicates(points: Sequence[Dict[str, Any]]) -> List[_Point]:
    """Collapse replicate rows at the same density into one averaged point."""
    by_density: Dict[float, List[Dict[str, Any]]] = {}
    for p in points:
        d = p.get("density")
        if d is None:
            continue
        by_density.setdefault(round(float(d), 4), []).append(p)

    def _avg(rows, key, default=0.0):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else default

    out: List[_Point] = []
    for density, rows in sorted(by_density.items()):
        out.append(_Point(
            density=density,
            concurrency=int(_avg(rows, "concurrency")),
            throughput=_avg(rows, "throughput_per_min"),
            p95_latency=_avg(rows, "p95_trial_latency_s"),
            cpu_avg=_avg(rows, "cpu_avg"),
            cpu_peak=_avg(rows, "cpu_peak"),
            runqueue_max=_avg(rows, "runqueue_max", default=None),
            ctx_sw_per_s=_avg(rows, "ctx_sw_per_s", default=None),
            mem_avail_mb_min=_avg(rows, "mem_avail_mb_min", default=float("inf")),
            iowait_pct_avg=_avg(rows, "iowait_pct_avg"),
        ))
    return out


def _kneedle(xs: List[float], ys: List[float]) -> Optional[int]:
    """Return the index of the knee via max distance from the endpoint chord.

    Normalizes both axes to [0,1], then finds the point of maximum vertical
    distance below the straight line joining the first and last points — the
    classic Kneedle elbow for a concave-increasing-then-flattening curve like
    throughput(density). Returns None if the curve is too short or degenerate.
    """
    n = len(xs)
    if n < KNEE_MIN_POINTS:
        return None
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    xr = (x1 - x0) or 1e-9
    yr = (max(ys) - min(ys)) or 1e-9
    nx = [(x - x0) / xr for x in xs]
    ny = [(y - min(ys)) / yr for y in ys]
    # Chord from first to last normalized point. A diminishing-returns
    # throughput curve is concave: its points sit ABOVE the chord, and the
    # knee is where that gap (actual - chord) is largest. A linear curve lies
    # on the chord (gap 0 → no knee); an accelerating/convex curve sits below
    # it (gap negative → no knee). Only positive gaps count.
    cx0, cy0 = nx[0], ny[0]
    cx1, cy1 = nx[-1], ny[-1]
    slope = (cy1 - cy0) / ((cx1 - cx0) or 1e-9)
    best_i, best_d = None, 1e-9  # require a strictly positive gap
    for i in range(1, n - 1):
        chord_y = cy0 + slope * (nx[i] - cx0)
        dist = ny[i] - chord_y  # positive when actual sits above the chord
        if dist > best_d:
            best_d, best_i = dist, i
    return best_i


def _classify_bottleneck(pt: _Point, logical_cpus: int) -> str:
    """Classify the dominant bottleneck at a saturated operating point."""
    if (logical_cpus > 0 and pt.runqueue_max is not None
            and pt.runqueue_max > RUNQUEUE_OVERSUB_RATIO * logical_cpus):
        return "scheduler_oversubscription"
    if pt.cpu_peak >= CPU_SATURATION_PEAK and pt.cpu_avg >= CPU_SATURATION_AVG:
        return "cpu_bound"
    if pt.mem_avail_mb_min < MEM_LOW_MB:
        return "memory_bound"
    if pt.iowait_pct_avg > IOWAIT_HIGH_PCT:
        return "io_bound"
    return "headroom_remaining"


_BOTTLENECK_RECS = {
    "scheduler_oversubscription": [
        "Run queue depth far exceeds logical CPUs — the scheduler, not raw compute, is the limit.",
        "Pin agents to NUMA nodes (socket_pinned) or cap concurrency below the knee.",
        "More cores will not help until oversubscription is reduced.",
    ],
    "cpu_bound": [
        "CPU pegged at the knee (peak ~100%, high sustained avg) — genuinely core-bound.",
        "Higher core count or frequency will raise the saturation ceiling.",
        "This is the SKU-scaling story: more Xeon cores buy more agents.",
    ],
    "memory_bound": [
        "Available memory bottoms out near the knee — memory capacity is the limit.",
        "Add DRAM or reduce per-agent footprint before adding cores.",
    ],
    "io_bound": [
        "High iowait at the knee — blocked on disk/container I/O, not CPU.",
        "Check the Docker image build/pull path and container storage before scaling cores.",
    ],
    "headroom_remaining": [
        "No saturation signature within the swept density range — the box has headroom.",
        "Extend the sweep to higher density to find the true knee.",
    ],
}


class ScalingAnalyzer:
    """Concurrency-scaling knee detection + bottleneck classification.

    Registered as an analyzer for discovery; invoked at SWEEP scope via
    :meth:`analyze_sweep`. ``input_layers`` is ``l1_system`` because that is the
    measurement the sweep cells produce, but the analyzer actually consumes the
    sweep-point rollups, not raw per-span records.
    """

    name: str = "scaling"
    input_layers = frozenset(["l1_system"])

    def analyze(
        self,
        records: Sequence[MeasurementRecord],
        *,
        context: Optional[AnalysisContext] = None,
    ) -> Iterable[AnalysisResult]:
        """Protocol conformance. Per-run records don't carry sweep structure,
        so this is a no-op — use :meth:`analyze_sweep`."""
        return ()

    def analyze_sweep(
        self,
        sweep_points: Sequence[Dict[str, Any]],
        *,
        logical_cpus: int,
        per_task_points: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    ) -> Optional[AnalysisResult]:
        """Analyze a full density sweep. Returns one aggregate result.

        Parameters
        ----------
        sweep_points:
            Per-cell rollups (rows from ``sweep_points``), aggregated across
            tasks. One or more per density (replicates averaged).
        logical_cpus:
            Logical CPU count of the box (for the oversubscription ratio).
        per_task_points:
            Optional ``{task_id: [point, ...]}`` so the verdict's evidence
            carries a per-task knee/bottleneck breakdown for the UI.
        """
        pts = _aggregate_replicates(sweep_points)
        if len(pts) < 2:
            return AnalysisResult(
                verdict="insufficient_data",
                confidence=0.0,
                evidence={"n_points": len(pts), "curve": [self._pt_dict(p) for p in pts]},
                recommendations=["Need at least 2 density points; run more sweep cells."],
                analyzer_name=self.name,
            )

        densities = [p.density for p in pts]
        throughputs = [p.throughput for p in pts]
        knee_i = _kneedle(densities, throughputs)

        curve = [self._pt_dict(p) for p in pts]

        if knee_i is None:
            # No clear elbow: either still scaling (headroom) or too few points.
            last = pts[-1]
            bottleneck = _classify_bottleneck(last, logical_cpus)
            knee = None
            verdict = (
                "no_knee_within_swept_range"
                if bottleneck == "headroom_remaining"
                else f"saturating:{bottleneck}"
            )
            confidence = 0.4
            recs = list(_BOTTLENECK_RECS.get(bottleneck, []))
        else:
            knee_pt = pts[knee_i]
            # The throughput elbow and the limiting signal need not coincide:
            # past the knee the box is saturated, and the bottleneck (CPU peg,
            # runqueue blow-up, memory floor) often shows a point or two later.
            # Scan the saturated region (knee → end) for the first real
            # bottleneck; fall back to the knee point's own classification.
            bottleneck = _classify_bottleneck(knee_pt, logical_cpus)
            for p in pts[knee_i:]:
                b = _classify_bottleneck(p, logical_cpus)
                if b != "headroom_remaining":
                    bottleneck = b
                    break
            # vcpu_basis is the density divisor (density = concurrency / basis);
            # recover it from the knee point so we can express throughput PER
            # CORE — the efficiency metric that makes different-core SKUs
            # comparable. (analyze_sweep takes logical_cpus, not basis.)
            vcpu_basis = (round(knee_pt.concurrency / knee_pt.density)
                          if knee_pt.density else 0)
            efficiency_at_knee = (round(knee_pt.throughput / vcpu_basis, 3)
                                  if vcpu_basis else None)
            # Confidence rises if p95 corroborates the throughput elbow.
            p95_climbs = pts[-1].p95_latency > knee_pt.p95_latency * 1.1
            knee = {
                "density": knee_pt.density,
                "concurrency": knee_pt.concurrency,
                "throughput_at_knee": round(knee_pt.throughput, 2),
                "p95_at_knee": round(knee_pt.p95_latency, 2),
                "cpu_avg_at_knee": round(knee_pt.cpu_avg, 1),
                "cpu_peak_at_knee": round(knee_pt.cpu_peak, 1),
                "runqueue_max_at_knee": (round(knee_pt.runqueue_max, 1)
                                         if knee_pt.runqueue_max is not None else None),
                "vcpu_basis": vcpu_basis,
                "efficiency_at_knee": efficiency_at_knee,
                "p95_corroborated": bool(p95_climbs),
            }
            verdict = f"saturation_knee@density={knee_pt.density:g}:{bottleneck}"
            confidence = 0.8 if p95_climbs else 0.6
            recs = list(_BOTTLENECK_RECS.get(bottleneck, []))
            recs.insert(
                0,
                f"Knee at density {knee_pt.density:g} (~{knee_pt.concurrency} agents "
                f"on {logical_cpus} logical CPUs); bottleneck: {bottleneck}.",
            )

        recs.append(
            "NOTE: bottleneck thresholds are EMR estimates — calibrate against "
            "validated sweeps before publishing."
        )

        runqueue_unmeasured = all(p.runqueue_max is None for p in pts)
        if runqueue_unmeasured:
            recs.append(
                "CAVEAT: run-queue depth was not measured (cpuset-scoped sweeps "
                "cannot read it from /proc/stat), so the scheduler_oversubscription "
                "rule could not be evaluated — it is untested, not ruled out."
            )

        # Per-task breakdown (optional) so the UI can profile e.g. mteb-retrieve.
        per_task_evidence: Dict[str, Any] = {}
        if per_task_points:
            for task_id, tpts in per_task_points.items():
                tp = _aggregate_replicates(tpts)
                if len(tp) < 2:
                    continue
                t_knee_i = _kneedle([p.density for p in tp], [p.throughput for p in tp])
                if t_knee_i is not None:
                    kp = tp[t_knee_i]
                    per_task_evidence[task_id] = {
                        "knee_density": kp.density,
                        "bottleneck": _classify_bottleneck(kp, logical_cpus),
                        "curve": [self._pt_dict(p) for p in tp],
                    }
                else:
                    per_task_evidence[task_id] = {
                        "knee_density": None,
                        "bottleneck": _classify_bottleneck(tp[-1], logical_cpus),
                        "curve": [self._pt_dict(p) for p in tp],
                    }

        evidence: Dict[str, Any] = {
            "curve": curve,
            "knee": knee,
            "bottleneck": bottleneck,
            "logical_cpus": logical_cpus,
            "n_points": len(pts),
            "runqueue_unmeasured": runqueue_unmeasured,
        }
        if per_task_evidence:
            evidence["per_task"] = per_task_evidence

        return AnalysisResult(
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            recommendations=recs,
            analyzer_name=self.name,
        )

    @staticmethod
    def _pt_dict(p: _Point) -> Dict[str, Any]:
        return {
            "density": p.density,
            "concurrency": p.concurrency,
            "throughput": round(p.throughput, 2),
            "p95_latency": round(p.p95_latency, 2),
            "cpu_avg": round(p.cpu_avg, 1),
            "cpu_peak": round(p.cpu_peak, 1),
            "runqueue_max": (round(p.runqueue_max, 1)
                             if p.runqueue_max is not None else None),
            "ctx_sw_per_s": (round(p.ctx_sw_per_s, 0)
                             if p.ctx_sw_per_s is not None else None),
            "iowait_pct_avg": round(p.iowait_pct_avg, 1),
        }


__all__ = ["ScalingAnalyzer"]
