#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Plotly figure builders for the concurrency-scaling views.

Two views, both driven off a sweep's ``sweep_points`` rows and the
ScalingAnalyzer verdict:

  * :func:`throughput_knee_figure` — the headline aggregate curve: density vs
    throughput and p95 latency (dual axis) with the saturation knee marked.
  * :func:`per_task_profile_figure` — one task's profile across density
    (e.g. mteb-retrieve): its latency + CPU, with its own knee/bottleneck.

Figures take plain dict rows (as returned by
:meth:`SQLiteResultStore.query_sweep_points` and the analyzer's evidence) so the
builders carry no Streamlit/DB coupling and can be unit-tested directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# One-way import of the analyzer's calibrated thresholds so the pressure scores
# and the signature panel match the classification logic (single source of truth
# — the bottleneck shown here can never disagree with the analyzer's verdict).
from src.analyzers.scaling import (
    RUNQUEUE_OVERSUB_RATIO,
    CPU_SATURATION_PEAK,
    CPU_SATURATION_AVG,
    MEM_LOW_MB,
    IOWAIT_HIGH_PCT,
)

# Exec-facing limiting-factor classes → (color, short label). One ordering,
# shared by the band, the heatstrip, and BOTTLENECK_STYLE semantics.
LIMITING_FACTOR_STYLE = {
    "compute":   ("#d32f2f", "Compute"),     # CPU pegged
    "memory":    ("#7b1fa2", "Memory"),      # DRAM headroom low
    "scheduler": ("#f57c00", "Scheduler"),   # runqueue oversubscribed
    "io":        ("#0288d1", "I/O"),
    "headroom":  ("#388e3c", "Headroom"),    # not saturated
}
_LIMITING_ORDER = ["compute", "memory", "scheduler", "io", "headroom"]

# Bottleneck → display colour/label, shared across views.
BOTTLENECK_STYLE = {
    "cpu_bound": ("#d32f2f", "CPU-bound"),
    "scheduler_oversubscription": ("#f57c00", "Scheduler-oversubscribed"),
    "memory_bound": ("#7b1fa2", "Memory-bound"),
    "io_bound": ("#0288d1", "I/O-bound"),
    "headroom_remaining": ("#388e3c", "Headroom remaining"),
}


def _avg_by_density(points: Sequence[Dict[str, Any]], key: str) -> List[tuple]:
    """Average ``key`` across replicates at each density → sorted (density, val)."""
    buckets: Dict[float, List[float]] = {}
    for p in points:
        d = p.get("density")
        v = p.get(key)
        if d is None or v is None:
            continue
        buckets.setdefault(round(float(d), 4), []).append(float(v))
    return [(d, sum(vs) / len(vs)) for d, vs in sorted(buckets.items())]


def _pressure_scores(point: Dict[str, Any], logical_cpus: int) -> Dict[str, float]:
    """Normalized [0,1] pressure per limiting factor for one density cell.

    Each score is value/threshold clamped to [0,1] — 1.0 means "at the
    saturation threshold" (the SAME thresholds ScalingAnalyzer classifies on, so
    the dominant factor here equals the analyzer's bottleneck). 'headroom' is the
    slack left by the others, so the four pressures + headroom sum to ~1 for the
    stacked band.
    """
    def clamp(x):
        return max(0.0, min(1.0, x))
    cpu_peak = point.get("cpu_peak") or 0.0
    cpu_avg = point.get("cpu_avg") or 0.0
    runq = point.get("runqueue_max") or 0.0
    mem = point.get("mem_avail_mb_min")
    iow = point.get("iowait_pct_avg") or 0.0

    # Compute: how close cpu_peak/avg are to the pegged thresholds (avg gates it
    # so a bursty-but-idle box doesn't read as compute-bound).
    compute = clamp(cpu_peak / CPU_SATURATION_PEAK) * clamp(cpu_avg / CPU_SATURATION_AVG)
    # SCHEDULER: runqueue depth vs the oversubscription line (ratio x cores).
    #
    # SCOPE MISMATCH, and it inflates this score. runqueue_max is
    # /proc/stat procs_running, which is HOST-WIDE — every runnable task on all
    # 288 cores, including other tenants and kernel threads — while
    # logical_cpus here is the 16-core cpuset the agents were pinned to.
    # Measured: 205 at n=24, i.e. 205/(1.5x16) = 8.5x "oversubscribed", when the
    # 24 agents can account for at most a fraction of that queue. The cell
    # records runqueue_is_host_wide=True precisely so this is auditable.
    #
    # There is no per-cpuset runnable count in /proc — procs_running has no
    # per-cpu form — so the honest options are to drop the signal or to scale it
    # against the whole machine it was actually measured on. Scaling against the
    # host keeps a real signal (queue depth does rise 29 -> 205 across the
    # ladder) without claiming the agents caused all of it.
    _m = point.get("metadata") if isinstance(point.get("metadata"), dict) else {}
    host_wide = point.get("runqueue_is_host_wide") or _m.get("runqueue_is_host_wide")
    host_cpus = point.get("logical_cpus_host") or _m.get("logical_cpus_host")
    # Only rescale when we KNOW the host width. Falling back to the cpuset when
    # logical_cpus_host is absent would silently reintroduce the 8x inflation on
    # older cells, so those keep the old basis and stay comparable with whatever
    # they were previously read as.
    sched_basis = host_cpus if (host_wide and host_cpus) else logical_cpus
    sched = (clamp(runq / (RUNQUEUE_OVERSUB_RATIO * sched_basis))
             if sched_basis else 0.0)
    # MEMORY: free-capacity pressure only, and on this class of machine it is
    # structurally unreachable — which is WHY the Memory band never appears.
    # Measured: 704 GB available against MEM_LOW_MB=2048, i.e. 344x above the
    # threshold at every density.
    #
    # The signal that actually moves is BANDWIDTH (0.46 -> 1.41 GB/s across the
    # ladder, 3x), but it cannot be turned into a saturation score here: scoring
    # needs a denominator, and this platform's DRAM peak is ESTIMATED
    # (detect.py reports dram_bw_is_measured=False, 614 GB/s inferred from the
    # uarch table). Dividing a real reading by a guessed peak manufactures a
    # percentage that looks measured. So bandwidth is reported in the resource
    # spectrum in GB/s, where the reader judges the trend, and is deliberately
    # kept OUT of this band. The band saying "headroom" is then correct and not a
    # blind spot — it means no measured saturation threshold was crossed.
    memory = clamp((MEM_LOW_MB * 2 - mem) / (MEM_LOW_MB * 2)) if (mem is not None) else 0.0

    # I/O: iowait against its threshold, unchanged.
    #
    # Deliberately NOT normalised against the busiest cell in the sweep: that
    # would make the busiest cell score 1.0 by construction and render as
    # "I/O-bound" no matter how little I/O it did. Disk throughput has no
    # measured device ceiling here, so there is nothing honest to divide by — the
    # disk axis belongs in the resource spectrum, where it is shown in MB/s and
    # the reader judges it, not in a saturation score that implies a limit.
    io = clamp(iow / IOWAIT_HIGH_PCT)
    saturation = max(compute, sched, memory, io)
    return {
        "compute": compute, "memory": memory, "scheduler": sched, "io": io,
        "headroom": clamp(1.0 - saturation),
    }


def _dominant_factor(point: Dict[str, Any], logical_cpus: int) -> str:
    """The single limiting factor an exec sees for a cell — argmax of the
    pressures, but only if something actually crosses saturation; else headroom.
    Mirrors _classify_bottleneck's precedence (scheduler > compute > memory > io)."""
    s = _pressure_scores(point, logical_cpus)
    if s["headroom"] >= 0.5:
        return "headroom"
    # precedence ties to the analyzer's ordering
    for f in ("scheduler", "compute", "memory", "io"):
        if s[f] >= 1.0:
            return f
    return max(("compute", "memory", "scheduler", "io"), key=lambda k: s[k])


def limiting_factor_band_figure(
    points: Sequence[Dict[str, Any]],
    logical_cpus: int,
    knee: Optional[Dict[str, Any]] = None,
    bottleneck: Optional[str] = None,
) -> go.Figure:
    """Exec headline: 'what's eating the box' as a normalized 0-100% stacked band
    vs density. At a glance, where does pressure shift compute→memory→scheduler
    as agents pile on, and how much headroom is left. Replaces the data-heavy
    3-panel signature as the lead chart (that moves to an expander)."""
    # Average each pressure across replicates at each density.
    densities = sorted({round(float(p["density"]), 4) for p in points if p.get("density") is not None})
    series = {f: [] for f in _LIMITING_ORDER}
    for d in densities:
        cells = [p for p in points if round(float(p.get("density", -1)), 4) == d]
        scored = [_pressure_scores(c, logical_cpus) for c in cells]
        for f in _LIMITING_ORDER:
            series[f].append(sum(s[f] for s in scored) / len(scored) if scored else 0.0)
    # Normalize each density's stack to 100%.
    fig = go.Figure()
    for f in _LIMITING_ORDER:
        color, label = LIMITING_FACTOR_STYLE[f]
        ys = []
        for i, d in enumerate(densities):
            total = sum(series[g][i] for g in _LIMITING_ORDER) or 1.0
            ys.append(series[f][i] / total * 100.0)
        fig.add_trace(go.Scatter(
            x=densities, y=ys, mode="lines", name=label, stackgroup="one",
            line=dict(width=0.5, color=color), fillcolor=color,
            hovertemplate=f"{label}: %{{y:.0f}}%<extra></extra>",
        ))
    if knee and knee.get("density") is not None:
        fig.add_vline(x=knee["density"], line_dash="dash", line_color="#212121", line_width=2,
                      annotation_text="knee", annotation_position="top")
    fig.update_layout(
        title="What's limiting the box, as agent density rises",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="Share of limiting pressure (%)", range=[0, 100]),
        legend=dict(orientation="h", y=-0.2),
        height=400, margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def bottleneck_heatstrip_figure(
    points: Sequence[Dict[str, Any]],
    logical_cpus: int,
) -> go.Figure:
    """One-row traffic-light strip: the DOMINANT limiting factor at each density.
    The fastest possible read — 'green headroom until ~here, then it goes red
    (compute)'. Cell color = dominant factor; hover gives the density + factor."""
    densities = sorted({round(float(p["density"]), 4) for p in points if p.get("density") is not None})
    factors, labels, colors = [], [], []
    for d in densities:
        cells = [p for p in points if round(float(p.get("density", -1)), 4) == d]
        # use the worst (most-saturated) replicate's dominant factor at this density
        dom = _dominant_factor(max(cells, key=lambda c: _pressure_scores(c, logical_cpus)["compute"]
                                   + _pressure_scores(c, logical_cpus)["memory"]), logical_cpus) if cells else "headroom"
        factors.append(_LIMITING_ORDER.index(dom))
        labels.append(LIMITING_FACTOR_STYLE[dom][1])
    # Discrete heatmap: one row, integer class per density.
    fig = go.Figure(go.Heatmap(
        z=[factors], x=densities, y=["limiting factor"],
        text=[labels], texttemplate="%{text}", hoverinfo="x+text",
        colorscale=[[i / (len(_LIMITING_ORDER) - 1), LIMITING_FACTOR_STYLE[f][0]]
                    for i, f in enumerate(_LIMITING_ORDER)],
        zmin=0, zmax=len(_LIMITING_ORDER) - 1, showscale=False,
    ))
    fig.update_layout(
        title="Dominant bottleneck at each density",
        xaxis_title="Density (concurrency / vCPU)",
        height=140, margin=dict(l=60, r=40, t=50, b=40),
    )
    return fig


def throughput_knee_figure(
    points: Sequence[Dict[str, Any]],
    knee: Optional[Dict[str, Any]] = None,
    bottleneck: Optional[str] = None,
) -> go.Figure:
    """Aggregate scaling curve: throughput + p95 vs density, with the knee marked."""
    thr = _avg_by_density(points, "throughput_per_min")
    p95 = _avg_by_density(points, "p95_trial_latency_s")

    fig = go.Figure()
    if thr:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in thr], y=[v for _, v in thr],
            mode="lines+markers", name="Throughput (trials/min)",
            line=dict(color="#1976d2", width=3), yaxis="y1",
        ))
    if p95:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in p95], y=[v for _, v in p95],
            mode="lines+markers", name="p95 trial latency (s)",
            line=dict(color="#e64a19", width=2, dash="dot"), yaxis="y2",
        ))

    if knee and knee.get("density") is not None:
        color, label = BOTTLENECK_STYLE.get(bottleneck or "", ("#616161", bottleneck or "knee"))
        kd = knee["density"]
        fig.add_vline(x=kd, line_dash="dash", line_color=color, line_width=2)
        fig.add_annotation(
            x=kd, y=1.0, yref="paper", showarrow=False, yanchor="bottom",
            text=f"knee @ density {kd:g} (~{knee.get('concurrency','?')} agents) — {label}",
            font=dict(color=color, size=12),
        )

    fig.update_layout(
        title="Agentic CPU scaling — throughput vs density (agents per vCPU)",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="Throughput (trials/min)", side="left"),
        yaxis2=dict(title="p95 latency (s)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.2),
        height=420, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def throughput_compare_figure(sweeps: Sequence[Dict[str, Any]]) -> go.Figure:
    """Overlay several sweeps' throughput-vs-density curves for A/B comparison.

    ``sweeps`` is a list of ``{"label": str, "points": [sweep_point rows],
    "knee": optional knee dict}``. Each sweep gets one throughput line (averaged
    across replicates by density) plus its knee marker, so e.g. NUMA-pinned vs
    unpinned, or replay vs off, sit on one chart.
    """
    palette = ["#1976d2", "#e64a19", "#388e3c", "#7b1fa2", "#f9a825", "#00838f"]
    fig = go.Figure()
    for i, sw in enumerate(sweeps):
        color = palette[i % len(palette)]
        thr = _avg_by_density(sw.get("points", []), "throughput_per_min")
        if not thr:
            continue
        label = sw.get("label", f"sweep {i + 1}")
        fig.add_trace(go.Scatter(
            x=[d for d, _ in thr], y=[v for _, v in thr],
            mode="lines+markers", name=label,
            line=dict(color=color, width=3),
        ))
        knee = sw.get("knee")
        if knee and knee.get("density") is not None:
            fig.add_vline(x=knee["density"], line_dash="dash", line_color=color, line_width=1.5)
    fig.update_layout(
        title="Sweep comparison — throughput vs density",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="Throughput (trials/min)"),
        legend=dict(orientation="h", y=-0.2),
        height=420, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def efficiency_figure(
    points: Sequence[Dict[str, Any]],
    vcpu_basis: int,
    knee: Optional[Dict[str, Any]] = None,
    bottleneck: Optional[str] = None,
) -> go.Figure:
    """Throughput-PER-CORE vs density — the efficiency curve.

    throughput_per_min / vcpu_basis normalizes for core count, so the knee on
    this curve is the architecturally meaningful "work per core before
    diminishing returns" — comparable across different-core-count SKUs.
    """
    basis = max(int(vcpu_basis or 0), 1)
    thr = _avg_by_density(points, "throughput_per_min")
    eff = [(d, v / basis) for d, v in thr]

    fig = go.Figure()
    if eff:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in eff], y=[v for _, v in eff],
            mode="lines+markers", name="Throughput / core (trials/min/core)",
            line=dict(color="#00838f", width=3),
        ))
    if knee and knee.get("density") is not None:
        color, label = BOTTLENECK_STYLE.get(bottleneck or "", ("#616161", bottleneck or "knee"))
        fig.add_vline(x=knee["density"], line_dash="dash", line_color=color, line_width=2)
        fig.add_annotation(
            x=knee["density"], y=1.0, yref="paper", showarrow=False, yanchor="bottom",
            text=f"knee @ density {knee['density']:g} — {label}",
            font=dict(color=color, size=12),
        )
    fig.update_layout(
        title="Efficiency — throughput per core vs density",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="Throughput per core (trials/min/core)"),
        legend=dict(orientation="h", y=-0.2),
        height=360, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def efficiency_compare_figure(
    sweeps: Sequence[Dict[str, Any]],
    vcpu_bases: Optional[Sequence[int]] = None,
) -> go.Figure:
    """Overlay each sweep's throughput-per-core curve for A/B comparison.

    ``sweeps`` is ``[{"label", "points", "vcpu_basis"(optional)}]``; ``vcpu_bases``
    is an optional parallel list overriding per-sweep basis.
    """
    palette = ["#00838f", "#e64a19", "#388e3c", "#7b1fa2", "#1976d2", "#f9a825"]
    fig = go.Figure()
    for i, sw in enumerate(sweeps):
        basis = max(int((vcpu_bases[i] if vcpu_bases and i < len(vcpu_bases)
                         else sw.get("vcpu_basis")) or 0), 1)
        thr = _avg_by_density(sw.get("points", []), "throughput_per_min")
        if not thr:
            continue
        fig.add_trace(go.Scatter(
            x=[d for d, _ in thr], y=[v / basis for _, v in thr],
            mode="lines+markers", name=sw.get("label", f"sweep {i + 1}"),
            line=dict(color=palette[i % len(palette)], width=3),
        ))
    fig.update_layout(
        title="Efficiency comparison — throughput per core vs density",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="Throughput per core (trials/min/core)"),
        legend=dict(orientation="h", y=-0.2),
        height=360, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def cpu_runqueue_figure(points: Sequence[Dict[str, Any]]) -> go.Figure:
    """Saturation-signature panel: CPU peak/avg and runqueue vs density."""
    cpu_peak = _avg_by_density(points, "cpu_peak")
    cpu_avg = _avg_by_density(points, "cpu_avg")
    runq = _avg_by_density(points, "runqueue_max")

    fig = go.Figure()
    if cpu_peak:
        fig.add_trace(go.Scatter(x=[d for d, _ in cpu_peak], y=[v for _, v in cpu_peak],
                                 mode="lines+markers", name="CPU peak %",
                                 line=dict(color="#d32f2f")))
    if cpu_avg:
        fig.add_trace(go.Scatter(x=[d for d, _ in cpu_avg], y=[v for _, v in cpu_avg],
                                 mode="lines+markers", name="CPU avg %",
                                 line=dict(color="#fbc02d")))
    if runq:
        fig.add_trace(go.Scatter(x=[d for d, _ in runq], y=[v for _, v in runq],
                                 mode="lines+markers", name="Tasks waiting to run (peak)",
                                 line=dict(color="#512da8", dash="dot"), yaxis="y2"))
    fig.update_layout(
        title="Saturation signature — CPU & run queue vs density",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="CPU %", range=[0, 105]),
        yaxis2=dict(title="Tasks waiting to run", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.2),
        height=360, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def cpu_runqueue_compare_figure(sweeps: Sequence[Dict[str, Any]]) -> go.Figure:
    """Overlay CPU-peak% and tasks-waiting-to-run for several sweeps — shows WHY a
    knee moved (e.g. NUMA pinning lowering the run queue at the same density).

    ``sweeps`` is ``[{"label", "points", "logical_cpus"(optional)}]``. CPU on the
    left axis (solid), runqueue on the right (dashed); both colored per sweep so
    the legend ties line→sweep.
    """
    palette = ["#d32f2f", "#1976d2", "#388e3c", "#7b1fa2", "#f57c00", "#00838f"]
    fig = go.Figure()
    for i, sw in enumerate(sweeps):
        color = palette[i % len(palette)]
        label = sw.get("label", f"sweep {i + 1}")
        cpu = _avg_by_density(sw.get("points", []), "cpu_peak")
        rq = _avg_by_density(sw.get("points", []), "runqueue_max")
        if cpu:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in cpu], y=[v for _, v in cpu],
                mode="lines+markers", name=f"{label} · CPU peak %",
                line=dict(color=color, width=3)))
        if rq:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in rq], y=[v for _, v in rq],
                mode="lines+markers", name=f"{label} · tasks waiting",
                line=dict(color=color, width=2, dash="dot"), yaxis="y2"))
    fig.update_layout(
        title="Saturation-signature comparison — CPU & run queue vs density",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="CPU %", range=[0, 105]),
        yaxis2=dict(title="Tasks waiting to run", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.25),
        height=400, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def saturation_signature_figure(
    points: Sequence[Dict[str, Any]],
    logical_cpus: int,
    bottleneck: Optional[str] = None,
    knee: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Proof panel for the bottleneck classification: the stored-but-unshown
    signals that justify a memory_bound / io_bound / scheduler verdict.

    Three stacked rows (shared x = density):
      1. Run queue depth, with reference lines at ``logical_cpus`` (1 thread/core)
         and ``1.5×logical_cpus`` (the scheduler-oversubscription threshold).
      2. Context-switch rate (scheduler thrash signal).
      3. I/O wait % and available memory (MB) on a secondary axis (io/memory
         bound signals).
    A knee vline (bottleneck-colored) spans all rows.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Run queue depth (vs core count)",
                        "Context-switch rate (/s)",
                        "I/O wait % and available memory"),
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    rq = _avg_by_density(points, "runqueue_max")
    # "runqueue" alone is opaque, and the number is host-wide. Name what it counts
    # and whose cores it counts across, or a 205 on a 16-core cpuset reads as
    # catastrophic oversubscription when most of the queue is other work.
    _m0 = next((p.get("metadata") for p in points
                if isinstance(p.get("metadata"), dict)), {}) or {}
    _host_cpus = _m0.get("logical_cpus_host")
    _host_wide = _m0.get("runqueue_is_host_wide")
    _rq_name = ("Tasks waiting to run (whole machine)" if _host_wide
                else "Tasks waiting to run")
    if rq:
        fig.add_trace(go.Scatter(x=[d for d, _ in rq], y=[v for _, v in rq],
                                 mode="lines+markers", name=_rq_name,
                                 line=dict(color="#512da8")), row=1, col=1)
    # Draw the reference lines against the width the metric was MEASURED over. On
    # a pinned sub-range the host-wide queue vs the cpuset width is a scope
    # mismatch, and the "oversubscribed" line would be crossed at 2 agents.
    _rq_basis = _host_cpus if (_host_wide and _host_cpus) else logical_cpus
    if _rq_basis and _rq_basis > 0:
        _scope = "machine" if (_host_wide and _host_cpus) else "cpuset"
        fig.add_hline(y=_rq_basis, line_dash="dot", line_color="#9e9e9e",
                      annotation_text=f"{_rq_basis} = 1 per core ({_scope})",
                      row=1, col=1)
        fig.add_hline(y=RUNQUEUE_OVERSUB_RATIO * _rq_basis, line_dash="dash",
                      line_color="#f57c00",
                      annotation_text=f"{RUNQUEUE_OVERSUB_RATIO:g}× — oversubscribed",
                      row=1, col=1)

    ctx = _avg_by_density(points, "ctx_sw_per_s")
    if ctx:
        fig.add_trace(go.Scatter(x=[d for d, _ in ctx], y=[v for _, v in ctx],
                                 mode="lines+markers", name="ctx switches/s",
                                 line=dict(color="#00838f")), row=2, col=1)

    iow = _avg_by_density(points, "iowait_pct_avg")
    mem = _avg_by_density(points, "mem_avail_mb_min")
    if iow:
        fig.add_trace(go.Scatter(x=[d for d, _ in iow], y=[v for _, v in iow],
                                 mode="lines+markers", name="iowait %",
                                 line=dict(color="#c2185b")), row=3, col=1, secondary_y=False)
    if mem:
        fig.add_trace(go.Scatter(x=[d for d, _ in mem], y=[v for _, v in mem],
                                 mode="lines+markers", name="mem avail (MB)",
                                 line=dict(color="#388e3c", dash="dot")),
                      row=3, col=1, secondary_y=True)

    if knee and knee.get("density") is not None:
        color, _ = BOTTLENECK_STYLE.get(bottleneck or "", ("#616161", ""))
        fig.add_vline(x=knee["density"], line_dash="dash", line_color=color, line_width=2)

    fig.update_xaxes(title_text="Density (concurrency / vCPU)", row=3, col=1)
    fig.update_layout(height=620, margin=dict(l=60, r=60, t=50, b=50),
                      legend=dict(orientation="h", y=-0.12),
                      title="Bottleneck-proof signals")
    return fig


def per_task_profile_figure(
    task_id: str,
    curve: Sequence[Dict[str, Any]],
    knee_density: Optional[float] = None,
    bottleneck: Optional[str] = None,
) -> go.Figure:
    """One task's profile across density: latency + CPU peak, with its knee.

    ``curve`` is the per-task point list (from the analyzer's
    ``evidence.per_task[task_id].curve``).
    """
    xs = [c["density"] for c in curve]
    lat = [c.get("p95_latency", 0.0) for c in curve]
    cpu = [c.get("cpu_peak", 0.0) for c in curve]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=lat, mode="lines+markers",
                             name="p95 latency (s)", line=dict(color="#e64a19", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=cpu, mode="lines+markers",
                             name="CPU peak %", line=dict(color="#d32f2f", dash="dot"),
                             yaxis="y2"))
    if knee_density is not None:
        color, label = BOTTLENECK_STYLE.get(bottleneck or "", ("#616161", bottleneck or "knee"))
        fig.add_vline(x=knee_density, line_dash="dash", line_color=color)
        fig.add_annotation(x=knee_density, y=1.0, yref="paper", showarrow=False,
                           yanchor="bottom", text=f"knee — {label}",
                           font=dict(color=color, size=12))
    fig.update_layout(
        title=f"Task profile: {task_id}",
        xaxis_title="Density (concurrency / vCPU)",
        yaxis=dict(title="p95 latency (s)"),
        yaxis2=dict(title="CPU peak %", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
        legend=dict(orientation="h", y=-0.2),
        height=360, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


# ── Resource spectrum (task signature) ────────────────────────────────────────
# Per-component display: (label, unit, y-format). One row per component with its
# OWN y-scale, because the components span three orders of magnitude —
# quota_fill ~0.78 against net_mb_s ~0. A shared axis would collapse four of six
# panels onto the origin.
# quota_demand is DELIBERATELY NOT a row: it is the x-axis, so plotting it as a
# panel draws a y=x diagonal that carries no information while consuming a sixth
# of the figure and inviting a reader to look for a trend in it.
# (component key, legend label, short unit for the peak annotation). The unit is
# what makes "100%" unambiguous in a merged chart — 100% of a fraction and 100%
# of MB/s are different kinds of statement, so the legend states the real peak
# and its unit next to every series name.
_SPECTRUM_ROWS = [
    ("quota_fill",      "CPU used, of what was promised", "of quota"),
    ("serialization",   "Time NOT computing",             "of wall time"),
    ("mem_bw_gb_s",     "Memory bandwidth",               "GB/s (whole socket)"),
    ("disk_write_mb_s", "Disk writes",                    "MB/s (whole host)"),
    ("net_mb_s",        "Network",                        "MB/s (whole host)"),
]

_SPECTRUM_COLORS = ["#1976d2", "#d32f2f", "#7b1fa2", "#f57c00", "#0288d1", "#388e3c"]


def _spectrum_xy(points: Sequence[Any], comp: str) -> tuple:
    """(x, y) over points that HAVE this component, x = quota demand.

    Points missing the component are dropped, not zero-filled: a gap in the line
    means unmeasured, which is different from measured-as-zero and must look
    different.
    """
    xs, ys = [], []
    for p in points:
        x = p.components.get("quota_demand")
        y = p.components.get(comp)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


# Resource GROUPS, one panel each. Grouping by resource rather than by metric is
# what lets every panel use NATIVE units: the three CPU series are all fractions
# of something, memory is GB/s alone, disk MB/s alone. That removes the need for
# the "% of its own peak" normalisation an all-in-one chart required — heights
# inside a panel are directly comparable and mean what they say.
#
# (key, legend label, colour, is_fraction)
_CPU_SERIES = [
    ("__retention__", "Work finished per agent (vs its own best)", "#424242", True),
    ("quota_fill",    "CPU used, of what was promised",            "#1976d2", True),
    ("serialization", "Time NOT computing (waiting)",              "#d32f2f", True),
]
_GROUPS = [
    ("CPU", "% (each line is a share — see legend)", _CPU_SERIES),
    ("Memory bandwidth", "GB/s · whole socket, incl. other tenants",
     [("mem_bw_gb_s", "Memory bandwidth", "#7b1fa2", False)]),
    ("Disk", "MB/s written · whole host, incl. other tenants",
     [("disk_write_mb_s", "Disk writes", "#f57c00", False)]),
    ("Network", "MB/s · whole host, incl. other tenants",
     [("net_mb_s", "Network", "#0288d1", False)]),
]


def _series_xy(points: Sequence[Any], comp: str):
    """(x, y, replicates) for one series. y keeps None so the line BREAKS there.

    Dropping the x too would let Plotly draw a straight segment through an
    unmeasured point, which is exactly the "gaps mean unmeasured" promise the
    caption makes. y=None plus connectgaps=False is what actually keeps it.
    """
    xs, ys, reps = [], [], []
    for p in points:
        x = p.components.get("quota_demand")
        if x is None:
            continue
        xs.append(x)
        if comp == "__retention__":
            ys.append(p.retention)
        else:
            ys.append(p.components.get(comp))
        reps.append(getattr(p, "replicates", 1))
    return xs, ys, reps


def resource_spectrum_figure(points: Sequence[Any]) -> go.Figure:
    """One panel per RESOURCE — CPU, memory bandwidth, disk, network — vs quota demand.

    ``points`` is the output of
    :func:`src.analyzers.task_signature.signature_for_sweep`.

    Grouped by resource rather than one panel per metric, which keeps each panel
    internally single-unit and therefore in NATIVE units. That is the reason this
    shape is preferable to a merged single chart: the components span 37x in raw
    units (disk 28.7 MB/s against a 0.78 CPU fraction), so one shared axis needs
    a "% of its own peak" rescale that makes heights meaningless. Here a height
    means what it says.

    The CPU panel carries the outcome (work finished per agent) alongside the two
    CPU explanations, because that is the comparison a reader makes: "we lost
    speed here — was the CPU full, or was it waiting?"

    A panel whose only series was measured as zero everywhere is dropped and
    named in the title instead, so a flat zero line does not consume a panel while
    staying distinguishable from unmeasured (which is a break in a line).

    x is quota demand, not agent count: two tasks whose declared ``cpus`` differ
    are not at the same load at the same agent count.
    """
    if not points:
        return go.Figure(layout=dict(
            title="No signature points — import a sweep first", height=300))

    live, flat_zero, absent = [], [], []
    for title, unit, series in _GROUPS:
        vals = [v for comp, _l, _c, _f in series
                for v in _series_xy(points, comp)[1] if v is not None]
        if not vals:
            absent.append(title)
        elif all(v == 0 for v in vals):
            flat_zero.append(title)
        else:
            live.append((title, unit, series))
    if not live:
        return go.Figure(layout=dict(
            title="No varying resources in this sweep", height=300))

    fig = make_subplots(
        rows=len(live), cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[f"{t} — {u}" for t, u, _ in live],
    )

    for row, (_title, _unit, series) in enumerate(live, start=1):
        for comp, lbl, colour, is_frac in series:
            xs, ys, reps = _series_xy(points, comp)
            if not any(v is not None for v in ys):
                continue
            # Fractions display as percentages; throughputs in native units.
            disp = [(v * 100.0 if (v is not None and is_frac) else v) for v in ys]
            thin = [r < 3 for r in reps]
            wide = comp == "__retention__"
            fig.add_trace(
                go.Scatter(
                    x=xs, y=disp, mode="lines+markers", name=lbl,
                    connectgaps=False,
                    legendgroup=_title, showlegend=True,
                    line=dict(color=colour, width=3.5 if wide else 2,
                              dash="dot" if any(thin) else "solid"),
                    # Hollow marker = fewer than 3 replicates at that point, so
                    # the confidence differs point by point rather than only
                    # per-figure.
                    marker=dict(color=colour,
                                size=[11 if t else 8 for t in thin],
                                symbol=["circle-open" if t else "circle"
                                        for t in thin]),
                    customdata=[[r] for r in reps],
                    hovertemplate=(f"{lbl}<br>%{{y:.2f}}"
                                   + ("%" if is_frac else "")
                                   + "<br>%{customdata[0]} rep(s)<extra></extra>"),
                ),
                row=row, col=1,
            )
        if all(f for _c, _l, _col, f in series):
            fig.update_yaxes(range=[0, 105], ticksuffix="%", row=row, col=1)

    # Quota demand 1.0 = every pinned core promised to some agent. Annotated on
    # the top panel only; repeating the text on each panel is noise, and only
    # drawn when the sweep actually reaches it, since an unconditional line
    # implies a threshold the data never tested.
    if any((p.components.get("quota_demand") or 0) >= 1.0 for p in points):
        for row in range(1, len(live) + 1):
            fig.add_vline(
                x=1.0, line=dict(color="#616161", width=1.5, dash="dash"),
                row=row, col=1,
                **(dict(annotation_text="every core promised",
                        annotation_position="top left",
                        annotation_font=dict(size=11, color="#424242"))
                   if row == 1 else {}),
            )

    task = points[0].task or "unknown task"
    notes = []
    if flat_zero:
        notes.append("measured zero throughout: " + ", ".join(flat_zero))
    if absent:
        notes.append("not measured: " + ", ".join(absent))
    fig.update_layout(
        title=(f"How each resource responds as agents are added — {task}"
               + (f"<br><sub>{' · '.join(notes)}</sub>" if notes else "")),
        height=250 * len(live) + 110,
        margin=dict(l=70, r=40, t=90 if notes else 70, b=60),
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Agents' promised share of the pinned cores "
                   "(1.0 = every core promised)",
        row=len(live), col=1)
    return fig


def _collect_caveats(points: Sequence[Any]) -> str:
    """Fold per-point caveats into one annotation line.

    These belong ON the chart, not in a doc: a socket-scoped bandwidth number and
    a cpuset-scoped CPU number look identical once plotted, and a reader who
    assumes both are per-cell draws a wrong conclusion.
    """
    seen: Dict[str, None] = {}
    for p in points:
        for c in getattr(p, "caveats", []) or []:
            seen.setdefault(c, None)
    missing = sorted({m for p in points for m in (getattr(p, "missing", []) or [])})
    parts = list(seen)
    if missing:
        parts.append("UNMEASURED: " + ", ".join(missing))
    return " · ".join(parts)


def signature_compare_figure(sweeps: Sequence[Dict[str, Any]]) -> go.Figure:
    """Overlay two tasks' spectra on one quota-demand axis.

    ``sweeps`` is ``[{"label": str, "points": [SignaturePoint]}]``.

    This answers the originating question directly. Plotted against agent count,
    circuit-fibsqrt and overfull-hbox appear to knee at 2x different densities;
    plotted against quota demand both sit at x = 1.0 and the apparent difference
    collapses. The panels then show why the APPROACH differs — hbox's
    serialization starts high and stays high, fib's only breaks past x = 1.0.
    """
    live = [s for s in sweeps if s.get("points")]
    if len(live) < 2:
        return go.Figure(layout=dict(
            title="Need two sweeps with signature points to compare", height=300))

    all_pts = [p for s in live for p in s["points"]]
    # Same flat-zero collapse as the single view: a component measured as zero at
    # every point in BOTH sweeps is a one-line fact, not a panel. Applying it in
    # only one of the two figures left them inconsistent.
    rows, flat_zero = [], []
    for r in _SPECTRUM_ROWS:
        present = [p.components.get(r[0]) for p in all_pts
                   if p.components.get(r[0]) is not None]
        if not present:
            continue
        if all(v == 0 for v in present):
            flat_zero.append(r[1])
        else:
            rows.append(r)
    # Retention leads: the question is "where does throughput fall off", and the
    # resource panels exist to explain it. Omitting it here (as the first version
    # did) left the reader comparing resources with no outcome to compare them
    # against.
    rows = [("__retention__", "Per-agent throughput retained",
             "fraction of each task's own peak")] + rows

    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[f"{lbl} — {unit}" for _, lbl, unit in rows],
    )
    palette = ["#1976d2", "#d32f2f", "#388e3c", "#f57c00"]
    for si, sw in enumerate(live):
        color = palette[si % len(palette)]
        # A single-replicate sweep gets markers WITHOUT a connecting line. hbox's
        # retention bounces 0.50 -> 0.18 -> 1.00 -> 0.56 -> 0.16 on one replicate
        # per point; a confident connected path through that asserts a shape the
        # data cannot support.
        one_rep = any(getattr(p, "replicates", 1) < 2 for p in sw["points"])
        mode = "markers" if one_rep else "lines+markers"
        for i, (comp, _lbl, _unit) in enumerate(rows, start=1):
            if comp == "__retention__":
                x = [p.components.get("quota_demand") for p in sw["points"]
                     if p.retention is not None
                     and p.components.get("quota_demand") is not None]
                y = [p.retention for p in sw["points"]
                     if p.retention is not None
                     and p.components.get("quota_demand") is not None]
            else:
                x, y = _spectrum_xy(sw["points"], comp)
            label = sw["label"] + (" (1 replicate)" if one_rep else "")
            fig.add_trace(
                go.Scatter(x=x, y=y, mode=mode, name=label,
                           legendgroup=sw["label"], showlegend=(i == 1),
                           line=dict(color=color, width=2),
                           marker=dict(size=8, color=color)),
                row=i, col=1,
            )

    if any((p.components.get("quota_demand") or 0) >= 1.0 for p in all_pts):
        for i in range(1, len(rows) + 1):
            fig.add_vline(
                x=1.0, line=dict(color="#616161", width=1, dash="dot"), row=i, col=1,
                **(dict(annotation_text="granted CPU = cpuset",
                        annotation_position="top right",
                        annotation_font=dict(size=10, color="#616161"))
                   if i == 1 else {}),
            )

    fig.update_layout(
        title=("Task signature comparison — aligned on quota demand"
               + (f"  ·  measured zero throughout: {', '.join(flat_zero)}"
                  if flat_zero else "")),
        height=170 * len(rows) + 80,
        margin=dict(l=70, r=40, t=70, b=50),
        legend=dict(orientation="h", y=-0.06),
    )
    fig.update_xaxes(
        title_text="quota demand = agents × task_cpus ÷ cpuset cores "
                   "(1.0 = granted CPU equals the cpuset)",
        row=len(rows), col=1)
    return fig


__all__ = [
    "BOTTLENECK_STYLE",
    "LIMITING_FACTOR_STYLE",
    "limiting_factor_band_figure",
    "bottleneck_heatstrip_figure",
    "throughput_knee_figure",
    "throughput_compare_figure",
    "efficiency_figure",
    "efficiency_compare_figure",
    "cpu_runqueue_figure",
    "cpu_runqueue_compare_figure",
    "saturation_signature_figure",
    "per_task_profile_figure",
    "resource_spectrum_figure",
    "signature_compare_figure",
]
