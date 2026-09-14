#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Task signature — a resource spectrum per (task, density) operating point.

Why this exists: none of the eight existing analyzers produces a resource
breakdown. Each emits a single verdict string, and `ScalingAnalyzer` — the only
sweep-scope one — classifies into five bottleneck classes that cannot name
network, memory bandwidth, or disk at all. So "task A knees at 1.0 and task B at
0.5" was reportable, but "and here is why" was not.

The signature is six components per operating point, computed from
`sweep_points` rows (plus their metadata blob) that the shell runner already
writes:

    quota_demand    declared load: concurrency x task_cpus / cores_in_cpuset
    quota_fill      measured: share of GRANTED cpu actually used
    serialization   measured: share of wall time NOT spent on cpu
    mem_bw_gb_s     measured (socket-scoped): IMC read+write bandwidth
    disk_write_mb_s measured (host-scoped): vmstat bo
    net_mb_s        measured (host-scoped): docker bridge rx+tx delta

**Components are omitted, never zero-filled.** Zero-filling is the defect this
module exists to avoid: `scaling.py:76-78` returns 0.0 for a missing average and
`inf` for missing memory, which makes an all-unmeasured sweep classify every cell
`headroom_remaining` and still emit a confident knee. A caller must be able to
tell "measured as zero" from "not measured", so every component carries a
provenance tag and absent inputs yield None.

Scope tags are load-bearing and deliberately ugly. `mem_bw_gb_s` is
SOCKET-scoped (uncore_imc ignores `-C`), and disk/net are HOST-scoped (vmstat and
the bridge counters have no cgroup view). On a shared box those include other
tenants. A reader who treats them as per-cell is wrong, so the tag travels with
the number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Component -> what the number can honestly be claimed to be. Anything reading a
# signature must respect these; they are why the module refuses to zero-fill.
PROVENANCE: Dict[str, str] = {
    "quota_demand": "declared",            # from task.toml, not measured
    "quota_fill": "measured_cpuset",
    "serialization": "derived_cpuset",
    "mem_bw_gb_s": "measured_socket",      # includes other tenants
    "disk_write_mb_s": "measured_host",    # includes other tenants
    "net_mb_s": "measured_host",           # includes other tenants
}

COMPONENTS: Sequence[str] = tuple(PROVENANCE)


@dataclass
class SignaturePoint:
    """One (task, density) operating point's resource spectrum."""

    task: Optional[str]
    concurrency: int
    density: Optional[float]
    replicates: int = 1
    components: Dict[str, Optional[float]] = field(default_factory=dict)
    # Outcome metrics the spectrum is meant to explain.
    per_agent_throughput: Optional[float] = None
    retention: Optional[float] = None
    p95_latency_s: Optional[float] = None
    ipc: Optional[float] = None
    throttled_pct: Optional[float] = None
    # Which components could not be computed, and why the point may mislead.
    missing: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "concurrency": self.concurrency,
            "density": self.density,
            "replicates": self.replicates,
            "components": dict(self.components),
            "provenance": {k: PROVENANCE[k] for k in self.components},
            "per_agent_throughput": self.per_agent_throughput,
            "retention": self.retention,
            "p95_latency_s": self.p95_latency_s,
            "ipc": self.ipc,
            "throttled_pct": self.throttled_pct,
            "missing": list(self.missing),
            "caveats": list(self.caveats),
        }


def _meta(row: Mapping[str, Any]) -> Dict[str, Any]:
    """sweep_points spills unknown keys into a metadata JSON column."""
    m = row.get("metadata")
    return m if isinstance(m, dict) else {}


def _get(row: Mapping[str, Any], key: str) -> Any:
    """Read a field from the row or its metadata blob, whichever has it."""
    if row.get(key) is not None:
        return row[key]
    return _meta(row).get(key)


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    """Mean over present values, or None. NEVER 0.0 for an empty input."""
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _serialization(row: Mapping[str, Any]) -> Optional[float]:
    """Share of a trial's wall time NOT explained by CPU work.

    cpu_avg is cpuset-scoped %, so cpu_avg/100 * cores = cores busy; times the
    cell wall gives CPU-seconds; divided by completed trials gives CPU-seconds
    per trial. Compare against the trial's own wall time: the shortfall is time
    the trial existed but was not computing — queueing, I/O, or an internally
    serialized loop.

    This is the component that separates the two measured tasks. fib sits at
    ~0.17-0.22 until its knee; hbox sits at ~0.44-0.61 throughout, i.e. half its
    wall time is not CPU at all, which is why it saturates without ever filling
    its CPU quota.
    """
    cpu_avg = _get(row, "cpu_avg")
    cores = _get(row, "cores_in_cpuset")
    elapsed = _get(row, "elapsed_s")
    completed = _get(row, "completed_trials")
    trial_wall = _get(row, "p50_trial_total_s") or _get(row, "p50_agent_exec_s")
    if None in (cpu_avg, cores, elapsed, completed, trial_wall):
        return None
    if not completed or not trial_wall:
        return None
    cpu_s_per_trial = (cpu_avg / 100.0) * cores * elapsed / completed
    frac = 1.0 - (cpu_s_per_trial / trial_wall)
    # Clamp: a negative value means CPU-seconds exceeded wall time, which happens
    # when the cpuset was busy with work outside these trials. Report the clamp
    # rather than a nonsensical negative.
    return round(max(0.0, min(1.0, frac)), 4)


def signature_for_row(row: Mapping[str, Any]) -> SignaturePoint:
    """Build a signature for one sweep_points row."""
    n = int(_get(row, "concurrency") or 0)
    comps: Dict[str, Optional[float]] = {}
    missing: List[str] = []
    caveats: List[str] = []

    comps["quota_demand"] = _get(row, "quota_demand")
    comps["quota_fill"] = _get(row, "quota_fill")
    comps["serialization"] = _serialization(row)

    bw = _get(row, "mem_bw_mib_s_socket")
    # `if bw` would treat a MEASURED 0.0 as absent — the exact conflation this
    # module exists to prevent, and the one that already bit net_mb_s (0.0 in
    # every real cell because the images are pre-cached, which is a finding, not
    # a gap). Test explicitly for None.
    comps["mem_bw_gb_s"] = round(bw / 1024.0, 4) if bw is not None else None
    comps["disk_write_mb_s"] = _get(row, "disk_write_mb_s_host")
    comps["net_mb_s"] = _get(row, "net_total_mb_s_host")

    for k, v in comps.items():
        if v is None:
            missing.append(k)

    # Caveats that change how a number should be read, surfaced per point rather
    # than buried in a doc nobody opens next to the chart.
    if _get(row, "counters_multiplexed"):
        caveats.append(
            f"counters multiplexed ({_get(row, 'counter_enabled_pct_min')}% enabled)"
            " — IPC and cache figures are scaled estimates"
        )
    seen = _get(row, "ctr_containers_seen")
    if seen is not None and n and seen < n:
        caveats.append(
            f"container sampler saw {seen} of {n} containers — per-agent"
            " aggregates cover a subset"
        )
    if comps["mem_bw_gb_s"] is not None:
        caveats.append("mem_bw is SOCKET-scoped; includes other tenants")
    if comps["disk_write_mb_s"] or comps["net_mb_s"]:
        caveats.append("disk/net are HOST-scoped; include other tenants")

    thr = _get(row, "throughput_per_min")
    return SignaturePoint(
        task=_get(row, "task"),
        concurrency=n,
        density=_get(row, "density"),
        components=comps,
        # thr == 0.0 is a real outcome (a cell where every trial failed), not a
        # missing measurement; only n == 0 makes the division meaningless.
        per_agent_throughput=(round(thr / n, 4) if thr is not None and n else None),
        p95_latency_s=_get(row, "p95_trial_latency_s"),
        ipc=_get(row, "ipc"),
        throttled_pct=_get(row, "ctr_throttled_pct_periods_mean"),
        missing=missing,
        caveats=caveats,
    )


def signature_for_sweep(rows: Sequence[Mapping[str, Any]]) -> List[SignaturePoint]:
    """Aggregate replicates into one signature per concurrency, sorted by it.

    Replicates are averaged per component. Retention is computed against the PEAK
    per-agent cell, not the lowest-concurrency one: measured on this hardware the
    lowest-density cells can be startup-dominated and worse than the peak, so
    baselining on them overstates scaling.
    """
    by_n: Dict[int, List[SignaturePoint]] = {}
    for r in rows:
        p = signature_for_row(r)
        by_n.setdefault(p.concurrency, []).append(p)

    out: List[SignaturePoint] = []
    for n in sorted(by_n):
        group = by_n[n]
        agg = SignaturePoint(
            task=group[0].task,
            concurrency=n,
            density=group[0].density,
            replicates=len(group),
            components={
                k: _mean([g.components.get(k) for g in group]) for k in COMPONENTS
            },
            per_agent_throughput=_mean([g.per_agent_throughput for g in group]),
            p95_latency_s=_mean([g.p95_latency_s for g in group]),
            ipc=_mean([g.ipc for g in group]),
            throttled_pct=_mean([g.throttled_pct for g in group]),
        )
        agg.missing = [k for k, v in agg.components.items() if v is None]
        # Union of caveats, deduped, order preserved.
        seen_c: Dict[str, None] = {}
        for g in group:
            for c in g.caveats:
                seen_c.setdefault(c, None)
        agg.caveats = list(seen_c)
        if len(group) < 3:
            agg.caveats.append(
                f"only {len(group)} replicate(s) — no variance estimate, so this"
                " point has no confidence interval"
            )
        out.append(agg)

    peak = max(
        (p.per_agent_throughput for p in out if p.per_agent_throughput is not None),
        default=None,
    )
    if peak:
        for p in out:
            if p.per_agent_throughput is not None:
                p.retention = round(p.per_agent_throughput / peak, 4)
    return out


def _note_ramp(lines: List[str], usable: Sequence[SignaturePoint],
               peak: SignaturePoint) -> None:
    """Call out points BELOW the peak at lower concurrency.

    These are not degradation — they are the ramp region, where container startup
    and a near-idle box dominate a short trial. Measured: hbox's n=2 and n=4 sit
    at 0.50 and 0.18 retention *below* its n=8 peak. Saying nothing invites a
    reader to read that dip as a knee.
    """
    ramp = [p for p in usable
            if p.concurrency < peak.concurrency and (p.retention or 1) < 0.8]
    if ramp:
        ns = ", ".join(f"n={p.concurrency}" for p in ramp)
        lines.append(
            f"Below the peak ({ns}) per-agent throughput is also low. That is the "
            f"ramp region — startup-dominated on a near-idle box — not saturation. "
            f"Do not read it as a knee."
        )


def explain(points: Sequence[SignaturePoint]) -> List[str]:
    """Plain-language reading of what the spectrum says about the knee.

    Deliberately conservative: it names what the data supports and says
    'undetermined' otherwise, rather than asserting a bottleneck the components
    cannot distinguish.
    """
    lines: List[str] = []
    usable = [p for p in points if p.retention is not None]
    if not usable:
        return ["no point has a per-agent throughput; nothing to explain"]

    peak = max(usable, key=lambda p: p.retention or 0)
    pk_pa = f"{peak.per_agent_throughput:.2f}" if peak.per_agent_throughput else "?"
    lines.append(
        f"**Fastest per copy at n={peak.concurrency}** "
        f"({peak.concurrency} copies at once, quota demand "
        f"{peak.components.get('quota_demand')}): each finished "
        f"{pk_pa} tasks/min. Every number below is relative to this."
    )

    # Degradation can only be looked for PAST the peak. Scanning the whole sweep
    # reported hbox's n=2 as the "first >20% loss" even though its peak is at
    # n=8 — i.e. it named a point BEFORE the peak as the onset of decline, which
    # is incoherent. Points below the peak at lower concurrency are the ramp
    # region (startup-dominated), not degradation.
    after_peak = [p for p in usable if p.concurrency > peak.concurrency]
    degraded = [p for p in after_peak if (p.retention or 1) < 0.8]
    if not degraded:
        if after_peak:
            lines.append(
                f"**No saturation found.** Out to n={after_peak[-1].concurrency} "
                f"copies, none lost more than 20% of that speed — this workload "
                f"has headroom left on this hardware."
            )
        else:
            lines.append(
                f"**Cannot tell where it saturates.** The fastest point is also "
                f"the most agents tested (n={peak.concurrency}), so the ladder "
                f"needs extending before a limit can be found."
            )
        _note_ramp(lines, usable, peak)
        return lines

    first = degraded[0]
    lost = (1.0 - (first.retention or 0)) * 100.0
    lines.append(
        f"**Saturates at n={first.concurrency}** (quota demand "
        f"{first.components.get('quota_demand')}): each copy now runs "
        f"{lost:.0f}% slower than at its best. Adding agents past here costs "
        f"more per-agent speed than it buys in total work."
    )

    fill = first.components.get("quota_fill")
    ser = first.components.get("serialization")
    thr = first.throttled_pct
    if fill is not None and fill >= 0.7:
        lines.append(
            f"**Why: it ran out of CPU.** At that point the agents used "
            f"{fill * 100:.0f}% of the CPU they were promised — they are genuinely "
            f"compute-limited, so more cores (or faster ones) would help."
        )
    elif fill is not None:
        # No gap between the bands: anything below the CPU-saturation gate is
        # explained by how much of its wall time is off-CPU. An earlier version
        # gated on fill < 0.4 and left 0.4-0.7 unexplained, which silently
        # dropped the explanation for the one cell that mattered (fib at n=24
        # sits at fill 0.53).
        if ser is not None and ser >= 0.4:
            lines.append(
                f"**Why: NOT a shortage of CPU.** The agents used only "
                f"{fill * 100:.0f}% of the CPU they were promised, and "
                f"{ser * 100:.0f}% of each task's wall time was spent not "
                f"computing at all — waiting on I/O, or on its own sequential "
                f"steps. More cores would not fix this."
            )
        elif ser is not None:
            lines.append(
                f"**Why: undetermined.** The agents used {fill * 100:.0f}% of "
                f"their promised CPU and spent {ser * 100:.0f}% of wall time "
                f"off-CPU — neither compute-limited nor clearly stalled. The "
                f"measured components do not separate the cause."
            )
        else:
            lines.append(
                f"**Why: undetermined.** The agents used {fill * 100:.0f}% of "
                f"their promised CPU, but off-CPU time was not measured."
            )
    if thr is not None and thr >= 20.0:
        lines.append(
            f"**Bursty, not idle.** {thr:.0f}% of scheduling periods hit the "
            f"per-container CPU cap, so demand spikes above the ceiling even "
            f"though the average looks low."
        )
    elif thr is not None and thr < 5.0 and (fill or 1) < 0.4:
        lines.append(
            f"**Genuinely idle, not capped.** Only {thr:.1f}% of scheduling "
            f"periods hit the CPU cap — the agents were waiting, not throttled."
        )
    if first.missing:
        lines.append(
            "**Not measured here:** " + ", ".join(first.missing)
            + ". A limit in any of those cannot be ruled out."
        )
    _note_ramp(lines, usable, peak)
    return lines


__all__ = [
    "SignaturePoint", "COMPONENTS", "PROVENANCE",
    "signature_for_row", "signature_for_sweep", "explain",
]
