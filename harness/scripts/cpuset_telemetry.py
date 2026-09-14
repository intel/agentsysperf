#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Sample CPU / runqueue / memory telemetry scoped to a CPUSET.

Why this exists rather than reusing `L1SystemMeasurement`: that probe reads the
host-aggregate `cpu` line of /proc/stat and host-wide `procs_running`
(`l1_system/probe.py:161-165`). On a 16-core cpuset inside a 288-core box that
reports ~5% CPU while those 16 cores are pegged — so every
`ScalingAnalyzer._classify_bottleneck` threshold (CPU_SATURATION_AVG=80,
CPU_SATURATION_PEAK=95) is unreachable and a saturated cell classifies as
`headroom_remaining`. Measured exactly that on the 16-core sweep.

This sampler sums only the `cpuN` lines in the cpuset, so `cpu_avg` is the
utilization of the cores under test. Both scopes are emitted: `*_cpuset` is what
the analyzer should threshold on, `*_host` is kept for provenance so the
difference is auditable rather than hidden.

Runqueue is host-wide (`procs_running` has no per-cpu form in /proc/stat) and is
labelled as such — a runnable-process count cannot be attributed to a cpuset
without walking every task's affinity, which is not worth the sampling cost.

Usage:
    cpuset_telemetry.py --cpus 0-15 --interval 1.0 --out telemetry.json
    # runs until SIGINT/SIGTERM, then writes the rollup
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def expand_cpus(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _read_stat(cpus: Sequence[int]) -> Optional[Dict[str, float]]:
    """Sum jiffies for the cpuset and for the host, plus ctxt/procs_running."""
    try:
        text = Path("/proc/stat").read_text()
    except OSError:
        return None
    want = {f"cpu{i}" for i in cpus}
    out = {
        "cpuset_total": 0.0, "cpuset_idle": 0.0, "cpuset_iowait": 0.0,
        "host_total": 0.0, "host_idle": 0.0, "host_iowait": 0.0,
        "ctxt": 0.0, "procs_running": 0.0,
    }
    seen = 0
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "cpu":                       # host aggregate
            v = [float(x) for x in f[1:]]
            out["host_total"] = sum(v)
            out["host_idle"] = v[3] + (v[4] if len(v) > 4 else 0.0)
            out["host_iowait"] = v[4] if len(v) > 4 else 0.0
        elif f[0] in want:
            v = [float(x) for x in f[1:]]
            out["cpuset_total"] += sum(v)
            # idle + iowait, matching the host-line convention above
            out["cpuset_idle"] += v[3] + (v[4] if len(v) > 4 else 0.0)
            out["cpuset_iowait"] += v[4] if len(v) > 4 else 0.0
            seen += 1
        elif f[0] == "ctxt":
            out["ctxt"] = float(f[1])
        elif f[0] == "procs_running":
            out["procs_running"] = float(f[1])
    if seen != len(cpus):
        # A cpu vanished (offline) — refuse rather than silently average over
        # fewer cores than requested.
        return None
    return out


def _read_mem_mb() -> Dict[str, Optional[float]]:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return {"avail": None, "used": None}
    kv = {}
    for line in text.splitlines():
        k, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            kv[k] = float(parts[0]) / 1024.0     # kB -> MB
    total, avail = kv.get("MemTotal"), kv.get("MemAvailable")
    return {"avail": avail, "used": (total - avail) if (total and avail) else None}


def _pct(vals: List[float], p: float) -> Optional[float]:
    """Linear-interpolating percentile (not the max-for-small-n kind)."""
    v = sorted(vals)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpus", required=True, help="cpuset spec, e.g. 0-15 or 0,2,4")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cpus = expand_cpus(args.cpus)
    if not cpus:
        print(f"ERROR: empty cpuset from {args.cpus!r}", file=sys.stderr)
        return 1

    stop = {"now": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("now", True))

    samples: List[Dict[str, float]] = []
    prev, prev_t = _read_stat(cpus), time.monotonic()
    t0 = prev_t
    while not stop["now"]:
        time.sleep(args.interval)
        now = time.monotonic()
        cur = _read_stat(cpus)
        if cur is None or prev is None:
            prev, prev_t = cur, now
            continue
        dt = max(now - prev_t, 1e-6)

        def util(scope: str) -> float:
            td = cur[f"{scope}_total"] - prev[f"{scope}_total"]
            idl = cur[f"{scope}_idle"] - prev[f"{scope}_idle"]
            return 100.0 * (1.0 - idl / td) if td > 0 else 0.0

        def iowait(scope: str) -> float:
            td = cur[f"{scope}_total"] - prev[f"{scope}_total"]
            iow = cur[f"{scope}_iowait"] - prev[f"{scope}_iowait"]
            return 100.0 * (iow / td) if td > 0 else 0.0

        mem = _read_mem_mb()
        samples.append({
            "rel_ts": now - t0,
            "cpu_pct_cpuset": util("cpuset"),
            "cpu_pct_host": util("host"),
            "iowait_pct_cpuset": iowait("cpuset"),
            "ctx_sw_per_s": (cur["ctxt"] - prev["ctxt"]) / dt,
            "procs_running_host": cur["procs_running"],
            "mem_avail_mb": mem["avail"],
        })
        prev, prev_t = cur, now

    if not samples:
        json.dump({"error": "no samples", "cpus": args.cpus}, args.out.open("w"), indent=2)
        return 1

    cs = [s["cpu_pct_cpuset"] for s in samples]
    hs = [s["cpu_pct_host"] for s in samples]
    roll = {
        "cpuset": args.cpus,
        "cores_in_cpuset": len(cpus),
        "n_samples": len(samples),
        "interval_s": args.interval,
        # These are what ScalingAnalyzer must threshold on.
        "cpu_avg": round(sum(cs) / len(cs), 3),
        "cpu_p95": round(_pct(cs, 95) or 0.0, 3),
        "cpu_peak": round(max(cs), 3),
        "iowait_pct_avg": round(
            sum(s["iowait_pct_cpuset"] for s in samples) / len(samples), 4),
        "ctx_sw_per_s": round(
            sum(s["ctx_sw_per_s"] for s in samples) / len(samples), 1),
        # Host-wide: procs_running has no per-cpu form. Named so nobody mistakes
        # it for a cpuset-scoped runqueue.
        "runqueue_max": max(s["procs_running_host"] for s in samples),
        "runqueue_is_host_wide": True,
        "mem_avail_mb_min": round(min(s["mem_avail_mb"] for s in samples), 1),
        # Provenance: how badly a host-scoped reading would have understated it.
        "cpu_avg_host": round(sum(hs) / len(hs), 3),
        "cpu_peak_host": round(max(hs), 3),
        "scope_note": (
            "cpu_* are scoped to the cpuset; cpu_*_host are host-aggregate. "
            "A host-scoped reading understates cpuset load by ~cores_total/"
            "cores_in_cpuset and makes saturation thresholds unreachable."
        ),
        "samples": samples,
    }
    json.dump(roll, args.out.open("w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
