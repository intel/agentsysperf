#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Sample per-container cgroup v2 telemetry for the duration of a sweep cell.

Why this exists: everything else in the sweep is cpuset- or socket-scoped, so it
can say "the 16 cores were 74% busy" but not "each agent used X". This reads
`/sys/fs/cgroup/system.slice/docker-*.scope`, which gives exact per-container CPU
seconds, peak memory, disk bytes, and — the reason this was built — CFS throttling.

Throttling is the measurement that decides an open question. `overfull-hbox`
declares `cpus=2` and fills only ~21-26% of that allowance on average, so it is
not average-quota-saturated; but an average hides bursts. `nr_throttled` /
`throttled_usec` say whether it repeatedly slams into its 2-CPU ceiling (bursty
CPU demand) or genuinely idles (serialized on something else, e.g. its sequential
pdflatex loop). Those are different architectural stories and the existing
metrics cannot separate them.

Two properties that make this worth having over PMU-based measurement here:
  * No PMU involved, so it is immune to the SEP driver breaking per-process perf
    counters host-wide on this box.
  * No root needed — verified readable as an unprivileged user.

Containers are DISCOVERED EACH TICK, not once: harbor creates and destroys one
container per trial, so a sampler that enumerates once at startup would miss
every container of a multi-wave cell. Cost is one readdir plus a few small file
reads per container per tick (~0.02 ms per read, measured).

Usage:
    container_telemetry.py --out ctr.json [--interval 1.0] [--label-filter docker-]
    # runs until SIGINT/SIGTERM, then writes the rollup
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CGROUP_ROOT = Path("/sys/fs/cgroup/system.slice")


def _read_kv(path: Path) -> Dict[str, float]:
    """Parse a flat ``key value`` cgroup file (cpu.stat, memory.stat)."""
    out: Dict[str, float] = {}
    try:
        for line in path.read_text().split("\n"):
            f = line.split()
            if len(f) >= 2:
                try:
                    out[f[0]] = float(f[1])
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_cpu_max(path: Path) -> Optional[float]:
    """Return the quota in CPUs, or None for 'max' (unlimited).

    cpu.max is "<quota_usec> <period_usec>"; 200000/100000 = 2.0 CPUs. This is
    what harbor writes from task.toml [environment] cpus, so it is the ceiling one
    agent can consume regardless of how many cores the cpuset has.
    """
    try:
        q, p = path.read_text().split()
    except (OSError, ValueError):
        return None
    if q == "max":
        return None
    try:
        return float(q) / float(p)
    except (ValueError, ZeroDivisionError):
        return None


def _read_io(path: Path) -> Dict[str, float]:
    """Sum rbytes/wbytes across all devices in io.stat."""
    tot = {"rbytes": 0.0, "wbytes": 0.0, "rios": 0.0, "wios": 0.0}
    try:
        for line in path.read_text().split("\n"):
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                if k in tot:
                    try:
                        tot[k] += float(v)
                    except ValueError:
                        pass
    except OSError:
        pass
    return tot


def _read_pressure_full_avg10(path: Path) -> Optional[float]:
    """PSI 'full' avg10 — share of time ALL tasks in the cgroup were stalled.

    NOT co-tenant-immune: PSI rises when someone else steals your CPU, so it
    attributes the victim rather than the cause. Useful as a stall signal, not as
    proof that this container is the problem.
    """
    try:
        for line in path.read_text().split("\n"):
            if line.startswith("full"):
                for tok in line.split():
                    if tok.startswith("avg10="):
                        return float(tok.split("=")[1])
    except (OSError, ValueError):
        pass
    return None


def snapshot(only_cids: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """One reading of every docker scope currently present.

    ``only_cids`` restricts to containers we own. This box is SHARED — a bare
    scan picked up two other tenants' long-running containers (an autoclaw proxy
    and a hermes agent) alongside the trial container, and their cumulative
    counters would have been folded into the cell aggregates. Without the filter
    the numbers are silently wrong rather than merely noisy.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        scopes = [p for p in CGROUP_ROOT.iterdir()
                  if p.name.startswith("docker-") and p.name.endswith(".scope")]
    except OSError:
        return out
    for s in scopes:
        cid = s.name[len("docker-"):-len(".scope")][:12]
        if only_cids is not None and cid not in only_cids:
            continue
        cpu = _read_kv(s / "cpu.stat")
        io = _read_io(s / "io.stat")
        out[cid] = {
            "usage_usec": cpu.get("usage_usec"),
            "nr_periods": cpu.get("nr_periods"),
            "nr_throttled": cpu.get("nr_throttled"),
            "throttled_usec": cpu.get("throttled_usec"),
            "quota_cpus": _read_cpu_max(s / "cpu.max"),
            "memory_current": _read_int(s / "memory.current"),
            "memory_peak": _read_int(s / "memory.peak"),
            "rbytes": io["rbytes"],
            "wbytes": io["wbytes"],
            "cpu_pressure_full_avg10": _read_pressure_full_avg10(s / "cpu.pressure"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument(
        "--image-filter", default=None,
        help="Only sample containers whose image matches this substring "
             "(e.g. 'alexgshaw/'). Required on a shared box: without it, other "
             "tenants' containers are folded into the cell aggregates.",
    )
    ap.add_argument(
        "--baseline-cids", default="",
        help="Comma-separated short container ids to EXCLUDE (pre-existing "
             "containers captured before the cell started).",
    )
    args = ap.parse_args()

    excluded = {c.strip() for c in args.baseline_cids.split(",") if c.strip()}

    def owned() -> Optional[set]:
        """Short ids of containers we should sample, or None for 'all'."""
        if args.image_filter is None and not excluded:
            return None
        import subprocess
        import sys
        try:
            out = subprocess.run(
                ["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Image}}"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            print("WARNING: docker ps failed; refusing to sample unfiltered containers", file=sys.stderr)
            return set()
        keep = set()
        for line in out.splitlines():
            cid, _, image = line.partition("\t")
            short = cid[:12]
            if short in excluded:
                continue
            if args.image_filter and args.image_filter not in image:
                continue
            keep.add(short)
        return keep

    stop = {"now": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("now", True))

    # cid -> first and last seen snapshot. Counters are cumulative per container,
    # so the delta over its lifetime is what we want; a container that appears and
    # vanishes between ticks is simply missed (recorded via n_ticks).
    first: Dict[str, Dict[str, Any]] = {}
    last: Dict[str, Dict[str, Any]] = {}
    peak_mem: Dict[str, int] = {}
    ticks = 0
    t0 = time.monotonic()

    while not stop["now"]:
        # Re-resolve ownership each tick: harbor creates and destroys one
        # container per trial, so the owned set changes throughout a cell.
        snap = snapshot(owned())
        ticks += 1
        for cid, v in snap.items():
            first.setdefault(cid, v)
            last[cid] = v
            mp = v.get("memory_peak")
            if mp is not None:
                peak_mem[cid] = max(peak_mem.get(cid, 0), mp)
        time.sleep(args.interval)

    wall = time.monotonic() - t0
    containers: List[Dict[str, Any]] = []
    for cid, f in first.items():
        l = last[cid]

        def delta(k: str) -> Optional[float]:
            a, b = f.get(k), l.get(k)
            return (b - a) if (a is not None and b is not None) else None

        cpu_s = (delta("usage_usec") or 0.0) / 1e6
        thr_s = (delta("throttled_usec") or 0.0) / 1e6
        periods = delta("nr_periods") or 0.0
        throttled = delta("nr_throttled") or 0.0
        containers.append({
            "cid": cid,
            "quota_cpus": l.get("quota_cpus"),
            "cpu_seconds": round(cpu_s, 3),
            # Fraction of its granted quota the container actually used, over the
            # sampling window. 1.0 means it pinned its ceiling the whole time.
            "quota_fill": (
                round(cpu_s / (wall * l["quota_cpus"]), 4)
                if l.get("quota_cpus") and wall > 0 else None
            ),
            # THE throttling signal: what share of CFS periods hit the ceiling.
            # High with a low quota_fill == bursty demand against the cap, which
            # an average cannot show.
            "nr_periods": periods,
            "nr_throttled": throttled,
            "throttled_pct_periods": (
                round(100.0 * throttled / periods, 2) if periods else None
            ),
            "throttled_seconds": round(thr_s, 3),
            "memory_peak_mb": (
                round(peak_mem[cid] / 1048576.0, 1) if cid in peak_mem else None
            ),
            "disk_read_mb": round((delta("rbytes") or 0.0) / 1048576.0, 2),
            "disk_write_mb": round((delta("wbytes") or 0.0) / 1048576.0, 2),
            "cpu_pressure_full_avg10": l.get("cpu_pressure_full_avg10"),
        })

    def agg(key: str, fn) -> Optional[float]:
        vals = [c[key] for c in containers if c.get(key) is not None]
        return round(fn(vals), 3) if vals else None

    roll = {
        "n_containers_seen": len(containers),
        "n_ticks": ticks,
        "interval_s": args.interval,
        "window_s": round(wall, 2),
        # Cell-level aggregates. These are the ones the sweep rollup consumes.
        "ctr_quota_cpus": containers[0]["quota_cpus"] if containers else None,
        "ctr_cpu_seconds_total": agg("cpu_seconds", sum),
        "ctr_quota_fill_mean": agg("quota_fill", lambda v: sum(v) / len(v)),
        "ctr_throttled_pct_periods_mean": agg(
            "throttled_pct_periods", lambda v: sum(v) / len(v)),
        "ctr_throttled_pct_periods_max": agg("throttled_pct_periods", max),
        "ctr_throttled_seconds_total": agg("throttled_seconds", sum),
        "ctr_memory_peak_mb_max": agg("memory_peak_mb", max),
        "ctr_disk_write_mb_total": agg("disk_write_mb", sum),
        "ctr_disk_read_mb_total": agg("disk_read_mb", sum),
        "ctr_cpu_pressure_full_avg10_max": agg("cpu_pressure_full_avg10", max),
        "scope_note": (
            "Per-container cgroup v2, discovered each tick. Containers that start "
            "AND finish between two ticks are missed entirely; compare "
            "n_containers_seen against the cell's expected trial count before "
            "trusting the aggregates. No PMU involved, so unaffected by the SEP "
            "driver. cpu_pressure is not co-tenant-immune."
        ),
        "containers": containers,
    }
    json.dump(roll, args.out.open("w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
