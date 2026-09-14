#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Density-scaling EMON collection driver — one workload mix, deadlock-safe.

Runs a synthetic agent-density sweep for a SINGLE workload mix, collecting EMON
EDP telemetry per density and post-processing it to core-filtered TMA CSVs.
This is the driver that produced the clean 8-density `compile` dataset.

Why this exists (vs. run_experiment.py):
  - Deadlock-safe: agents are daemon procs and the result queue is drained by a
    background thread, so a slow/killed worker cannot wedge the parent on a
    broken queue pipe (a failure mode hit with the naive join+drain).
  - Prompt EMON stop: EMON is stopped immediately after the compute window so
    straggler agents can't pad the .dat with an idle tail that dilutes TMA.
  - Core-filtered pyEDP: post-processing excludes idle cores (whose 48-bit PMU
    counters overflow under long system-wide collection), so there are zero
    "excessively large counts" exclusions.
  - Separate pyEDP interpreter: the repo venv lacks polars/pyarrow; pyEDP needs
    its own env (see --pyedp-python / PYEDP_PYTHON).

Setup (one-time) for the pyEDP interpreter:
    python3 -m venv --system-site-packages /tmp/pyedp_venv
    PYTHONNOUSERSITE=1 /tmp/pyedp_venv/bin/pip install polars pyarrow
  (needs numpy<2 + pandas from system site-packages; verified with numpy 1.26.)

Examples:
    # Single mix, full density sweep, with EMON:
    PYTHONPATH=. python3 -m experiments.scaling.density_emon_runner \
        --mix compile --densities 1 2 4 8 12 16 24 32

    # One density, no EMON (throughput/latency only, fast):
    PYTHONPATH=. python3 -m experiments.scaling.density_emon_runner \
        --mix ml_train --densities 16 --no-emon --turns 60
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .agent_worker import agent_worker
from .config import ExperimentConfig, ORCHESTRATOR_CORES, PhaseMix, Placement
from .pinning import active_cores, assign_cpusets, format_core_ranges, set_mixed_rotation
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

# SEP / pyEDP locations (override via env if the install moves).
SEP_DIR = Path(os.environ.get("SEP_DIR", "/opt/intel/sep"))
EMON_BIN = SEP_DIR / "bin64" / "emon"
# Metric database must match the host microarchitecture — TMA formulas differ
# per platform, so the wrong XML yields plausible-looking but wrong metrics.
# `emon -v` prints the right name on the "EMON Database" line; set EMON_DB to
# it (e.g. EMON_DB=graniterapids_server).
#
# Note that on some platforms the base `<platform>_server.xml` may define few
# or no TMA metrics — point EMON_DB at whichever metric database on the host
# actually defines the events you need.
EMON_DB = os.environ.get("EMON_DB", "graniterapids_server")
METRICS_XML = Path(os.environ.get(
    "EMON_METRICS_XML", SEP_DIR / "config" / "edp" / f"{EMON_DB}.xml"))
CHART_FMT = Path(os.environ.get(
    "EMON_CHART_FMT", SEP_DIR / "config" / "edp" / f"chart_format_{EMON_DB}.txt"))
# pyEDP needs polars/pyarrow + numpy<2; the repo venv can't run it. Default to a
# dedicated interpreter (see module docstring for one-time setup).
PYEDP_PYTHON = os.environ.get("PYEDP_PYTHON", f"{_TMP}/pyedp_venv/bin/python3")


def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _emon_available() -> bool:
    return EMON_BIN.exists()


def run_density(mix: PhaseMix, density: int, cores_per_agent: int, turns: int,
                out_base: Path, use_emon: bool, oversubscribe: bool = False,
                placement: Placement = Placement.SPREAD) -> dict:
    """Run one (mix, density) cell. Returns a metrics dict; writes emon.dat +
    throughput.json into the cell dir, and (if EMON) core-filtered pyEDP CSVs."""
    cfg = ExperimentConfig(density=density, placement=placement,
                           phase_mix=mix, cores_per_agent=cores_per_agent, turns=turns,
                           oversubscribe=oversubscribe)
    cell = out_base / f"d{density}_{placement.value}_{mix.value}"
    cell.mkdir(parents=True, exist_ok=True)
    dat = cell / "emon.dat"
    cfilter = format_core_ranges(active_cores(cfg))
    print(f"\n{'='*64}\n{mix.value} | d={density} | {placement.value} | "
          f"{cores_per_agent}c/agent | {turns} turns | filter {cfilter}", flush=True)

    try:
        os.sched_setaffinity(0, ORCHESTRATOR_CORES)
    except (OSError, AttributeError):
        pass

    agents = assign_cpusets(cfg)

    # Persist the agent->core->workload mapping. assign_cpusets() is deterministic,
    # but recording it means the mix layout is archived with the run instead of
    # only reproducible from the config (needed to interpret per-core EMON).
    mapping = [{"agent_id": a.agent_id, "cores": sorted(a.cpuset),
                "workload": a.phase_mix.value} for a in agents]
    (cell / "mapping.json").write_text(json.dumps(mapping, indent=2))
    with (cell / "mapping.csv").open("w") as fh:
        fh.write("agent_id,core,workload\n")
        for a in agents:
            for c in sorted(a.cpuset):
                fh.write(f"{a.agent_id},{c},{a.phase_mix.value}\n")

    emon_proc = None
    if use_emon and _emon_available():
        emon_proc = subprocess.Popen([str(EMON_BIN), "-collect-edp", "-f", str(dat)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)  # warmup
    elif use_emon:
        print("  WARNING: EMON requested but binary not found — running without it", flush=True)

    barrier = multiprocessing.Barrier(density + 1, timeout=60)
    q = multiprocessing.Queue()
    wd = cell / "workdirs"
    wd.mkdir(exist_ok=True)

    # Continuous drain thread — prevents queue-pipe deadlock if a worker dies mid-put.
    results, stop_drain = [], threading.Event()

    def _drain():
        while True:
            try:
                results.append(q.get(timeout=0.5))
            except Exception:
                # After stop is requested, keep draining until the queue is empty.
                if stop_drain.is_set():
                    break

    dt = threading.Thread(target=_drain, daemon=True)
    dt.start()

    procs = []
    for ac in agents:
        p = multiprocessing.Process(target=agent_worker, args=(ac, barrier, q, wd),
                                    name=f"a{ac.agent_id}")
        p.daemon = True
        p.start()
        procs.append(p)

    t0 = time.time()
    try:
        barrier.wait(timeout=60)
    except Exception as e:
        print(f"  barrier: {e}", flush=True)

    for p in procs:
        p.join(timeout=max(300, turns * 2))  # generous cap; then treat as straggler
        if p.is_alive():
            print(f"  terminating straggler {p.name}", flush=True)
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
    wall = time.time() - t0

    time.sleep(1.0)  # cooldown
    if emon_proc:
        subprocess.run([str(EMON_BIN), "-stop"], capture_output=True, timeout=30)
    stop_drain.set()
    dt.join(timeout=3)

    ok = [r for r in results if getattr(r, "error", None) is None]
    per_agent = [r.throughput() for r in ok]
    turn_ms = [t["total_s"] * 1000 for r in ok for t in (r.turn_durations or [])]
    rec = {
        "mix": mix.value, "density": density, "cores_per_agent": cores_per_agent,
        "placement": placement.value,
        "turns": turns, "agents_ok": len(ok), "wall_s": round(wall, 1),
        "mean_tps": round(sum(per_agent) / len(per_agent), 3) if per_agent else 0.0,
        "agg_tps": round(sum(per_agent), 3),
        "p50_ms": round(_pct(turn_ms, 50), 1), "p95_ms": round(_pct(turn_ms, 95), 1),
        "p99_ms": round(_pct(turn_ms, 99), 1),
        "emon_dat": str(dat) if emon_proc else None,
        "core_filter": cfilter,
    }
    (cell / "throughput.json").write_text(json.dumps(rec, indent=2))
    print(f"  ok={len(ok)}/{density} wall={wall:.0f}s mean={rec['mean_tps']} "
          f"agg={rec['agg_tps']} p50={rec['p50_ms']} p95={rec['p95_ms']} "
          f"p99={rec['p99_ms']}", flush=True)

    if emon_proc and dat.exists():
        _post_process(dat, cfilter)
    return rec


def _post_process(dat: Path, cfilter: str) -> None:
    """Core-filtered pyEDP post-process via the dedicated interpreter."""
    if not Path(PYEDP_PYTHON).exists():
        print(f"  pyEDP SKIPPED — interpreter {PYEDP_PYTHON} not found "
              f"(set PYEDP_PYTHON or see module docstring)", flush=True)
        return
    out = dat.parent / "emon_filtered.csv"
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{SEP_DIR}/config/edp:{SEP_DIR}/config/edp/pyedp"
    cmd = [PYEDP_PYTHON, "-m", "pyedp.mpp", "--socket-view",
           "--core-filter", cfilter, "-p", "1",
           "-i", str(dat), "-o", str(out),
           "-m", str(METRICS_XML), "-f", str(CHART_FMT)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800, env=env)
    except subprocess.SubprocessError as e:
        print(f"  pyEDP error: {e}", flush=True)
        return
    # The metric CSVs are what we consume; pyEDP writes them before the
    # optional chart step, which can fail on a metric the platform never
    # collected (e.g. "metric_core c6 residency %" on Clearwater Forest).
    # Treat the CSVs as the success criterion but never swallow the reason:
    # a silent "FAIL 0s" cost a debugging cycle here.
    ok = any((dat.parent / f"emon_filtered.csv_{_view}_view_details.csv").exists()
             for _view in ("system", "socket"))
    print(f"  pyEDP {'OK' if ok else 'FAIL'} {time.time()-t0:.0f}s", flush=True)
    if proc.returncode != 0:
        _tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        _msg = _tail[-1] if _tail else f"exit {proc.returncode}"
        print(f"  pyEDP {'warning (CSVs written)' if ok else 'error'}: {_msg}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", required=True,
                    choices=[m.value for m in PhaseMix],
                    help="Synthetic workload mix to run.")
    ap.add_argument("--densities", type=int, nargs="+",
                    default=[1, 2, 4, 8, 12, 16, 24, 32],
                    help="Agent densities to sweep.")
    ap.add_argument("--cores-per-agent", type=int, default=2,
                    help="Cores per agent (2 fits the full sweep in the 92-core pool; "
                         "4 exceeds it beyond d=16).")
    ap.add_argument("--turns", type=int, default=600,
                    help="Turns/agent. ~600 gives a ~5-min EMON window; "
                         "~60 is enough for throughput/latency without EMON.")
    ap.add_argument("--placement", default="spread",
                    choices=[p.value for p in Placement],
                    help="Agent-to-core placement strategy (default: spread).")
    ap.add_argument("--no-emon", action="store_true", help="Skip EMON (throughput only).")
    ap.add_argument("--oversubscribe", action="store_true",
                    help="Allow >pool agents to share cores (cpusets wrap the pool). "
                         "Needed to drive past full-pool occupancy into the "
                         "scheduling-contention (oversubscription) knee.")
    ap.add_argument("--mix-set", nargs="+", default=None,
                    choices=[m.value for m in PhaseMix if m.value != "mixed"],
                    help="With --mix mixed, restrict the round-robin to this subset of "
                         "workloads (e.g. --mix mixed --mix-set io_heavy raytrace interpreter linalg). "
                         "Each agent gets a different task type, cycling through the set.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output base dir (default: /tmp/agentsysperf_density_<mix>).")
    args = ap.parse_args()

    mix = PhaseMix(args.mix)
    if args.mix_set:
        if mix != PhaseMix.MIXED:
            print("  NOTE: --mix-set only applies with --mix mixed; ignoring.", flush=True)
        else:
            set_mixed_rotation([PhaseMix(m) for m in args.mix_set])
            print(f"  MIXED rotation restricted to: {args.mix_set}", flush=True)
    out_base = args.output_dir or Path(f"{_TMP}/agentsysperf_density_{mix.value}")
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Density sweep: mix={mix.value} densities={args.densities} "
          f"cpa={args.cores_per_agent} turns={args.turns} emon={not args.no_emon}\n"
          f"Output: {out_base}", flush=True)

    all_recs = []
    for d in args.densities:
        try:
            all_recs.append(run_density(mix, d, args.cores_per_agent, args.turns,
                                        out_base, use_emon=not args.no_emon,
                                        oversubscribe=args.oversubscribe,
                                        placement=Placement(args.placement)))
        except ValueError as e:  # e.g. density doesn't fit the core pool
            print(f"  SKIP d={d}: {e}", flush=True)
    (out_base / "summary.json").write_text(json.dumps(all_recs, indent=2))
    print(f"\nDONE — {len(all_recs)} densities → {out_base}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
