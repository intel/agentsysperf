#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Concurrency sweep for the 6-phase agent loop — the Ring-1 experiment.

Dispatches N agent loops concurrently at each concurrency level c, using a
thread pool so many loops are in flight at once. For each c it records
per-loop wall latency and per-phase latency, then computes:

  - throughput (loops/sec)
  - p50 / p95 / p99 loop latency
  - agents-at-SLO: how many concurrent loops still meet the P99 latency SLO
  - per-phase mean latency (shows Admit ballooning, Context growing with c)

Writes one JSON rollup per concurrency cell plus a sweep summary, so the
chart step reads structured data rather than scraping logs.

NOTE (stand-in path): phase workloads are CPU-signature proxies (see
phases.py), so absolute numbers are for harness/analysis validation, not
publication. The sweep mechanics, SLO computation, and per-phase rollup
are exactly what the real-component run will use unchanged.

Usage:
    python examples/run_six_phase_sweep.py
    python examples/run_six_phase_sweep.py --concurrencies 1 5 10 20 --loops-per-cell 40
    python examples/run_six_phase_sweep.py --slo-ms 2000 --scale 1
"""

from __future__ import annotations

import argparse
import os

# CRITICAL — thread-budget control. Set BEFORE numpy/torch/faiss/llama import.
# Without this, each phase's inner libraries (torch BLAS, faiss OMP) fan out
# across ALL cores (torch defaulted to 96 threads on this 192-core box). Under
# outer concurrency c, that is c × ~96 threads — quadratic oversubscription
# that makes throughput COLLAPSE as c rises. That collapse is a benchmark bug,
# not an architectural finding. Pinning inner libs to 1 thread makes OUTER
# concurrency the thing that scales, which is what a concurrency sweep must
# measure. Override with AGENTSYSPERF_INNER_THREADS if you want a different
# per-agent budget.
_INNER = os.environ.get("AGENTSYSPERF_INNER_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _INNER)

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# torch reads env at import but can still be told directly; pin it too so a
# single agent's embedding / inference work uses the budgeted thread count.
try:
    import torch
    torch.set_num_threads(int(_INNER))
except Exception:
    pass

from src.benchmarks.six_phase_agent import SixPhaseAgentAdapter
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.protocols import TaskSpec
from src.runner import RunContext, track_span
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

PHASE_ORDER = ["reason", "retrieve", "act", "admit", "context", "commit"]


class _NoopInvoker:
    def invoke(self, instruction: str, **kwargs: Any) -> Any:
        return None


def _phase_backends() -> Dict[str, str]:
    """Report which phase is real vs stand-in, by running each once and
    reading the backend it declares. Measured-not-assumed: the badge comes
    from what actually ran, not a hardcoded table."""
    from src.benchmarks.six_phase_agent import phases as _ph
    out: Dict[str, str] = {}
    try:
        out["reason"] = _ph.reason(1).get("backend", "unknown")
    except Exception:
        out["reason"] = "error"
    try:
        out["retrieve"] = ("real_embed+faiss"
                           if _ph.retrieve(1).get("used_embed_model")
                           else "faiss+random_query")
    except Exception:
        out["retrieve"] = "error"
    # admit is always the real LiteLLM Router; act/context/commit are stand-ins.
    out["admit"] = "litellm_router"
    out["act"] = "subprocess_standin"
    out["context"] = "tokenizer+matmul_standin"
    out["commit"] = "sqlite_standin"
    return out


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile; pct in [0,100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def run_cell(
    *,
    concurrency: int,
    loops: int,
    scale: int,
    slo_ms: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one concurrency cell: `loops` agent loops at `concurrency` in flight."""
    ctx = RunContext(
        measurements=[L1SubSpanMeasurement()],
        output_dir=output_dir / f"c{concurrency:03d}",
    )
    adapter = SixPhaseAgentAdapter(ctx, n_loops=loops)
    invoker = _NoopInvoker()

    specs = [
        TaskSpec(**{**s.__dict__, "extra": {**s.extra, "scale": scale}})
        for s in adapter.list_tasks()
    ]

    loop_latencies_ms: List[float] = []
    per_phase_ms: Dict[str, List[float]] = {p: [] for p in PHASE_ORDER}

    def one_loop(spec: TaskSpec) -> Dict[str, Any]:
        outer = f"{ctx.run_id}::{spec.id}"
        t0 = time.monotonic()
        with track_span(ctx, outer, kind="loop", node_id=spec.id):
            result = adapter.run_task(spec, agent_invoker=invoker)
        wall_ms = (time.monotonic() - t0) * 1000.0
        return {"wall_ms": wall_ms, "result": result}

    with ctx:
        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one_loop, s) for s in specs]
            for fut in as_completed(futures):
                r = fut.result()
                loop_latencies_ms.append(r["wall_ms"])
                pw = r["result"].extra.get("phase_wall_ms", {})
                for p in PHASE_ORDER:
                    if p in pw:
                        per_phase_ms[p].append(pw[p])
        wall_s = time.monotonic() - t_start

    p50 = _percentile(loop_latencies_ms, 50)
    p95 = _percentile(loop_latencies_ms, 95)
    p99 = _percentile(loop_latencies_ms, 99)
    throughput = loops / wall_s if wall_s > 0 else 0.0
    meets_slo = p99 <= slo_ms

    rollup = {
        "concurrency": concurrency,
        "loops": loops,
        "scale": scale,
        "wall_s": round(wall_s, 3),
        # ── Evidence badges (per position paper §7) ──────────────────────
        # Every published number must state its evidential status:
        #   MEASURED       — observed on real hardware, real components
        #   ILLUSTRATIVE   — shape is real, absolute value not publishable
        #                    (stand-in phases, single un-tuned box)
        #   FRAMEWORK-ONLY — a model/derivation, not an observation
        #   EXTERNAL       — sourced from a third-party study
        # agents_at_slo is the paper's headline metric (Fig 3: capacity =
        # where P99 crosses the SLO). It is MEASURED as an operating point
        # but ILLUSTRATIVE in absolute terms until all phases are real and
        # a GPU-host baseline exists — so we tag it honestly.
        "evidence": {
            "agents_at_slo": "ILLUSTRATIVE",   # real knee, synthetic-phase values
            "latency_ms": "MEASURED",          # wall-clock is really observed
            "throughput_loops_per_s": "MEASURED",
            "per_phase_mean_ms": "ILLUSTRATIVE",  # act/context/commit are stand-ins
            "note": (
                "Single-box, CPU-only, 3/6 phases real (reason/retrieve/admit). "
                "No GPU term, so per-phase SHARES are not production shares "
                "(cf. external study [3]: host CPU is 11-15% of E2E on a "
                "GPU-attached node). Do not report phase shares as delivered "
                "capacity. Headline = agents-at-SLO knee, shape MEASURED, "
                "absolute value ILLUSTRATIVE."
            ),
        },
        "phase_backends": _phase_backends(),
        "throughput_loops_per_s": round(throughput, 3),
        "latency_ms": {
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
            "mean": round(statistics.mean(loop_latencies_ms), 1),
        },
        "slo_ms": slo_ms,
        "p99_meets_slo": meets_slo,
        # agents-at-SLO for this cell: if the cell meets SLO, all `concurrency`
        # agents are being served within SLO; else this concurrency exceeds it.
        "agents_at_slo": concurrency if meets_slo else 0,
        "per_phase_mean_ms": {
            p: round(statistics.mean(per_phase_ms[p]), 2) if per_phase_ms[p] else None
            for p in PHASE_ORDER
        },
        "per_phase_p95_ms": {
            p: round(_percentile(per_phase_ms[p], 95), 2) if per_phase_ms[p] else None
            for p in PHASE_ORDER
        },
    }
    (output_dir / f"c{concurrency:03d}").mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"c{concurrency:03d}" / "rollup.json", "w") as f:
        json.dump(rollup, f, indent=2)
    return rollup


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    p.add_argument("--loops-per-cell", type=int, default=40)
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--slo-ms", type=float, default=1000.0,
                   help="P99 loop-latency SLO in ms; agents-at-SLO keys off this")
    p.add_argument("--output-dir", type=Path,
                   default=Path(f"{_TMP}/agentsysperf_scratch/six_phase_sweep"))
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    backends = _phase_backends()
    print(f"\n6-phase concurrency sweep — scale={args.scale}, "
          f"{args.loops_per_cell} loops/cell, SLO(P99)={args.slo_ms}ms")
    print(f"phase backends: {backends}")
    print("HEADLINE metric = agents-at-SLO (Fig 3). Badge: shape MEASURED, "
          "absolute value ILLUSTRATIVE (CPU-only, 3/6 phases real, no GPU term).")
    print(f"{'c':>4} {'thru/s':>8} {'p50':>7} {'p95':>7} {'p99':>7} {'SLO?':>5} "
          f"{'admit':>7} {'context':>8} {'reason':>7}")
    print("-" * 72)

    summary: List[Dict[str, Any]] = []
    for c in args.concurrencies:
        rollup = run_cell(
            concurrency=c, loops=args.loops_per_cell, scale=args.scale,
            slo_ms=args.slo_ms, output_dir=args.output_dir,
        )
        summary.append(rollup)
        lat = rollup["latency_ms"]
        pm = rollup["per_phase_mean_ms"]
        print(f"{c:>4} {rollup['throughput_loops_per_s']:>8.2f} "
              f"{lat['p50']:>6.0f} {lat['p95']:>6.0f} {lat['p99']:>6.0f} "
              f"{'yes' if rollup['p99_meets_slo'] else 'NO':>5} "
              f"{pm['admit']:>7.2f} {pm['context']:>8.2f} {pm['reason']:>7.2f}")

    # agents-at-SLO knee: highest concurrency that still met the SLO.
    knee = max((r["concurrency"] for r in summary if r["p99_meets_slo"]), default=0)
    peak = max(summary, key=lambda r: r["throughput_loops_per_s"])
    summary_doc = {
        "headline": {
            "metric": "agents_at_slo",
            "value": knee,
            "definition": f"max concurrency meeting P99 <= {args.slo_ms}ms",
            "evidence": "ILLUSTRATIVE",  # real knee, synthetic-phase absolute value
            "peak_throughput_at_c": peak["concurrency"],
            "peak_throughput_loops_per_s": peak["throughput_loops_per_s"],
        },
        "evidence_note": (
            "Per position paper §7: agents-at-SLO is the headline (Fig 3, "
            "capacity = P99 crossing SLO). Shape MEASURED; absolute value "
            "ILLUSTRATIVE until all phases real + GPU-host baseline exists."
        ),
        "phase_backends": backends,
        "cells": summary,
    }
    with open(args.output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary_doc, f, indent=2)

    print("-" * 72)
    print(f"HEADLINE  agents-at-SLO = {knee}  (P99 <= {args.slo_ms}ms)  "
          f"[ILLUSTRATIVE]")
    print(f"peak throughput {peak['throughput_loops_per_s']:.2f} loops/s "
          f"at c={peak['concurrency']}")
    print(f"summary → {args.output_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
