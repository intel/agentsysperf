#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Run a concurrency-scaling sweep on EMR (TB2 + Terminus-2, deterministic replay).

Finds the saturation knee — how many agents per vCPU the Xeon sustains before
throughput floors — and classifies the bottleneck (CPU / scheduler / memory /
I/O). Results persist to SQLite (sweeps + sweep_points + a scaling verdict) and
render in the live dashboard's Scaling tab.

Examples
--------
  # Dry run — exercise the whole pipeline with synthetic cells (no Harbor):
  python examples/run_scaling_sweep_tb2.py --dry-run

  # Real sweep with deterministic replay from a recorded fixture:
  python examples/run_scaling_sweep_tb2.py \
      --fixture /path/to/your/fixture.jsonl \
      --densities 0.25 0.5 1.0 1.5 2.0 3.0

  # Real sweep with EMON hardware telemetry per cell:
  python examples/run_scaling_sweep_tb2.py --emon \
      --fixture /path/to/your/fixture.jsonl \
      --densities 0.25 0.5 1.0 1.5 2.0 3.0

NOTE: a real replay sweep requires a fixture recorded with the SAME agent you
replay (Harbor/Terminus-2 here). A fixture recorded by a different agent will
MISS and abort under strict-miss. See src/replay/fixture.py.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.sweep import DEFAULT_TB2_TASKS, HarborSweep, SweepSpec
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

# Shared with `agentsysperf sweep run` so the two cannot drift. To profile a
# different task (e.g. mteb-retrieve), record a new fixture that includes it —
# see src/replay/fixture.py.
DEFAULT_TASKS = list(DEFAULT_TB2_TASKS)


def main() -> int:
    p = argparse.ArgumentParser(description="AgentSysPerf concurrency-scaling sweep")
    p.add_argument("--densities", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--fixture", type=Path, default=None)
    p.add_argument("--dataset-path", type=Path, default=None,
                   help="Local TB2 dataset dir (uses harbor --path instead of registry -d). "
                        "Required on hosts without egress to the Harbor registry.")
    p.add_argument("--agent-timeout-multiplier", type=float, default=2.0,
                   help="Multiplier on each task's timeout_sec for the agent phase. "
                        "Lower it (e.g. 0.1) so replay-miss stalls fail fast instead of "
                        "idling for the full task timeout and polluting EMON.")
    p.add_argument("--llm-mode", choices=["replay", "off", "record"], default="replay")
    p.add_argument("--basis", choices=["physical_cores", "logical_cpus"],
                   default="physical_cores")
    p.add_argument("--numa", choices=["unpinned", "socket_pinned", "interleaved"],
                   default="unpinned")
    p.add_argument("--output-dir", type=Path, default=Path(f"{_TMP}/agentsysperf_sweep"))
    p.add_argument("--emon", action="store_true",
                   help="Collect EMON EDP during each sweep cell for TMA analysis.")
    p.add_argument("--dry-run", action="store_true",
                   help="Synthetic cells (no Harbor) to smoke-test the pipeline.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.llm_mode == "off":
        fixture = None
    else:
        fixture = args.fixture

    spec = SweepSpec(
        densities=args.densities,
        replicates=args.replicates,
        attempts=args.attempts,
        tasks=args.tasks,
        vcpu_basis_kind=args.basis,
        llm_mode="off" if args.dry_run else args.llm_mode,
        fixture=fixture,
        dataset_path=args.dataset_path,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
        numa_policy=args.numa,
        emon=args.emon,
        output_dir=args.output_dir,
    )
    sweep = HarborSweep(spec)
    sweep_id = sweep.run(dry_run=args.dry_run)
    print(f"\nSweep complete: {sweep_id}")
    print(f"DB: {sweep.store.db_path}")
    print("View in the dashboard: streamlit run live_dashboard.py --server.port 8502")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
