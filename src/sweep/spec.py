#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""SweepSpec — the parameters of one concurrency-scaling sweep.

A sweep characterizes how a Xeon box behaves as agent concurrency rises. The
x-axis is **density = concurrency / vcpu_basis** ("agents per vCPU"), so a knee
found on EMR is directly comparable to one found on a smaller box — the whole
point of the colleague's normalization. On EMR (256 logical / 128 physical
cores) we default the basis to physical cores and sweep densities into the
high tail, because EMR sustains far more concurrent agents than the 8–16 vCPU
AWS instances the original study used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


# Default Terminal-Bench 2 task list for sweeps. Replay only hits for tasks
# present in the user's fixture, so the --tasks flag (or this default) must
# match the recorded fixture or the sweep aborts under strict-miss.
# Lives here rather than in a driver script because both the CLI
# (`agentsysperf sweep run`) and examples/run_scaling_sweep_tb2.py need it.
DEFAULT_TB2_TASKS = (
    "terminal-bench/reshard-c4-data",
    "terminal-bench/make-doom-for-mips",
    "terminal-bench/constraints-scheduling",
    "terminal-bench/train-fasttext",
    "terminal-bench/video-processing",
    "terminal-bench/schemelike-metacircular-eval",
    "terminal-bench/build-cython-ext",
    "terminal-bench/largest-eigenval",
    "terminal-bench/path-tracing",
    "terminal-bench/write-compressor",
)


@dataclass
class SweepSpec:
    """One concurrency sweep: densities × replicates over a task set."""

    # The density operating points to sweep. concurrency = round(density × basis).
    densities: Sequence[float] = field(
        default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    )
    replicates: int = 1
    attempts: int = 1                     # Harbor -k: passes through the task set
    tasks: Sequence[str] = field(default_factory=list)
    benchmark: str = "tb2"

    # Density normalization basis.
    vcpu_basis: Optional[int] = None      # None → resolved from platform at runtime
    vcpu_basis_kind: str = "physical_cores"  # or "logical_cpus"

    # LLM data plane.
    llm_mode: str = "replay"              # "replay" | "off" | "record"
    fixture: Optional[Path] = None

    # Dataset resolution. When set, Harbor loads tasks from this local directory
    # (--path) instead of fetching the dataset from the registry (-d). Required
    # on hosts without egress to the Harbor registry (raw.githubusercontent.com).
    dataset_path: Optional[Path] = None

    # NUMA policy. MVP runs unpinned (closest to the single-NUMA AWS baseline)
    # and just RECORDS the effect; pinning is a future swept axis.
    numa_policy: str = "unpinned"         # "unpinned" | "socket_pinned" | "interleaved"

    # Per-cell knobs.
    agent_timeout_multiplier: float = 2.0
    proxy_port: int = 4001
    sample_interval_s: float = 1.0

    emon: bool = False                   # collect EMON EDP during each cell

    output_dir: Path = Path(f"{_TMP}/agentsysperf_sweep")

    def concurrency_for(self, density: float) -> int:
        """Agents to run in parallel for a density point (>=1)."""
        basis = self.vcpu_basis or 1
        return max(1, round(density * basis))

    def cells(self) -> List[tuple]:
        """Enumerate (density, concurrency, replicate) cells in run order."""
        out = []
        for d in self.densities:
            c = self.concurrency_for(d)
            for r in range(self.replicates):
                out.append((d, c, r))
        return out


__all__ = ["SweepSpec", "DEFAULT_TB2_TASKS"]
