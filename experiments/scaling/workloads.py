#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Workload implementations derived from TB2 synthetic task signatures.

Each workload faithfully reproduces the CPU signature of a real TB2 task type
(branch-heavy, FP-heavy, memory-BW-heavy, etc.) using the functions defined in
harness/scripts/synthetic_tasks.py. This replaces the previous generic
array-traversal workloads with microarchitecturally accurate profiles.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from .config import PhaseMix

# Load synthetic_tasks.py by file path (harness/ is not a Python package)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_SCRIPT = _REPO_ROOT / "harness" / "scripts" / "synthetic_tasks.py"


def _load_synthetic_module():
    """Import harness/scripts/synthetic_tasks.py and return the module."""
    if not _SYNTHETIC_SCRIPT.exists():
        raise FileNotFoundError(
            f"Synthetic workloads not found at {_SYNTHETIC_SCRIPT}. "
            f"Has the harness/ tree moved?"
        )
    spec = importlib.util.spec_from_file_location(
        "_density_synthetic_tasks", _SYNTHETIC_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {_SYNTHETIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_synthetic = _load_synthetic_module()


@dataclass
class TurnResult:
    """Result from executing one agent turn (reason + act)."""
    turn_idx: int
    reason_duration_s: float
    act_duration_s: float
    act_output: str


@dataclass
class WorkloadSpec:
    """Specification for a density experiment workload.

    reason_fn: callable from synthetic_tasks.py that exercises the target
               microarchitectural axis (called with duration_s per turn)
    reason_duration_s: how long to run the reason phase per turn
    commands: shell commands for the act phase (one per turn)
    setup_commands: one-time environment setup
    description: human-readable description of the CPU signature
    """
    reason_fn: Callable[[float], Dict[str, Any]]
    reason_duration_s: float
    commands: List[str]
    setup_commands: List[str]
    description: str


# Map PhaseMix profiles to real TB2 task signatures.
# Each profile uses the synthetic_tasks function that best represents
# the CPU behavior of that agent workload class.

WORKLOAD_SPECS: Dict[PhaseMix, WorkloadSpec] = {
    PhaseMix.COMPILE: WorkloadSpec(
        reason_fn=_synthetic.compile_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="Branch-heavy lexing/parsing (high branch rate, instruction-stream-bound)",
    ),
    PhaseMix.ML_TRAIN: WorkloadSpec(
        reason_fn=_synthetic.ml_train_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="Vectorized FP + memory streaming (high mem-BW, FP-heavy)",
    ),
    PhaseMix.LINALG: WorkloadSpec(
        reason_fn=_synthetic.linalg_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="Dense matmul (memory-BW-bound, AMX-candidate, large working set)",
    ),
    PhaseMix.COMPRESS: WorkloadSpec(
        reason_fn=_synthetic.compress_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="LZ-style compression (cache-friendly, high IPC, hash-chain lookups)",
    ),
    PhaseMix.RAYTRACE: WorkloadSpec(
        reason_fn=_synthetic.raytrace_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="FP-heavy with random memory access (path tracing)",
    ),
    PhaseMix.SAT: WorkloadSpec(
        reason_fn=_synthetic.sat_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="Branch-heavy + random memory (constraint solving, low IPC)",
    ),
    PhaseMix.INTERPRETER: WorkloadSpec(
        reason_fn=_synthetic.interpreter_like,
        reason_duration_s=0.5,
        commands=[
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo tick",
            "echo done",
        ],
        setup_commands=[],
        description="Unpredictable branches (bytecode dispatch, very low IPC)",
    ),
    PhaseMix.IO_HEAVY: WorkloadSpec(
        reason_fn=_synthetic.io_like,
        reason_duration_s=0.5,
        commands=[
            "find /usr -type f -name '*.conf' 2>/dev/null | head -50",
            "ls -laR /tmp 2>/dev/null | wc -l",
            "find . -type f | xargs wc -l 2>/dev/null | sort -n | tail -5",
            "grep -r 'root' /etc/passwd 2>/dev/null | head -10",
            "du -sh /usr/lib 2>/dev/null",
            "find /usr/share -name '*.txt' 2>/dev/null | head -20",
            "sort /etc/services 2>/dev/null | uniq -c | sort -rn | head -10",
            "cat /proc/meminfo 2>/dev/null | head -20",
        ],
        setup_commands=[],
        description="Sequential file reads + page faults (ETL-style I/O)",
    ),
    PhaseMix.MIXED: WorkloadSpec(
        reason_fn=_synthetic.compile_like,  # placeholder; per-agent assignment overrides
        reason_duration_s=0.5,
        commands=["echo tick"] * 8,
        setup_commands=[],
        description="Mixed fleet (each agent assigned a different workload type)",
    ),
}


def get_workload_spec(mix: PhaseMix) -> WorkloadSpec:
    """Get the workload spec for a phase mix.

    For MIXED, returns compile_like as default; the orchestrator assigns
    different mixes per agent via pinning._agent_mix().
    """
    return WORKLOAD_SPECS[mix]


def run_reason_phase(spec: WorkloadSpec, turn_idx: int) -> float:
    """Execute the Reason phase using the synthetic task function.

    Each turn runs the workload for spec.reason_duration_s, producing
    the characteristic CPU signature (branch pattern, cache behavior,
    FP throughput, memory access pattern) of that workload type.

    Returns the wall-clock duration in seconds.
    """
    import time
    t0 = time.time()
    spec.reason_fn(spec.reason_duration_s)
    return time.time() - t0
