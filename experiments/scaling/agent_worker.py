#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Single-agent subprocess worker.

This module is the entry point for each agent process. It:
1. Pins itself to the assigned cpuset
2. Waits on a barrier for synchronized start
3. Runs the TB2 workload loop with measurements
4. Returns results via a multiprocessing Queue
"""

from __future__ import annotations

import array
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .config import AgentConfig
from .workloads import get_workload_spec, run_reason_phase, WorkloadSpec


@dataclass
class AgentResult:
    """Results returned by a single agent worker."""
    agent_id: int
    phase_mix: str
    cpuset: List[int]
    turns_completed: int
    total_duration_s: float
    turn_durations: List[Dict[str, float]]
    perf_events: Dict[str, float]
    error: Optional[str] = None

    def throughput(self) -> float:
        """Turns per second."""
        if self.total_duration_s > 0:
            return self.turns_completed / self.total_duration_s
        return 0.0

    def mean_turn_ms(self) -> float:
        """Mean turn duration in milliseconds."""
        if not self.turn_durations:
            return 0.0
        return 1000 * sum(t["total_s"] for t in self.turn_durations) / len(self.turn_durations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "phase_mix": self.phase_mix,
            "cpuset": self.cpuset,
            "turns_completed": self.turns_completed,
            "total_duration_s": self.total_duration_s,
            "throughput_turns_per_s": self.throughput(),
            "mean_turn_ms": self.mean_turn_ms(),
            "turn_durations": self.turn_durations,
            "perf_events": self.perf_events,
            "error": self.error,
        }


def _pin_to_cpuset(cpuset: set) -> None:
    """Pin the current process to the given CPU set."""
    try:
        os.sched_setaffinity(0, cpuset)
    except (OSError, AttributeError) as e:
        print(f"  WARNING: Could not pin to cpuset {cpuset}: {e}", file=sys.stderr)


def _setup_environment(spec: WorkloadSpec, work_dir: Path) -> None:
    """Run setup commands in the agent's working directory."""
    import subprocess
    for cmd in spec.setup_commands:
        subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(work_dir),
            capture_output=True,
            timeout=30,
        )


def _exec_command(cmd: str, work_dir: Path, timeout: float = 30.0) -> str:
    """Execute a shell command and return output."""
    import subprocess
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout[:200]
    except (subprocess.TimeoutExpired, Exception):
        return ""


def _collect_perf_summary(pid: int, duration_s: float) -> Dict[str, float]:
    """Collect aggregated perf counters for this process (best-effort).

    Uses perf stat with a short snapshot. Returns empty dict if perf unavailable.
    """
    import subprocess

    events = "cycles,instructions,cache-references,cache-misses,LLC-load-misses,context-switches"
    cmd = ["perf", "stat", "-e", events, "-x", ",", "-p", str(pid), "--", "sleep", "0.01"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        counters = {}
        for line in result.stderr.splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    counters[parts[2].strip()] = float(parts[0])
                except (ValueError, IndexError):
                    pass
        return counters
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}


def agent_worker(
    config: AgentConfig,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue,
    work_dir_base: Path,
) -> None:
    """Entry point for a single agent worker process.

    Pins to cpuset, waits on barrier, runs workload, puts result in queue.
    """
    agent_id = config.agent_id
    work_dir = work_dir_base / f"agent_{agent_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Pin to assigned cores
    _pin_to_cpuset(config.cpuset)

    # Step 2: Get workload spec and setup
    spec = get_workload_spec(config.phase_mix)
    _setup_environment(spec, work_dir)

    # Step 3: Wait for all agents to be ready
    try:
        barrier.wait(timeout=30)
    except Exception as e:
        result_queue.put(AgentResult(
            agent_id=agent_id,
            phase_mix=config.phase_mix.value,
            cpuset=sorted(config.cpuset),
            turns_completed=0,
            total_duration_s=0,
            turn_durations=[],
            perf_events={},
            error=f"Barrier wait failed: {e}",
        ))
        return

    # Step 4: Run workload
    turn_durations = []
    pid = os.getpid()
    start_time = time.time()

    try:
        # Cycle the command list to honor the requested turn count. The workload
        # command list is short (~8 entries); slicing to config.turns would cap a
        # run at ~4s, far too short for EMON. Repeating reaches the intended
        # duration (e.g. 600 turns x ~0.5s reason ≈ 5 min) for HW collection.
        commands = [spec.commands[i % len(spec.commands)] for i in range(config.turns)]

        for turn_idx in range(len(commands)):
            turn_start = time.time()

            # Reason phase
            reason_start = time.time()
            run_reason_phase(spec, turn_idx)
            reason_dur = time.time() - reason_start

            # Act phase
            act_start = time.time()
            _exec_command(commands[turn_idx], work_dir)
            act_dur = time.time() - act_start

            total_turn = time.time() - turn_start
            turn_durations.append({
                "turn_idx": turn_idx,
                "reason_s": reason_dur,
                "act_s": act_dur,
                "total_s": total_turn,
            })

    except Exception as e:
        result_queue.put(AgentResult(
            agent_id=agent_id,
            phase_mix=config.phase_mix.value,
            cpuset=sorted(config.cpuset),
            turns_completed=len(turn_durations),
            total_duration_s=time.time() - start_time,
            turn_durations=turn_durations,
            perf_events={},
            error=str(e),
        ))
        return

    total_duration = time.time() - start_time

    # Step 5: Quick perf snapshot (best-effort)
    perf_events = _collect_perf_summary(pid, total_duration)

    # Step 6: Return results
    result_queue.put(AgentResult(
        agent_id=agent_id,
        phase_mix=config.phase_mix.value,
        cpuset=sorted(config.cpuset),
        turns_completed=len(turn_durations),
        total_duration_s=total_duration,
        turn_durations=turn_durations,
        perf_events=perf_events,
    ))
