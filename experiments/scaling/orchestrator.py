#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Experiment orchestrator: spawns agents, manages EMON, collects results."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_worker import AgentResult, agent_worker
from .config import ExperimentConfig, ORCHESTRATOR_CORES
from .pinning import assign_cpusets, active_cores, format_core_ranges


# SEP / pyEDP locations. A SEP install directory carries its version and a build
# hash in its name, so any literal path is valid on exactly one host: override via
# $SEP_DIR. /opt/intel/sep is the version-independent symlink a normal install
# leaves behind. EMON_DB must match the host microarchitecture -- TMA formulas
# differ per platform, so the wrong XML yields plausible but wrong metrics
# (`emon -v` prints the right name on its "EMON Database" line).
SEP_DIR = Path(os.environ.get("SEP_DIR", "/opt/intel/sep"))
EMON_BIN = SEP_DIR / "bin64" / "emon"
PYEDP_DIR = SEP_DIR / "config" / "edp"
EMON_DB = os.environ.get("EMON_DB", "graniterapids_server")
METRICS_XML = PYEDP_DIR / f"{EMON_DB}.xml"
CHART_FORMAT = PYEDP_DIR / f"chart_format_{EMON_DB}.txt"


@dataclass
class ExperimentResult:
    """Results from one experiment configuration."""
    config: ExperimentConfig
    agent_results: List[AgentResult]
    emon_csv: Optional[str] = None
    emon_metrics_count: int = 0
    total_duration_s: float = 0.0
    error: Optional[str] = None

    def mean_throughput(self) -> float:
        """Mean per-agent throughput (turns/s)."""
        rates = [r.throughput() for r in self.agent_results if r.error is None]
        return sum(rates) / len(rates) if rates else 0.0

    def aggregate_throughput(self) -> float:
        """Total system throughput (all agents combined turns/s)."""
        return sum(r.throughput() for r in self.agent_results if r.error is None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "density": self.config.density,
                "placement": self.config.placement.value,
                "phase_mix": self.config.phase_mix.value,
                "cores_per_agent": self.config.cores_per_agent,
            },
            "mean_throughput_turns_per_s": self.mean_throughput(),
            "aggregate_throughput_turns_per_s": self.aggregate_throughput(),
            "total_duration_s": self.total_duration_s,
            "emon_csv": self.emon_csv,
            "emon_metrics_count": self.emon_metrics_count,
            "agents": [r.to_dict() for r in self.agent_results],
            "error": self.error,
        }


def _emon_available() -> bool:
    """Check if EMON is available."""
    if not EMON_BIN.exists():
        return False
    result = subprocess.run(
        [str(EMON_BIN), "-v"],
        capture_output=True, text=True, timeout=10,
    )
    combined = result.stdout + result.stderr
    return "SEP Driver Version:" in combined and "Error" not in combined.split("SEP Driver Version:")[1].split("\n")[0]


def _start_emon(output_file: Path) -> Optional[subprocess.Popen]:
    """Start EMON EDP collection."""
    if not _emon_available():
        return None
    cmd = [str(EMON_BIN), "-collect-edp", "-f", str(output_file)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.5)
    return proc


def _stop_emon(proc: Optional[subprocess.Popen]) -> None:
    """Stop EMON collection."""
    if proc is None:
        return
    subprocess.run([str(EMON_BIN), "-stop"], capture_output=True, timeout=10)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _post_process_emon(dat_file: Path, core_filter: Optional[str] = None) -> Optional[Path]:
    """Run pyEDP post-processing. Returns CSV path or None.

    ``core_filter`` (pyEDP range syntax, e.g. "0-31,92-95") restricts processing
    to the cores this experiment actually used. This skips idle cores, whose
    48-bit PMU counters overflow during system-wide collection and emit
    'excluded due to excessively large counts' warnings.
    """
    if not dat_file.exists():
        return None

    csv_prefix = dat_file.with_suffix(".csv")
    env = os.environ.copy()
    env["PATH"] = f"{SEP_DIR / 'bin64'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{PYEDP_DIR}:{PYEDP_DIR / 'pyedp'}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "pyedp.mpp",
        "--socket-view",
        "-i", str(dat_file),
        "-o", str(csv_prefix),
        "-m", str(METRICS_XML),
        "-f", str(CHART_FORMAT),
    ]
    if core_filter:
        cmd += ["--core-filter", core_filter]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Try summary first, fall back to details
    for suffix in ("_system_view_summary.csv", "_system_view_details.csv",
                   "_socket_view_summary.csv", "_socket_view_details.csv"):
        candidate = Path(f"{csv_prefix}{suffix}")
        if candidate.exists():
            return candidate
    return None


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    verbose: bool = True,
) -> ExperimentResult:
    """Run a single experiment configuration.

    Spawns N agent workers, optionally collects EMON, waits for results.
    """
    run_dir = output_dir / f"d{config.density}_{config.placement.value}_{config.phase_mix.value}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n  ┌─ Density={config.density} | {config.placement.value} | {config.phase_mix.value}")

    # Pin orchestrator to reserved cores
    try:
        os.sched_setaffinity(0, ORCHESTRATOR_CORES)
    except (OSError, AttributeError):
        pass

    # Assign cpusets to agents
    try:
        agent_configs = assign_cpusets(config)
    except ValueError as e:
        if verbose:
            print(f"  │  SKIP: {e}")
        return ExperimentResult(config=config, agent_results=[], error=str(e))

    if verbose:
        print(f"  │  Agents: {len(agent_configs)}, cores/agent: {config.cores_per_agent}")

    # Start EMON
    emon_dat = run_dir / "emon.dat"
    emon_proc = _start_emon(emon_dat)
    if emon_proc and verbose:
        print(f"  │  EMON: collecting (PID {emon_proc.pid})")
    elif verbose:
        print(f"  │  EMON: unavailable (L3 perf only)")

    # EMON warmup
    time.sleep(config.emon_warmup_s)

    # Create barrier and result queue
    barrier = multiprocessing.Barrier(config.density + 1, timeout=30)
    result_queue = multiprocessing.Queue()
    work_dir_base = run_dir / "workdirs"
    work_dir_base.mkdir(parents=True, exist_ok=True)

    # Spawn agent workers
    processes = []
    for ac in agent_configs:
        p = multiprocessing.Process(
            target=agent_worker,
            args=(ac, barrier, result_queue, work_dir_base),
            name=f"agent-{ac.agent_id}",
        )
        p.start()
        processes.append(p)

    # Release barrier (orchestrator is participant N+1)
    run_start = time.time()
    try:
        barrier.wait(timeout=30)
    except Exception as e:
        if verbose:
            print(f"  │  ERROR: Barrier failed: {e}")
        _stop_emon(emon_proc)
        return ExperimentResult(config=config, agent_results=[], error=f"Barrier: {e}")

    if verbose:
        print(f"  │  All {config.density} agents started simultaneously")

    # Wait for all agents to finish
    for p in processes:
        p.join(timeout=config.agent_timeout_s)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    total_duration = time.time() - run_start

    # EMON cooldown + stop
    time.sleep(config.emon_cooldown_s)
    _stop_emon(emon_proc)

    # Collect results from queue
    agent_results = []
    while not result_queue.empty():
        try:
            agent_results.append(result_queue.get_nowait())
        except Exception:
            break

    # Sort by agent_id
    agent_results.sort(key=lambda r: r.agent_id)

    # Post-process EMON
    emon_csv = None
    emon_metrics = 0
    if emon_proc and emon_dat.exists():
        if verbose:
            print(f"  │  EMON: post-processing ({emon_dat.stat().st_size // 1024} KB)...")
        core_filter = format_core_ranges(active_cores(config))
        csv_path = _post_process_emon(emon_dat, core_filter=core_filter)
        if csv_path:
            emon_csv = str(csv_path)
            with open(csv_path) as fh:
                emon_metrics = sum(1 for _ in fh) - 1
            if verbose:
                print(f"  │  EMON: {emon_metrics} metrics → {csv_path.name}")

    # Print summary
    if verbose:
        successful = [r for r in agent_results if r.error is None]
        if successful:
            mean_tp = sum(r.throughput() for r in successful) / len(successful)
            mean_turn = sum(r.mean_turn_ms() for r in successful) / len(successful)
            print(f"  │  Results: {len(successful)}/{config.density} agents OK")
            print(f"  │  Mean throughput: {mean_tp:.2f} turns/s, mean turn: {mean_turn:.0f}ms")
        else:
            print(f"  │  Results: NO successful agents")
        print(f"  └─ Duration: {total_duration:.1f}s")

    # Save results JSON
    result = ExperimentResult(
        config=config,
        agent_results=agent_results,
        emon_csv=emon_csv,
        emon_metrics_count=emon_metrics,
        total_duration_s=total_duration,
    )

    results_file = run_dir / "results.json"
    results_file.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    return result
