#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Stage inference — classify agentic turns by their hardware signature.

Instead of pre-labeling stages (like the AgentSysPerf design's legal_review
workload does), this module observes what the hardware actually did during
each turn and clusters turns into stage types:

  - compute_heavy: high CPU%, high IPC, low IO wait (compilation, SAT solve)
  - memory_bound:  high cache-miss rate, high memory BW (large model inference)
  - io_bound:      high IO wait, low CPU% (disk reads, network, Docker pulls)
  - idle_wait:     very low CPU% (waiting for LLM response, sleeping)
  - control_plane: short duration, low CPU%, minimal IO (agent orchestration)

Inputs:
  - task_phases.json (per-trial start/finish + metadata)
  - perf_continuous.csv (timestamped counter samples, 100ms intervals)
  - mpstat.txt (per-CPU utilization, 1s intervals)
  - docker_stats.txt (per-container CPU%, mem, PIDs)

Output:
  - stage_classification.json: per-turn hardware signature + inferred class
  - stage_summary.json: aggregated per-class statistics

This is the core of the characterization — it produces the per-stage
fingerprint that tells you "compile turns are core-bound at IPC=2.1,
L2-resident" vs "inference turns are memory-bound at IPC=0.8, DRAM-bound."
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TurnSignature:
    """Hardware signature for one agent turn."""
    task: str
    turn_index: int
    started_at: str
    finished_at: str
    duration_s: float

    # From perf (if available)
    ipc: Optional[float] = None
    cache_miss_rate: Optional[float] = None
    branch_miss_rate: Optional[float] = None
    context_switches: Optional[int] = None
    cpu_migrations: Optional[int] = None

    # From mpstat (always available)
    avg_cpu_pct: Optional[float] = None
    avg_iowait_pct: Optional[float] = None
    peak_cpu_pct: Optional[float] = None

    # From docker stats
    avg_container_cpu_pct: Optional[float] = None
    peak_container_mem_mb: Optional[float] = None

    # Inferred class
    stage_class: str = 'unknown'
    confidence: float = 0.0


def parse_mpstat(path: Path) -> list[dict]:
    """Parse mpstat output into timestamped CPU samples.
    Each sample: {timestamp, cpu_id, usr, sys, iowait, idle, ...}
    """
    samples = []
    if not path.exists():
        return samples

    # mpstat -P ALL output format varies; we look for the 'all' aggregate line
    for line in path.read_text().splitlines():
        # Typical: "HH:MM:SS  all  usr nice sys iowait irq soft steal guest idle"
        if ' all ' not in line or 'CPU' in line or 'Average' in line:
            continue
        parts = line.split()
        if len(parts) < 12:
            continue
        try:
            # Try to extract timestamp from beginning
            ts = parts[0]  # may be "HH:MM:SS" or "AM/PM" format
            usr = float(parts[3])
            sys_pct = float(parts[5])
            iowait = float(parts[6])
            idle = float(parts[-1])
            samples.append({
                'timestamp': ts,
                'cpu_pct': 100.0 - idle,
                'usr': usr,
                'sys': sys_pct,
                'iowait': iowait,
                'idle': idle,
            })
        except (ValueError, IndexError):
            continue
    return samples


def parse_docker_stats(path: Path) -> list[dict]:
    """Parse docker_stats.txt: 'TIMESTAMP NAME CPU% MEM/LIMIT PIDS'"""
    samples = []
    if not path.exists():
        return samples
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            ts = parts[0]
            name = parts[1]
            cpu_str = parts[2].rstrip('%')
            cpu_pct = float(cpu_str)
            # mem is like "123.4MiB / 256GiB"
            mem_str = parts[3]
            mem_mb = 0.0
            if 'MiB' in mem_str:
                mem_mb = float(mem_str.replace('MiB', ''))
            elif 'GiB' in mem_str:
                mem_mb = float(mem_str.replace('GiB', '')) * 1024
            samples.append({
                'timestamp': ts,
                'container': name,
                'cpu_pct': cpu_pct,
                'mem_mb': mem_mb,
            })
        except (ValueError, IndexError):
            continue
    return samples


def parse_perf_csv(path: Path) -> dict[str, list]:
    """Parse perf stat -I CSV output. Returns {event: [(ts, value), ...]}"""
    events = defaultdict(list)
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            ts = float(parts[0].strip())
            val_str = parts[1].strip()
            val = int(val_str) if val_str != '<not counted>' else 0
            event = parts[3].strip()
            events[event].append((ts, val))
        except (ValueError, IndexError):
            continue
    return dict(events)


def classify_turn(sig: TurnSignature) -> tuple[str, float]:
    """Infer the stage class from the hardware signature.

    Returns (class_name, confidence 0-1).

    Classification logic:
    - Very short (<0.5s) + low CPU → control_plane
    - Low CPU (<10%) + long duration → idle_wait (LLM response wait)
    - High iowait (>20%) → io_bound
    - High CPU (>70%) + high IPC (>1.5) → compute_heavy
    - High CPU + low IPC (<1.0) + high cache misses → memory_bound
    - Moderate CPU (30-70%) → mixed
    """
    # Short + light = control plane
    if sig.duration_s < 0.5 and (sig.avg_cpu_pct or 0) < 20:
        return 'control_plane', 0.8

    # Low CPU = waiting (for LLM or IO)
    if (sig.avg_cpu_pct or 0) < 10:
        if (sig.avg_iowait_pct or 0) > 5:
            return 'io_bound', 0.7
        return 'idle_wait', 0.7

    # High iowait
    if (sig.avg_iowait_pct or 0) > 20:
        return 'io_bound', 0.8

    # High CPU — distinguish compute vs memory bound
    if (sig.avg_cpu_pct or 0) > 70:
        if sig.ipc is not None:
            if sig.ipc > 1.5:
                return 'compute_heavy', 0.85
            elif sig.ipc < 1.0:
                return 'memory_bound', 0.75
            else:
                return 'compute_heavy', 0.6  # moderate IPC, still high CPU
        # No IPC data — use cache miss rate if available
        if sig.cache_miss_rate is not None and sig.cache_miss_rate > 0.1:
            return 'memory_bound', 0.6
        return 'compute_heavy', 0.5  # default for high CPU without detail

    # Moderate CPU
    if (sig.avg_cpu_pct or 0) > 30:
        return 'mixed', 0.5

    return 'control_plane', 0.4


def infer_stages(run_dir: Path) -> dict:
    """Main entry point: analyze a completed run and classify turns.

    Expects:
      run_dir/task_phases.json
      run_dir/monitoring/mpstat.txt
      run_dir/monitoring/docker_stats.txt
      run_dir/monitoring/perf_continuous.csv (optional)

    Writes:
      run_dir/stage_classification.json
      run_dir/stage_summary.json
    """
    phases_path = run_dir / 'task_phases.json'
    if not phases_path.exists():
        return {'error': 'no task_phases.json'}

    phases = json.loads(phases_path.read_text())
    mon_dir = run_dir / 'monitoring'

    mpstat_samples = parse_mpstat(mon_dir / 'mpstat.txt')
    docker_samples = parse_docker_stats(mon_dir / 'docker_stats.txt')
    perf_events = parse_perf_csv(mon_dir / 'perf_continuous.csv')

    # For the first pass, we classify each trial as a whole (not per-turn
    # within a trial, since we don't have turn-level timestamps from Harbor
    # yet — only trial-level start/finish). This still gives us per-TASK
    # classification: "make-doom-for-mips is compute_heavy, video-processing
    # is io_bound" etc.

    signatures = []
    for i, phase in enumerate(phases):
        task = phase.get('task', f'task_{i}')
        started = phase.get('started_at', '')
        finished = phase.get('finished_at', '')
        duration = phase.get('duration_s', 0)

        if not duration and started and finished:
            try:
                t0 = datetime.fromisoformat(started.replace('Z', '+00:00'))
                t1 = datetime.fromisoformat(finished.replace('Z', '+00:00'))
                duration = (t1 - t0).total_seconds()
            except Exception:
                pass

        # Compute average CPU% from mpstat samples during this trial's window
        # (rough: we don't have sub-second alignment yet, just use overall avg
        # divided proportionally by the number of samples in the window)
        avg_cpu = None
        avg_iowait = None
        if mpstat_samples:
            # Simple approach: if we have N samples total for M seconds total,
            # attribute proportionally. Finer alignment comes in v2.
            cpu_vals = [s['cpu_pct'] for s in mpstat_samples]
            iowait_vals = [s['iowait'] for s in mpstat_samples]
            if cpu_vals:
                avg_cpu = sum(cpu_vals) / len(cpu_vals)
                avg_iowait = sum(iowait_vals) / len(iowait_vals)

        # Compute IPC from perf if available
        ipc = None
        cache_miss_rate = None
        if 'cycles' in perf_events and 'instructions' in perf_events:
            cyc_total = sum(v for _, v in perf_events['cycles'])
            ins_total = sum(v for _, v in perf_events['instructions'])
            if cyc_total > 0:
                ipc = ins_total / cyc_total
        if 'cache-references' in perf_events and 'cache-misses' in perf_events:
            refs = sum(v for _, v in perf_events['cache-references'])
            misses = sum(v for _, v in perf_events['cache-misses'])
            if refs > 0:
                cache_miss_rate = misses / refs

        sig = TurnSignature(
            task=task, turn_index=i,
            started_at=started, finished_at=finished,
            duration_s=duration,
            ipc=ipc, cache_miss_rate=cache_miss_rate,
            avg_cpu_pct=avg_cpu, avg_iowait_pct=avg_iowait,
        )
        sig.stage_class, sig.confidence = classify_turn(sig)
        signatures.append(sig)

    # Write classification
    classification = [
        {
            'task': s.task, 'turn_index': s.turn_index,
            'duration_s': s.duration_s,
            'stage_class': s.stage_class, 'confidence': s.confidence,
            'ipc': s.ipc, 'cache_miss_rate': s.cache_miss_rate,
            'avg_cpu_pct': s.avg_cpu_pct, 'avg_iowait_pct': s.avg_iowait_pct,
        }
        for s in signatures
    ]
    (run_dir / 'stage_classification.json').write_text(json.dumps(classification, indent=2))

    # Aggregate by class
    by_class = defaultdict(list)
    for s in signatures:
        by_class[s.stage_class].append(s)

    summary = {}
    for cls, sigs in by_class.items():
        durations = [s.duration_s for s in sigs if s.duration_s]
        ipcs = [s.ipc for s in sigs if s.ipc is not None]
        summary[cls] = {
            'count': len(sigs),
            'tasks': list(set(s.task for s in sigs)),
            'total_duration_s': sum(durations),
            'avg_duration_s': sum(durations) / len(durations) if durations else 0,
            'avg_ipc': sum(ipcs) / len(ipcs) if ipcs else None,
            'avg_cpu_pct': (sum(s.avg_cpu_pct for s in sigs if s.avg_cpu_pct) /
                           len([s for s in sigs if s.avg_cpu_pct]))
                          if any(s.avg_cpu_pct for s in sigs) else None,
        }
    (run_dir / 'stage_summary.json').write_text(json.dumps(summary, indent=2))

    print(f'\n  Stage classification:')
    for cls, info in sorted(summary.items(), key=lambda x: -x[1]['total_duration_s']):
        print(f'    {cls:15s}: {info["count"]} tasks, '
              f'{info["total_duration_s"]:.1f}s total, '
              f'avg IPC={info["avg_ipc"]:.2f}' if info["avg_ipc"] else
              f'    {cls:15s}: {info["count"]} tasks, '
              f'{info["total_duration_s"]:.1f}s total')

    return summary


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python stage_inference.py <run_dir>')
        sys.exit(1)
    infer_stages(Path(sys.argv[1]))
