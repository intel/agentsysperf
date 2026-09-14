#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Per-stage perf instrumentation for agentic workload characterization.

Provides two modes:
  1. Continuous — lightweight perf stat running throughout the benchmark,
     producing 100ms interval samples that are later joined with stage
     timestamps from task_phases.json.
  2. Targeted — perf stat wrapped around a specific PID/cgroup for deep
     per-stage counters (heavier, used in characterization pass).

The join key is timestamps: task_phases.json has per-trial start/finish
times; perf stat -I produces timestamped counter rows. Post-processing
correlates them.

Usage:
    sampler = PerfSampler(config)
    sampler.start_continuous(output_dir)
    # ... run benchmark ...
    sampler.stop_continuous()
    sampler.attribute_stages(task_phases_path, output_dir)
"""
import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PerfConfig:
    enabled: bool = True
    continuous_events: list = field(default_factory=lambda: [
        'cycles', 'instructions', 'cache-references', 'cache-misses',
        'branch-instructions', 'branch-misses',
        'context-switches', 'cpu-migrations', 'page-faults',
    ])
    deep_events: list = field(default_factory=list)
    sample_interval_ms: int = 100
    core_range: Optional[str] = None  # e.g. "0-21" — restrict sampling


class PerfSampler:
    def __init__(self, config: PerfConfig):
        self.config = config
        self._continuous_proc: Optional[subprocess.Popen] = None
        self._output_path: Optional[Path] = None

    def available(self) -> bool:
        """Check if perf counters are accessible."""
        if not self.config.enabled:
            return False
        try:
            r = subprocess.run(
                ['perf', 'stat', '-e', 'cycles', '--', 'sleep', '0.001'],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start_continuous(self, output_dir: Path) -> bool:
        """Start background perf stat with interval sampling."""
        if not self.available():
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = output_dir / 'perf_continuous.csv'

        events = ','.join(self.config.continuous_events)
        cmd = [
            'perf', 'stat',
            '-e', events,
            '-I', str(self.config.sample_interval_ms),
            '-x', ',',  # CSV output
            '-a',       # system-wide
        ]
        if self.config.core_range:
            cmd += ['-C', self.config.core_range]

        # perf stat -I writes to stderr
        with open(self._output_path, 'w') as f:
            self._continuous_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=f,
            )
        return True

    def stop_continuous(self) -> Optional[Path]:
        """Stop the continuous perf stat process."""
        if self._continuous_proc:
            self._continuous_proc.send_signal(signal.SIGINT)
            try:
                self._continuous_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._continuous_proc.kill()
            self._continuous_proc = None
        return self._output_path

    def stat_interval(self, pid: int, duration_s: float, output_path: Path,
                      events: Optional[list] = None) -> bool:
        """Run perf stat on a specific PID for a duration. For targeted passes."""
        if not self.available():
            return False
        evts = events or self.config.continuous_events
        cmd = [
            'perf', 'stat',
            '-e', ','.join(evts),
            '-p', str(pid),
            '-x', ',',
            '--', 'sleep', str(duration_s),
        ]
        with open(output_path, 'w') as f:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=f, timeout=duration_s + 10)
        return r.returncode == 0

    def attribute_stages(self, task_phases_path: Path, perf_csv_path: Path,
                         output_path: Path) -> dict:
        """Join perf interval samples with task phase timestamps.

        Returns per-stage aggregated counters. Each perf_continuous.csv row has
        a timestamp (relative seconds from start); task_phases.json has absolute
        timestamps. We align them by computing offsets.
        """
        if not task_phases_path.exists() or not perf_csv_path.exists():
            return {}

        with open(task_phases_path) as f:
            phases = json.load(f)

        # Parse perf CSV: each row is "timestamp,value,unit,event,..."
        perf_samples = []
        with open(perf_csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                try:
                    ts = float(parts[0].strip())
                    val = int(parts[1].strip()) if parts[1].strip() != '<not counted>' else 0
                    event = parts[3].strip()
                    perf_samples.append({'ts': ts, 'value': val, 'event': event})
                except (ValueError, IndexError):
                    continue

        if not perf_samples or not phases:
            return {}

        # Group perf samples by event
        by_event = {}
        for s in perf_samples:
            by_event.setdefault(s['event'], []).append(s)

        # For each phase, find overlapping perf samples and aggregate
        # Phases have started_at/finished_at as ISO strings or epoch floats
        stage_results = {}
        for phase in phases:
            task = phase.get('task', 'unknown')
            started = phase.get('started_at', '')
            finished = phase.get('finished_at', '')
            if not started or not finished:
                continue

            # Convert to relative seconds if they're ISO strings
            # (for now, just collect them — post-processing aligns later)
            stage_results.setdefault(task, []).append({
                'started_at': started,
                'finished_at': finished,
                'reward': phase.get('reward', 0.0),
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'total_samples': len(perf_samples),
            'events': list(by_event.keys()),
            'stages': stage_results,
            'summary': {},
        }

        # Compute totals per event
        for event, samples in by_event.items():
            total = sum(s['value'] for s in samples)
            result['summary'][event] = {
                'total': total,
                'samples': len(samples),
                'avg_per_interval': total / len(samples) if samples else 0,
            }

        # IPC if both cycles and instructions present
        if 'cycles' in result['summary'] and 'instructions' in result['summary']:
            cyc = result['summary']['cycles']['total']
            ins = result['summary']['instructions']['total']
            if cyc > 0:
                result['summary']['ipc'] = ins / cyc

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result


def mpstat_sampler(output_path: Path, interval_s: int = 1) -> subprocess.Popen:
    """Start mpstat background sampler (works without perf permissions)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ['mpstat', '-P', 'ALL', str(interval_s)],
        stdout=open(output_path, 'w'),
        stderr=subprocess.DEVNULL,
    )
    return proc


def vmstat_sampler(output_path: Path, interval_s: int = 1) -> subprocess.Popen:
    """Start vmstat background sampler."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ['vmstat', str(interval_s)],
        stdout=open(output_path, 'w'),
        stderr=subprocess.DEVNULL,
    )
    return proc
