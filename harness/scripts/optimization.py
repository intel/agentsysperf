#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Optimization profile applier — configures NUMA, hugepages, core isolation
before a benchmark run. Requires sudo for most operations.

Each profile is a set of system-level settings applied before the run and
(where possible) reverted after. The benchmark reports which profile was
active, and the harness verifies engagement via perf counters post-run.

Usage:
    from optimization import apply_profile, revert_profile, check_prerequisites

    issues = check_prerequisites()
    if issues:
        print("Cannot apply profiles:", issues)

    apply_profile(profile_cfg)
    # ... run benchmark ...
    revert_profile(profile_cfg)
"""
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProfileConfig:
    name: str
    numa: str = "unpinned"         # "unpinned" | "node0" | "node1" | ...
    hugepages: str = "4k"          # "4k" | "2M" | "1G"
    core_isolation: bool = False
    core_range: Optional[str] = None   # e.g. "0-21"
    description: str = ""


def _run(cmd: str, check: bool = True) -> tuple[int, str]:
    """Run a command from a string, return (returncode, output).

    Split into argv rather than handed to a shell. Both callers are of the form
    `sudo sh -c "echo N > /sys/..."` — the INNER `sh -c` is what performs the
    redirect as root and is genuinely required, but the outer shell that
    shell=True added on top of it was pure attack surface. shlex.split keeps the
    quoted `echo N > path` intact as a single argv element, so the behaviour is
    unchanged.
    """
    r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if check and r.returncode != 0:
        return r.returncode, r.stderr.strip()
    return r.returncode, r.stdout.strip()


def _has_sudo() -> bool:
    """Check if we can run sudo without a password prompt."""
    r = subprocess.run(['sudo', '-n', 'true'], capture_output=True)
    return r.returncode == 0


def check_prerequisites() -> list[str]:
    """Check what's available for optimization. Returns list of issues."""
    issues = []
    if not _has_sudo():
        issues.append("sudo without password not available (needed for NUMA/hugepages/isolation)")
    if not Path('/usr/bin/numactl').exists():
        issues.append("numactl not installed (apt install numactl)")
    if not Path('/sys/kernel/mm/hugepages').exists():
        issues.append("hugepages sysfs not found")
    return issues


def get_numa_topology() -> dict:
    """Return NUMA node → CPU list mapping."""
    nodes = {}
    numa_dir = Path('/sys/devices/system/node')
    if not numa_dir.exists():
        return nodes
    for node_dir in sorted(numa_dir.glob('node*')):
        cpulist_path = node_dir / 'cpulist'
        if cpulist_path.exists():
            nodes[node_dir.name] = cpulist_path.read_text().strip()
    return nodes


def apply_profile(cfg: ProfileConfig) -> dict:
    """Apply the optimization profile. Returns a report of what was done.

    NOTE: Most operations require sudo. If sudo is unavailable, the profile
    is recorded as 'requested but not applied' and the run proceeds — the
    results just reflect the system's default configuration.
    """
    report = {'profile': cfg.name, 'applied': [], 'skipped': [], 'errors': []}

    if cfg.name == 'base':
        report['applied'].append('base profile — no changes applied')
        return report

    has_sudo = _has_sudo()

    # NUMA policy
    if cfg.numa != 'unpinned':
        if has_sudo:
            # For NUMA-local allocation, we use numactl in the runner command
            # rather than system-wide settings. Record the intent here.
            report['applied'].append(f'numa={cfg.numa} (applied via numactl in runner)')
        else:
            report['skipped'].append(f'numa={cfg.numa} (no sudo)')

    # Hugepages
    if cfg.hugepages == '2M':
        hp_path = '/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages'
        if has_sudo:
            # Allocate 4GB worth of 2M pages (2048 pages)
            rc, out = _run(f'sudo sh -c "echo 2048 > {hp_path}"', check=False)
            if rc == 0:
                report['applied'].append('hugepages=2M (2048 pages allocated)')
            else:
                report['errors'].append(f'hugepages allocation failed: {out}')
        else:
            report['skipped'].append('hugepages=2M (no sudo)')
    elif cfg.hugepages == '1G':
        # 1G pages typically need boot-time allocation
        report['skipped'].append('hugepages=1G (requires boot-time reservation)')

    # Core isolation
    if cfg.core_isolation and cfg.core_range:
        if has_sudo:
            # Core isolation via cgroup cpuset for the benchmark process.
            # Full isolcpus requires reboot; we use taskset in the runner instead.
            report['applied'].append(
                f'core_isolation={cfg.core_range} (applied via taskset in runner)')
        else:
            report['skipped'].append(f'core_isolation={cfg.core_range} (no sudo)')

    return report


def revert_profile(cfg: ProfileConfig) -> dict:
    """Revert applied optimizations."""
    report = {'profile': cfg.name, 'reverted': []}

    if cfg.name == 'base':
        return report

    has_sudo = _has_sudo()

    # Revert hugepages
    if cfg.hugepages == '2M' and has_sudo:
        hp_path = '/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages'
        _run(f'sudo sh -c "echo 0 > {hp_path}"', check=False)
        report['reverted'].append('hugepages reset to 0')

    return report


def numactl_prefix(cfg: ProfileConfig) -> str:
    """Return the numactl command prefix for the given profile.
    Used by the runner to wrap the harbor command."""
    if cfg.numa == 'unpinned':
        return ''
    # Extract node number from "node0", "node1", etc.
    node = cfg.numa.replace('node', '')
    return f'numactl --cpunodebind={node} --membind={node}'


def taskset_prefix(cfg: ProfileConfig) -> str:
    """Return taskset prefix for core isolation."""
    if not cfg.core_isolation or not cfg.core_range:
        return ''
    return f'taskset -c {cfg.core_range}'


def get_run_prefix(cfg: ProfileConfig) -> str:
    """Combined prefix for running the benchmark with the profile applied."""
    parts = []
    numa = numactl_prefix(cfg)
    if numa:
        parts.append(numa)
    task = taskset_prefix(cfg)
    if task:
        parts.append(task)
    return ' '.join(parts)
