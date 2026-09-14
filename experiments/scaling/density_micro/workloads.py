#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Workload implementations for each phase mix profile."""

from __future__ import annotations

import array
import time
from dataclasses import dataclass
from typing import List

from .config import PhaseMix


@dataclass
class TurnResult:
    """Result from executing one agent turn (reason + act)."""
    turn_idx: int
    reason_duration_s: float
    act_duration_s: float
    act_output: str


# ─── Workload Definitions ─────────────────────────────────────────────────────
# Each workload defines:
#   - array_size: number of doubles for the Reason phase array traversal
#   - passes: number of full passes over the array per turn
#   - stride: access stride (64 = cache-line stride)
#   - commands: shell commands for the Act phase
#   - setup_commands: one-time environment setup

@dataclass
class WorkloadSpec:
    array_base_size: int
    array_growth_per_turn: int
    passes: int
    stride: int
    commands: List[str]
    setup_commands: List[str]


WORKLOAD_SPECS = {
    PhaseMix.COMPUTE_HEAVY: WorkloadSpec(
        array_base_size=1_000_000,   # 8MB (1M doubles x 8 bytes)
        array_growth_per_turn=200_000,
        passes=16,
        stride=64,
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
    ),
    PhaseMix.IO_HEAVY: WorkloadSpec(
        array_base_size=8_000,       # 64KB — fits L1
        array_growth_per_turn=0,
        passes=2,
        stride=64,
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
        setup_commands=[
            "dd if=/dev/urandom of=data.bin bs=1M count=10 2>/dev/null",
            "mkdir -p tree/{a,b,c}/{d,e,f}",
            "for f in tree/{a,b,c}/{d,e,f}/file.txt; do head -c 1024 /dev/urandom > $f; done",
        ],
    ),
    PhaseMix.BALANCED: WorkloadSpec(
        array_base_size=250_000,     # 2MB — fills L2, spills to L3
        array_growth_per_turn=50_000,
        passes=8,
        stride=64,
        commands=[
            "ls -la",
            "find . -type f -name '*.bin'",
            "du -b *.bin 2>/dev/null || echo none",
            "find . -type f -exec wc -c {} + 2>/dev/null | sort -n | tail -3",
            "find . -type f -exec wc -c {} + 2>/dev/null | sort -rn | head -1",
            "wc -c data.bin 2>/dev/null || echo 0",
            "awk 'BEGIN{s=0; for(i=1;i<=10000;i++) s+=i; print s}'",
            "echo done",
        ],
        setup_commands=[
            "dd if=/dev/zero of=small.bin bs=1024 count=10 2>/dev/null",
            "dd if=/dev/zero of=medium.bin bs=1024 count=100 2>/dev/null",
            "dd if=/dev/zero of=data.bin bs=1024 count=2000 2>/dev/null",
        ],
    ),
}

# MIXED uses the other three, assigned per-agent in pinning.py


def get_workload_spec(mix: PhaseMix) -> WorkloadSpec:
    """Get the workload spec for a phase mix (MIXED resolves per-agent)."""
    if mix == PhaseMix.MIXED:
        return WORKLOAD_SPECS[PhaseMix.BALANCED]
    return WORKLOAD_SPECS[mix]


def run_reason_phase(spec: WorkloadSpec, turn_idx: int) -> float:
    """Execute the Reason phase: array traversal simulating inference.

    Returns the wall-clock duration in seconds.
    """
    size = spec.array_base_size + turn_idx * spec.array_growth_per_turn
    arr = array.array('d', range(size))

    t0 = time.time()
    total = 0.0
    for _ in range(spec.passes):
        for i in range(0, len(arr), spec.stride):
            total += arr[i]
    # Prevent dead-code elimination
    if total == float('inf'):
        pass
    duration = time.time() - t0
    return duration
