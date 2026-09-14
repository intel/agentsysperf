#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Archetype workloads for analyzer threshold calibration.

Each archetype is a synthetic workload with a **known bottleneck** label.
Running these archetypes with L1+L3 measurements produces a labeled
dataset for training analyzer thresholds.

Archetypes cover the key bottleneck classes for agentic AI workloads:
- memory_bound: high LLC miss rate, saturated memory bandwidth
- core_bound: high IPC, execution port saturation
- io_bound: blocked on disk/network, low CPU%
- frontend_starved: low IPC due to instruction fetch bottlenecks
- balanced: no dominant bottleneck
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Archetype:
    """A workload archetype with ground-truth bottleneck label."""

    name: str
    true_bottleneck: str  # ground truth label
    run_fn: Callable[[], None]
    description: str
    duration_estimate_s: float = 5.0


def _memory_bound_stream() -> None:
    """Memory bandwidth saturator (STREAM-like).

    Allocates large arrays and performs vector addition in a loop.
    Expected: high LLC miss rate, memory-bound classification.
    """
    import numpy as np
    # 200 MB arrays * 3 = 600 MB total, exceeds L3 cache
    n = 25_000_000
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)

    # Stream triad: c = a + b (memory bandwidth bound)
    for _ in range(20):
        c[:] = a + b
        a[:] = c  # write-back to keep memory busy


def _memory_bound_random_access() -> None:
    """Random memory access pattern (pointer chasing).

    Expected: high LLC miss rate, low IPC, memory-bound.
    """
    import numpy as np
    n = 10_000_000
    indices = np.arange(n, dtype=np.int64)
    np.random.shuffle(indices)
    data = np.zeros(n, dtype=np.int64)

    # Pointer chasing (cache-hostile)
    idx = 0
    for _ in range(n):
        idx = indices[idx % n]
        data[idx] += 1


def _core_bound_matmul() -> None:
    """Matrix multiplication (compute-bound).

    Expected: high IPC, core-bound classification.
    On AMX-enabled Xeon with bf16, this should saturate tile units.
    """
    import numpy as np
    n = 1024
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    # Matmul loop (FP32, core-bound)
    for _ in range(10):
        c = a @ b
        a = c  # keep result hot


def _core_bound_fibonacci() -> None:
    """Recursive computation (core-bound, no memory pressure).

    Expected: high IPC, low cache miss, core-bound.
    """
    def fib(n: int) -> int:
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    # Compute fib(30) multiple times (pure CPU)
    for _ in range(100):
        _ = fib(28)


def _io_bound_disk() -> None:
    """Disk I/O bound workload.

    Expected: low CPU%, blocked on I/O, io_bound classification.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "testfile.dat"

        # Write 100 MB in chunks (I/O bound)
        chunk = b"x" * (1024 * 1024)  # 1 MB
        with open(fpath, "wb") as f:
            for _ in range(100):
                f.write(chunk)
                f.flush()  # force disk write

        # Read back (I/O bound)
        with open(fpath, "rb") as f:
            while f.read(1024 * 1024):
                pass


def _io_bound_sleep() -> None:
    """Network wait simulation (idle, I/O bound).

    Simulates waiting for external API response (e.g., hosted LLM).
    Expected: very low CPU%, low IPC, io_bound or network_bound.
    """
    # Simulate 5 API calls with 200ms latency each
    for _ in range(5):
        time.sleep(0.2)


def _frontend_starved_branch() -> None:
    """Branch-heavy workload (frontend bottleneck).

    High branch misprediction rate → frontend starvation.
    Expected: low IPC, low cache miss, frontend_starved.
    """
    import random

    # Random branches (high misprediction)
    data = [random.randint(0, 100) for _ in range(1_000_000)]
    count = 0
    for _ in range(10):
        for val in data:
            if val > 50:
                if val > 75:
                    count += 1
                else:
                    count -= 1
            else:
                if val < 25:
                    count += 2
                else:
                    count -= 2


def _balanced_mixed() -> None:
    """Mixed workload (balanced, no dominant bottleneck).

    Expected: moderate IPC, moderate cache miss, balanced.
    """
    import numpy as np

    # Mix of compute + memory access
    n = 5_000_000
    a = np.random.randn(n).astype(np.float32)

    for _ in range(5):
        # Some compute
        b = np.sqrt(np.abs(a)) + np.sin(a)
        # Some memory access
        c = a[::2] + a[1::2]
        a = np.concatenate([b[:len(c)], c])[:n]


# Registry of all archetypes
ARCHETYPES = [
    Archetype(
        name="memory_bound_stream",
        true_bottleneck="memory_bound",
        run_fn=_memory_bound_stream,
        description="STREAM triad (vector add), saturates memory bandwidth",
        duration_estimate_s=8.0,
    ),
    Archetype(
        name="memory_bound_random",
        true_bottleneck="memory_bound",
        run_fn=_memory_bound_random_access,
        description="Random pointer chasing, high LLC miss rate",
        duration_estimate_s=12.0,
    ),
    Archetype(
        name="core_bound_matmul",
        true_bottleneck="core_bound",
        run_fn=_core_bound_matmul,
        description="Matrix multiplication (FP32), execution port saturation",
        duration_estimate_s=10.0,
    ),
    Archetype(
        name="core_bound_fibonacci",
        true_bottleneck="core_bound",
        run_fn=_core_bound_fibonacci,
        description="Recursive Fibonacci, pure CPU compute",
        duration_estimate_s=6.0,
    ),
    Archetype(
        name="io_bound_disk",
        true_bottleneck="io_bound",
        run_fn=_io_bound_disk,
        description="Disk I/O (write + read 100 MB), blocked on storage",
        duration_estimate_s=4.0,
    ),
    Archetype(
        name="io_bound_sleep",
        true_bottleneck="io_bound",
        run_fn=_io_bound_sleep,
        description="Sleep simulation (network API wait), idle CPU",
        duration_estimate_s=1.5,
    ),
    Archetype(
        name="frontend_starved_branch",
        true_bottleneck="frontend_starved",
        run_fn=_frontend_starved_branch,
        description="Random branches, high misprediction, instruction fetch bottleneck",
        duration_estimate_s=8.0,
    ),
    Archetype(
        name="balanced_mixed",
        true_bottleneck="balanced",
        run_fn=_balanced_mixed,
        description="Mixed compute + memory, no dominant bottleneck",
        duration_estimate_s=5.0,
    ),
]


__all__ = ["ARCHETYPES", "Archetype"]
