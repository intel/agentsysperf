#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Resource planning for task-sized Docker stream workers."""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    """The CPU set and optional memory cap applied to one task container.

    An empty CPU set represents the default, unpinned scheduler. The memory
    limit remains meaningful in that mode and is still applied by Docker.
    """

    cpuset: tuple[int, ...]
    memory_mb: int | None = None

    def cpuset_str(self) -> str:
        """Render Docker's cpuset grammar."""
        if not self.cpuset:
            return ""
        values = sorted(self.cpuset)
        parts: list[str] = []
        start = previous = values[0]
        for cpu in values[1:]:
            if cpu == previous + 1:
                previous = cpu
                continue
            parts.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = cpu
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        return ",".join(parts)


@dataclass(frozen=True)
class TaskSlot:
    """One worker slot matching tasks with ``cpu_budget == size``."""

    size: int
    budget: ResourceBudget


def _parse_cpulist(value: str) -> list[int]:
    cpus: list[int] = []
    for token in value.strip().split(","):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ValueError(f"invalid kernel cpulist token {token!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        cpus.extend(range(start, end + 1))
    return sorted(set(cpus))


def available_cores_by_node() -> list[list[int]]:
    """Return this process's available CPUs grouped by NUMA node.

    The process affinity mask is the hard upper bound, so callers work
    correctly inside an externally pinned shell or cgroup.
    """
    available = set(os.sched_getaffinity(0))
    paths = sorted(glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"))
    if not paths:
        return [sorted(available)]

    nodes: list[list[int]] = []
    for path in paths:
        with open(path) as handle:
            node = sorted(available.intersection(_parse_cpulist(handle.read())))
        if node:
            nodes.append(node)
    return nodes or [sorted(available)]


def plan_slot_counts(
    total_cores: int, task_histogram: Mapping[int, int]
) -> dict[int, int]:
    """Allocate a pinned total-core budget across every required task class.

    Every task size receives at least one slot. Additional slots are assigned
    to the class with the highest queued-work-per-slot ratio. A remainder that
    cannot fit another class is intentionally left unused rather than changing
    a task's requested container size.
    """
    if total_cores <= 0:
        raise ValueError(f"total_cores must be positive, got {total_cores}")

    counts = {size: count for size, count in task_histogram.items() if count > 0}
    if not counts:
        raise ValueError("task histogram is empty")
    if any(size <= 0 for size in counts):
        raise ValueError(f"task CPU budgets must be positive: {sorted(counts)}")

    required_cores = sum(size for size in counts)
    if required_cores > total_cores:
        raise ValueError(
            f"total_cores={total_cores} cannot provide one slot for every "
            f"requested task size {sorted(counts)} (needs at least "
            f"{required_cores})"
        )

    slots = {size: 1 for size in counts}
    remaining = total_cores - required_cores
    while True:
        candidates = [size for size in slots if size <= remaining]
        if not candidates:
            break
        # Work is proportional to task frequency. CPU size is already charged
        # when this slot consumes `size` cores from the remaining budget.
        size = max(candidates, key=lambda item: (counts[item] / slots[item], item))
        slots[size] += 1
        remaining -= size
    return slots


def plan_unpinned_slot_counts(
    slot_count: int,
    task_histogram: Mapping[int, int],
    *,
    max_slots: int = 4096,
) -> dict[int, int]:
    """Allocate exactly ``slot_count`` unpinned workers across task classes.

    The allocation follows queued work per currently assigned slot. CPU size
    is only a scheduling class here; it does not constrain the number of
    workers because the host scheduler is intentionally allowed to
    oversubscribe available CPUs.
    """
    if slot_count <= 0:
        raise ValueError(f"slot_count must be positive, got {slot_count}")
    if max_slots <= 0:
        raise ValueError(f"max_slots must be positive, got {max_slots}")
    if slot_count > max_slots:
        raise ValueError(f"slot_count={slot_count} exceeds max_slots={max_slots}")

    counts = {size: count for size, count in task_histogram.items() if count > 0}
    if not counts:
        raise ValueError("task histogram is empty")
    if any(size <= 0 for size in counts):
        raise ValueError(f"task CPU budgets must be positive: {sorted(counts)}")
    if slot_count < len(counts):
        raise ValueError(
            f"slot_count={slot_count} cannot provide one slot for every "
            f"requested task size {sorted(counts)}"
        )

    slots = {size: 1 for size in counts}
    for _ in range(slot_count - len(counts)):
        size = max(
            slots,
            key=lambda item: (counts[item] / slots[item], counts[item], -item),
        )
        slots[size] += 1
    return slots


def _take_contiguous(node: Sequence[int], size: int) -> tuple[int, ...] | None:
    run: list[int] = []
    for cpu in node:
        if run and cpu != run[-1] + 1:
            run = []
        run.append(cpu)
        if len(run) == size:
            return tuple(run)
    return None


def pack_task_slots(
    slot_counts: Mapping[int, int],
    nodes: Sequence[Sequence[int]],
) -> list[TaskSlot]:
    """Pack requested task slots into disjoint, NUMA-local contiguous cpusets.

    Unlike the former pooled path, placement failure is fatal. A benchmark
    point must not replace a task's requested container size with another one.
    """
    free = [list(node) for node in nodes]
    requested = [
        size
        for size in sorted(slot_counts, reverse=True)
        for _ in range(slot_counts[size])
    ]
    slots: list[TaskSlot] = []
    node_index = 0
    for size in requested:
        placed = False
        for offset in range(len(free)):
            index = (node_index + offset) % len(free)
            cpuset = _take_contiguous(free[index], size)
            if cpuset is None:
                continue
            selected = set(cpuset)
            free[index] = [cpu for cpu in free[index] if cpu not in selected]
            slots.append(TaskSlot(size=size, budget=ResourceBudget(cpuset)))
            node_index = (index + 1) % len(free)
            placed = True
            break
        if not placed:
            raise ValueError(
                f"cannot place a {size}-CPU task slot within one NUMA node; "
                f"remaining CPUs per node are {[len(node) for node in free]}"
            )
    return slots


def plan_task_slots(
    total_cores: int,
    task_histogram: Mapping[int, int],
    *,
    nodes: Sequence[Sequence[int]] | None = None,
) -> list[TaskSlot]:
    """Build all runc worker slots for a task mix and total core budget."""
    node_cpus = [
        list(node)
        for node in (nodes if nodes is not None else available_cores_by_node())
    ]
    available = sum(len(node) for node in node_cpus)
    if total_cores > available:
        raise ValueError(
            f"total_cores={total_cores} exceeds this process's available "
            f"CPU budget ({available})"
        )

    # Limit placement to the requested budget while retaining NUMA locality.
    selected_nodes: list[list[int]] = []
    remaining = total_cores
    for node in node_cpus:
        take = min(len(node), remaining)
        if take:
            selected_nodes.append(node[:take])
            remaining -= take
        if not remaining:
            break

    return pack_task_slots(
        plan_slot_counts(total_cores, task_histogram), selected_nodes
    )


def plan_unpinned_task_slots(
    slot_count: int,
    task_histogram: Mapping[int, int],
    *,
    max_slots: int = 4096,
) -> list[TaskSlot]:
    """Build exactly ``slot_count`` scheduler slots without CPU pinning."""
    slot_counts = plan_unpinned_slot_counts(
        slot_count, task_histogram, max_slots=max_slots
    )
    return [
        TaskSlot(size=size, budget=ResourceBudget(()))
        for size in sorted(slot_counts)
        for _ in range(slot_counts[size])
    ]


__all__ = [
    "ResourceBudget",
    "TaskSlot",
    "available_cores_by_node",
    "pack_task_slots",
    "plan_slot_counts",
    "plan_task_slots",
    "plan_unpinned_slot_counts",
    "plan_unpinned_task_slots",
]
