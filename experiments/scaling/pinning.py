#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""CPU affinity assignment for placement strategies."""

from __future__ import annotations

from typing import List, Set

from .config import (
    AGENT_POOL_CORES,
    NUMA_NODES,
    AgentConfig,
    ExperimentConfig,
    PhaseMix,
    Placement,
)


def _node_cores(node_id: int) -> List[int]:
    """Get sorted core list for a NUMA node, excluding orchestrator cores."""
    return sorted(set(NUMA_NODES[node_id]) & AGENT_POOL_CORES)


def active_cores(config: ExperimentConfig) -> List[int]:
    """Cores actually used by this experiment: all agent cpusets + orchestrator.

    Used to build a pyEDP --core-filter so post-processing skips idle cores,
    whose 48-bit PMU counters overflow during long system-wide collection and
    trigger 'excluded due to excessively large counts' warnings.
    """
    from .config import ORCHESTRATOR_CORES

    used: Set[int] = set(ORCHESTRATOR_CORES)
    for ac in assign_cpusets(config):
        used |= ac.cpuset
    return sorted(used)


def format_core_ranges(cores: List[int]) -> str:
    """Compress a sorted core list to pyEDP range syntax, e.g. [0,1,2,4] -> '0-2,4'."""
    if not cores:
        return ""
    cores = sorted(set(cores))
    ranges: List[str] = []
    start = prev = cores[0]
    for c in cores[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)


def assign_cpusets(config: ExperimentConfig) -> List[AgentConfig]:
    """Assign CPU sets to each agent based on placement strategy.

    Returns a list of AgentConfig, one per agent, with cpuset populated.
    Raises ValueError if there aren't enough cores for the requested density.
    """
    n = config.density
    cpa = config.cores_per_agent
    needed = n * cpa

    if config.placement == Placement.INTRA_NODE:
        pool = _node_cores(0)
    elif config.placement == Placement.CROSS_NODE:
        pool = _node_cores(0) + _node_cores(1)
    else:
        pool = _node_cores(0) + _node_cores(1) + _node_cores(2)

    oversub = getattr(config, "oversubscribe", False)
    if needed > len(pool) and not oversub:
        raise ValueError(
            f"Not enough cores: need {needed} ({n} agents x {cpa} cores), "
            f"but placement '{config.placement.value}' only has {len(pool)} available "
            f"(set oversubscribe=True to share cores)"
        )

    agents = []
    for i in range(n):
        if oversub:
            if not pool:
                raise ValueError(f"No cores available for placement '{config.placement.value}'")
            if cpa > len(pool):
                raise ValueError(
                    f"cores_per_agent={cpa} exceeds available core pool ({len(pool)}) for placement "
                    f"'{config.placement.value}'"
                )
            # Wrap around the pool so agents beyond capacity share cores.
            start = (i * cpa) % len(pool)
            cpuset = {pool[(start + k) % len(pool)] for k in range(cpa)}
        elif config.placement == Placement.SPREAD:
            # Round-robin across nodes
            cpuset = _spread_assign(i, n, cpa)
        else:
            # Contiguous allocation from pool
            start = i * cpa
            cpuset = set(pool[start:start + cpa])

        mix = _agent_mix(i, n, config.phase_mix)
        agents.append(AgentConfig(
            agent_id=i,
            cpuset=cpuset,
            phase_mix=mix,
            turns=config.turns,
        ))

    return agents


def _spread_assign(agent_idx: int, total_agents: int, cores_per_agent: int) -> Set[int]:
    """Round-robin assignment across all 3 NUMA nodes."""
    nodes = [_node_cores(0), _node_cores(1), _node_cores(2)]
    node_id = agent_idx % 3
    # How many agents have been placed on this node before us
    slot_on_node = agent_idx // 3
    start = slot_on_node * cores_per_agent
    node_pool = nodes[node_id]

    if start + cores_per_agent > len(node_pool):
        # Overflow — fall back to next node with space
        for fallback_node in range(3):
            fallback_slot = sum(1 for j in range(agent_idx) if j % 3 == fallback_node)
            fallback_start = fallback_slot * cores_per_agent
            if fallback_start + cores_per_agent <= len(nodes[fallback_node]):
                return set(nodes[fallback_node][fallback_start:fallback_start + cores_per_agent])
        raise ValueError(f"Cannot place agent {agent_idx} with spread strategy")

    return set(node_pool[start:start + cores_per_agent])


_MIXED_ROTATION = [
    PhaseMix.COMPILE,
    PhaseMix.ML_TRAIN,
    PhaseMix.LINALG,
    PhaseMix.COMPRESS,
    PhaseMix.RAYTRACE,
    PhaseMix.SAT,
    PhaseMix.INTERPRETER,
    PhaseMix.IO_HEAVY,
]

# Optional override for the MIXED rotation. When set (a non-empty list of
# PhaseMix), MIXED experiments round-robin over THIS subset instead of the full
# 8-way default — lets a run target a specific heterogeneous combination
# (e.g. io_heavy + raytrace + interpreter + linalg).
_MIXED_ROTATION_OVERRIDE: List[PhaseMix] = []


def set_mixed_rotation(mixes: List[PhaseMix]) -> None:
    """Restrict the MIXED rotation to a custom subset (empty list = default)."""
    global _MIXED_ROTATION_OVERRIDE
    _MIXED_ROTATION_OVERRIDE = list(mixes)


def _agent_mix(agent_idx: int, total_agents: int, experiment_mix: PhaseMix) -> PhaseMix:
    """Determine the phase mix for a specific agent.

    For MIXED experiments, assigns different mixes to different agents
    in round-robin across all TB2 workload types (or a custom subset if
    set_mixed_rotation() was called). For all other experiments, every agent
    runs the same mix.
    """
    if experiment_mix != PhaseMix.MIXED:
        return experiment_mix

    rotation = _MIXED_ROTATION_OVERRIDE or _MIXED_ROTATION
    return rotation[agent_idx % len(rotation)]
