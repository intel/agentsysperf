#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Experiment configuration: topology, density levels, placement strategies."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


# ─── Topology ─────────────────────────────────────────────────────────────────
#
# Defaults describe Granite Rapids SNC3 (96 physical cores, 3 nodes x 32).
# They are the reference topology behind the published density data, so they
# stay as-is. To run on a differently-shaped host, discover the topology from
# sysfs instead of editing these:
#
#     AGENTSYSPERF_TOPOLOGY=auto         # read /sys NUMA layout
#     AGENTSYSPERF_ORCH_CORES=4          # reserve N cores for orchestrator+EMON
#
# Verified on Clearwater Forest (288 cores, no HT, 3 nodes x 96).


def _discover_numa_nodes() -> Dict[int, List[int]]:
    """Map node_id -> online CPU list from sysfs.

    Raises RuntimeError rather than guessing: a silently wrong core map would
    pin agents onto the wrong nodes and quietly invalidate the placement study.
    """
    base = "/sys/devices/system/node"
    nodes: Dict[int, List[int]] = {}
    try:
        entries = sorted(
            d for d in os.listdir(base)
            if d.startswith("node") and d[4:].isdigit()
        )
    except OSError as exc:
        raise RuntimeError(f"cannot read {base}: {exc}") from exc
    for entry in entries:
        node_id = int(entry[4:])
        try:
            with open(f"{base}/{entry}/cpulist") as fh:
                spec = fh.read().strip()
        except OSError as exc:
            raise RuntimeError(f"cannot read {entry}/cpulist: {exc}") from exc
        cpus: List[int] = []
        for part in filter(None, spec.split(",")):
            if "-" in part:
                lo, hi = part.split("-")
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        if cpus:
            nodes[node_id] = sorted(cpus)
    if not nodes:
        raise RuntimeError(f"no NUMA nodes with CPUs found under {base}")
    return nodes


if os.environ.get("AGENTPERF_TOPOLOGY", "").lower() == "auto":
    NUMA_NODES = _discover_numa_nodes()
if os.environ.get("AGENTSYSPERF_TOPOLOGY", "").lower() == "auto":
    NUMA_NODES = _discover_numa_nodes()
    # pinning._spread_assign hardcodes _node_cores(0..2) and agent_idx % 3.
    # On a 2-node host that raises KeyError; on a 4+-node host it silently
    # confines every agent to nodes 0-2, quietly invalidating the placement
    # study. Fail fast until _spread_assign is generalized over NUMA_NODES.
    if set(NUMA_NODES) != {0, 1, 2}:
        raise RuntimeError(
            f"AGENTSYSPERF_TOPOLOGY=auto found NUMA nodes {sorted(NUMA_NODES)}; "
            "the placement strategies in pinning.py assume exactly nodes 0, 1, 2"
        )
    _ALL_CORES = sorted(c for cores in NUMA_NODES.values() for c in cores)
    _N_ORCH = int(os.environ.get("AGENTSYSPERF_ORCH_CORES", "4"))
    if _N_ORCH >= len(_ALL_CORES):
        raise RuntimeError(
            f"AGENTSYSPERF_ORCH_CORES={_N_ORCH} leaves no cores for agents "
            f"(host has {len(_ALL_CORES)})"
        )
    # Reserve the highest-numbered cores, mirroring the GNR default (92-95).
    ORCHESTRATOR_CORES = set(_ALL_CORES[-_N_ORCH:]) if _N_ORCH > 0 else set()
    AGENT_POOL_CORES = set(_ALL_CORES) - ORCHESTRATOR_CORES
    # No fixed HT-sibling offset is safe on a discovered host. Defined so the
    # name exists in both branches; callers needing siblings should read
    # /sys/devices/system/cpu/cpuN/topology/thread_siblings_list.
    HT_OFFSET = None
else:
    NUMA_NODES = {
        0: list(range(0, 32)),    # physical cores
        1: list(range(32, 64)),
        2: list(range(64, 96)),
    }

    HT_OFFSET = 96  # HT sibling of core N is at N + 96

    ORCHESTRATOR_CORES = {92, 93, 94, 95}

    AGENT_POOL_CORES = set(range(0, 92))  # 92 cores available for agents


class Placement(str, Enum):
    INTRA_NODE = "intra_node"   # all agents on node0
    CROSS_NODE = "cross_node"   # split across node0 + node1
    SPREAD = "spread"           # round-robin across all 3 nodes


class PhaseMix(str, Enum):
    COMPILE = "compile"             # branch-heavy, instruction-stream-bound
    ML_TRAIN = "ml_train"           # vectorized FP, memory-BW
    LINALG = "linalg"              # dense matmul, memory-BW, AMX-candidate
    COMPRESS = "compress"           # cache-friendly, high IPC
    RAYTRACE = "raytrace"           # FP-heavy, random memory access
    SAT = "sat"                     # branch-heavy, random memory, low IPC
    INTERPRETER = "interpreter"     # unpredictable branches, very low IPC
    IO_HEAVY = "io_heavy"           # sequential file reads, page faults
    MIXED = "mixed"                 # per-agent assignment from the above


DENSITY_LEVELS = [1, 2, 4, 8, 12, 16, 24, 32]

DEFAULT_CORES_PER_AGENT = 4

DEFAULT_TURNS = 8


@dataclass
class AgentConfig:
    """Configuration for a single agent worker."""
    agent_id: int
    cpuset: Set[int]
    phase_mix: PhaseMix
    turns: int = DEFAULT_TURNS


@dataclass
class ExperimentConfig:
    """Configuration for one experiment run (one density level)."""
    density: int
    placement: Placement
    phase_mix: PhaseMix
    cores_per_agent: int = DEFAULT_CORES_PER_AGENT
    turns: int = DEFAULT_TURNS
    emon_warmup_s: float = 2.0
    emon_cooldown_s: float = 1.0
    agent_timeout_s: float = 120.0
    # When True, allow more agents than cores: cpusets wrap around the pool
    # (modulo) so multiple agents share cores. Used to drive past pool-full into
    # the oversubscription regime where the scheduling-contention knee appears.
    # Default False preserves dedicated-core assignment.
    oversubscribe: bool = False


@dataclass
class ExperimentMatrix:
    """Full experiment matrix to iterate."""
    densities: List[int] = field(default_factory=lambda: list(DENSITY_LEVELS))
    placements: List[Placement] = field(
        default_factory=lambda: [Placement.INTRA_NODE, Placement.CROSS_NODE, Placement.SPREAD]
    )
    phase_mixes: List[PhaseMix] = field(
        default_factory=lambda: [PhaseMix.COMPILE, PhaseMix.ML_TRAIN, PhaseMix.LINALG, PhaseMix.MIXED]
    )
    cores_per_agent: int = DEFAULT_CORES_PER_AGENT
    turns: int = DEFAULT_TURNS

    def configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        result = []
        for density in self.densities:
            for placement in self.placements:
                for mix in self.phase_mixes:
                    result.append(ExperimentConfig(
                        density=density,
                        placement=placement,
                        phase_mix=mix,
                        cores_per_agent=self.cores_per_agent,
                        turns=self.turns,
                    ))
        return result

    def subset(
        self,
        densities: List[int] | None = None,
        placements: List[Placement] | None = None,
        phase_mixes: List[PhaseMix] | None = None,
    ) -> ExperimentMatrix:
        """Create a subset of the matrix for quick iteration."""
        return ExperimentMatrix(
            densities=densities or self.densities,
            placements=placements or self.placements,
            phase_mixes=phase_mixes or self.phase_mixes,
            cores_per_agent=self.cores_per_agent,
            turns=self.turns,
        )
