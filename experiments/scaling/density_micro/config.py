#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Experiment configuration: topology, density levels, placement strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set


# ─── Granite Rapids SNC3 Topology ─────────────────────────────────────────────

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
    COMPUTE_HEAVY = "compute_heavy"
    IO_HEAVY = "io_heavy"
    BALANCED = "balanced"
    MIXED = "mixed"


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


@dataclass
class ExperimentMatrix:
    """Full experiment matrix to iterate."""
    densities: List[int] = field(default_factory=lambda: list(DENSITY_LEVELS))
    placements: List[Placement] = field(
        default_factory=lambda: [Placement.INTRA_NODE, Placement.CROSS_NODE, Placement.SPREAD]
    )
    phase_mixes: List[PhaseMix] = field(
        default_factory=lambda: [PhaseMix.COMPUTE_HEAVY, PhaseMix.IO_HEAVY, PhaseMix.BALANCED, PhaseMix.MIXED]
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
