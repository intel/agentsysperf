#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Analyzer plugins shipped with AgentSysPerf core.

Analyzers transform raw MeasurementRecords into domain insights.
External analyzers register via the ``agentsysperf.analyzers`` entry point.
"""

from src.analyzers.cache import CacheAnalyzer
from src.analyzers.cpu_bound import CPUBoundAnalyzer
from src.analyzers.memory_leak import MemoryLeakAnalyzer
from src.analyzers.memory_bandwidth import MemoryBandwidthAnalyzer
from src.analyzers.phase_profiler import PhaseProfiler
from src.analyzers.scaling import ScalingAnalyzer

# EmonAnalyzer is Intel-specific (its TMA layers depend on the closed EMON
# database). It ships in the separate, unreleased ``agentsysperf-emon`` plugin
# and registers via the ``agentsysperf.analyzers`` entry point when installed —
# it is intentionally NOT re-exported here.

__all__ = [
    "CPUBoundAnalyzer",
    "CacheAnalyzer",
    "MemoryLeakAnalyzer",
    "MemoryBandwidthAnalyzer",
    "PhaseProfiler",
    "ScalingAnalyzer",
]
