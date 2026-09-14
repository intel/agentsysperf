#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""PerfSpect measurement plugin for AgentSysPerf.

Wraps Intel PerfSpect's `metrics` subcommand to collect TMA (Top-down
Microarchitecture Analysis), cache behavior, memory bandwidth, and power
metrics during benchmark spans.

See ``measurement`` for the PerfSpect and TMA references the collection
settings and metric interpretation are based on.
"""

from .measurement import PerfSpectMeasurement

__all__ = ["PerfSpectMeasurement"]
