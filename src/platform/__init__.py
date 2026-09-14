#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Platform detection for AgentSysPerf.

Auto-discovers hardware capabilities (CPU vendor, L3 size, DRAM bandwidth,
NUMA topology) so analyzers and measurements use detected values instead
of hardcoded Intel-specific constants.

Works on Intel, AMD, and ARM platforms.
"""

from .detect import PlatformInfo, detect_platform

__all__ = ["PlatformInfo", "detect_platform"]
