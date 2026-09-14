#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""L1-system measurement: node-level CPU / runqueue / context-switch / memory.

Registered via the ``agentsysperf.measurements`` entry point as ``l1_system``.
"""
from src.measurements.l1_system.probe import L1SystemMeasurement

__all__ = ["L1SystemMeasurement"]
