#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tau-Bench benchmark adapter for AgentSysPerf.

Tau-Bench tests agent capabilities on real-world retail and travel scenarios,
evaluating tool use, planning, and customer interaction quality.
"""

from .adapter import TauBenchAdapter

__all__ = ["TauBenchAdapter"]
