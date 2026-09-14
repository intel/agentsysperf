#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Terminal-Bench adapter for AgentSysPerf.

Terminal-Bench is a shell-based benchmark for agentic AI systems. Tasks
involve file manipulation, system administration, coding, and debugging
in a Linux shell environment.
"""

from src.benchmarks.terminal_bench.adapter import TerminalBenchAdapter

__all__ = ["TerminalBenchAdapter"]
