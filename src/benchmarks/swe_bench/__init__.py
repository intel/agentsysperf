#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""SWE-Bench benchmark adapter for AgentSysPerf.

SWE-Bench (Software Engineering Bench) tests agent capabilities on real-world
software engineering tasks: fixing GitHub issues in open-source repositories.
"""

from .adapter import SWEBenchAdapter

__all__ = ["SWEBenchAdapter"]
