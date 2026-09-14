#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""OpenClaw benchmark adapter for AgentSysPerf.

OpenClaw is a legal reasoning benchmark where agents must analyze case law,
statutes, and legal questions to provide accurate legal analysis.
"""

from .adapter import OpenClawAdapter

__all__ = ["OpenClawAdapter"]
