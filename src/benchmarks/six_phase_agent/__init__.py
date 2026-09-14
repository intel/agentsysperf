#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Six-phase agent-loop benchmark: reason→retrieve→act→admit→context→commit."""

from .adapter import SixPhaseAgentAdapter

__all__ = ["SixPhaseAgentAdapter"]
