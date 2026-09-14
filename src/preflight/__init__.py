#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Pre-benchmark system validation for AgentSysPerf.

Checks that the host is properly configured for performance measurement
before running benchmarks. See ``system_check`` for the kernel documentation
each check's boundary is taken from.

Usage:
    from src.preflight import SystemCheck
    check = SystemCheck()
    report = check.run()
    if not report.ready:
        print(report.summary())
"""

from .system_check import SystemCheck, CheckResult, SystemReport

__all__ = ["SystemCheck", "CheckResult", "SystemReport"]
