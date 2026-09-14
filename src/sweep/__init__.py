#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Concurrency-scaling sweeps for agentic CPU benchmarking.

A sweep runs a workload at rising agent concurrency (normalized as
density = concurrency / vcpu_basis) to find the saturation knee — the
"agents per vCPU" the box sustains before throughput floors. See
:class:`~src.sweep.spec.SweepSpec` for the parameters and
:class:`~src.sweep.harbor_sweep.HarborSweep` for the EMR runner.
"""
from src.sweep.harbor_sweep import HarborSweep
from src.sweep.spec import DEFAULT_TB2_TASKS, SweepSpec

__all__ = ["SweepSpec", "HarborSweep", "DEFAULT_TB2_TASKS"]
