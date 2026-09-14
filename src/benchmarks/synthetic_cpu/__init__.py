#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Synthetic CPU workloads as a :class:`BenchmarkAdapter`.

Nine deterministic CPU-signature exercisers (compile, ml_train, linalg,
io, compress, raytrace, sat, interpreter, control) that each stress a
distinct microarchitectural axis. Originally lived as
``harness/scripts/synthetic_tasks.py``; this adapter wraps them inside
the AgentSysPerf plugin contract so they compose with measurement plugins
(L3 perf, future L2 py-spy, L5 PCM/RAPL/eBPF) and report through the
universal record schema.

These tasks do not call an LLM. They run the workload function
directly in-process, ignoring the ``agent_invoker`` parameter the
:class:`BenchmarkAdapter` protocol mandates. That is intentional:
synthetic CPU is for plugin testing, baseline characterization, and
hardware regression detection — not agent benchmarking.
"""

from src.benchmarks.synthetic_cpu.adapter import SyntheticCpuAdapter

__all__ = ["SyntheticCpuAdapter"]
