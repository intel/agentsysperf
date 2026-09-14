#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""AgentSysPerf: pluggable benchmark suite for end-to-end agentic AI stacks.

Top-level package. The runtime exposes four extension points:

- ``BenchmarkAdapter`` — a workload (Terminal-Bench, SWE-Bench, synthetic CPU).
- ``Measurement`` — a probe (L1 sub-spans, L3 perf-stat, L5 PCM, VTune, ...).
- ``OptimizationProfilePlugin`` — a configuration + engagement verifier
  (base, AMX-only, full-Xeon, OpenVINO, ...).
- ``HardwareTelemetryPlugin`` — counter source that doubles as the
  optimization-engagement verifier.

Plugins ship in external packages and register via ``pyproject.toml``
entry points (``agentsysperf.benchmarks``, ``agentsysperf.measurements``,
``agentsysperf.optimization_profiles``, ``agentsysperf.hardware_telemetry``).

The core integrity principles — measured-not-assumed, verify-don't-trust,
abort-don't-degrade — are enforced by the protocols themselves and stated
under *Integrity principles* in the README.
"""

from src.protocols import (
    AgentInvoker,
    BenchmarkAdapter,
    HardwareTelemetryPlugin,
    Measurement,
    MeasurementRecord,
    NullMeasurement,
    OptimizationProfilePlugin,
    TaskResult,
    TaskSpec,
    discover_benchmarks,
    discover_hardware_telemetry,
    discover_measurements,
    discover_optimization_profiles,
)
from src.runner import RunContext, track_span

__all__ = [
    "AgentInvoker",
    "BenchmarkAdapter",
    "HardwareTelemetryPlugin",
    "Measurement",
    "MeasurementRecord",
    "NullMeasurement",
    "OptimizationProfilePlugin",
    "RunContext",
    "TaskResult",
    "TaskSpec",
    "discover_benchmarks",
    "discover_hardware_telemetry",
    "discover_measurements",
    "discover_optimization_profiles",
    "track_span",
]
