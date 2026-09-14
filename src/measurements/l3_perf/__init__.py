#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""L3: hardware-counter sampling via Linux ``perf stat``.

Two plugin classes live here, sharing a small subprocess helper:

- :class:`L3PerfMeasurement` — implements
  :class:`src.protocols.Measurement`. Runs continuous
  ``perf stat -I 100 -a`` for the duration of a benchmark run, then
  attributes per-event counts to span windows by wall-clock overlap
  on ``finalize_span``.

- :class:`PerfStatTelemetry` — implements
  :class:`src.protocols.HardwareTelemetryPlugin`. Point-in-time
  reader: takes a list of events + a window, runs ``perf stat`` for
  exactly that window, returns a :class:`CounterReading`. Used by
  :class:`OptimizationProfilePlugin.verify_engaged` to prove a profile's
  optimizations engaged.

Both wrap the ``harness/scripts/perf_sampler.py`` sampling logic inside the
AgentSysPerf plugin contract.
"""

from src.measurements.l3_perf.probe import L3PerfMeasurement
from src.measurements.l3_perf.telemetry import PerfStatTelemetry

__all__ = ["L3PerfMeasurement", "PerfStatTelemetry"]
