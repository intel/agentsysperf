#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Point-in-time perf-stat telemetry for engagement verification.

This module satisfies the
:class:`src.protocols.HardwareTelemetryPlugin` Protocol. Whereas
:class:`L3PerfMeasurement` runs perf continuously across a whole run,
this plugin issues a one-shot ``perf stat -- sleep <window_s>`` for the
exact set of events the caller asked about, then returns a
:class:`CounterReading` snapshot.

The intended consumer is
:meth:`OptimizationProfilePlugin.verify_engaged`. After the SUT warms
up, the profile asks this telemetry source whether (e.g.)
``amx_active_cycles`` is > 0 over a 1-second window. If yes, the
profile is engaged; if no, the run aborts — per the design's
"abort, don't degrade" principle.

Note on event availability: ``perf list`` reports thousands of vendor
events on Xeon, but the set actually accessible at a given paranoid
level / SKU varies. This plugin advertises only generic events as
``available_events`` because per the no-fabricated-PMU-codes rule in
the README's *Integrity principles* we must NOT fabricate PMU
codes for unknown SKUs. Optimization profiles that need
vendor-specific events ship their own telemetry plugin (e.g.
``intel_pcm`` for uncore PMC + RAPL).

Consequence for the shipped Xeon profiles: the names they declare in
``SPEC.verify_counters`` are derived metrics
(``amx_active_cycle_ratio``, ``hugepage_fault_ratio``, ...), and
``available_events`` is matched against them literally — see the naming
contract on :class:`HardwareTelemetryPlugin`. This plugin advertises raw
generic perf events, so it satisfies none of them and every ``amx_*``
profile fails engagement verification here by construction. That is
honest, not a bug; but note it is a *namespace* gap as much as a missing
vendor plugin, so shipping ``intel_pcm`` alone would not close it.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.measurements.l3_perf._perf_subprocess import (
    DEFAULT_EVENTS,
    _counts_are_physical,
    detect_cpu_model,
    parse_perf_csv_summary,
    perf_available,
    run_perf_window,
)
from src.protocols import CounterReading

logger = logging.getLogger(__name__)


class PerfStatTelemetry:
    """Linux ``perf stat`` as a :class:`HardwareTelemetryPlugin`.

    Cheap to construct; expensive to call (each ``read_counters`` runs
    a real ``sleep <window_s>``). Profiles should call it once after
    warmup, not in a hot loop.
    """

    name: str = "perf_stat"

    # Generic events available on essentially every x86-64 CPU. These
    # are the ones the L3 measurement plugin uses by default. Vendor
    # events (e.g. AMX active cycles) require a vendor-specific
    # telemetry plugin.
    available_events: frozenset = frozenset(DEFAULT_EVENTS)

    def __init__(self) -> None:
        self._cpu_model = detect_cpu_model()
        self._available = perf_available()
        if not self._available:
            logger.warning(
                "PerfStatTelemetry: perf binary unavailable or paranoid "
                "level blocks counter access; read_counters() will return "
                "empty values. Optimization profiles that depend on this "
                "telemetry will fail engagement verification."
            )

    def read_counters(
        self,
        events: Sequence[str],
        *,
        window_s: float = 1.0,
        target: Optional[str] = None,
    ) -> CounterReading:
        """Sample ``events`` for ``window_s`` and return the readings.

        Events not in :attr:`available_events` are silently dropped from
        the request — never fabricated, never zeroed. Consumers see
        them missing from the resulting :class:`CounterReading.values`,
        which is the signal to either ship a different telemetry plugin
        or relax the profile's threshold.
        """
        kept = [e for e in events if e in self.available_events]
        dropped = [e for e in events if e not in self.available_events]
        if dropped:
            logger.debug(
                "PerfStatTelemetry: %d events not in available_events, "
                "dropping: %s", len(dropped), dropped,
            )

        if not self._available or not kept:
            return CounterReading(
                values={}, window_s=window_s,
                cpu_model=self._cpu_model,
                notes=("perf unavailable" if not self._available
                       else "no events in scope"),
            )

        stderr_text = run_perf_window(
            events=kept, window_s=window_s, target=target,
        )
        if stderr_text is None:
            return CounterReading(
                values={}, window_s=window_s,
                cpu_model=self._cpu_model,
                notes="perf window run failed",
            )

        values, unmeasured = parse_perf_csv_summary(stderr_text)

        # Same corruption mode L3PerfMeasurement guards against: while an
        # exclusive-PMU collector holds the counters (or under severe event
        # multiplexing) perf exits 0 and emits well-formed CSV in which every
        # event reads back a near-identical ~1e18 count. Reporting those as
        # readings would make verify_engaged() return a confident, wrong
        # verdict, so discard the whole window — abort, don't degrade.
        if values and not _counts_are_physical(values, window_s):
            logger.warning(
                "PerfStatTelemetry: discarding window — counter values are "
                "physically impossible (cycles=%.3g over %.1fs). perf exited "
                "cleanly but the counts cannot be real. Usual causes: another "
                "tool holds the PMU (an exclusive-PMU collector cannot run "
                "alongside perf), or severe event multiplexing.",
                values.get("cycles", 0.0), window_s,
            )
            return CounterReading(
                values={}, window_s=window_s,
                cpu_model=self._cpu_model,
                notes="counter values physically impossible (PMU contention?)",
            )

        if unmeasured:
            logger.warning(
                "PerfStatTelemetry: %d of %d requested events came back "
                "<not counted>/<not supported> and are reported as unmeasured "
                "rather than zero: %s",
                len(unmeasured), len(kept), unmeasured,
            )

        notes = ""
        if not values:
            notes = "no counters parsed from perf output"
        elif unmeasured:
            notes = f"not counted on this host: {', '.join(sorted(unmeasured))}"
        return CounterReading(
            values=values,
            window_s=window_s,
            cpu_model=self._cpu_model,
            notes=notes,
        )


__all__ = ["PerfStatTelemetry"]
