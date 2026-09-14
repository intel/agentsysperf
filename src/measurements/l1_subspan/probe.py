#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""L1: sub-span decomposition — the canonical Measurement reference plugin.

For every closed span, emits one :class:`MeasurementRecord` carrying
per-thread CPU time, wall duration, and resource metrics that the
:class:`PsutilSampler` already captured.  No new sampling, no kernel
access — pure observation of state the sampler produced.

This is the simplest real measurement plugin.  External developers
writing L2/L3/L4/L5/VTune probes should read this file as the
worked example of the Measurement Protocol lifecycle.

Per-span emission shape::

    MeasurementRecord(
        span_id="run-abc::task-1",
        layer="l1",
        payload={
            "kind": "synthetic_cpu",
            "node_id": "linalg",
            "duration_us": 2_014_332,
            "cpu_time_s": 1.98,
            "cpu_pct_mean": 99.1,
            "cpu_pct_peak": 100.0,
            "rss_kb_peak": 312_448,
            "num_threads_peak": 4,
            "sample_count": 100,
        },
    )
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.protocols import MeasurementRecord, NullMeasurement

logger = logging.getLogger(__name__)


class L1SubSpanMeasurement(NullMeasurement):
    """L1 reference: emit per-span CPU + duration + resource records.

    Passive plugin — depends on :class:`PsutilSampler` running in the
    :class:`RunContext`.  Declares ``_needs_sampler = True`` so the
    context auto-starts the sampler when L1 is registered.
    """

    name: str = "l1_subspan"
    layer: str = "l1"
    _needs_sampler: bool = True

    def __init__(self, *, _ctx: Optional[Any] = None) -> None:
        super().__init__()
        self._ctx = _ctx

    def set_context(self, ctx: Any) -> None:
        """Attach the RunContext (called automatically by RunContext.start)."""
        self._ctx = ctx

    def start(self, *, run_id: str, output_dir: Path) -> None:
        return None

    def observe_span(self, *, span_id: str, kind: str, node_id: str) -> None:
        return None

    def finalize_span(self, span_id: str) -> Iterable[MeasurementRecord]:
        """Read the closed SpanRecord from the registry and emit one record."""
        if self._ctx is None:
            return

        tid = threading.get_native_id()
        record = self._ctx.registry.current(tid)
        if record is None or record.span_id != span_id:
            logger.debug(
                "L1 finalize_span(%s): registry top didn't match (got %s)",
                span_id, getattr(record, "span_id", None),
            )
            return

        payload: Dict[str, Any] = {
            "kind": record.kind,
            "node_id": record.node_id,
            "duration_us": record.duration_us(),
            "cpu_time_s": float(record.cpu_time_s),
            "cpu_pct_mean": float(record.cpu_pct_mean),
            "cpu_pct_peak": float(record.cpu_pct_peak),
            "rss_kb_peak": int(record.rss_kb_peak),
            "num_threads_peak": int(record.num_threads_peak),
            "sample_count": int(record.sample_count),
        }
        if record.phase:
            payload["phase"] = record.phase
        yield MeasurementRecord(
            span_id=span_id,
            layer=self.layer,
            payload=payload,
        )

    def stop(self) -> Iterable[MeasurementRecord]:
        return ()


__all__ = ["L1SubSpanMeasurement"]
