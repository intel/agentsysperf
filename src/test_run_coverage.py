#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the post-run coverage report (W1.4).

The defect this closes is not a wrong number — it is silence. A run whose
``emon`` probe found no SEP driver and a run whose ``emon`` probe collected 629
metrics both printed "9 records" and exited 0, so a 3-of-5 measurement run was
indistinguishable from a 5-of-5 one without reading the source.

Two invariants matter and are tested directly:

1. **Coverage is derived from what landed, never from what was discovered.**
   A probe that registered, started, and produced nothing must read as silent.
   Asserting against ``discover_measurements()`` would restate the registry.
2. **Every silent plugin carries a reason.** A bare "3/8" tells a user they
   lost something without telling them what or why, which is the same dead end
   as printing nothing.
"""

from __future__ import annotations

from src.protocols import AnalysisResult, MeasurementRecord
from src.run_driver import _compute_coverage


class _Probe:
    """Minimal Measurement stand-in: only the attributes coverage reads."""

    def __init__(self, layer: str, available: bool | None = None) -> None:
        self.layer = layer
        if available is not None:
            self._available = available


class _Analyzer:
    def __init__(self, name: str, input_layers: frozenset) -> None:
        self.name = name
        self.input_layers = input_layers


def _rec(layer: str) -> MeasurementRecord:
    return MeasurementRecord(span_id="s1", layer=layer, payload={})


def _verdict(analyzer_name: str) -> AnalysisResult:
    return AnalysisResult(
        verdict="v", confidence=1.0, evidence={}, analyzer_name=analyzer_name,
    )


def test_probe_that_emitted_nothing_reads_as_silent_not_active():
    """The core invariant: registration is not collection.

    ``perfspect`` reports ``_available=True`` whenever the binary is on disk,
    which is why the UX assessment found it "available" and collecting nothing
    on every documented benchmark. Coverage must key off the records.
    """
    coverage = _compute_coverage(
        measurements={
            "l1_subspan": _Probe("l1"),
            "perfspect": _Probe("perfspect", available=True),
        },
        analyzers={},
        records=[_rec("l1")],
        verdicts=[],
        analyzer_errors={},
    )

    assert coverage.measurements_active == ["l1_subspan"]
    assert [n for n, _ in coverage.measurements_silent] == ["perfspect"]
    assert coverage.measurements_total == 2


def test_unavailable_probe_says_so_rather_than_just_counting():
    """An `_available=False` probe is a host-capability fact, not a mystery."""
    coverage = _compute_coverage(
        measurements={"emon": _Probe("emon", available=False)},
        analyzers={},
        records=[],
        verdicts=[],
        analyzer_errors={},
    )

    (_, reason), = coverage.measurements_silent
    assert "unavailable" in reason


def test_silent_analyzer_names_the_layer_it_needed():
    """`input_layers` becomes load-bearing instead of decorative.

    ``cpu_bound`` needs l3. On a host where ``perf`` is blocked it can only be
    silent, and the run should say which layer was missing rather than leaving
    the user to infer it.
    """
    coverage = _compute_coverage(
        measurements={"l1_subspan": _Probe("l1")},
        analyzers={
            "memory_leak": _Analyzer("memory_leak", frozenset({"l1"})),
            "cpu_bound": _Analyzer("cpu_bound", frozenset({"l1", "l3"})),
        },
        records=[_rec("l1")],
        verdicts=[_verdict("memory_leak")],
        analyzer_errors={},
    )

    assert coverage.analyzers_emitted == ["memory_leak"]
    (name, reason), = coverage.analyzers_silent
    assert name == "cpu_bound"
    assert "l3" in reason
    # The layer it *did* have must not be reported as missing.
    assert "l1" not in reason.replace("l3", "")


def test_analyzer_that_raised_reports_the_failure_not_a_missing_layer():
    """A crash and a missing prerequisite are different problems.

    Reporting "needs layer l1" for an analyzer that had l1 and threw would send
    the user to fix a host that is already correct.
    """
    coverage = _compute_coverage(
        measurements={"l1_subspan": _Probe("l1")},
        analyzers={"memory_leak": _Analyzer("memory_leak", frozenset({"l1"}))},
        records=[_rec("l1")],
        verdicts=[],
        analyzer_errors={"memory_leak": "boom"},
    )

    (_, reason), = coverage.analyzers_silent
    assert "failed" in reason and "boom" in reason


def test_analyzer_with_all_layers_and_no_verdict_is_reported_honestly():
    """Having the data and declining to speak is a third, distinct state.

    ``scaling`` is sweep-scoped: it sees l1_system on every run and emits
    nothing outside a sweep. That must not be dressed up as a missing layer.
    """
    coverage = _compute_coverage(
        measurements={"l1_system": _Probe("l1_system")},
        analyzers={"scaling": _Analyzer("scaling", frozenset({"l1_system"}))},
        records=[_rec("l1_system")],
        verdicts=[],
        analyzer_errors={},
    )

    (_, reason), = coverage.analyzers_silent
    assert "single-run scope" in reason


def test_every_silent_plugin_carries_a_reason():
    """No silent plugin may be reported as a bare count."""
    coverage = _compute_coverage(
        measurements={
            "l1_subspan": _Probe("l1"),
            "l3_perf": _Probe("l3", available=False),
            "perfspect": _Probe("perfspect", available=True),
        },
        analyzers={
            "cpu_bound": _Analyzer("cpu_bound", frozenset({"l1", "l3"})),
            "memory_leak": _Analyzer("memory_leak", frozenset({"l1"})),
        },
        records=[_rec("l1")],
        verdicts=[_verdict("memory_leak")],
        analyzer_errors={},
    )

    for _, reason in coverage.measurements_silent + coverage.analyzers_silent:
        assert reason and reason.strip()

    lines = coverage.as_lines()
    assert "measurements 1/3 active" in lines[0]
    assert "analyzers 1/2 emitted" in lines[1]
    # The reasons reach the rendered line, not just the dataclass.
    assert "l3_perf" in lines[0] and "perfspect" in lines[0]
    assert "cpu_bound" in lines[1]


def test_clean_run_renders_without_a_parenthetical():
    """When nothing was skipped the line stays short and says so."""
    coverage = _compute_coverage(
        measurements={"l1_subspan": _Probe("l1")},
        analyzers={"memory_leak": _Analyzer("memory_leak", frozenset({"l1"}))},
        records=[_rec("l1")],
        verdicts=[_verdict("memory_leak")],
        analyzer_errors={},
    )

    assert coverage.as_lines() == [
        "measurements 1/1 active",
        "analyzers 1/1 emitted",
    ]
