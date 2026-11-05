#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""One PMU owner per run.

The defect: ``run_benchmark`` registers every discovered probe, so a default
serial run started ``l3_perf`` *and* ``perfspect`` against the same hardware
counters. Measured on a serial Terminal-Bench replay (fix-git, prove-plus-comm):
every ``l3_perf`` hardware event fell to 0% counter-enabled at its minimum (mean
41-89%), against 100% on every event with perfspect left out. perf scales the
counts to cover the gaps, so the stored rows looked like data.

Invariants pinned here:

1. With two PMU probes registered, exactly one is started.
2. ``l3_perf`` wins by default; ``AGENTSYSPERF_PMU_OWNER`` overrides it.
3. A probe that cannot collect never wins.
4. The loser reads as silent with a reason, not as a broken probe.
"""

from __future__ import annotations

import pytest

from src.measurements.l3_perf import L3PerfMeasurement
from src.measurements.perfspect import PerfSpectMeasurement
from src.run_driver import _measurement_skip_reason
from src.runner import PMU_OWNER_ENV, RunContext


class _Probe:
    layer = "x"

    def __init__(self, name: str, *, pmu: bool = True, able: bool = True) -> None:
        self.name = name
        self.pmu_exclusive = pmu
        self._able = able
        self.started = False

    def pmu_available(self) -> bool:
        return self._able

    def start(self, *, run_id, output_dir) -> None:
        self.started = True

    def stop(self):
        return []


@pytest.fixture(autouse=True)
def _no_owner_override(monkeypatch):
    monkeypatch.delenv(PMU_OWNER_ENV, raising=False)


def _start(tmp_path, *probes):
    ctx = RunContext(measurements=list(probes), output_dir=tmp_path)
    ctx.start()
    ctx.stop()
    return ctx


def test_real_probes_declare_pmu_exclusive():
    assert L3PerfMeasurement.pmu_exclusive is True
    assert PerfSpectMeasurement.pmu_exclusive is True
    assert PerfSpectMeasurement.name == "perfspect"


def test_l3_perf_owns_the_pmu_by_default(tmp_path):
    l3, ps, l1 = _Probe("l3_perf"), _Probe("perfspect"), _Probe("l1_subspan", pmu=False)
    ctx = _start(tmp_path, ps, l1, l3)

    assert l3.started and l1.started
    assert not ps.started
    assert ps not in ctx.measurements
    reason = _measurement_skip_reason(ps)
    assert "l3_perf" in reason and "owns the hardware counters" in reason
    assert PMU_OWNER_ENV in reason


def test_env_override_chooses_the_owner(tmp_path, monkeypatch):
    monkeypatch.setenv(PMU_OWNER_ENV, "perfspect")
    l3, ps = _Probe("l3_perf"), _Probe("perfspect")
    _start(tmp_path, l3, ps)

    assert ps.started
    assert not l3.started
    assert "perfspect" in _measurement_skip_reason(l3)


def test_a_probe_that_cannot_collect_never_wins(tmp_path):
    l3, ps = _Probe("l3_perf", able=False), _Probe("perfspect")
    _start(tmp_path, l3, ps)

    assert ps.started
    assert not l3.started
    assert "cannot collect on this host" in _measurement_skip_reason(l3)


def test_an_override_naming_a_probe_that_cannot_collect_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv(PMU_OWNER_ENV, "perfspect")
    l3, ps = _Probe("l3_perf"), _Probe("perfspect", able=False)
    _start(tmp_path, l3, ps)

    assert l3.started
    assert not ps.started


def test_a_single_pmu_probe_is_untouched(tmp_path):
    l3, l1 = _Probe("l3_perf"), _Probe("l1_subspan", pmu=False)
    ctx = _start(tmp_path, l3, l1)

    assert l3.started and l1.started
    assert ctx.measurements == [l3, l1]
    assert not hasattr(l3, "_pmu_election_denied")
