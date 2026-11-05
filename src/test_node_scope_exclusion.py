#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Node-wide layers are excluded from concurrent ``run-streams`` slots.

The defect: with ``--slots N`` (N > 1) every worker ran ``run_benchmark`` with
every discovered layer, so N ``perf stat -a`` sessions, N perfspect collectors
and N ``/proc/stat`` samplers each stored the *node's* activity -- all N slots'
work -- as their own task's row, and the L3 analyzers published per-task
verdicts (``memory_bound``, ``dram_bound``) from it. Reproduced with 3 slots on
a 288-CPU host: the same ``l1_system`` node sample appeared under two tasks.

Invariants pinned here:

1. Excluded probes are never registered, so they never start a collector.
2. They still read as silent *with a reason*, and the run says which ones.
3. ``run-streams`` sets the flag exactly when more than one slot shares the node.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.measurements.l1_system import L1SystemMeasurement
from src.measurements.l3_perf import L3PerfMeasurement
from src.measurements.perfspect import PerfSpectMeasurement
from src.run_driver import (
    RunConfig,
    _compute_coverage,
    _split_node_scoped,
    run_benchmark,
)
from src.storage.sqlite_store import SQLiteResultStore
from src.streams import orchestrator


class _Probe:
    def __init__(self, layer: str, node_scoped: bool) -> None:
        self.layer = layer
        self.node_scoped = node_scoped


def test_node_wide_probes_declare_it_and_the_process_probe_does_not():
    assert L1SystemMeasurement.node_scoped is True
    assert L3PerfMeasurement.node_scoped is True
    assert PerfSpectMeasurement.node_scoped is True
    assert getattr(L1SubSpanMeasurement, "node_scoped", False) is False


def test_split_is_a_no_op_unless_requested():
    probes = {"l1_subspan": _Probe("l1", False), "l3_perf": _Probe("l3", True)}
    kept, excluded = _split_node_scoped(probes, RunConfig())
    assert kept == list(probes.values())
    assert excluded == []
    assert not hasattr(probes["l3_perf"], "_node_scope_excluded")


def test_split_drops_node_scoped_and_coverage_says_why():
    probes = {
        "l1_subspan": _Probe("l1", False),
        "l1_system": _Probe("l1_system", True),
        "l3_perf": _Probe("l3", True),
    }
    kept, excluded = _split_node_scoped(probes, RunConfig(exclude_node_scoped=True))

    assert kept == [probes["l1_subspan"]]
    assert excluded == ["l1_system", "l3_perf"]

    coverage = _compute_coverage(
        measurements=probes, analyzers={}, records=[], verdicts=[],
        analyzer_errors={},
    )
    reasons = dict(coverage.measurements_silent)
    assert "concurrent runs share this node" in reasons["l3_perf"]
    assert "concurrent runs share this node" in reasons["l1_system"]


def test_run_benchmark_never_registers_an_excluded_layer(tmp_path, monkeypatch):
    """End to end: no l1_system rows, and the stored run names the exclusion."""
    from src.benchmarks.synthetic_cpu.adapter import SyntheticCpuAdapter

    monkeypatch.setattr(SyntheticCpuAdapter, "default_duration_s", 0.2)
    monkeypatch.setattr(
        "src.run_driver.discover_measurements",
        lambda: {"l1_subspan": L1SubSpanMeasurement(),
                 "l1_system": L1SystemMeasurement()},
    )
    store = SQLiteResultStore(tmp_path)
    cfg = RunConfig(benchmark="synthetic_cpu", tasks=["control"],
                    run_id="excluded", exclude_node_scoped=True)

    summary = run_benchmark(cfg, store=store, stamp=1)

    assert store.query_measurements(summary.run_id, layer="l1")
    assert store.query_measurements(summary.run_id, layer="l1_system") == []
    meta = store.get_run(summary.run_id)["metadata"]
    assert meta["node_scoped_layers_excluded"] == ["l1_system"]
    assert [n for n, _ in summary.coverage.measurements_silent] == ["l1_system"]


class _Captured(Exception):
    pass


@pytest.mark.parametrize("slots,expected", [(1, False), (3, True)])
def test_run_streams_excludes_node_layers_only_when_slots_share_the_node(
    tmp_path, monkeypatch, slots, expected,
):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    tasks = [SimpleNamespace(task_id=f"terminal-bench/t{i}", cpus=1) for i in range(3)]
    monkeypatch.setattr(orchestrator, "_selected_tasks", lambda *_: tasks)
    monkeypatch.setattr(orchestrator, "_admit_stream_capacity", lambda *a, **k: {})
    monkeypatch.setattr(orchestrator, "_preflight_docker_network_capacity", lambda *_: None)
    monkeypatch.setattr(orchestrator, "_ensure_task_images", lambda *_: None)
    seen = []

    def capture(stream_id, slot, task_queue, result_queue, base_cfg, lock):
        seen.append(base_cfg)
        raise _Captured

    monkeypatch.setattr(orchestrator, "_WorkerArgs", capture)

    with pytest.raises(_Captured):
        orchestrator.run_task_sized_streams(
            RunConfig(benchmark="terminal-bench"),
            [t.task_id for t in tasks],
            slots=slots,
        )
    assert seen[0].exclude_node_scoped is expected
