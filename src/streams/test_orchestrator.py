#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

from __future__ import annotations

import queue
import subprocess

import pytest

from src.run_driver import RunConfig, RunSummary
from src.streams import orchestrator
from src.streams.resources import ResourceBudget, TaskSlot
from src.storage.sqlite_store import SQLiteResultStore


def test_selected_tasks_preserves_requested_order(monkeypatch):
    class Task:
        def __init__(self, task_id):
            self.task_id = task_id
            self.cpus = 1

    observed = {}

    def load(**kwargs):
        observed.update(kwargs)
        return [Task("terminal-bench/b"), Task("terminal-bench/a")]

    monkeypatch.setattr(orchestrator, "_load_harbor_tasks_cached", load)
    tasks = orchestrator._selected_tasks(RunConfig(), ["a", "b"])
    assert [task.task_id for task in tasks] == ["terminal-bench/a", "terminal-bench/b"]
    assert observed["task_names"] == ["terminal-bench/a", "terminal-bench/b"]


def test_worker_uses_slot_budget_and_distinct_proxy_port(monkeypatch):
    observed = []
    monkeypatch.setattr(orchestrator.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(orchestrator, "_docker_storage_stats", lambda: {})
    monkeypatch.setattr(
        orchestrator,
        "_cleanup_point_containers",
        lambda _: {"containers_after": 0, "networks_after": 0},
    )
    monkeypatch.setattr(orchestrator, "_record_point_metadata", lambda *_: None)

    def fake_run(cfg):
        observed.append(cfg)
        return RunSummary(
            run_id=cfg.run_id,
            total=1,
            passed=0,
            records=0,
            db_path="db",
            execution_failures=2,
            oracle_failures=1,
        )

    monkeypatch.setattr(orchestrator, "run_benchmark", fake_run)
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    task_queue.put(("terminal-bench/example", 0))
    task_queue.put(None)
    orchestrator._worker(
        orchestrator._WorkerArgs(
            stream_id=3,
            slot=TaskSlot(2, ResourceBudget((10, 11))),
            task_queue=task_queue,
            result_queue=result_queue,
            base_cfg=RunConfig(run_id="point", proxy_port=4100),
        )
    )

    assert observed[0].run_id == "point_s3_example"
    assert observed[0].proxy_port == 4103
    assert observed[0].adapter_kwargs["resource_budget"].cpuset == (10, 11)
    result = result_queue.get_nowait()
    assert result["error"] is None
    assert result["execution_failures"] == 2
    assert result["oracle_failures"] == 1


def test_unpinned_worker_skips_affinity(monkeypatch):
    monkeypatch.setattr(
        orchestrator.os,
        "sched_setaffinity",
        lambda *_: (_ for _ in ()).throw(AssertionError("affinity was applied")),
    )
    monkeypatch.setattr(orchestrator, "_docker_storage_stats", lambda: {})
    monkeypatch.setattr(
        orchestrator,
        "_cleanup_point_containers",
        lambda _: {"containers_after": 0, "networks_after": 0},
    )
    monkeypatch.setattr(orchestrator, "_record_point_metadata", lambda *_: None)
    monkeypatch.setattr(
        orchestrator,
        "run_benchmark",
        lambda cfg: RunSummary(
            run_id=cfg.run_id, total=1, passed=1, records=0, db_path="db"
        ),
    )
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    task_queue.put(("terminal-bench/example", 0))
    task_queue.put(None)

    orchestrator._worker(
        orchestrator._WorkerArgs(
            stream_id=0,
            slot=TaskSlot(4, ResourceBudget(())),
            task_queue=task_queue,
            result_queue=result_queue,
            base_cfg=RunConfig(run_id="point"),
        )
    )
    assert result_queue.get_nowait()["error"] is None


def test_worker_propagates_primary_error_without_cleanup_replacement(monkeypatch):
    monkeypatch.setattr(orchestrator, "_docker_storage_stats", lambda: {})
    monkeypatch.setattr(
        orchestrator,
        "_cleanup_point_containers",
        lambda _: (_ for _ in ()).throw(orchestrator.CleanupError("cleanup")),
    )
    monkeypatch.setattr(orchestrator, "_record_point_metadata", lambda *_: None)
    monkeypatch.setattr(
        orchestrator,
        "run_benchmark",
        lambda _cfg: (_ for _ in ()).throw(ValueError("primary")),
    )
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    task_queue.put(("terminal-bench/example", 0))
    task_queue.put(None)

    with pytest.raises(orchestrator.WorkerError, match="primary"):
        orchestrator._worker(
            orchestrator._WorkerArgs(
                stream_id=0,
                slot=TaskSlot(1, ResourceBudget(())),
                task_queue=task_queue,
                result_queue=result_queue,
                base_cfg=RunConfig(run_id="point"),
            )
        )

    result = result_queue.get_nowait()
    assert result["fatal"] is True
    assert "primary" in result["error"]
    assert "cleanup" in result["cleanup_error"]


def test_worker_cleanup_uses_sanitized_compose_project_name(monkeypatch):
    prefixes = []
    monkeypatch.setattr(orchestrator, "_docker_storage_stats", lambda: {})
    monkeypatch.setattr(
        orchestrator,
        "_cleanup_point_containers",
        lambda prefix: prefixes.append(prefix) or {
            "containers_after": 0,
            "networks_after": 0,
        },
    )
    monkeypatch.setattr(orchestrator, "_record_point_metadata", lambda *_: None)
    monkeypatch.setattr(
        orchestrator,
        "run_benchmark",
        lambda cfg: RunSummary(
            run_id=cfg.run_id, total=1, passed=1, records=0, db_path="db"
        ),
    )
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    task_queue.put(("terminal-bench/example", 0))
    task_queue.put(None)

    orchestrator._worker(
        orchestrator._WorkerArgs(
            stream_id=0,
            slot=TaskSlot(1, ResourceBudget(())),
            task_queue=task_queue,
            result_queue=result_queue,
            base_cfg=RunConfig(run_id="run-abc"),
        )
    )
    assert prefixes == ["run-abc_s0_example"]


def test_admission_rejects_unpinned_slots_above_affinity_cap(monkeypatch):
    monkeypatch.setattr(orchestrator, "_available_affinity_cpus", lambda: 4)
    monkeypatch.setattr(
        orchestrator,
        "_docker_capacity",
        lambda: {"cpus": 8, "memory_mb": 10_000},
    )
    monkeypatch.setattr(orchestrator, "_available_memory_mb", lambda: 10_000)
    tasks = [
        type("Task", (), {"task_id": "a", "cpus": 1, "memory_mb": 512})(),
    ]
    slots = [TaskSlot(1, ResourceBudget(())) for _ in range(5)]
    with pytest.raises(ValueError, match="host-safe active cap"):
        orchestrator._admit_stream_capacity(
            tasks, slots, pin_cores=False, base_port=4001, max_slots=4096
        )


def test_admission_records_unpinned_mode_and_declared_capacity(monkeypatch):
    monkeypatch.setattr(orchestrator, "_available_affinity_cpus", lambda: 8)
    monkeypatch.setattr(
        orchestrator,
        "_docker_capacity",
        lambda: {"cpus": 8, "memory_mb": 10_000},
    )
    monkeypatch.setattr(orchestrator, "_available_memory_mb", lambda: 10_000)
    checked = []
    monkeypatch.setattr(
        orchestrator,
        "_check_proxy_ports",
        lambda base, count: checked.append((base, count)),
    )
    tasks = [
        type("Task", (), {"task_id": "a", "cpus": 1, "memory_mb": 512})(),
    ]
    slots = [TaskSlot(1, ResourceBudget(())) for _ in range(2)]
    report = orchestrator._admit_stream_capacity(
        tasks, slots, pin_cores=False, base_port=4100, max_slots=4096
    )
    assert report["mode"] == "unpinned"
    assert report["intentional_oversubscription"] is True
    assert report["declared_memory_mb"] == 1024
    assert checked == [(4100, 2)]


def test_network_preflight_bounds_large_requests(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator._preflight_docker_network_capacity(4096)

    creates = [call for call in calls if call[:3] == ["docker", "network", "create"]]
    removes = [call for call in calls if call[:3] == ["docker", "network", "rm"]]
    assert len(creates) == 64
    assert len(removes) == 64


def test_point_metadata_records_cleanup_and_storage_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    store = SQLiteResultStore.open()
    store.store_run_metadata(
        run_id="point_task",
        metadata={
            "start_time": 1,
            "benchmark_id": "terminal-bench",
            "status": "complete",
        },
    )
    store.close()

    orchestrator._record_point_metadata(
        "point_task",
        {
            "cleanup": {"containers_removed": 1, "networks_removed": 1},
            "docker_storage_before": {"raw": "before"},
            "docker_storage_after": {"raw": "after"},
        },
    )

    store = SQLiteResultStore.open()
    metadata = store.get_run("point_task")["metadata"]
    store.close()
    assert metadata["cleanup"]["containers_removed"] == 1
    assert metadata["docker_storage_before"]["raw"] == "before"
    assert metadata["docker_storage_after"]["raw"] == "after"


def test_cleanup_only_targets_point_prefix(monkeypatch):
    calls = []
    containers = ["a", "b"]
    networks = ["point_s0_a_default", "unrelated", "point_s1_b_default"]

    class Proc:
        def __init__(self, stdout="", returncode=0, stderr=""):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = stderr

    def fake_run(command, **_):
        calls.append(command)
        if command[:2] == ["docker", "ps"]:
            return Proc(" ".join(containers))
        if command[:3] == ["docker", "network", "ls"]:
            return Proc("\n".join(networks))
        if command[:3] == ["docker", "rm", "-f"]:
            containers.clear()
            return Proc()
        if command[:3] == ["docker", "network", "rm"]:
            networks.remove(command[3])
            return Proc()
        return Proc()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    report = orchestrator._cleanup_point_containers("point", timeout_s=0)
    assert ["docker", "rm", "-f", "a", "b"] in calls
    assert ["docker", "network", "rm", "point_s0_a_default"] in calls
    assert ["docker", "network", "rm", "point_s1_b_default"] in calls
    assert ["docker", "network", "rm", "unrelated"] not in calls
    assert report["containers_after"] == 0
    assert report["networks_after"] == 0


def test_cleanup_retries_transient_docker_failure(monkeypatch):
    attempts = {"rm": 0}

    def fake_run(command, **_):
        if command[:2] == ["docker", "ps"]:
            return type(
                "Proc", (), {"stdout": "container", "returncode": 0, "stderr": ""}
            )()
        if command[:3] == ["docker", "network", "ls"]:
            return type("Proc", (), {"stdout": "", "returncode": 0, "stderr": ""})()
        if command[:3] == ["docker", "rm", "-f"]:
            attempts["rm"] += 1
            return type(
                "Proc",
                (),
                {
                    "stdout": "",
                    "returncode": 1 if attempts["rm"] == 1 else 0,
                    "stderr": "busy" if attempts["rm"] == 1 else "",
                },
            )()
        return type("Proc", (), {"stdout": "", "returncode": 0, "stderr": ""})()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _: None)
    original = orchestrator._list_scoped_containers

    def list_containers(prefix):
        if attempts["rm"] > 1:
            return []
        return original(prefix)

    monkeypatch.setattr(orchestrator, "_list_scoped_containers", list_containers)
    report = orchestrator._cleanup_point_containers(
        "point", timeout_s=1, retry_interval_s=0
    )
    assert attempts["rm"] == 2
    assert report["containers_removed"] == 1


def test_cleanup_fails_closed_when_matching_resource_remains(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_list_scoped_containers",
        lambda _: ["container"],
    )
    monkeypatch.setattr(orchestrator, "_list_scoped_networks", lambda _: [])
    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Proc", (), {"stdout": "", "returncode": 1, "stderr": "busy"}
        )(),
    )
    with pytest.raises(orchestrator.CleanupError, match="remaining containers"):
        orchestrator._cleanup_point_containers("point", timeout_s=0, retry_interval_s=0)
