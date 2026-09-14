#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Run Terminal-Bench tasks through task-sized runc workers."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import socket
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.benchmarks.terminal_bench.harbor_environment import (
    sanitize_compose_project_name,
)
from src.run_driver import RunConfig, _load_harbor_tasks_cached, run_benchmark
from src.streams.resources import (
    TaskSlot,
    plan_task_slots,
    plan_unpinned_task_slots,
)

logger = logging.getLogger(__name__)
_DOCKER_NETWORK_PROBE_LIMIT = 64


def _short_name(name: str) -> str:
    return name.split("/")[-1]


class WorkerError(RuntimeError):
    """A task worker failed before it could produce a normal benchmark result."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        traceback_text: str | None = None,
        cleanup_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.traceback_text = traceback_text
        self.cleanup_error = cleanup_error


class WorkerSupervisionError(WorkerError):
    """The parent could not observe all workers within the supervision bound."""


def _available_affinity_cpus() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise RuntimeError("cannot determine the process CPU affinity") from exc
    if cpus <= 0:
        raise RuntimeError("the process CPU affinity mask is empty")
    return cpus


def _available_memory_mb() -> int:
    """Return MemAvailable, failing closed when the host does not expose it."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("cannot determine available host memory") from exc
    raise RuntimeError("MemAvailable is not exposed by this host")


def _docker_capacity() -> dict[str, int]:
    """Read Docker's CPU and memory capacity before creating any resources."""
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Docker capacity check could not run") from exc
    if proc.returncode:
        raise RuntimeError(
            f"Docker capacity check failed: {proc.stderr.strip() or proc.returncode}"
        )
    try:
        info = json.loads(proc.stdout)
        cpus = int(info["NCPU"])
        memory_mb = int(info["MemTotal"]) // (1024 * 1024)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Docker capacity output was not valid") from exc
    if cpus <= 0 or memory_mb <= 0:
        raise RuntimeError(f"Docker reported invalid capacity: cpus={cpus}, memory_mb={memory_mb}")
    return {"cpus": cpus, "memory_mb": memory_mb}


def _check_proxy_ports(base_port: int, count: int) -> None:
    """Fail closed if the worker proxy ports are out of range or occupied."""
    if not 1 <= base_port <= 65535 or base_port + count - 1 > 65535:
        raise ValueError(
            f"proxy port range {base_port}-{base_port + count - 1} is invalid"
        )
    for port in range(base_port, base_port + count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"proxy port {port} is unavailable") from exc
        finally:
            sock.close()


def _declared_memory_demand_mb(tasks: Sequence[Any], slots: Sequence[TaskSlot]) -> int:
    memory_by_size: dict[int, int] = {}
    for task in tasks:
        size = int(getattr(task, "cpus", 1))
        memory = int(getattr(task, "memory_mb", 0))
        if memory <= 0:
            raise ValueError(f"task {task.task_id!r} has invalid memory budget {memory}")
        memory_by_size[size] = max(memory_by_size.get(size, 0), memory)
    try:
        return sum(memory_by_size[slot.size] for slot in slots)
    except KeyError as exc:
        raise ValueError(f"no memory budget was declared for task size {exc.args[0]}") from exc


def _admit_stream_capacity(
    tasks: Sequence[Any],
    slots: Sequence[TaskSlot],
    *,
    pin_cores: bool,
    base_port: int,
    max_slots: int,
) -> dict[str, Any]:
    """Validate every host resource owned by one live worker slot."""
    if not slots:
        raise ValueError("stream planning produced no worker slots")
    if len(slots) > max_slots:
        raise ValueError(
            f"planned slot count {len(slots)} exceeds max_slots={max_slots}"
        )

    affinity_cpus = _available_affinity_cpus()
    docker = _docker_capacity()
    if len(slots) > affinity_cpus and not pin_cores:
        raise ValueError(
            f"requested {len(slots)} unpinned slots exceeds the "
            f"host-safe active cap of {affinity_cpus} affinity CPUs; "
            "higher oversubscription is not enabled"
        )
    if pin_cores and sum(len(slot.budget.cpuset) for slot in slots) > affinity_cpus:
        raise ValueError(
            f"pinned stream slots require more than {affinity_cpus} affinity CPUs"
        )
    if len(slots) > docker["cpus"]:
        raise ValueError(
            f"requested {len(slots)} worker slots exceeds Docker capacity "
            f"of {docker['cpus']} CPUs"
        )

    declared_memory_mb = _declared_memory_demand_mb(tasks, slots)
    memory_capacity_mb = min(_available_memory_mb(), docker["memory_mb"])
    memory_limit_mb = int(memory_capacity_mb * 0.8)
    if declared_memory_mb > memory_limit_mb:
        raise ValueError(
            f"declared live container memory {declared_memory_mb} MiB exceeds "
            f"the admission limit of {memory_limit_mb} MiB"
        )

    _check_proxy_ports(base_port, len(slots))
    return {
        "mode": "pinned" if pin_cores else "unpinned",
        "cpuset_applied": pin_cores,
        "cpu_sizes_are_scheduling_classes": not pin_cores,
        "intentional_oversubscription": not pin_cores,
        "requested_slots": len(slots),
        "affinity_cpus": affinity_cpus,
        "docker_cpus": docker["cpus"],
        "declared_memory_mb": declared_memory_mb,
        "admission_memory_mb": memory_limit_mb,
        "proxy_port_start": base_port,
        "proxy_port_end": base_port + len(slots) - 1,
    }


def _preflight_docker_network_capacity(slot_count: int) -> None:
    """Verify Docker can create one bridge network per concurrent worker."""
    probe_count = min(slot_count, _DOCKER_NETWORK_PROBE_LIMIT)
    if probe_count < slot_count:
        logger.warning(
            "Bounding Docker network capacity preflight to %d probes for "
            "%d requested worker slots",
            probe_count,
            slot_count,
        )
    prefix = f"_agentsysperf_netprobe_{os.getpid()}_"
    created: list[str] = []
    primary_error: BaseException | None = None
    try:
        for index in range(probe_count):
            name = f"{prefix}{index}"
            try:
                proc = subprocess.run(
                    ["docker", "network", "create", name],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=30,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("docker is required for task-sized streams") from exc
            if proc.returncode:
                raise RuntimeError(
                    f"Docker could create only {len(created)}/{probe_count} "
                    f"network probes for {slot_count} requested stream "
                    f"slots: {proc.stderr.strip()}"
                )
            created.append(name)
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_errors: list[str] = []
        for name in created:
            try:
                proc = subprocess.run(
                    ["docker", "network", "rm", name],
                    capture_output=True,
                    check=False,
                    text=True,
                )
                if proc.returncode:
                    cleanup_errors.append(f"{name}: {proc.stderr.strip()}")
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if cleanup_errors and primary_error is None:
            primary_error = CleanupError(
                "Docker network capacity probe cleanup failed: "
                + "; ".join(cleanup_errors)
            )
    if primary_error is not None:
        raise primary_error


class CleanupError(RuntimeError):
    """Raised when scoped Docker cleanup cannot reach a verified zero state."""


def _worker_error_from_result(result: Mapping[str, Any]) -> WorkerError:
    """Reconstruct the primary worker failure without losing its traceback."""
    error = result.get("error") or "worker failed without an error message"
    if result.get("error_type") == "CleanupError" and not result.get("cleanup_error"):
        return CleanupError(error)
    return WorkerError(
        error,
        run_id=result.get("run_id"),
        traceback_text=result.get("traceback"),
        cleanup_error=result.get("cleanup_error"),
    )


def _list_scoped_containers(run_id_prefix: str) -> list[str]:
    proc = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"name=^{re.escape(run_id_prefix)}_",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if getattr(proc, "returncode", 0):
        raise CleanupError(
            f"docker container listing failed: {getattr(proc, 'stderr', '').strip()}"
        )
    return proc.stdout.split()


def _list_scoped_networks(run_id_prefix: str) -> list[str]:
    proc = subprocess.run(
        ["docker", "network", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if getattr(proc, "returncode", 0):
        raise CleanupError(
            f"docker network listing failed: {getattr(proc, 'stderr', '').strip()}"
        )
    return [
        name
        for name in proc.stdout.splitlines()
        if name.startswith(f"{run_id_prefix}_")
    ]


def _docker_storage_stats() -> dict[str, Any]:
    """Capture Docker storage usage without deleting any cached images."""
    try:
        proc = subprocess.run(
            ["docker", "system", "df", "--format", "{{json .}}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode:
        return {"error": proc.stderr.strip() or f"exit {proc.returncode}"}
    return {"raw": proc.stdout.strip()}


def _cleanup_point_containers(
    run_id_prefix: str,
    *,
    timeout_s: float = 120.0,
    retry_interval_s: float = 0.25,
) -> dict[str, Any]:
    """Remove and verify only resources belonging to one run prefix.

    Docker may report a stopped container or attached network briefly after
    Harbor returns. Re-listing after every removal attempt turns teardown into
    a barrier and makes transient Docker races visible to the caller.
    """
    if not run_id_prefix:
        raise ValueError("run_id_prefix must be non-empty")

    deadline = time.monotonic() + max(0.0, timeout_s)
    attempts = 0
    first_containers: list[str] | None = None
    first_networks: list[str] | None = None
    last_error: Exception | None = None
    removed_containers: set[str] = set()
    removed_networks: set[str] = set()

    while True:
        attempts += 1
        try:
            containers = _list_scoped_containers(run_id_prefix)
            networks = _list_scoped_networks(run_id_prefix)
            if first_containers is None:
                first_containers = list(containers)
                first_networks = list(networks)

            if not containers and not networks:
                return {
                    "containers_before": len(first_containers or ()),
                    "networks_before": len(first_networks or ()),
                    "containers_removed": len(removed_containers),
                    "networks_removed": len(removed_networks),
                    "containers_after": 0,
                    "networks_after": 0,
                    "attempts": attempts,
                }

            if containers:
                proc = subprocess.run(
                    ["docker", "rm", "-f", *containers],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                if getattr(proc, "returncode", 0) == 0:
                    removed_containers.update(containers)
                else:
                    last_error = CleanupError(
                        "docker container removal failed: "
                        f"{getattr(proc, 'stderr', '').strip()}"
                    )

            for network in networks:
                proc = subprocess.run(
                    ["docker", "network", "rm", network],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                if getattr(proc, "returncode", 0) == 0:
                    removed_networks.add(network)
                else:
                    last_error = CleanupError(
                        f"docker network removal failed for {network}: "
                        f"{getattr(proc, 'stderr', '').strip()}"
                    )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if time.monotonic() >= deadline:
            try:
                remaining_containers = _list_scoped_containers(run_id_prefix)
                remaining_networks = _list_scoped_networks(run_id_prefix)
            except Exception as exc:  # noqa: BLE001
                remaining_containers = ["<listing failed>"]
                remaining_networks = ["<listing failed>"]
                last_error = exc
            if not remaining_containers and not remaining_networks:
                return {
                    "containers_before": len(first_containers or ()),
                    "networks_before": len(first_networks or ()),
                    "containers_removed": len(removed_containers),
                    "networks_removed": len(removed_networks),
                    "containers_after": 0,
                    "networks_after": 0,
                    "attempts": attempts,
                }
            raise CleanupError(
                f"scoped Docker cleanup failed for {run_id_prefix!r}; "
                f"remaining containers={remaining_containers}, "
                f"networks={remaining_networks}; last error={last_error}"
            ) from last_error

        time.sleep(min(retry_interval_s, max(0.0, deadline - time.monotonic())))


def _selected_tasks(base_cfg: RunConfig, task_names: Sequence[str]) -> list[Any]:
    requested = [_short_name(name) for name in task_names]
    if not requested:
        raise ValueError("task_names must be non-empty")

    resolved = list(
        _load_harbor_tasks_cached(
            dataset_name=base_cfg.dataset_name,
            ref=base_cfg.dataset_ref,
            task_names=[f"terminal-bench/{name}" for name in requested],
            limit=None,
        )
    )
    by_name = {_short_name(task.task_id): task for task in resolved}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Harbor did not resolve requested task(s): {missing}")
    return [by_name[name] for name in requested]


def _task_histogram(tasks: Iterable[Any]) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for task in tasks:
        size = int(getattr(task, "cpus", 1))
        if size <= 0:
            raise ValueError(f"task {task.task_id!r} has invalid CPU budget {size}")
        histogram[size] = histogram.get(size, 0) + 1
    return histogram


def _ensure_task_images(tasks: Iterable[Any]) -> None:
    """Require all selected task images before timing starts."""
    from src.benchmarks.terminal_bench.prebuild import ensure_task_images

    ensure_task_images(tasks)


@dataclass(frozen=True)
class _WorkerArgs:
    stream_id: int
    slot: TaskSlot
    task_queue: Any
    result_queue: Any
    base_cfg: RunConfig
    cleanup_lock: Any = None


def _worker(args: _WorkerArgs) -> None:
    if args.slot.budget.cpuset:
        os.sched_setaffinity(0, set(args.slot.budget.cpuset))
    while True:
        item = args.task_queue.get()
        if item is None:
            return
        name, repeat_index = item
        short = _short_name(name)
        suffix = f"_r{repeat_index}" if repeat_index else ""
        cfg = dataclasses.replace(
            args.base_cfg,
            tasks=[name],
            run_id=f"{args.base_cfg.run_id}_s{args.stream_id}_{short}{suffix}",
            adapter_kwargs={
                **(args.base_cfg.adapter_kwargs or {}),
                "resource_budget": args.slot.budget,
            },
            proxy_port=(args.base_cfg.proxy_port or 4001) + args.stream_id,
        )
        summary = None
        primary_error: WorkerError | None = None
        cleanup_report: dict[str, Any] | None = None
        cleanup_error: Exception | None = None
        storage_before = _docker_storage_stats()
        try:
            summary = run_benchmark(cfg)
        except Exception as exc:  # noqa: BLE001
            primary_error = WorkerError(
                f"worker task {cfg.run_id} failed: {type(exc).__name__}: {exc}",
                run_id=cfg.run_id,
                traceback_text=traceback.format_exc(),
            )
        finally:
            cleanup_prefix = sanitize_compose_project_name(cfg.run_id)
            try:
                if args.cleanup_lock is None:
                    cleanup_report = _cleanup_point_containers(cleanup_prefix)
                else:
                    with args.cleanup_lock:
                        cleanup_report = _cleanup_point_containers(cleanup_prefix)
            except Exception as exc:  # noqa: BLE001
                cleanup_error = exc
            storage_after = _docker_storage_stats()

        if primary_error is not None and cleanup_error is not None:
            primary_error.cleanup_error = (
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        error = primary_error or cleanup_error
        result = {
            "stream_id": args.stream_id,
            "task": name,
            "task_size": args.slot.size,
            "repeat": repeat_index,
            "run_id": summary.run_id if summary is not None else cfg.run_id,
            "passed": summary.passed if summary is not None else 0,
            "total": summary.total if summary is not None else 0,
            "execution_failures": (
                summary.execution_failures if summary is not None else 0
            ),
            "oracle_failures": (
                summary.oracle_failures if summary is not None else 0
            ),
            "error": f"{type(error).__name__}: {error}" if error else None,
            "cleanup": cleanup_report,
            "docker_storage_before": storage_before,
            "docker_storage_after": storage_after,
            "fatal": primary_error is not None or cleanup_error is not None,
            "error_type": type(error).__name__ if error else None,
            "traceback": primary_error.traceback_text if primary_error else None,
            "cleanup_error": (
                f"{type(cleanup_error).__name__}: {cleanup_error}"
                if cleanup_error and primary_error
                else None
            ),
        }
        args.result_queue.put(result)
        if summary is not None:
            _record_point_metadata(
                summary.run_id,
                {
                    "stream_id": args.stream_id,
                    "task_size": args.slot.size,
                    "cleanup": cleanup_report,
                    "docker_storage_before": storage_before,
                    "docker_storage_after": storage_after,
                },
            )
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error


def _record_point_metadata(run_id: str, point_metadata: Mapping[str, Any]) -> None:
    """Merge per-run Docker lifecycle evidence into the canonical store."""
    store = None
    try:
        from src.storage.sqlite_store import SQLiteResultStore

        store = SQLiteResultStore.open()
        current = store.get_run(run_id) or {}
        known = (
            "start_time",
            "end_time",
            "hardware_sku",
            "model",
            "total_tasks",
            "passed_tasks",
            "benchmark_id",
            "optimization_profile",
            "numa_policy",
            "agentsysperf_version",
            "result_digest",
            "owner_id",
            "owner_kind",
            "host_id",
            "status",
        )
        metadata = dict(current.get("metadata") or {})
        metadata.update(point_metadata)
        store.store_run_metadata(
            run_id=run_id,
            metadata={
                **{key: current[key] for key in known if key in current},
                **metadata,
            },
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "failed to record Docker lifecycle metadata for %s", run_id, exc_info=True
        )
    finally:
        if store is not None:
            store.close()


def _shutdown_workers(processes: Sequence[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is None:
            continue
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def run_task_sized_streams(
    base_cfg: RunConfig,
    task_names: Sequence[str],
    *,
    slots: int | None = None,
    total_cores: int | None = None,
    pin_cores: bool = False,
    max_slots: int = 4096,
    stream_multiple: float = 2.0,
    launch_stagger_s: float = 0.0,
    supervision_timeout_s: float | None = None,
    on_task_done: Callable[[dict[str, Any], int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run a task mix on matching-size runc worker slots.

    The selected Terminal-Bench tasks are resolved and image-prepared in the
    parent. Workers only receive slots that exactly match their queue's task
    CPU budget. Unpinned scheduling is the default and permits oversubscription;
    ``pin_cores=True`` retains the NUMA-local CPU-set behavior.
    """
    if base_cfg.benchmark not in ("terminal-bench", "terminal_bench"):
        raise ValueError("task-sized streams support terminal-bench only")
    if stream_multiple <= 0:
        raise ValueError(f"stream_multiple must be positive, got {stream_multiple}")
    if max_slots <= 0:
        raise ValueError(f"max_slots must be positive, got {max_slots}")
    if slots is not None and total_cores is not None:
        raise ValueError("slots and total_cores are mutually exclusive")
    if pin_cores and slots is not None:
        raise ValueError("slots cannot be combined with pin_cores")
    if pin_cores and total_cores is None:
        raise ValueError("pin_cores requires total_cores")
    requested = total_cores if slots is None else slots
    if requested is None:
        raise ValueError("one of slots or total_cores is required")
    if requested <= 0:
        raise ValueError(f"slot count must be positive, got {requested}")

    tasks = _selected_tasks(base_cfg, task_names)
    if pin_cores:
        planned_slots = plan_task_slots(requested, _task_histogram(tasks))
    else:
        planned_slots = plan_unpinned_task_slots(
            requested,
            _task_histogram(tasks),
            max_slots=max_slots,
        )
    admission = _admit_stream_capacity(
        tasks,
        planned_slots,
        pin_cores=pin_cores,
        base_port=base_cfg.proxy_port or 4001,
        max_slots=max_slots,
    )
    _preflight_docker_network_capacity(len(planned_slots))
    _ensure_task_images(tasks)

    base_run_id = base_cfg.run_id or f"task_sized_streams_{int(time.time())}"
    base_cfg = dataclasses.replace(
        base_cfg,
        run_id=base_run_id,
        preresolved_tasks=tasks,
        adapter_kwargs={**(base_cfg.adapter_kwargs or {}), "force_build": False},
        stream_metadata=admission,
    )

    repeats = max(1, math.ceil(stream_multiple * len(planned_slots) / len(tasks)))
    items = [
        (name, repeat_index) for repeat_index in range(repeats) for name in task_names
    ]
    task_size = {_short_name(task.task_id): int(task.cpus) for task in tasks}

    mp_context = mp.get_context("spawn")
    queues = {
        size: mp_context.Queue() for size in {slot.size for slot in planned_slots}
    }
    slots_by_size: dict[int, list[TaskSlot]] = {}
    for slot in planned_slots:
        slots_by_size.setdefault(slot.size, []).append(slot)
    for name, repeat_index in items:
        queues[task_size[_short_name(name)]].put((name, repeat_index))
    for size, size_slots in slots_by_size.items():
        for _ in size_slots:
            queues[size].put(None)

    from src.storage.sqlite_store import SQLiteResultStore

    SQLiteResultStore.open().close()
    results: list[dict[str, Any]] = []
    result_queue = mp_context.Queue()
    cleanup_lock = mp_context.Lock()
    processes: list[Any] = []
    stream_id = 0
    for size, size_slots in sorted(slots_by_size.items()):
        for slot in size_slots:
            processes.append(
                mp_context.Process(
                    target=_worker,
                    args=(
                        _WorkerArgs(
                            stream_id,
                            slot,
                            queues[size],
                            result_queue,
                            base_cfg,
                            cleanup_lock,
                        ),
                    ),
                )
            )
            stream_id += 1

    primary_exception: BaseException | None = None
    try:
        for index, process in enumerate(processes):
            process.start()
            if launch_stagger_s and index < len(processes) - 1:
                time.sleep(launch_stagger_s)
        completed = 0
        idle_after_exit = 0
        supervision_deadline = time.monotonic() + (
            supervision_timeout_s
            if supervision_timeout_s is not None
            else max(
                60.0,
                base_cfg.timeout_s
                * math.ceil(len(items) / max(1, len(processes)))
                + 120.0,
            )
        )
        while completed < len(items):
            if time.monotonic() >= supervision_deadline:
                raise WorkerSupervisionError(
                    "stream supervision deadline exceeded "
                    f"after {completed}/{len(items)} task results"
                )
            try:
                result = result_queue.get(timeout=0.5)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    idle_after_exit += 1
                    if idle_after_exit >= 4:
                        raise WorkerSupervisionError(
                            f"all stream workers exited after "
                            f"{completed}/{len(items)} task results"
                        )
                continue
            idle_after_exit = 0
            results.append(result)
            completed += 1
            if on_task_done is not None:
                on_task_done(result, completed, len(items))
            if result.get("fatal"):
                raise _worker_error_from_result(result)
        if completed != len(items):
            raise WorkerSupervisionError(
                f"stream workers produced {completed}/{len(items)} task results"
            )
        for process in processes:
            process.join(timeout=10)
        failed_workers = [
            process for process in processes if process.exitcode not in (None, 0)
        ]
        if failed_workers:
            raise WorkerError(
                "stream worker failed: "
                + ", ".join(
                    f"pid={process.pid} exitcode={process.exitcode}"
                    for process in failed_workers
                )
            )
    except BaseException as exc:
        primary_exception = exc
        raise
    finally:
        _shutdown_workers(processes)
        try:
            _cleanup_point_containers(sanitize_compose_project_name(base_run_id))
        except Exception:
            if primary_exception is None:
                raise
            logger.error(
                "stream cleanup failed after primary error; preserving primary error",
                exc_info=True,
            )
    return results


__all__ = [
    "CleanupError",
    "WorkerError",
    "WorkerSupervisionError",
    "run_task_sized_streams",
]
