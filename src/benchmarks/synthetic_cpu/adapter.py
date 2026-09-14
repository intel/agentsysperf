#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""SyntheticCpuAdapter: 9 CPU-signature workloads as a :class:`BenchmarkAdapter`.

Each task is a deterministic CPU exerciser targeting one
microarchitectural axis (branch mispredict rate, FP throughput, cache
miss rate, page faults, etc.). The workloads themselves live in
``harness/scripts/synthetic_tasks.py`` and are imported as a sibling
module — duplicating them here would invite drift.

Composition with measurement plugins is the whole point of this
adapter: opening a span around ``run_task`` lets L3PerfMeasurement,
future L2 py-spy, and L5 PCM/RAPL plugins each emit per-task records
keyed by the same ``span_id``. A driver under ``examples/`` shows the
full composition.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from src.protocols import (
    AgentInvoker,
    TaskResult,
    TaskSpec,
)

logger = logging.getLogger(__name__)


# Resolved at module import: path to harness/scripts/synthetic_tasks.py.
# We compute this from the agentsysperf package location so it works when
# the package is installed editable; for a future split-repo install
# the harness module will move under src/benchmarks/synthetic_cpu/
# and this resolution becomes unnecessary.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SYNTHETIC_SCRIPT = _REPO_ROOT / "harness" / "scripts" / "synthetic_tasks.py"


def _load_workloads() -> Dict[str, Tuple[str, Callable[[float], Dict[str, Any]]]]:
    """Import ``harness/scripts/synthetic_tasks.py`` and return its TASKS dict.

    The module is loaded by file path because ``harness/`` is not a
    Python package (no ``__init__.py``) and isn't on ``sys.path`` by
    default. Cached after first call.
    """
    if not _SYNTHETIC_SCRIPT.exists():
        raise FileNotFoundError(
            f"SyntheticCpuAdapter: workload script not found at "
            f"{_SYNTHETIC_SCRIPT}. Has the harness/ tree moved?"
        )
    spec = importlib.util.spec_from_file_location(
        "_agentsysperf_synthetic_tasks", _SYNTHETIC_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {_SYNTHETIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, "TASKS")


class SyntheticCpuAdapter:
    """Nine CPU-signature workloads exposed as a :class:`BenchmarkAdapter`.

    ``list_tasks`` yields one :class:`TaskSpec` per workload. Each task's
    ``id`` is the short workload name (``compile``, ``linalg``, ...);
    ``category`` is ``"synthetic_cpu"``; ``instruction`` carries the
    one-line description. ``extra["duration_s"]`` (default 3.0) is how
    long the workload exerciser runs — overridable via ``TaskSpec.extra``
    so callers can run shorter for smoke or longer for measurement.

    ``run_task`` ignores ``agent_invoker``: these workloads do not invoke
    an agent. The parameter is kept to satisfy the Protocol so the same
    runner code can drive both agent benchmarks and synthetic ones.

    The dead-parameter pattern only works because the per-task
    measurement span is opened around ``run_task``, not around
    ``agent_invoker.invoke``. See ``BenchmarkAdapter.run_task`` for the
    span convention; in particular, do not "fix" this by moving span
    bracketing onto ``invoke`` — that would silently emit zero
    measurement records for synthetic_cpu and any future non-agent
    adapter.
    """

    name: str = "synthetic_cpu"
    version: str = "0.1.0"
    default_duration_s: float = 3.0

    def __init__(self) -> None:
        self._tasks_cache: Optional[
            Dict[str, Tuple[str, Callable[[float], Dict[str, Any]]]]
        ] = None

    # ── BenchmarkAdapter Protocol ─────────────────────────────────────

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        tasks = self._tasks()
        include_set = set(include) if include else None
        exclude_set = set(exclude) if exclude else set()

        emitted = 0
        for name, (description, _func) in tasks.items():
            if include_set is not None and name not in include_set:
                continue
            if name in exclude_set:
                continue
            yield TaskSpec(
                id=name,
                instruction=description,
                category="synthetic_cpu",
                difficulty="deterministic",
                cpu_budget=1,
                memory_mb=512,
                timeout_s=60.0,
                extra={"duration_s": self.default_duration_s},
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def run_task(
        self,
        task: TaskSpec,
        *,
        agent_invoker: AgentInvoker,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TaskResult:
        """Run one synthetic workload to completion, return its TaskResult.

        ``agent_invoker`` is unused — synthetic CPU tasks have no agent
        loop. ``on_step`` is called once with a ``{"phase": "begin"}``
        notification and once with ``{"phase": "end", ...}`` carrying
        the workload's own result dict, so a runner observing progress
        gets the same hook shape it would for a real benchmark.
        """
        del agent_invoker  # protocol arg; not applicable to this adapter
        tasks = self._tasks()
        if task.id not in tasks:
            return TaskResult(
                task_id=task.id, passed=False, reward=0.0,
                error=f"unknown synthetic_cpu task id: {task.id!r}",
            )
        description, func = tasks[task.id]
        duration_s = float(task.extra.get("duration_s", self.default_duration_s))

        if on_step is not None:
            on_step({"phase": "begin", "task": task.id, "duration_s": duration_s})

        t0 = time.monotonic()
        try:
            workload_result = func(duration_s)
        except Exception as exc:  # workload-internal failures shouldn't kill the run
            logger.exception("synthetic_cpu task %r raised", task.id)
            return TaskResult(
                task_id=task.id, passed=False, reward=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = time.monotonic() - t0

        if on_step is not None:
            on_step({"phase": "end", "task": task.id, "elapsed_s": elapsed,
                     "result": workload_result})

        # passed=True iff the workload produced its expected output shape.
        # All TASKS callables return a dict with at least 'iterations'
        # (a count of how many full passes ran in the budget); use that
        # as the basic sanity check.
        passed = isinstance(workload_result, dict) and workload_result.get("iterations", 0) > 0
        return TaskResult(
            task_id=task.id,
            passed=passed,
            reward=1.0 if passed else 0.0,
            extra={
                "description": description,
                "elapsed_s": elapsed,
                "duration_s_requested": duration_s,
                "workload_result": workload_result,
            },
        )

    def teardown(self) -> None:
        """No persistent resources to release."""
        return None

    # ── Internals ─────────────────────────────────────────────────────

    def _tasks(self) -> Dict[str, Tuple[str, Callable[[float], Dict[str, Any]]]]:
        if self._tasks_cache is None:
            self._tasks_cache = _load_workloads()
        return self._tasks_cache


__all__ = ["SyntheticCpuAdapter"]
