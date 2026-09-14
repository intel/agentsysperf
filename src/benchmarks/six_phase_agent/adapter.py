#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""SixPhaseAgentAdapter: one agent loop as six measured phase spans.

**Status: experimental / unsupported.**  This adapter is not yet
part of the stable benchmark suite.  APIs, task schema, and phase
workloads may change without notice.  Not recommended for external
publication or production benchmarking until promoted to ``stable``.

Each task is one full agent-loop iteration. ``run_task`` opens a nested
:func:`track_span` per phase (reason → retrieve → act → admit → context
→ commit), so every measurement plugin (L1, L3, future L5/accelerator)
emits per-phase records keyed by ``<task_id>::<phase>``. This is what
lets the analysis attribute latency, IPC, cache, and mem-BW to each
phase independently — the whole point of the 6-phase design.

The adapter needs a :class:`RunContext` to open phase spans, so unlike
the synthetic_cpu reference it is constructed WITH the context. The
outer per-task span is still opened by the driver around ``run_task``
(the load-bearing span convention); the phase spans are nested inside.

``agent_invoker`` is unused — the phase workloads are self-contained CPU
exercisers (see :mod:`.phases`). The real LLM/tool components swap in
behind the phase-function seam without changing span boundaries.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from src.protocols import AgentInvoker, TaskResult, TaskSpec
from src.runner import RunContext, track_span

from .phases import PHASES

logger = logging.getLogger(__name__)


class SixPhaseAgentAdapter:
    """Six agent-loop phases, each a measured span, as a BenchmarkAdapter.

    ``list_tasks`` yields ``n_loops`` identical agent-loop tasks (one
    TaskSpec per loop iteration) so a concurrency sweep can dispatch many
    loops in parallel. ``extra["scale"]`` scales the per-phase work.
    """

    name: str = "six_phase_agent"
    version: str = "0.1.0"
    default_scale: int = 1

    def __init__(self, ctx: RunContext, *, n_loops: int = 1, warmup: bool = True) -> None:
        # The adapter opens phase spans, so it holds the run context.
        self._ctx = ctx
        self._n_loops = n_loops
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Pay one-time init costs (LiteLLM Router setup, tokenizer load)
        before measurement so the first measured loop reflects steady state,
        not cold start. Without this the Router's first completion() eats
        ~2s of model-list/routing-strategy init and pollutes the admit span.
        """
        from .phases import admit, context, retrieve, reason
        try:
            reason(scale=1)    # loads the shared llama.cpp model
            admit(scale=1)     # builds + warms the shared Router
            retrieve(scale=1)  # builds the shared FAISS index + loads embed model
            context(scale=1)   # loads the tokenizer
        except Exception:
            logger.warning("warmup failed; first loop may include init cost",
                           exc_info=True)

    # ── BenchmarkAdapter Protocol ─────────────────────────────────────

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        emitted = 0
        for i in range(self._n_loops):
            task_id = f"loop-{i:04d}"
            if include is not None and task_id not in include:
                continue
            if exclude is not None and task_id in exclude:
                continue
            yield TaskSpec(
                id=task_id,
                instruction="one full agent loop: reason→retrieve→act→admit→context→commit",
                category="six_phase_agent",
                difficulty="deterministic",
                cpu_budget=1,
                memory_mb=1024,
                timeout_s=120.0,
                extra={"scale": self.default_scale},
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
        """Run one agent loop; open a nested span per phase."""
        del agent_invoker  # phase workloads are self-contained
        scale = int(task.extra.get("scale", self.default_scale))

        phase_results: Dict[str, Any] = {}
        phase_wall_ms: Dict[str, float] = {}
        t_loop = time.monotonic()

        for phase_name, func in PHASES.items():
            span_id = f"{task.id}::{phase_name}"
            if on_step is not None:
                on_step({"phase": phase_name, "task": task.id, "event": "begin"})
            t0 = time.monotonic()
            with track_span(
                self._ctx, span_id, kind="phase", node_id=phase_name, phase=phase_name,
            ):
                try:
                    phase_results[phase_name] = func(scale)
                except Exception as exc:
                    logger.exception("phase %r raised in task %r", phase_name, task.id)
                    phase_results[phase_name] = {"error": f"{type(exc).__name__}: {exc}"}
            phase_wall_ms[phase_name] = (time.monotonic() - t0) * 1000.0

        loop_ms = (time.monotonic() - t_loop) * 1000.0

        # passed iff every phase returned a dict with no 'error' key.
        passed = all(
            isinstance(r, dict) and "error" not in r for r in phase_results.values()
        )
        return TaskResult(
            task_id=task.id,
            passed=passed,
            reward=1.0 if passed else 0.0,
            extra={
                "loop_ms": loop_ms,
                "phase_wall_ms": phase_wall_ms,
                "phase_results": phase_results,
                "scale": scale,
            },
        )

    def teardown(self) -> None:
        return None


__all__ = ["SixPhaseAgentAdapter"]
