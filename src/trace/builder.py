#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Build :class:`StepTrace` rows from agent-loop turn records.

The LiteLLM terminal agent loop records a :class:`TurnRecord` per turn (tokens,
latency, command, tier, exit status, stop reason). Historically that data was
consumed by the scoring oracle and discarded. This module converts a task's
turns into the versioned :class:`StepTrace` schema so step-level analytics
(tokens, cost, tool outcomes) persist alongside the hardware measurements.

Each turn yields:
- one ``LLM_CALL`` row (always — every turn makes a model call), and
- one ``TOOL_CALL`` row (only when the turn executed a shell command).

All rows are children of a single ``AGENT_STEP`` row for the task.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

from src.trace.schema import SpanKind, StepTrace

logger = logging.getLogger(__name__)


def _cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute USD cost for a completion via LiteLLM's price tables.

    Returns 0.0 if pricing is unavailable for the model (best-effort; never
    raises). Langfuse computes its own cost server-side; this gives the
    offline SQLite store a consistent figure too.
    """
    if not tokens_in and not tokens_out:
        return 0.0
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=tokens_in, completion_tokens=tokens_out
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return 0.0


def turns_to_steptraces(
    *,
    run_id: str,
    task_id: str,
    turns: Sequence[Any],   # Sequence[TurnRecord]
    model: str,
    task_duration_s: float = 0.0,
    runtime_id: str = "",
) -> List[StepTrace]:
    """Convert a task's turn records into StepTrace rows.

    Parameters mirror what the agent loop has on hand at task completion. The
    span_id convention matches the hardware sub-span naming so trace rows and
    measurement records line up: ``{task_id}/turn_{i}_llm`` and
    ``{task_id}/turn_{i}_cmd``, both parented to ``{task_id}``.
    """
    rows: List[StepTrace] = []
    rid = runtime_id or run_id

    def _us(ns: int) -> int:
        """epoch-ns -> epoch-us (StepTrace's unit); 0 stays 0 (not recorded)."""
        return int(ns) // 1000 if ns else 0

    child_rows: List[StepTrace] = []
    starts: List[int] = []   # epoch-us span starts, for the parent envelope
    ends: List[int] = []

    for t in turns:
        turn_idx = getattr(t, "turn", 0)
        tin = int(getattr(t, "prompt_tokens", 0) or 0)
        tout = int(getattr(t, "completion_tokens", 0) or 0)
        gen_ms = float(getattr(t, "generation_ms", 0.0) or 0.0)
        turn_err = getattr(t, "error", None)

        # LLM call — one per turn.
        llm_start = _us(getattr(t, "llm_start_ns", 0))
        llm_end = _us(getattr(t, "llm_end_ns", 0))
        if llm_start:
            starts.append(llm_start)
        if llm_end:
            ends.append(llm_end)
        child_rows.append(
            StepTrace(
                runtime_id=rid,
                run_id=run_id,
                parent_span_id=task_id,
                span_id=f"{task_id}/turn_{turn_idx}_llm",
                span_kind=SpanKind.LLM_CALL,
                node_id=f"llm_call_{turn_idx}",
                start_ts_us=llm_start,
                end_ts_us=llm_end,
                duration_us=int(gen_ms * 1000),
                model_id=model,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=_cost_usd(model, tin, tout),
                routing_reason=(getattr(t, "stop_reason", "") or None),
                status="error" if turn_err else "ok",
                error=turn_err,
                extra={"turn": turn_idx, "action": getattr(t, "action", "")},
            )
        )

        # Tool call — only when a shell command actually executed.
        command = getattr(t, "command", "") or ""
        if command:
            exit_code = getattr(t, "exit_code", None)
            cmd_ms = float(getattr(t, "command_ms", 0.0) or 0.0)
            cmd_start = _us(getattr(t, "cmd_start_ns", 0))
            cmd_end = _us(getattr(t, "cmd_end_ns", 0))
            if cmd_start:
                starts.append(cmd_start)
            if cmd_end:
                ends.append(cmd_end)
            child_rows.append(
                StepTrace(
                    runtime_id=rid,
                    run_id=run_id,
                    parent_span_id=task_id,
                    span_id=f"{task_id}/turn_{turn_idx}_cmd",
                    span_kind=SpanKind.TOOL_CALL,
                    node_id=f"cmd_{turn_idx}",
                    start_ts_us=cmd_start,
                    end_ts_us=cmd_end,
                    duration_us=int(cmd_ms * 1000),
                    tool_name="shell",
                    resource_tier=(getattr(t, "command_tier", "") or None),
                    wall_clock_ms=cmd_ms,
                    status="ok" if (exit_code in (0, None)) else "error",
                    error=None if exit_code in (0, None) else f"exit_code={exit_code}",
                    extra={"turn": turn_idx, "command": command[:200],
                           "exit_code": exit_code},
                )
            )

    # Parent agent-step row spans the envelope of all child timestamps.
    task_start = min(starts) if starts else 0
    task_end = max(ends) if ends else 0
    rows.append(
        StepTrace(
            runtime_id=rid,
            run_id=run_id,
            parent_span_id=None,
            span_id=task_id,
            span_kind=SpanKind.AGENT_STEP,
            node_id="task",
            start_ts_us=task_start,
            end_ts_us=task_end,
            duration_us=int(task_duration_s * 1_000_000),
            model_id=model,
            status="ok",
        )
    )
    rows.extend(child_rows)
    return rows
