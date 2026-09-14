#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Opt-in Langfuse tracing for the LiteLLM-backed agent loop.

Enabling is config-only: set ``AGENTSYSPERF_LANGFUSE=1`` plus the standard
Langfuse env vars (``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY``,
``LANGFUSE_SECRET_KEY``). When enabled, LiteLLM's built-in ``langfuse_otel``
callback (v3 SDK, OpenTelemetry-based) instruments every
``litellm.completion()`` call — emitting a per-call
generation with model, prompt/completion, token usage, latency, and
auto-computed cost. AgentSysPerf attaches ``run_id``/``task_id`` as trace metadata
so Langfuse traces correlate with AgentSysPerf's hardware measurements.

This is the Phase-1 quick win: step + task-level LLM analytics with near-zero
changes to the agent loop. Phases 2-3 add native SQLite persistence and OTel
fan-out (Tempo + hardware correlation).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level guard so we register the callback exactly once per process.
_ENABLED: Optional[bool] = None


def langfuse_requested() -> bool:
    """True if the operator opted into Langfuse tracing via env."""
    return os.environ.get("AGENTSYSPERF_LANGFUSE", "").strip() in ("1", "true", "True", "yes")


def enable_langfuse() -> bool:
    """Register LiteLLM's Langfuse callback once. Idempotent.

    Returns True if tracing is active, False otherwise (not requested, missing
    creds, or SDK not installed). Never raises — tracing is best-effort and must
    not break a benchmark run.
    """
    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED

    if not langfuse_requested():
        _ENABLED = False
        return False

    missing = [
        k for k in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not os.environ.get(k)
    ]
    if missing:
        logger.warning(
            "AGENTSYSPERF_LANGFUSE set but missing env: %s — tracing disabled.",
            ", ".join(missing),
        )
        _ENABLED = False
        return False

    try:
        import litellm  # noqa: F401  (import side effect: ensures it's present)

        # LiteLLM ships the Langfuse logger; appending to success_callback makes
        # it instrument every completion(). Append (not overwrite) so we don't
        # clobber any callbacks already configured. "langfuse_otel" is the v3-SDK
        # (OpenTelemetry) callback; the legacy "langfuse" callback imports the
        # v2-only SDK, which conflicts with core deps (see pyproject trace extra).
        cbs = list(getattr(litellm, "success_callback", []) or [])
        if "langfuse_otel" not in cbs:
            cbs.append("langfuse_otel")
        litellm.success_callback = cbs
        # Also trace failed calls so error turns show up in the trace.
        fcbs = list(getattr(litellm, "failure_callback", []) or [])
        if "langfuse_otel" not in fcbs:
            fcbs.append("langfuse_otel")
        litellm.failure_callback = fcbs
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to enable Langfuse callback: %s — tracing disabled.", e)
        _ENABLED = False
        return False

    logger.info(
        "Langfuse tracing enabled -> %s (project traces via LiteLLM callback).",
        os.environ.get("LANGFUSE_HOST"),
    )
    _ENABLED = True
    return True


def completion_metadata(*, run_id: str, task_id: str, turn_idx: int) -> Dict[str, Any]:
    """Build the ``metadata`` dict to pass into ``litellm.completion``.

    LiteLLM's Langfuse callback reads these keys to group/label traces:
    - ``trace_name``/``session_id`` group a task's turns into one session,
    - ``trace_user_id``/``tags`` enable per-run and per-task filtering,
    - ``run_id``/``task_id`` are carried through for correlation with AgentSysPerf
      hardware metrics (the bridge for Phase 3).
    Returns ``{}`` when tracing is inactive (so callers can splat it harmlessly).
    """
    if not enable_langfuse():
        return {}
    return {
        "metadata": {
            "trace_name": task_id,
            "session_id": task_id,
            "trace_user_id": run_id,
            "tags": ["agentsysperf", f"run:{run_id}"],
            "run_id": run_id,
            "task_id": task_id,
            "turn": turn_idx,
        }
    }
