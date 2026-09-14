#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Versioned trace and dispatch-plan schemas for AgentSysPerf.

Two stable, versioned schemas live here:

- :class:`StepTrace` — one row per recorded execution event (LLM call, tool
  call, routing decision, plan-build step, or agent-step boundary). Per-call
  rows reference an outer invocation span via ``parent_span_id``. AgentSysPerf
  builds these from agent-loop turn records and persists them for step-level
  analytics; the AgentFlow runtime (project AgentOptimizer) emits the same
  shape directly.
- :class:`ExecutionPlan` — a committed dispatch plan (per-node model /
  max_tokens / route_reason). Emitted by an optimizer pass and consulted at
  agent node boundaries.

Both carry ``schema_version: Literal["0.3"]`` so consumers can refuse
mixed-schema runs cleanly. This schema is intentionally byte-compatible with
AgentOptimizer's ``src/agentflow/core/schema.py`` (v0.3); see
``docs/trace_schema_v0_3.md`` for full field documentation.

Bump :data:`SCHEMA_VERSION` only on breaking changes. Additive fields default
to ``None``/zero and stay on the same version.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SCHEMA_VERSION",
    "SpanKind",
    "StepTrace",
    "ExecutionPlan",
    "PlanNodeHint",
]


# Bump only at breaking changes to StepTrace or ExecutionPlan. Additive
# fields default to None and stay on the same schema version.
SCHEMA_VERSION = "0.3"


class SpanKind(str, Enum):
    """Closed enum of recordable event kinds.

    Locked at 0.3. Adding kinds later requires a schema bump because
    downstream consumers (analyzers, reporters, exporters) dispatch on this
    field.
    """

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    ROUTE_DECISION = "route_decision"
    PLAN_BUILD = "plan_build"
    AGENT_STEP = "agent_step"


class StepTrace(BaseModel):
    """One row of recorded execution.

    Per-call rows (LLM_CALL / TOOL_CALL) reference an outer invocation span
    (AGENT_STEP) via ``parent_span_id``. Timestamps are microseconds since the
    epoch so the trace is consumed natively by Perfetto and converts cleanly to
    OpenTelemetry spans.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.3"] = SCHEMA_VERSION

    # Identification.
    runtime_id: str
    run_id: str
    parent_span_id: Optional[str] = None
    span_id: str
    span_kind: SpanKind

    # Logical placement in the workflow. ``node_id`` is the agent-graph node
    # name when known, otherwise the synthesized identifier the emitter assigns
    # to the invocation.
    session_id: str = ""
    node_id: str = ""

    # Time, in microseconds since epoch. Microseconds (not ms) so Perfetto
    # consumes the trace natively.
    start_ts_us: int = 0
    end_ts_us: int = 0
    duration_us: int = 0

    # LLM-call columns (populated when span_kind == LLM_CALL).
    deployment_id: Optional[str] = None
    model_id: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    routing_reason: Optional[str] = None
    quality_score: Optional[float] = None
    kv_cache_hit: Optional[bool] = None

    # Tool-call columns (populated when span_kind == TOOL_CALL).
    tool_name: Optional[str] = None
    resource_tier: Optional[str] = None
    dispatch_mode: Optional[str] = None
    wall_clock_ms: Optional[float] = None
    queue_wait_ms: Optional[float] = None
    worker_id: Optional[str] = None
    batch_id: int = 0
    batch_size: int = 1

    # Outcome.
    status: str = "ok"
    error: Optional[str] = None

    # Free-form provenance. Keep small; large blobs go to a sidecar. AgentSysPerf
    # stashes per-step hardware rollups (cpu_time_s, rss_kb_peak, ipc, ...) here
    # when correlating a step with its measurement records.
    extra: Dict[str, Any] = Field(default_factory=dict)


class PlanNodeHint(BaseModel):
    """Per-node hint applied at agent-graph node boundaries.

    v1 schema is deliberately minimal. v2 adds tier, fallback chain,
    prefix-cache key, parallelize, expected cost / latency. Adding fields is
    additive; removing or renaming requires a schema bump.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The chat-model deployment to use for this node, if specified. When None,
    # the runtime's router decides.
    model: Optional[str] = None
    # Override on max output tokens, if specified.
    max_tokens: Optional[int] = None
    # Free-form reason the optimizer used when emitting the hint; surfaced for
    # debugging and audit.
    route_reason: Optional[str] = None


class ExecutionPlan(BaseModel):
    """A committed dispatch plan an optimizer emits and the runtime reads.

    Loaded from YAML via :meth:`from_yaml` or constructed from a dict, then
    consulted at agent-graph node boundaries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.3"] = SCHEMA_VERSION
    nodes: Dict[str, PlanNodeHint] = Field(default_factory=dict)

    def hint_for(self, node_id: str) -> Optional[PlanNodeHint]:
        """Return the hint for ``node_id`` if present, else None."""
        return self.nodes.get(node_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str) -> "ExecutionPlan":
        from pathlib import Path

        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_yaml(self, path: str) -> None:
        from pathlib import Path

        import yaml

        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        )
