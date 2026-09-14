#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Contract tests for the AgentSysPerf trace schema (v0.3).

Mirrors AgentOptimizer's tests/test_phase2_5_trace_schema.py so the two
schemas stay byte-compatible: version lock, closed SpanKind enum, frozen +
extra-forbidden models, and ExecutionPlan YAML round-trip.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.trace import (
    SCHEMA_VERSION,
    ExecutionPlan,
    PlanNodeHint,
    SpanKind,
    StepTrace,
)


def _minimal_step(**overrides):
    base = dict(
        runtime_id="rt-1",
        run_id="run-1",
        span_id="run-1",
        span_kind=SpanKind.AGENT_STEP,
    )
    base.update(overrides)
    return StepTrace(**base)


def test_schema_version_locked():
    assert SCHEMA_VERSION == "0.3"
    assert _minimal_step().schema_version == "0.3"
    # A different version literal must be rejected.
    with pytest.raises(ValidationError):
        StepTrace(
            schema_version="0.4",  # type: ignore[arg-type]
            runtime_id="rt",
            run_id="r",
            span_id="r",
            span_kind=SpanKind.AGENT_STEP,
        )


def test_spankind_is_closed():
    assert {k.value for k in SpanKind} == {
        "llm_call",
        "tool_call",
        "route_decision",
        "plan_build",
        "agent_step",
    }


def test_steptrace_is_frozen_and_extra_forbidden():
    step = _minimal_step()
    with pytest.raises(ValidationError):  # frozen → no mutation
        step.tokens_in = 5  # type: ignore[misc]
    with pytest.raises(ValidationError):  # extra fields rejected
        _minimal_step(unknown_field="x")


def test_llm_call_fields_round_trip():
    step = _minimal_step(
        span_kind=SpanKind.LLM_CALL,
        parent_span_id="run-1",
        span_id="run-1-llm-0",
        model_id="gpt-4o-mini",
        tokens_in=120,
        tokens_out=48,
        cost_usd=0.0009,
        start_ts_us=1_000_000,
        end_ts_us=1_250_000,
        duration_us=250_000,
    )
    d = step.model_dump()
    assert d["span_kind"] == "llm_call"
    assert d["tokens_in"] == 120 and d["cost_usd"] == 0.0009
    # Reconstruct from the dump → identical model.
    assert StepTrace.model_validate(d) == step


def test_tool_call_defaults():
    step = _minimal_step(
        span_kind=SpanKind.TOOL_CALL,
        tool_name="run_shell",
        wall_clock_ms=12.5,
    )
    assert step.batch_id == 0 and step.batch_size == 1
    assert step.status == "ok" and step.error is None


def test_execution_plan_yaml_round_trip(tmp_path):
    plan = ExecutionPlan(
        nodes={
            "agent": PlanNodeHint(
                model="gpt-4o-mini", max_tokens=512, route_reason="cheap-tier"
            ),
            "tools": PlanNodeHint(),
        }
    )
    path = tmp_path / "plan.yaml"
    plan.to_yaml(str(path))
    loaded = ExecutionPlan.from_yaml(str(path))
    assert loaded == plan
    assert loaded.hint_for("agent").model == "gpt-4o-mini"
    assert loaded.hint_for("missing") is None


def test_execution_plan_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ExecutionPlan.from_dict({"nodes": {}, "bogus": 1})
