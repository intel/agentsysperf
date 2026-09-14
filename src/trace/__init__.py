#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Step-level trace subsystem for AgentSysPerf.

This package owns the versioned trace schema (:class:`StepTrace`,
:class:`SpanKind`, :class:`ExecutionPlan`) that powers trace- and step-level
analytics — per-step LLM/tool events with tokens, cost, routing, and
parent/child span linkage, on top of the hardware measurements AgentSysPerf
already collects.

The schema originates from the AgentFlow runtime (project AgentOptimizer,
``src/agentflow/core/schema.py``) which *emits* these rows. AgentSysPerf is a
*consumer*: it ingests, persists, and analyzes them. The schema is kept
byte-compatible (same ``schema_version``) so traces produced by either side
interoperate and the ``compare`` path can refuse mixed-schema runs.
"""

from src.trace.schema import (
    SCHEMA_VERSION,
    ExecutionPlan,
    PlanNodeHint,
    SpanKind,
    StepTrace,
)

__all__ = [
    "SCHEMA_VERSION",
    "SpanKind",
    "StepTrace",
    "ExecutionPlan",
    "PlanNodeHint",
]
