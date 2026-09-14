#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for turns_to_steptraces + SQLite span persistence round-trip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.trace.builder import turns_to_steptraces
from src.trace.schema import SpanKind
from src.storage.sqlite_store import SQLiteResultStore


@dataclass
class _Turn:
    turn: int
    action: str = "shell"
    command: str = ""
    command_tier: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_ms: float = 0.0
    command_ms: float = 0.0
    exit_code: Optional[int] = None
    stop_reason: str = ""
    error: Optional[str] = None
    llm_start_ns: int = 0
    llm_end_ns: int = 0
    cmd_start_ns: int = 0
    cmd_end_ns: int = 0


def _sample_turns():
    # Epoch-ns timestamps in order: turn0 llm -> turn0 cmd -> turn1 llm.
    base = 1_700_000_000_000_000_000
    return [
        _Turn(turn=0, command="ls -la", command_tier="cpu_light",
              prompt_tokens=100, completion_tokens=40, generation_ms=800,
              command_ms=12, exit_code=0, stop_reason="tool_calls",
              llm_start_ns=base, llm_end_ns=base + 800_000_000,
              cmd_start_ns=base + 800_000_000, cmd_end_ns=base + 812_000_000),
        _Turn(turn=1, action="submit", command="",  # submit turn: LLM only
              prompt_tokens=120, completion_tokens=10, generation_ms=600,
              stop_reason="stop",
              llm_start_ns=base + 812_000_000, llm_end_ns=base + 1_412_000_000),
    ]


def test_builder_shapes():
    rows = turns_to_steptraces(
        run_id="r1", task_id="tb/foo", turns=_sample_turns(),
        model="gpt-4o-mini", task_duration_s=2.5,
    )
    kinds = [r.span_kind for r in rows]
    # 1 agent_step + 2 llm_call + 1 tool_call (only turn 0 ran a command)
    assert kinds.count(SpanKind.AGENT_STEP) == 1
    assert kinds.count(SpanKind.LLM_CALL) == 2
    assert kinds.count(SpanKind.TOOL_CALL) == 1

    parent = next(r for r in rows if r.span_kind == SpanKind.AGENT_STEP)
    assert parent.span_id == "tb/foo" and parent.parent_span_id is None

    llm0 = next(r for r in rows if r.span_id == "tb/foo/turn_0_llm")
    assert llm0.parent_span_id == "tb/foo"
    assert llm0.tokens_in == 100 and llm0.tokens_out == 40
    assert llm0.cost_usd > 0.0   # gpt-4o-mini is in LiteLLM's price table

    tool0 = next(r for r in rows if r.span_id == "tb/foo/turn_0_cmd")
    assert tool0.span_kind == SpanKind.TOOL_CALL
    assert tool0.tool_name == "shell" and tool0.resource_tier == "cpu_light"
    assert tool0.status == "ok"


def test_absolute_timestamps_flow_through():
    rows = turns_to_steptraces(
        run_id="r1", task_id="tb/foo", turns=_sample_turns(),
        model="gpt-4o-mini", task_duration_s=2.5,
    )
    base_us = 1_700_000_000_000_000  # base ns / 1000

    llm0 = next(r for r in rows if r.span_id == "tb/foo/turn_0_llm")
    assert llm0.start_ts_us == base_us
    assert llm0.end_ts_us == base_us + 800_000          # 800ms in us

    tool0 = next(r for r in rows if r.span_id == "tb/foo/turn_0_cmd")
    assert tool0.start_ts_us == base_us + 800_000
    assert tool0.end_ts_us == base_us + 812_000

    # Parent span envelopes all children: earliest start, latest end.
    parent = next(r for r in rows if r.span_kind == SpanKind.AGENT_STEP)
    assert parent.start_ts_us == base_us
    assert parent.end_ts_us == base_us + 1_412_000

    # Spans are time-ordered turn 0 llm -> turn 0 cmd -> turn 1 llm.
    spans = sorted([r for r in rows if r.span_kind != SpanKind.AGENT_STEP],
                   key=lambda r: r.start_ts_us)
    assert [s.span_id for s in spans] == [
        "tb/foo/turn_0_llm", "tb/foo/turn_0_cmd", "tb/foo/turn_1_llm",
    ]


def test_missing_timestamps_default_zero():
    # Turns without ns timestamps (e.g. older runs) -> 0, no crash.
    rows = turns_to_steptraces(
        run_id="r1", task_id="t", model="gpt-4o-mini",
        turns=[_Turn(turn=0, command="ls", exit_code=0, generation_ms=10)],
    )
    for r in rows:
        assert r.start_ts_us == 0 and r.end_ts_us == 0


def test_tool_error_status():
    rows = turns_to_steptraces(
        run_id="r1", task_id="t", model="gpt-4o-mini",
        turns=[_Turn(turn=0, command="false", exit_code=1)],
    )
    tool = next(r for r in rows if r.span_kind == SpanKind.TOOL_CALL)
    assert tool.status == "error" and "exit_code=1" in (tool.error or "")


def test_store_and_query_round_trip(tmp_path):
    store = SQLiteResultStore(output_dir=tmp_path)
    store.store_run_metadata(run_id="r1", metadata={"start_time": 0, "model": "gpt-4o-mini"})
    rows = turns_to_steptraces(
        run_id="r1", task_id="tb/foo", turns=_sample_turns(),
        model="gpt-4o-mini", task_duration_s=2.5,
    )
    store.store_spans(run_id="r1", spans=rows)

    all_spans = store.query_spans("r1")
    assert len(all_spans) == 4

    llm = store.query_spans("r1", span_kind="llm_call")
    assert len(llm) == 2
    assert sum(s["tokens_in"] for s in llm) == 220

    by_task = store.query_spans("r1", task_id="tb/foo")
    assert len(by_task) == 4
    # extra decoded from JSON
    cmd = [s for s in by_task if s["span_kind"] == "tool_call"][0]
    assert cmd["extra"]["command"] == "ls -la"

    # idempotent: storing again does not duplicate
    store.store_spans(run_id="r1", spans=rows)
    assert len(store.query_spans("r1")) == 4

    store.close()
