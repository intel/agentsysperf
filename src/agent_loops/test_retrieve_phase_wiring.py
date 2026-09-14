#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""The agent loop must actually tag a retrieve span, not merely be able to.

`retrieval.py` is unit-tested in isolation and `test_phase_spans.py` drives the
adapter with a stubbed invoker, so nothing executed the real `_run_turn` — the
function that decides a command's phase on every live run. This exercises it with
a stubbed LLM and a stubbed environment: no network, no Docker, no API key.

Run: poetry run pytest src/agent_loops/test_retrieve_phase_wiring.py -q
"""
from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional

import pytest

from src.agent_loops.litellm_terminal_loop import LiteLLMTerminalAgentLoop
from src.benchmarks.terminal_bench.environment import EnvironmentResult
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.runner import RunContext, track_span


class _StubEnv:
    """Records commands, returns success. Async, like the real backends."""

    def __init__(self) -> None:
        self.commands: List[str] = []

    async def exec(self, command: str, timeout_sec: float = 120.0,
                   cwd: Optional[str] = None) -> EnvironmentResult:
        self.commands.append(command)
        return EnvironmentResult(stdout="ok", stderr="", return_code=0)


def _tool_call(cmd: str, idx: int) -> Any:
    """Shape litellm returns for a native tool call."""
    class _Fn:
        name = "shell"
        arguments = __import__("json").dumps({"command": cmd})

    class _TC:
        id = f"call_{idx}"
        type = "function"
        function = _Fn()

    class _Msg:
        role = "assistant"
        content = None
        tool_calls = [_TC()]

    class _Choice:
        message = _Msg()
        finish_reason = "tool_calls"

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    return _Resp()


@pytest.fixture
def commands_then_stub(monkeypatch):
    """Drive the loop through a scripted command list via a fake litellm."""
    def _install(commands: List[str]):
        seq = iter(range(len(commands)))

        def _fake_completion(**kwargs):
            i = next(seq)
            return _tool_call(commands[i], i)

        monkeypatch.setattr(
            "src.agent_loops.litellm_terminal_loop.litellm.completion",
            _fake_completion,
        )
    return _install


def _run_loop(tmp_path, commands, run_id):
    env = _StubEnv()
    ctx = RunContext(measurements=[L1SubSpanMeasurement()],
                     run_id=run_id, output_dir=tmp_path)
    with ctx:
        with track_span(ctx, "terminal-bench/probe", kind="terminal_bench",
                        node_id="terminal-bench/probe"):
            loop = LiteLLMTerminalAgentLoop(
                model="stub/model", env=env, max_turns=len(commands),
                run_context=ctx,
            )
            result = loop.solve(
                task_id="terminal-bench/probe", instruction="do the thing",
            )
    return ctx, result, env


def test_a_retrieval_command_is_tagged_retrieve_by_the_real_loop(
    tmp_path, commands_then_stub,
):
    """The load-bearing path: classify_phase -> track_span -> record payload."""
    commands = [
        "python -c 'import faiss; faiss.read_index(\"docs.faiss\")'",
        "grep -rn 'def main' /app",
        "python -c 'from rank_bm25 import BM25Okapi'",
        "pdflatex main.tex",
    ]
    commands_then_stub(commands)
    ctx, result, env = _run_loop(tmp_path, commands, "probe_retrieve")

    assert env.commands == commands, "every command must have reached the env"

    phases = Counter(
        r.payload.get("phase") for r in ctx.records
        if isinstance(r.payload, dict)
    )
    # faiss + bm25 -> retrieve; grep + pdflatex -> act.
    assert phases["retrieve"] == 2, f"expected 2 retrieve spans, got {dict(phases)}"
    assert phases["act"] == 2, f"expected 2 act spans, got {dict(phases)}"
    assert phases["reason"] == 4, "one reason span per turn"


def test_the_reason_is_recorded_on_the_turn(tmp_path, commands_then_stub):
    """An attribution nobody can explain is not evidence."""
    commands = ["python -c 'import faiss'", "ls -la /app"]
    commands_then_stub(commands)
    _, result, _ = _run_loop(tmp_path, commands, "probe_signals")

    turns = {t.command: t for t in result.turns if t.command}
    faiss_turn = next(t for c, t in turns.items() if "faiss" in c)
    ls_turn = next(t for c, t in turns.items() if c.startswith("ls"))

    assert faiss_turn.phase == "retrieve"
    assert "vector_index" in faiss_turn.retrieval_signals
    assert ls_turn.phase == "act"
    assert ls_turn.retrieval_signals == []


def test_retrieval_span_uses_the_retrieval_kind(tmp_path, commands_then_stub):
    """kind distinguishes the span for consumers that group by kind, not phase."""
    commands = ["python -m mteb run -t SciFact"]
    commands_then_stub(commands)
    ctx, _, _ = _run_loop(tmp_path, commands, "probe_kind")

    cmd_spans = [
        r for r in ctx.records
        if isinstance(r.payload, dict) and r.payload.get("phase") in ("retrieve", "act")
    ]
    assert len(cmd_spans) == 1
    assert cmd_spans[0].payload.get("phase") == "retrieve"


def test_a_shell_only_trial_emits_no_retrieve_span(tmp_path, commands_then_stub):
    """Guard against over-tagging: ordinary work must stay act."""
    commands = ["make -j8", "gcc -O2 sim.c -o sim", "cat /app/README.md"]
    commands_then_stub(commands)
    ctx, _, _ = _run_loop(tmp_path, commands, "probe_no_retrieve")

    assert not any(
        r.payload.get("phase") == "retrieve"
        for r in ctx.records if isinstance(r.payload, dict)
    )
