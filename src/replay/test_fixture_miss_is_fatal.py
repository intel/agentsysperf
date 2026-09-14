#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""A fixture miss must be permanent, non-retryable, and fatal to the trial.

These cover the two halves of the same defect: the proxy advertised a miss as a
transient 503 so clients retried it, and the agent loop then swallowed the error
and walked to max_turns (observed: turns 0..199 on a single miss), burying the
cause and wasting the run.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.replay import proxy
from src.replay.fixture import FIXTURE_MISS_MARKER, FIXTURE_MISS_STATUS


def _write_fixture(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "fixture.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return p


def _entry(trial_key: str, turn: int) -> dict:
    return {
        "trial_key": trial_key, "turn": turn, "latency_ms": 5,
        "wants_stream": False, "recorded_at": 0,
        "response": {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                  "finish_reason": "stop"}]},
    }


@pytest.fixture(autouse=True)
def _reset_proxy_state():
    """proxy state is module-global; restore it so tests do not leak into each other."""
    saved = (proxy._MODE, proxy._FIXTURE_PATH, proxy._STRICT_MISS,
             proxy._REPLAY_TRIAL, dict(proxy._fixture))
    yield
    (proxy._MODE, proxy._FIXTURE_PATH, proxy._STRICT_MISS,
     proxy._REPLAY_TRIAL, restored) = saved
    proxy._fixture.clear()
    proxy._fixture.update(restored)


def test_miss_status_is_not_retryable():
    """503 is defined as transient, so clients retry it. A miss never resolves."""
    assert FIXTURE_MISS_STATUS == 400
    assert 500 > FIXTURE_MISS_STATUS >= 400, "must be a client error, not a server error"


def test_strict_miss_raises_non_retryable_and_names_the_trial(tmp_path):
    proxy._fixture.clear()
    proxy._FIXTURE_PATH = _write_fixture(tmp_path, [_entry("aaaa1111bbbb2222", 0)])
    proxy._load_fixture(proxy._FIXTURE_PATH)
    proxy._STRICT_MISS = True
    proxy._REPLAY_TRIAL = None

    # A prompt that hashes to something not in the fixture.
    body = {"messages": [{"role": "user", "content": "a task that was never recorded"}]}
    with pytest.raises(HTTPException) as ei:
        asyncio.run(proxy._handle_replay(body))

    assert ei.value.status_code == FIXTURE_MISS_STATUS
    detail = ei.value.detail
    assert FIXTURE_MISS_MARKER in detail
    # The trial_key is the only handle on the missing entry, so it must be named.
    assert f"trial={proxy._trial_key(body['messages'])}" in detail
    # ...and the message must say retrying is pointless, since 400 alone does not.
    assert "permanent" in detail
    # The absent-trial case should list what IS loaded, so "wrong task" is
    # distinguishable from "ran off the end of a recorded trajectory".
    assert "aaaa1111bbbb2222" in detail


def test_miss_past_last_recorded_turn_reports_the_range(tmp_path):
    proxy._fixture.clear()
    key = "cccc3333dddd4444"
    proxy._FIXTURE_PATH = _write_fixture(tmp_path, [_entry(key, 0), _entry(key, 1)])
    proxy._load_fixture(proxy._FIXTURE_PATH)
    proxy._STRICT_MISS = True
    proxy._REPLAY_TRIAL = key  # flexible mode: serve this trial regardless of hash

    body = {"messages": [
        {"role": "user", "content": "anything"},
        {"role": "assistant", "content": "1"},
        {"role": "assistant", "content": "2"},  # turn_index == 2, fixture has 0..1
    ]}
    with pytest.raises(HTTPException) as ei:
        asyncio.run(proxy._handle_replay(body))

    assert ei.value.status_code == FIXTURE_MISS_STATUS
    assert "turns 0..1" in ei.value.detail
    assert "asked for 2" in ei.value.detail


def test_non_strict_miss_still_serves_empty(tmp_path):
    """--no-strict is an explicit opt-out and must keep working."""
    proxy._fixture.clear()
    proxy._FIXTURE_PATH = _write_fixture(tmp_path, [_entry("eeee5555ffff6666", 0)])
    proxy._load_fixture(proxy._FIXTURE_PATH)
    proxy._STRICT_MISS = False
    proxy._REPLAY_TRIAL = None

    resp = asyncio.run(proxy._handle_replay({"messages": [{"role": "user", "content": "unrecorded"}]}))
    assert resp["choices"][0]["message"]["content"] == ""


def test_agent_loop_aborts_on_first_miss_instead_of_walking_max_turns(monkeypatch):
    """One miss must end the trial, not repeat for every remaining turn."""
    from src.agent_loops import litellm_terminal_loop as loop_mod

    calls = {"n": 0}

    def fake_run_turn(self, turn_idx, conversation, parent_span_id="unknown"):
        calls["n"] += 1
        rec = loop_mod.TurnRecord(turn=turn_idx)
        rec.error = (f"Generation failed: litellm.BadRequestError: "
                     f"{FIXTURE_MISS_MARKER}: trial=deadbeefdeadbeef turn={turn_idx} "
                     f"— trial not in fixture at all. This is permanent, not transient: "
                     f"retrying cannot help.")
        return rec

    monkeypatch.setattr(loop_mod.LiteLLMTerminalAgentLoop, "_run_turn", fake_run_turn)

    loop = loop_mod.LiteLLMTerminalAgentLoop.__new__(loop_mod.LiteLLMTerminalAgentLoop)
    loop.max_turns = 200
    loop.model = "test-model"
    loop._use_native_tools = True
    loop._run_context = None
    loop.command_metrics = loop_mod.CommandMetrics()
    monkeypatch.setattr(loop, "_build_initial_conversation", lambda instruction: [], raising=False)

    result = loop.solve("some-task", "do the thing")

    assert calls["n"] == 1, f"expected abort after the first miss, made {calls['n']} calls"
    assert result.num_turns == 1
    assert result.error is not None and FIXTURE_MISS_MARKER in result.error
    assert "deadbeefdeadbeef" in result.error, "the trial_key must reach the caller"
