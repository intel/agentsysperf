#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P11 verification: replay-journal wiring in the run driver (no live LLM).

The full record->replay round-trip needs a real LLM (the one interactive,
key-gated smoke). These tests cover the WIRING: mode selection, fixture
validation gating, and the off-mode null context — all without launching the
proxy subprocess or calling an LLM.

Run: poetry run pytest src/storage/test_replay_wiring.py -q
"""
from __future__ import annotations

import json

import pytest

from src.run_driver import RunConfig, _maybe_proxy


def test_off_mode_is_null_context():
    cm = _maybe_proxy(RunConfig(benchmark="synthetic_cpu", replay_mode="off"))
    with cm as proxy:
        assert proxy is None  # no proxy, live/no-LLM path


def test_replay_validates_fixture_and_rejects_bad(tmp_path):
    # An empty/invalid fixture must fail loud BEFORE any run starts.
    bad = tmp_path / "bad.jsonl"
    bad.write_text("")  # zero trials
    with pytest.raises(ValueError, match="fixture failed validation"):
        _maybe_proxy(RunConfig(benchmark="terminal-bench", replay_mode="replay", fixture=bad))


def test_replay_missing_fixture_rejected(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _maybe_proxy(RunConfig(benchmark="terminal-bench", replay_mode="replay",
                              fixture=tmp_path / "nope.jsonl"))


def test_replay_accepts_valid_fixture(tmp_path):
    # A minimal valid fixture (one trial, one turn) passes validation and yields
    # a ReplayProxy manager (not started here — we don't enter the context).
    fix = tmp_path / "ok.jsonl"
    entry = {
        "trial_key": "abc123def4567890",
        "turn": 0,
        "latency_ms": 10,
        "wants_stream": False,
        "response": {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        "recorded_at": 0,
    }
    fix.write_text(json.dumps(entry) + "\n")
    cm = _maybe_proxy(RunConfig(benchmark="terminal-bench", replay_mode="replay", fixture=fix))
    from src.replay import ReplayProxy
    assert isinstance(cm, ReplayProxy)
    assert cm.mode == "replay" and cm.fixture == fix


def test_record_mode_builds_proxy_with_upstream(tmp_path):
    cm = _maybe_proxy(RunConfig(
        benchmark="terminal-bench", replay_mode="record",
        fixture=tmp_path / "out.jsonl",
        upstream="https://gw.example/v1"))
    from src.replay import ReplayProxy
    assert isinstance(cm, ReplayProxy) and cm.mode == "record"
    assert cm.upstream == "https://gw.example/v1"


def test_record_mode_requires_upstream(tmp_path, monkeypatch):
    # No --upstream and no $OPENAI_BASE_URL -> fail loud BEFORE any run.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="upstream"):
        _maybe_proxy(RunConfig(benchmark="terminal-bench", replay_mode="record",
                              fixture=tmp_path / "out.jsonl"))


def test_record_upstream_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-token-xyz")
    cm = _maybe_proxy(RunConfig(
        benchmark="terminal-bench", replay_mode="record",
        fixture=tmp_path / "out.jsonl", upstream="https://gw.example/v1"))
    assert cm.upstream_key == "real-token-xyz", "real upstream credential not threaded to proxy"
