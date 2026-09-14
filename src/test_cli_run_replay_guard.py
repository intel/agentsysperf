#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""`run --replay` must refuse a dataset-sliced task selection.

-n/--full slice the dataset in its own order; a fixture holds whichever trials
were recorded, keyed by sha256(first user message). The two sets coincide only
by luck, and every non-overlapping task is a fatal miss — so the combination is
refused up front rather than discovered turn by turn.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli import app


def _fixture(tmp_path: Path, n_trials: int = 2, turns: int = 3) -> Path:
    p = tmp_path / "fx.jsonl"
    lines = []
    for t in range(n_trials):
        for turn in range(turns):
            lines.append(json.dumps({
                "trial_key": f"{t:016x}", "turn": turn, "latency_ms": 1,
                "wants_stream": False, "recorded_at": 0,
                "response": {"choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": "x"}}]},
            }))
    p.write_text("\n".join(lines) + "\n")
    return p


def test_replay_with_num_tasks_is_refused(tmp_path):
    fx = _fixture(tmp_path)
    result = CliRunner().invoke(
        app, ["run", "-b", "terminal-bench", "-n", "2", "--replay", str(fx)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1, result.output
    assert "--replay requires --tasks" in result.output
    assert "-n 2" in result.output


def test_replay_with_full_is_refused(tmp_path):
    fx = _fixture(tmp_path)
    result = CliRunner().invoke(
        app, ["run", "-b", "terminal-bench", "--full", "--replay", str(fx)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1, result.output
    assert "--replay requires --tasks" in result.output
    assert "--full" in result.output


def test_replay_with_no_selector_at_all_is_refused(tmp_path):
    """Silently falling back to a default slice is the same trap."""
    fx = _fixture(tmp_path)
    result = CliRunner().invoke(
        app, ["run", "-b", "terminal-bench", "--replay", str(fx)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1, result.output
    assert "no task selector" in result.output


def test_refusal_reports_what_the_fixture_holds(tmp_path):
    """The user cannot act on the refusal without knowing the fixture's shape."""
    fx = _fixture(tmp_path, n_trials=2, turns=3)
    result = CliRunner().invoke(
        app, ["run", "-b", "terminal-bench", "-n", "1", "--replay", str(fx)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1, result.output
    assert "2 trial(s)" in result.output
    assert "6 turn(s)" in result.output


def test_refusal_flags_a_missing_fixture_rather_than_reporting_zeros(tmp_path):
    missing = tmp_path / "nope.jsonl"
    result = CliRunner().invoke(
        app, ["run", "-b", "terminal-bench", "-n", "1", "--replay", str(missing)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1, result.output
    assert "does not exist" in result.output


def test_record_with_num_tasks_is_still_allowed(tmp_path):
    """Recording a dataset slice is the correct way to CREATE a fixture."""
    result = CliRunner().invoke(
        app, ["run", "--help"], env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    # The guard must not have leaked onto --record.
    assert "--record" in result.output

    # And the help must not teach the combination the guard now refuses.
    assert "-n 2 --replay" not in result.output
