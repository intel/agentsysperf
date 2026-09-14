#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

from __future__ import annotations

from typer.testing import CliRunner

from src import streams
from src.cli import app


def test_run_streams_help_exposes_only_task_sized_controls():
    result = CliRunner().invoke(
        app,
        ["run-streams", "--help"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    for option in (
        "--tasks",
        "--model",
        "--timeout",
        "--max-turns",
        "--run-id",
        "--output",
        "--replay",
        "--verbose",
        "--dataset",
        "--ref",
        "--launch-stagger",
    ):
        assert option in result.output
    assert "--slots" in result.output
    assert "--total-cores" in result.output
    assert "--pin-cores" in result.output
    assert "--max-slots" in result.output
    assert "--stream-multiple" in result.output
    assert "4096" in result.output
    assert "2.0" in result.output
    assert "3600" in result.output
    assert "1000000" in result.output
    assert "0.0" in result.output
    assert "[default: 1]" in result.output
    assert "clean_tasks_23.txt" in result.output
    assert "terminal-bench/terminal-bench-2" in result.output
    assert "openai/agentsysperf-proxy" in result.output
    assert "Compatibility alias for --slots" in result.output
    assert "required CPU budget when --pin-cores is selected" in result.output
    assert "--streams" not in result.output
    assert "--cores-per-stream" not in result.output
    assert "--pool" not in result.output
    assert "--benchmark" not in result.output
    assert "--num-tasks" not in result.output
    assert "--full" not in result.output
    assert "--record" not in result.output
    assert "--upstream" not in result.output


def test_run_streams_without_tasks_uses_the_default_23_task_file(monkeypatch):
    captured = {}

    def fake_run(_cfg, task_names, **_kwargs):
        captured["task_names"] = task_names
        return [
            {
                "task": task_names[0],
                "stream_id": 0,
                "passed": 1,
                "total": 1,
                "error": None,
            }
        ]

    monkeypatch.setattr(streams, "run_task_sized_streams", fake_run)
    result = CliRunner().invoke(app, ["run-streams", "--slots", "1"])

    assert result.exit_code == 0, result.output
    assert captured["task_names"] == [
        "cancel-async-tasks",
        "chess-best-move",
        "circuit-fibsqrt",
        "custom-memory-heap-crash",
        "distribution-search",
        "dna-insert",
        "extract-elf",
        "extract-moves-from-video",
        "fix-git",
        "gcode-to-text",
        "git-leak-recovery",
        "headless-terminal",
        "install-windows-3-11",
        "overfull-hbox",
        "path-tracing",
        "polyglot-rust-c",
        "prove-plus-comm",
        "pytorch-model-recovery",
        "regex-chess",
        "video-processing",
        "vulnerable-secret",
        "winning-avg-corewars",
        "write-compressor",
    ]


def test_run_streams_rejects_conflicting_slot_options():
    result = CliRunner().invoke(
        app,
        [
            "run-streams",
            "--tasks",
            "example",
            "--slots",
            "2",
            "--total-cores",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_run_streams_rejects_slots_with_pin_cores():
    result = CliRunner().invoke(
        app,
        [
            "run-streams",
            "--tasks",
            "example",
            "--slots",
            "2",
            "--pin-cores",
        ],
    )
    assert result.exit_code == 1
    assert "cannot be combined" in result.output


def test_run_streams_rejects_pin_cores_without_cpu_budget():
    result = CliRunner().invoke(
        app,
        ["run-streams", "--tasks", "example", "--pin-cores"],
    )
    assert result.exit_code == 1
    assert "requires --total-cores" in result.output


def test_run_streams_all_passing_results_exit_zero(monkeypatch):
    monkeypatch.setattr(
        streams,
        "run_task_sized_streams",
        lambda *_args, **_kwargs: [
            {"task": "example", "stream_id": 0, "passed": 1, "total": 1, "error": None}
        ],
    )
    result = CliRunner().invoke(
        app,
        ["run-streams", "--tasks", "example", "--slots", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "1/1 passed" in result.output


def test_run_streams_oracle_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(
        streams,
        "run_task_sized_streams",
        lambda *_args, **_kwargs: [
            {
                "task": "example",
                "stream_id": 0,
                "passed": 0,
                "total": 1,
                "oracle_failures": 1,
                "error": None,
            }
        ],
    )
    result = CliRunner().invoke(
        app,
        ["run-streams", "--tasks", "example", "--slots", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "example: 0/1 oracle checks passed" in result.output


def test_run_streams_task_execution_failure_exits_one(monkeypatch):
    monkeypatch.setattr(
        streams,
        "run_task_sized_streams",
        lambda *_args, **_kwargs: [
            {
                "task": "example",
                "stream_id": 0,
                "passed": 0,
                "total": 1,
                "execution_failures": 1,
                "error": None,
            }
        ],
    )
    result = CliRunner().invoke(
        app,
        ["run-streams", "--tasks", "example", "--slots", "1"],
    )
    assert result.exit_code == 1
    assert "example: 1 task execution failure(s)" in result.output


def test_run_streams_explicit_error_exits_one(monkeypatch):
    monkeypatch.setattr(
        streams,
        "run_task_sized_streams",
        lambda *_args, **_kwargs: [
            {
                "task": "example",
                "stream_id": 0,
                "passed": 1,
                "total": 1,
                "error": "task failed",
            }
        ],
    )
    result = CliRunner().invoke(
        app,
        ["run-streams", "--tasks", "example", "--slots", "1"],
    )
    assert result.exit_code == 1
    assert "example: task failed" in result.output
