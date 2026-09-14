#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

from src.default_tasks import load_default_task_text

ROOT = Path(__file__).resolve().parents[1]
REPLAY_README = ROOT / "examples/terminal_bench_task_sized_replay/README.md"
SWEEP_SCRIPT = ROOT / "examples/terminal_bench_task_sized_replay/sweep_total_cores.py"


def test_root_readme_advertises_task_sized_replay_workflow():
    text = (ROOT / "README.md").read_text()
    assert "`run-streams` (task-sized Terminal-Bench streams)" in text
    assert "agentsysperf run-streams" in text
    assert "the default 23-task selection" in text
    assert "--slots 48" in text
    assert "examples/terminal_bench_task_sized_replay/README.md" in text


def test_replay_readme_covers_supported_workflow_contract():
    text = REPLAY_README.read_text()
    for phrase in (
        "runc",
        "Harbor",
        "replay",
        "--record",
        "--replay",
        "run-streams",
        "clean_tasks_23.txt",
        "When `--tasks` is omitted",
        "comma-separated list",
    ):
        assert phrase in text, f"Missing phrase in replay README: {phrase!r}"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("task_sized_sweep", SWEEP_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_task_results_db(path: Path, run_ids: tuple[str, ...] = ()) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE task_results "
            "(run_id TEXT, duration_s REAL, elapsed_s REAL, passed INTEGER)"
        )
        connection.executemany(
            "INSERT INTO task_results VALUES (?, ?, ?, ?)",
            [(f"{run_id}_example", 2.0, 1.0, 0) for run_id in run_ids],
        )
        connection.commit()


def test_sweep_uses_packaged_default_task_file(tmp_path, monkeypatch):
    module = _load_sweep_module()
    db_path = tmp_path / "results.db"
    _make_task_results_db(db_path)
    output_dir = tmp_path / "output"
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "monotonic", iter((0.0, 1.0)).__next__)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_total_cores.py",
            "--replay",
            str(tmp_path / "fixture.jsonl"),
            "--slots-values",
            "32",
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert module.main() == 1
    assert commands[0][3] == load_default_task_text().strip()


def test_sweep_forwards_documented_options_and_stops_on_failure(tmp_path, monkeypatch):
    module = _load_sweep_module()
    tasks_file = tmp_path / "tasks.txt"
    replay_file = tmp_path / "fixture.jsonl"
    db_path = tmp_path / "results.db"
    tasks_file.write_text("example")
    replay_file.write_text("{}\n")
    _make_task_results_db(db_path)
    output_dir = tmp_path / "output"
    commands = []
    return_codes = iter((0, 1))

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=next(return_codes))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "monotonic", iter((0.0, 1.0, 1.0, 2.0)).__next__)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_total_cores.py",
            "--tasks-file",
            str(tasks_file),
            "--replay",
            str(replay_file),
            "--slots-values",
            "32,48,64",
            "--max-slots",
            "4096",
            "--stream-multiple",
            "2.0",
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--ref",
            "1",
            "--timeout",
            "3600",
            "--max-turns",
            "1000000",
            "--launch-stagger",
            "0.5",
            "--verbose",
            "--run-id-prefix",
            "sweep",
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert module.main() == 1
    assert len(commands) == 2
    assert commands[0][0:8] == [
        "agentsysperf",
        "run-streams",
        "--tasks",
        "example",
        "--slots",
        "32",
        "--max-slots",
        "4096",
    ]
    assert "--stream-multiple" in commands[0]
    assert "--model" in commands[0]
    assert "--replay" in commands[0]
    assert str(replay_file) in commands[0]
    assert "--dataset" in commands[0]
    assert "--ref" in commands[0]
    assert "--timeout" in commands[0]
    assert "--max-turns" in commands[0]
    assert "--run-id" in commands[0]
    assert "--launch-stagger" in commands[0]
    assert "--verbose" in commands[0]
    assert (output_dir / "results.json").exists()
    assert json.loads((output_dir / "results.json").read_text())[1]["exit_code"] == 1


def test_sweep_continues_after_completed_oracle_failures(tmp_path, monkeypatch):
    module = _load_sweep_module()
    tasks_file = tmp_path / "tasks.txt"
    replay_file = tmp_path / "fixture.jsonl"
    db_path = tmp_path / "results.db"
    run_id_prefix = "sweep"
    tasks_file.write_text("example")
    replay_file.write_text("{}\n")
    _make_task_results_db(
        db_path,
        tuple(f"{run_id_prefix}_s{slots}" for slots in (32, 48, 64)),
    )
    output_dir = tmp_path / "output"
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.time,
        "monotonic",
        iter((0.0, 1.0, 1.0, 2.0, 2.0, 3.0)).__next__,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_total_cores.py",
            "--tasks-file",
            str(tasks_file),
            "--replay",
            str(replay_file),
            "--slots-values",
            "32,48,64",
            "--run-id-prefix",
            run_id_prefix,
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert module.main() == 0
    assert len(commands) == 3
    rows = json.loads((output_dir / "results.json").read_text())
    assert [row["slots"] for row in rows] == [32, 48, 64]
    assert [row["task_runs"] for row in rows] == [1, 1, 1]
    assert [row["passed"] for row in rows] == [0, 0, 0]
    assert [row["exit_code"] for row in rows] == [0, 0, 0]
