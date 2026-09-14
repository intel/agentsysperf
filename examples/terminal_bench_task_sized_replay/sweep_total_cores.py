#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Sweep task-sized runc workers over explicit slot budgets."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path

from src.default_tasks import load_default_task_text
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[49], quantiles[94], quantiles[98]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help="Task list file; defaults to the packaged 23-task selection",
    )
    parser.add_argument("--replay", type=Path, required=True,
                        help="Path to a recorded replay fixture (record with --record)")
    parser.add_argument(
        "--slots-values",
        required=True,
        help="Comma-separated slot budgets, e.g. 32,64,96",
    )
    parser.add_argument("--max-slots", type=int, default=4096)
    parser.add_argument("--stream-multiple", type=float, default=2.0)
    parser.add_argument("--model", default="openai/agentsysperf-proxy")
    parser.add_argument("--dataset", default="terminal-bench/terminal-bench-2")
    parser.add_argument("--ref", default="1")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-turns", type=int, default=1_000_000)
    parser.add_argument("--launch-stagger", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run-id-prefix", default=f"task_sized_{int(time.time())}")
    parser.add_argument(
        "--db", type=Path, default=Path.home() / ".agentsysperf" / "results.db"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.tasks_file is None:
        tasks = load_default_task_text().strip()
    else:
        tasks_path = (
            args.tasks_file
            if args.tasks_file.is_absolute()
            else root / args.tasks_file
        )
        tasks = tasks_path.read_text().strip()
    replay_path = args.replay if args.replay.is_absolute() else root / args.replay
    slot_values = [
        int(value) for value in args.slots_values.split(",") if value.strip()
    ]
    output_dir = (
        args.output_dir or Path(f"{_TMP}/agentsysperf_task_sized") / args.run_id_prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = False
    executable = "agentsysperf"
    for slots in slot_values:
        run_id = f"{args.run_id_prefix}_s{slots}"
        command = [
            executable,
            "run-streams",
            "--tasks",
            tasks,
            "--slots",
            str(slots),
            "--max-slots",
            str(args.max_slots),
            "--stream-multiple",
            str(args.stream_multiple),
            "--model",
            args.model,
            "--replay",
            str(replay_path),
            "--dataset",
            args.dataset,
            "--ref",
            args.ref,
            "--timeout",
            str(args.timeout),
            "--max-turns",
            str(args.max_turns),
            "--run-id",
            run_id,
        ]
        if args.launch_stagger:
            command.extend(["--launch-stagger", str(args.launch_stagger)])
        if args.verbose:
            command.append("--verbose")

        started = time.monotonic()
        with (output_dir / f"{run_id}.log").open("w") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        sweep_wall_clock_s = time.monotonic() - started

        with sqlite3.connect(args.db) as connection:
            task_rows = connection.execute(
                "SELECT duration_s, elapsed_s, passed FROM task_results "
                "WHERE run_id LIKE ? AND duration_s IS NOT NULL",
                (f"{run_id}_%",),
            ).fetchall()
        durations = sorted(float(duration) for duration, _, _ in task_rows)
        agent_elapsed_values = sorted(
            float(agent_elapsed)
            for _, agent_elapsed, _ in task_rows
            if agent_elapsed is not None and agent_elapsed >= 0
        )
        p50, p95, p99 = _percentiles(durations)
        execution_p50, execution_p95, execution_p99 = _percentiles(agent_elapsed_values)
        rows.append(
            {
                "slots": slots,
                "exit_code": completed.returncode,
                "task_runs": len(durations),
                "agent_elapsed_task_runs": len(agent_elapsed_values),
                "passed": sum(bool(passed) for _, _, passed in task_rows),
                "sweep_wall_clock_s": round(sweep_wall_clock_s, 3),
                "throughput_tasks_per_s": round(len(durations) / sweep_wall_clock_s, 5)
                if sweep_wall_clock_s
                else 0.0,
                "task_duration_p50_s": round(p50, 3),
                "task_duration_p95_s": round(p95, 3),
                "task_duration_p99_s": round(p99, 3),
                "agent_elapsed_p50_s": round(execution_p50, 3),
                "agent_elapsed_p95_s": round(execution_p95, 3),
                "agent_elapsed_p99_s": round(execution_p99, 3),
            }
        )
        if completed.returncode != 0:
            failed = True
            break

    (output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(output_dir / "results.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
