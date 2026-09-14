#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P11 verification: `agentsysperf run` via the synthetic_cpu benchmark (no LLM,
no network, no key) — exercises the full run->persist->read loop end to end.

Run: poetry run pytest src/storage/test_run_driver.py -q
"""
from __future__ import annotations

from typer.testing import CliRunner

from src.cli import app
from src.run_driver import RunConfig, _run_metadata, run_benchmark
from src.protocols import TaskResult, TaskSpec
from src.storage.sqlite_store import SQLiteResultStore

runner = CliRunner()


def test_run_synthetic_persists_to_store(tmp_path):
    store = SQLiteResultStore(tmp_path)
    cfg = RunConfig(benchmark="synthetic_cpu", num_tasks=3, run_id="synth_test")
    summary = run_benchmark(cfg, store=store, stamp=123)
    assert summary.run_id == "synth_test"
    assert summary.total == 3
    # tasks actually RAN (not swallowed by error isolation) — guards the
    # run_context-kwarg-mismatch regression.
    assert summary.passed == 3, "synthetic tasks did not pass (run_task signature mismatch?)"
    # run + tasks + measurements all landed via persist_run
    assert store.get_run("synth_test") is not None
    assert len(store.query_tasks("synth_test")) == 3
    assert len(store.query_measurements("synth_test")) > 0
    # provenance captured (hardware_sku from detect, owner from $USER)
    run = store.get_run("synth_test")
    assert run["benchmark_id"] == "synthetic-cpu"
    assert run["status"] == "complete"


def test_run_id_is_deterministic_with_stamp(tmp_path):
    store = SQLiteResultStore(tmp_path)
    cfg = RunConfig(benchmark="synthetic_cpu", num_tasks=1)  # no explicit run_id
    s = run_benchmark(cfg, store=store, stamp=999)
    assert s.run_id == "synthetic_cpu_999"


def test_run_persists_adapter_elapsed_s(tmp_path, monkeypatch):
    class Adapter:
        def run_task(self, task, *, agent_invoker):
            return TaskResult(
                task_id=task.id,
                passed=True,
                reward=1.0,
                extra={"measured_s": 4.0, "elapsed_s": 2.5},
            )

        def teardown(self):
            return None

    monkeypatch.setattr(
        "src.run_driver._build_adapter_and_tasks",
        lambda _cfg: (Adapter(), [TaskSpec(id="task", instruction="")]),
    )
    monkeypatch.setattr("src.run_driver._build_invoker", lambda _cfg: object())
    store = SQLiteResultStore(tmp_path)

    run_benchmark(RunConfig(benchmark="synthetic_cpu", run_id="elapsed"), store=store, stamp=1)

    row = store._get_connection().execute(
        "SELECT duration_s, elapsed_s FROM task_results WHERE run_id='elapsed'"
    ).fetchone()
    assert tuple(row) == (4.0, 2.5)


def test_run_persists_agent_effort_counters(tmp_path, monkeypatch):
    """num_turns AND num_commands must reach the store.

    num_commands was dropped when building the task_results row, so the column
    was NULL for every run ever stored — which reads as "the agent ran no
    commands" rather than "nobody recorded it". The counts were only recoverable
    by counting ``*/turn_*_cmd`` spans.
    """
    class Adapter:
        def run_task(self, task, *, agent_invoker):
            return TaskResult(
                task_id=task.id,
                passed=True,
                reward=1.0,
                extra={"num_turns": 17, "num_commands": 12},
            )

        def teardown(self):
            return None

    monkeypatch.setattr(
        "src.run_driver._build_adapter_and_tasks",
        lambda _cfg: (Adapter(), [TaskSpec(id="task", instruction="")]),
    )
    monkeypatch.setattr("src.run_driver._build_invoker", lambda _cfg: object())
    store = SQLiteResultStore(tmp_path)

    run_benchmark(RunConfig(benchmark="synthetic_cpu", run_id="effort"), store=store, stamp=1)

    row = store._get_connection().execute(
        "SELECT num_turns, num_commands FROM task_results WHERE run_id='effort'"
    ).fetchone()
    assert tuple(row) == (17, 12)


def test_run_summary_separates_execution_and_oracle_failures(tmp_path, monkeypatch):
    class Adapter:
        def __init__(self):
            self.results = iter(
                [
                    TaskResult(
                        task_id="oracle",
                        passed=False,
                        extra={"oracle_run": True},
                    ),
                    TaskResult(task_id="execution", passed=False, error="agent failed"),
                ]
            )

        def run_task(self, task, *, agent_invoker):
            return next(self.results)

        def teardown(self):
            return None

    monkeypatch.setattr(
        "src.run_driver._build_adapter_and_tasks",
        lambda _cfg: (
            Adapter(),
            [
                TaskSpec(id="oracle", instruction=""),
                TaskSpec(id="execution", instruction=""),
            ],
        ),
    )
    monkeypatch.setattr("src.run_driver._build_invoker", lambda _cfg: object())
    store = SQLiteResultStore(tmp_path)

    summary = run_benchmark(
        RunConfig(benchmark="synthetic_cpu", run_id="failure-kinds"),
        store=store,
        stamp=1,
    )

    assert summary.passed == 0
    assert summary.execution_failures == 1
    assert summary.oracle_failures == 1
    metadata = store.get_run("failure-kinds")["metadata"]
    assert metadata["execution_failures"] == 1
    assert metadata["oracle_failures"] == 1


def test_num_tasks_limits(tmp_path):
    store = SQLiteResultStore(tmp_path)
    s2 = run_benchmark(RunConfig(benchmark="synthetic_cpu", num_tasks=2, run_id="a"),
                       store=store, stamp=1)
    assert s2.total == 2


def test_run_metadata_records_stream_admission_mode():
    cfg = RunConfig(
        benchmark="terminal-bench",
        stream_metadata={
            "mode": "unpinned",
            "requested_slots": 4,
            "intentional_oversubscription": True,
        },
    )
    metadata = _run_metadata(cfg, 1)
    assert metadata["stream_scheduling"]["mode"] == "unpinned"
    assert metadata["stream_scheduling"]["requested_slots"] == 4
    assert metadata["numa_policy"] == "unpinned"


def test_run_metadata_leaves_numa_policy_empty_for_ordinary_runs():
    metadata = _run_metadata(RunConfig(benchmark="synthetic_cpu"), 1)
    assert metadata["numa_policy"] is None


def test_run_metadata_records_actual_cpuset():
    from src.streams.resources import ResourceBudget

    cfg = RunConfig(
        benchmark="terminal-bench",
        adapter_kwargs={"resource_budget": ResourceBudget((4, 5))},
    )
    metadata = _run_metadata(cfg, 1)
    assert metadata["numa_policy"] == "cpuset:4-5"


def test_cli_run_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    res = runner.invoke(app, ["run", "--benchmark", "synthetic_cpu",
                              "--num-tasks", "2", "--run-id", "cli_synth"])
    assert res.exit_code == 0, res.output
    assert "cli_synth" in res.output and "passed" in res.output
    # readable back via db show
    res2 = runner.invoke(app, ["db", "show", "cli_synth"])
    assert res2.exit_code == 0 and "synthetic" in res2.output


def test_cli_run_then_report(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    runner.invoke(app, ["run", "-b", "synthetic_cpu", "-n", "2", "--run-id", "r2r"])
    out = tmp_path / "r2r.md"
    res = runner.invoke(app, ["report", "r2r", "--format", "md", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists() and "AgentSysPerf Report" in out.read_text()


def test_record_replay_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    res = runner.invoke(app, ["run", "-b", "synthetic_cpu",
                              "--record", "a.jsonl", "--replay", "b.jsonl"])
    assert res.exit_code == 1
    assert "mutually exclusive" in res.output
