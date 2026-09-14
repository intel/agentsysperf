#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Terminal-Bench phase tagging: 4 phases, and never a fabricated `retrieve`.

The adapter owns `admit` (env provisioning + setup) and `commit` (oracle); the
agent loop owns `reason` and `act`. `retrieve` means semantic retrieval —
embeddings / vector index — so a Terminal-Bench agent's `grep`/`find`/`cat`
shell actions are `act`, not retrieval. Tagging them `retrieve` would make
PhaseProfiler recommend HNSW quantization for a file search, so these tests pin
the absence as much as the presence.

Run: poetry run pytest src/benchmarks/terminal_bench/test_phase_spans.py -q
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.benchmarks.terminal_bench.adapter import TerminalBenchAdapter
from src.measurements.l1_subspan import L1SubSpanMeasurement
from src.protocols import TaskResult, TaskSpec
from src.runner import RunContext, track_span


class _FakeEnv:
    def start(self):
        pass

    def cleanup(self):
        pass


class _FakeInvoker:
    """Stands in for the agent loop, emitting its reason/act spans."""

    def __init__(self, turns: int = 2) -> None:
        self._turns = turns

    def invoke(self, instruction, **kw):
        ctx = kw.get("run_context")
        task_id = (kw.get("metadata") or {}).get("task_id", "t")
        if ctx is not None:
            for turn in range(self._turns):
                with track_span(ctx, f"{task_id}/turn_{turn}_llm",
                                kind="inference", node_id=f"llm_{turn}",
                                phase="reason"):
                    pass
                with track_span(ctx, f"{task_id}/turn_{turn}_cmd",
                                kind="execution", node_id=f"cmd_{turn}",
                                phase="act"):
                    pass
        return {"submitted": True, "num_turns": self._turns}


def _adapter(*, oracle_raises: bool = False) -> TerminalBenchAdapter:
    """An adapter with every environment/oracle touchpoint stubbed."""
    a = TerminalBenchAdapter.__new__(TerminalBenchAdapter)
    a._make_environment = lambda task, *, run_context=None: _FakeEnv()
    a._start_environment = lambda env: None
    a._run_setup = lambda task, env: None

    def _oracle(task, env):
        if oracle_raises:
            raise RuntimeError("oracle blew up")
        return True

    a._run_oracle = _oracle
    a._teardown_environment = lambda env: None
    # **kw so this stub tracks _score's real signature: the task-sized-streams
    # work added a required measured_s keyword, and a stub pinned to the old
    # signature raises TypeError from inside run_task's except clause, which
    # reports as "the agent failed" rather than "the test stub is stale".
    a._score = lambda task, raw, *, elapsed_s, oracle_passed, **kw: TaskResult(
        task_id=task.id, passed=oracle_passed, reward=1.0,
    )
    return a


def _run(tmp_path, adapter=None, turns=2):
    spec = TaskSpec(id="terminal-bench/demo", instruction="do it", category="t")
    ctx = RunContext(measurements=[L1SubSpanMeasurement()],
                     output_dir=tmp_path, run_id="phasetest")
    with ctx:
        with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
            (adapter or _adapter()).run_task(
                spec, agent_invoker=_FakeInvoker(turns), run_context=ctx,
            )
    return ctx


def test_emits_exactly_four_phases(tmp_path):
    ctx = _run(tmp_path)
    phases = {r.payload.get("phase") for r in ctx.records if r.payload.get("phase")}
    assert phases == {"admit", "reason", "act", "commit"}


def test_never_fabricates_a_retrieve_phase(tmp_path):
    """A shell-only agent emits no retrieve span.

    `retrieve` is a real phase and IS emitted for semantic / index-backed work
    (see test_retrieval.py and the agent loop), but it must never be invented
    for a task that only ran shell commands — the phase count is a property of
    the workload, not a fixed 5.
    """
    ctx = _run(tmp_path)
    assert not any(r.payload.get("phase") == "retrieve" for r in ctx.records)


def test_a_retrieval_command_produces_a_retrieve_span(tmp_path):
    """End-to-end: a vector-search command lands as phase=retrieve.

    Proves the tag survives the whole path — classifier -> track_span ->
    MeasurementRecord payload — which is what PhaseProfiler and the dashboard
    read.
    """
    from src.benchmarks.terminal_bench.retrieval import classify_phase

    class _RetrievingInvoker:
        def invoke(self, instruction, **kw):
            ctx = kw.get("run_context")
            task_id = (kw.get("metadata") or {}).get("task_id", "t")
            commands = [
                "python -c 'import faiss; faiss.read_index(\"docs.faiss\")'",
                "grep -rn 'def main' /app",
            ]
            if ctx is not None:
                for turn, cmd in enumerate(commands):
                    with track_span(ctx, f"{task_id}/turn_{turn}_llm",
                                    kind="inference", node_id=f"llm_{turn}",
                                    phase="reason"):
                        pass
                    _phase = classify_phase(cmd)
                    with track_span(ctx, f"{task_id}/turn_{turn}_cmd",
                                    kind="retrieval" if _phase == "retrieve" else "execution",
                                    node_id=f"cmd_{turn}", phase=_phase):
                        pass
            return {"submitted": True, "num_turns": len(commands)}

    spec = TaskSpec(id="terminal-bench/vec", instruction="search", category="t")
    ctx = RunContext(measurements=[L1SubSpanMeasurement()],
                     run_id="phase_retrieve", output_dir=tmp_path)
    with ctx:
        _adapter().run_task(spec, agent_invoker=_RetrievingInvoker(),
                            run_context=ctx)

    counts = Counter(r.payload.get("phase") for r in ctx.records)
    assert counts["retrieve"] == 1, "the faiss command must be phase=retrieve"
    assert counts["act"] == 1, "the grep must stay phase=act"
    # All five phases present for a workload that genuinely has all five.
    assert {"admit", "retrieve", "reason", "act", "commit"} <= set(counts)


def test_admit_and_commit_are_emitted_once_per_task(tmp_path):
    counts = Counter(r.payload.get("phase") for r in _run(tmp_path, turns=3).records)
    assert counts["admit"] == 1
    assert counts["commit"] == 1
    # ...while the turn phases scale with turns, proving they're distinct spans.
    assert counts["reason"] == 3
    assert counts["act"] == 3


def test_phase_span_ids_nest_under_the_task(tmp_path):
    ids = {r.span_id for r in _run(tmp_path).records}
    assert "terminal-bench/demo/admit" in ids
    assert "terminal-bench/demo/commit" in ids


def test_no_run_context_is_a_noop_not_a_crash(tmp_path):
    """Unmeasured callers must still be able to run tasks."""
    spec = TaskSpec(id="terminal-bench/demo", instruction="do it", category="t")
    result = _adapter().run_task(
        spec, agent_invoker=_FakeInvoker(), run_context=None,
    )
    assert result.passed is True


def test_commit_span_closes_when_the_oracle_raises(tmp_path):
    """A failing oracle must not leak an open span or lose admit."""
    spec = TaskSpec(id="terminal-bench/demo", instruction="do it", category="t")
    ctx = RunContext(measurements=[L1SubSpanMeasurement()],
                     output_dir=tmp_path, run_id="phasetest")
    try:
        with ctx:
            with track_span(ctx, spec.id, kind="terminal_bench", node_id=spec.id):
                _adapter(oracle_raises=True).run_task(
                    spec, agent_invoker=_FakeInvoker(), run_context=ctx,
                )
    except RuntimeError:
        pass
    phases = Counter(r.payload.get("phase") for r in ctx.records)
    assert phases["admit"] == 1, "admit must survive a later failure"


def test_phase_breakdown_omits_retrieve_and_sums_to_100(tmp_path):
    """End-to-end: what PhaseProfiler reports for a Terminal-Bench run."""
    from src.analyzers.phase_profiler import PhaseProfiler

    results = list(PhaseProfiler().analyze(_run(tmp_path).records))
    assert results, "PhaseProfiler emitted no verdict"
    bd = results[0].evidence.get("phase_breakdown", {})
    assert "retrieve" not in bd
    assert set(bd) == {"admit", "reason", "act", "commit"}
    assert abs(sum(v["wall_pct"] for v in bd.values()) - 100.0) < 0.5
