#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Regression tests for Terminal-Bench oracle scoring.

These pin the two halves of a defect that made every Harbor-backed task score
0.0 no matter what the agent did:

1. ``tests/`` was never uploaded, so ``bash /tests/test.sh`` exited 127.
2. ``_run_oracle`` read 127 as "the agent failed" and ``_score`` treats the
   oracle as authoritative, so a *correct* solution was recorded as failed.

The Docker end-to-end proof (run the task's own ``solution/solve.sh``, then the
oracle, and require reward 1.0) is opt-in via ``AGENTSYSPERF_ORACLE_E2E=1``
because it builds a container and takes ~2 minutes.

Run: poetry run pytest src/benchmarks/terminal_bench/test_oracle_scoring.py -q
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from src.benchmarks.terminal_bench.adapter import TerminalBenchAdapter
from src.benchmarks.terminal_bench.environment import EnvironmentResult
from src.protocols import TaskSpec


# ── fixtures ────────────────────────────────────────────────────────────────

class _FakeHarborHandle:
    def __init__(self, task_path: Path) -> None:
        self.task_path = task_path
        self.task_id_str = "terminal-bench/fake@latest"


class _FakeRaw:
    """Stands in for TerminalBenchTask (the object list_tasks stashes)."""

    def __init__(self, task_path: Path) -> None:
        self.extra = {"harbor": _FakeHarborHandle(task_path)}
        self.setup_commands: List[str] = []
        self.oracle_command = ""


class _FakeEnv:
    """Scripts test.sh's exit code and the contents of the reward files.

    ``test_rc`` is what ``bash /tests/test.sh`` returns. Note that a real
    Terminal-Bench test.sh returns 0 whether the tests passed or failed — its
    last statement is the ``echo N > reward.txt`` branch — so ``test_rc=0`` is
    the norm and carries no verdict.
    """

    def __init__(
        self,
        test_rc: int = 0,
        *,
        reward_txt: Optional[str] = None,
        reward_json: Optional[str] = None,
        ctrf_json: Optional[str] = None,
    ) -> None:
        self._test_rc = test_rc
        self._reward_txt = reward_txt
        self._reward_json = reward_json
        self._ctrf_json = ctrf_json
        self.commands: List[str] = []

    async def exec(self, command: str, timeout_sec: float = 120.0,
                   cwd: Optional[str] = None) -> EnvironmentResult:
        self.commands.append(command)
        if "reward.json" in command:
            if self._reward_json is None:
                return EnvironmentResult(return_code=0)  # `cat 2>/dev/null` of nothing
            return EnvironmentResult(stdout=self._reward_json, return_code=0)
        if "reward.txt" in command:
            if self._reward_txt is None:
                return EnvironmentResult(return_code=0)
            return EnvironmentResult(stdout=self._reward_txt, return_code=0)
        if "ctrf.json" in command:
            if self._ctrf_json is None:
                return EnvironmentResult(return_code=0)
            return EnvironmentResult(stdout=self._ctrf_json, return_code=0)
        return EnvironmentResult(return_code=self._test_rc)


def _ctrf(*, tests: int, failed: int) -> str:
    """A CTRF report the way pytest-json-ctrf writes it."""
    import json as _json
    return _json.dumps({"results": {"summary": {"tests": tests, "failed": failed}}})


def _harbor_task(tmp_path: Path) -> TaskSpec:
    return TaskSpec(
        id="terminal-bench/fake",
        instruction="do the thing",
        extra={"raw": _FakeRaw(tmp_path)},
    )


# ── the verdict comes from the reward file, not the exit code ───────────────

def test_reward_one_passes(tmp_path):
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(0, reward_txt="1\n")
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is True


def test_reward_zero_fails_even_though_test_sh_exited_zero(tmp_path):
    """The core defect: test.sh always exits 0, so only reward.txt can say no.

    Measured on an unsolved overfull-hbox with no agent at all: test.sh exited 0
    while reward.txt held "0". Reading the exit code scored every task as passed.
    """
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(0, reward_txt="0\n")
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is False


def test_reward_json_wins_over_reward_txt(tmp_path):
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(0, reward_json='{"main": 0}', reward_txt="1")
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is False


def test_reward_json_requires_every_reward_positive(tmp_path):
    adapter = TerminalBenchAdapter()
    assert adapter._run_oracle(
        _harbor_task(tmp_path), _FakeEnv(0, reward_json='{"a": 1, "b": 1}'),
    ) is True
    assert adapter._run_oracle(
        _harbor_task(tmp_path), _FakeEnv(0, reward_json='{"a": 1, "b": 0}'),
    ) is False


def test_missing_reward_file_is_unscored_not_passed(tmp_path):
    """No verdict written → we do not know. Never fall back to the exit code."""
    adapter = TerminalBenchAdapter()
    assert adapter._run_oracle(_harbor_task(tmp_path), _FakeEnv(0)) is None


# ── CTRF: third tier, only when no reward file survived ─────────────────────

def test_ctrf_recovers_a_verdict_when_no_reward_file(tmp_path):
    """test.sh can die between pytest and its `echo N > reward.txt`.

    The CTRF report still holds the per-test outcome, so recovering from it beats
    reporting UNSCORED. Adopted from tianmu-li's _run_harbor_oracle.
    """
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(0, ctrf_json=_ctrf(tests=4, failed=0))
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is True

    env = _FakeEnv(0, ctrf_json=_ctrf(tests=4, failed=1))
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is False


def test_reward_file_wins_over_ctrf(tmp_path):
    """Reward files are Harbor's declared contract; CTRF is pytest's."""
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(0, reward_txt="0", ctrf_json=_ctrf(tests=4, failed=0))
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is False
    assert not any("ctrf" in c for c in env.commands), \
        "a readable reward file must short-circuit before CTRF is read"


def test_ctrf_with_zero_tests_is_unscored_not_passed(tmp_path):
    """0 tests + 0 failures is a collection error, not a clean pass."""
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(0, ctrf_json=_ctrf(tests=0, failed=0))
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is None


def test_unparseable_ctrf_is_unscored(tmp_path):
    adapter = TerminalBenchAdapter()
    for blob in ("not json", '{"results": {}}', "{}", "[]"):
        env = _FakeEnv(0, ctrf_json=blob)
        assert adapter._run_oracle(_harbor_task(tmp_path), env) is None, blob


def test_ctrf_is_not_consulted_when_the_test_script_is_missing(tmp_path):
    """Exit 127 is a harness fault — short-circuit before reading any verdict."""
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(127, ctrf_json=_ctrf(tests=4, failed=0))
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is None
    assert not any("ctrf" in c or "reward" in c for c in env.commands)


def test_unparseable_reward_is_unscored(tmp_path):
    adapter = TerminalBenchAdapter()
    assert adapter._run_oracle(
        _harbor_task(tmp_path), _FakeEnv(0, reward_txt="banana"),
    ) is None


def test_oracle_exit_127_is_unscored_not_failed(tmp_path):
    """127 means /tests/test.sh is absent — a harness fault, not a task failure.

    It must NOT come back as False, because _score would then overwrite a
    correct agent result with passed=False and report a legitimate-looking 0.
    """
    adapter = TerminalBenchAdapter()
    env = _FakeEnv(127)
    assert adapter._run_oracle(_harbor_task(tmp_path), env) is None
    assert not any("reward" in c for c in env.commands), \
        "a missing test script must short-circuit before reading rewards"


def test_unscored_oracle_falls_back_to_agent_verdict(tmp_path):
    """The end-to-end consequence: a passing agent survives a broken oracle."""
    adapter = TerminalBenchAdapter()
    task = _harbor_task(tmp_path)
    unscored = adapter._run_oracle(task, _FakeEnv(127))
    result = adapter._score(
        task,
        {"passed": True, "reward": 1.0},
        elapsed_s=1.0,
        measured_s=1.0,
        oracle_passed=unscored,
    )
    assert result.passed is True
    assert result.reward == 1.0
    assert result.extra["oracle_run"] is False, "must be visible as unscored"


def test_failed_oracle_still_overrides_an_optimistic_agent(tmp_path):
    """Guardrail: a reward of 0 beats an agent that claims success."""
    adapter = TerminalBenchAdapter()
    task = _harbor_task(tmp_path)
    result = adapter._score(
        task,
        {"passed": True, "reward": 1.0},
        elapsed_s=1.0,
        measured_s=1.0,
        oracle_passed=adapter._run_oracle(task, _FakeEnv(0, reward_txt="0")),
    )
    assert result.passed is False and result.reward == 0.0
    assert result.extra["agent_self_reported_passed"] is True


# ── tests/ upload wiring (no Docker) ────────────────────────────────────────

class _RecordingHarborEnv:
    """Minimal stand-in for Harbor's DockerEnvironment."""

    def __init__(self) -> None:
        self.uploads: List[Tuple[Any, str]] = []
        self.commands: List[str] = []

    async def start(self, force_build: bool = False) -> None:
        return None

    async def upload_dir(self, source_dir: Any, target_dir: str) -> None:
        self.uploads.append((source_dir, target_dir))

    async def exec(self, command: str, **kw: Any) -> EnvironmentResult:
        self.commands.append(command)
        return EnvironmentResult(return_code=0)

    async def stop(self, delete: bool = False) -> None:
        return None


def _minimal_task_dir(root: Path, *, with_tests: bool) -> Path:
    """Build the smallest tree harbor's Task() will load.

    ``with_tests=False`` also flips the verifier to SEPARATE mode: harbor's own
    ``Task._validate_tests`` refuses to load a shared-verifier task that has no
    ``tests/test.sh``, so a task legitimately lacking one is always a
    separate-verifier task.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(
        'version = "1.0"\n'
        "[task]\n"
        'name = "terminal-bench/fake"\n'
        'description = "fake"\n'
        "[environment]\n"
        'docker_image = "example/fake:latest"\n'
        "cpus = 1\n"
        "memory_mb = 1024\n"
        + ("" if with_tests else '[verifier]\nenvironment_mode = "separate"\n')
    )
    (root / "instruction.md").write_text("do the thing\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text("FROM example/fake:latest\n")
    if with_tests:
        (root / "tests").mkdir(exist_ok=True)
        (root / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return root


def _start_with_recording_env(monkeypatch, task_dir: Path) -> _RecordingHarborEnv:
    from harbor.environments.factory import EnvironmentFactory
    from src.benchmarks.terminal_bench.harbor_environment import (
        create_harbor_environment,
    )

    recorder = _RecordingHarborEnv()
    monkeypatch.setattr(
        EnvironmentFactory, "create_environment",
        staticmethod(lambda **kw: recorder), raising=True,
    )
    env = create_harbor_environment(task_path=task_dir, session_id="t")
    asyncio.run(env.start())
    return recorder


def test_start_uploads_tests_dir_to_slash_tests(monkeypatch, tmp_path):
    task_dir = _minimal_task_dir(tmp_path / "task", with_tests=True)
    rec = _start_with_recording_env(monkeypatch, task_dir)

    targets = [t for _, t in rec.uploads]
    assert "/tests" in targets, (
        "tests/ must be uploaded — the adapter's oracle runs /tests/test.sh and "
        "Harbor does not mount it for us"
    )
    src = next(s for s, t in rec.uploads if t == "/tests")
    assert Path(src).name == "tests"


def test_start_creates_the_logs_tree(monkeypatch, tmp_path):
    """Harbor test scripts redirect into /logs/verifier; it must exist."""
    task_dir = _minimal_task_dir(tmp_path / "task", with_tests=True)
    rec = _start_with_recording_env(monkeypatch, task_dir)
    assert any("/logs/verifier" in c for c in rec.commands)


def test_start_warns_but_survives_a_task_with_no_tests(monkeypatch, tmp_path, caplog):
    task_dir = _minimal_task_dir(tmp_path / "task", with_tests=False)
    with caplog.at_level("WARNING"):
        rec = _start_with_recording_env(monkeypatch, task_dir)
    assert not rec.uploads
    assert any("cannot score" in r.getMessage() for r in caplog.records), \
        "a missing tests/ must be loud, not silent"


# ── Docker end-to-end: a correct solution must score 1.0 ────────────────────

_E2E = os.environ.get("AGENTSYSPERF_ORACLE_E2E") == "1"
_CACHE = Path.home() / ".cache/harbor/tasks/packages/terminal-bench/overfull-hbox"


def _cached_overfull_hbox() -> Optional[Path]:
    if not _CACHE.is_dir():
        return None
    for d in sorted(_CACHE.iterdir()):
        if (d / "task.toml").is_file() and (d / "solution" / "solve.sh").is_file():
            return d
    return None


def _e2e_oracle(*, solve: bool) -> Tuple[int, str]:
    """Provision overfull-hbox, optionally solve it, and run the real oracle."""
    from src.benchmarks.terminal_bench.harbor_environment import (
        create_harbor_environment,
    )

    task_dir = _cached_overfull_hbox()
    assert task_dir is not None

    async def _run() -> Tuple[int, str]:
        session = "oracle_e2e_solved" if solve else "oracle_e2e_unsolved"
        env = create_harbor_environment(task_path=task_dir, session_id=session)
        try:
            await env.start()
            if solve:
                script = (task_dir / "solution" / "solve.sh").read_text()
                await env.exec(
                    "mkdir -p /solution && cat > /solution/solve.sh <<'ASP_EOF'\n"
                    + script + "\nASP_EOF",
                    timeout_sec=30,
                )
                s = await env.exec("bash /solution/solve.sh", timeout_sec=900)
                assert s.return_code == 0, f"solve.sh failed: {s.stderr[-800:]}"
            o = await env.exec("bash /tests/test.sh", timeout_sec=600)
            r = await env.exec("cat /logs/verifier/reward.txt", timeout_sec=15)
            return o.return_code, (r.stdout or "").strip()
        finally:
            await env.stop()

    return asyncio.run(_run())


@pytest.mark.skipif(not _E2E, reason="set AGENTSYSPERF_ORACLE_E2E=1 (builds a container, ~2 min)")
@pytest.mark.skipif(_cached_overfull_hbox() is None, reason="overfull-hbox not in the harbor cache")
def test_oracle_scores_the_tasks_own_solution_as_pass():
    """Ground truth: the task's own solve.sh is a perfect agent; oracle → reward 1."""
    rc, reward = _e2e_oracle(solve=True)
    assert reward == "1", f"reward.txt = {reward!r}, expected '1'"
    assert rc == 0


@pytest.mark.skipif(not _E2E, reason="set AGENTSYSPERF_ORACLE_E2E=1 (builds a container, ~3 min)")
@pytest.mark.skipif(_cached_overfull_hbox() is None, reason="overfull-hbox not in the harbor cache")
def test_unsolved_task_scores_zero_despite_test_sh_exiting_zero():
    """The negative case, and the one that matters most.

    With no agent at all the task is definitionally unsolved, yet test.sh still
    exits 0 — its last statement is `echo 0 > reward.txt`, which succeeds. Only
    the reward file distinguishes this from a pass. Reading the exit code is what
    produced a 10/10 score on a 10-task run.
    """
    rc, reward = _e2e_oracle(solve=False)
    assert rc == 0, (
        "test.sh is expected to exit 0 even on failure; if this ever changes, "
        "the exit-code trap this test guards has gone away"
    )
    assert reward == "0", f"reward.txt = {reward!r}, expected '0' for an unsolved task"
