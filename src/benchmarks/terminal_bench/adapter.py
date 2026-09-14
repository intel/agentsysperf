#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Terminal-Bench adapter for AgentSysPerf.

This is a **reference implementation** of :class:`BenchmarkAdapter`
intended to serve two purposes:

1. Drive Terminal-Bench tasks through any agent that satisfies the
   :class:`AgentInvoker` Protocol — same parquet schema and the same
   measurement plumbing as the SWE-Bench adapter, regardless of which
   runtime is wired in.
2. Be the canonical example external developers read when adding a
   new benchmark adapter.  Read it alongside ``docs/EXTENDING.md`` and
   the ``add-benchmark`` Claude Code agent.

Decoupling note:  this adapter imports ONLY from
:mod:`src.protocols` and from sibling benchmark code
(``dataset.py``, ``environment.py``).  No AgentFlow runtime imports.
A different ``AgentInvoker`` (e.g. a LiteLLM CustomLLM-backed invoker)
plugs into the same adapter without code changes.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

from src.protocols import (
    AgentInvoker,
    TaskResult,
    TaskSpec,
)
from src.runner import RunContext, track_span

if TYPE_CHECKING:
    # Annotation-only, so the name is bound for type checkers and linters
    # without adding a runtime import edge. A module-scope import here would
    # close the streams <-> harbor_environment cycle; `_pin_environment` needs
    # the name at RUNTIME and imports it locally for that reason.
    from src.streams.resources import ResourceBudget

logger = logging.getLogger(__name__)


class TerminalBenchAdapter:
    """Drives Terminal-Bench tasks through any :class:`AgentInvoker`.

    Tasks are loaded via :mod:`src.benchmarks.terminal_bench.dataset`
    (Harbor-managed by default; JSONL or synthetic samples available).
    Per-task environments are provisioned via
    :class:`StandaloneEnvironment` for development or the
    Harbor-managed Docker environment for full benchmark runs.

    Parameters
    ----------
    dataset_loader:
        Optional callable returning ``Iterable[TerminalBenchTask]``.
        Defaults to :func:`generate_sample_tasks` so tests + smoke
        runs work with no external dataset.
    environment_factory:
        Optional callable ``(TerminalBenchTask) -> EnvironmentBackend``
        that provisions a fresh env per task.  Defaults to
        :class:`StandaloneEnvironment` (no Docker).  Pass a Harbor
        adapter factory for full benchmark runs.
    """

    name = "terminal_bench"
    version = "0.1.0"

    def __init__(
        self,
        *,
        dataset_loader: Optional[Any] = None,
        environment_factory: Optional[Any] = None,
        resource_budget: Optional[ResourceBudget] = None,
        force_build: bool = True,
    ) -> None:
        self._dataset_loader = dataset_loader
        self._environment_factory = environment_factory
        self._resource_budget = resource_budget
        self._force_build = force_build

    # ─── BenchmarkAdapter Protocol ───────────────────────────────────

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Yield :class:`TaskSpec` matching the include/exclude filters.

        Filters use exact-match against ``task_id``; glob patterns
        are not supported in v1 (TODO if useful).
        """
        from src.benchmarks.terminal_bench.dataset import (
            generate_sample_tasks,
        )

        loader = self._dataset_loader or generate_sample_tasks
        raw_tasks = loader() if callable(loader) else iter(loader)

        include_set = set(include) if include else None
        exclude_set = set(exclude) if exclude else None
        emitted = 0

        # Helper: read a field whether ``raw`` is a TerminalBenchTask
        # dataclass or a plain dict.  We don't want ``a or b``-style
        # fallback because it conflates "missing" with falsy ("", 0,
        # False) — the agent_timeout_sec field is real and may be 0.
        def _field(obj: Any, name: str, default: Any) -> Any:
            if isinstance(obj, Mapping):
                return obj.get(name, default)
            return getattr(obj, name, default)

        for raw in raw_tasks:
            task_id = _field(raw, "task_id", None)
            if task_id is None:
                continue
            if include_set is not None and task_id not in include_set:
                continue
            if exclude_set is not None and task_id in exclude_set:
                continue

            yield TaskSpec(
                id=task_id,
                instruction=_field(raw, "instruction", ""),
                category=_field(raw, "category", ""),
                difficulty=_field(raw, "difficulty", ""),
                cpu_budget=_field(raw, "cpus", 1),
                memory_mb=_field(raw, "memory_mb", 2048),
                timeout_s=_field(raw, "agent_timeout_sec", 900.0),
                expects_internet=_field(raw, "allow_internet", False),
                # Hold the original task object for run_task to consume.
                # AgentSysPerf's TaskSpec is a frozen dataclass with an
                # extra mapping; we stash the source under "raw" to
                # avoid losing fields we didn't elevate to first-class.
                extra={"raw": raw},
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def run_task(
        self,
        task: TaskSpec,
        *,
        agent_invoker: AgentInvoker,
        on_step: Optional[Any] = None,
        run_context: Optional[RunContext] = None,
    ) -> TaskResult:
        """Provision env + run setup + drive the agent + score via oracle.

        Phase tagging: this adapter owns ``admit`` (environment provisioning +
        setup) and ``commit`` (oracle verification). ``reason``, ``act`` and
        ``retrieve`` are tagged inside the agent loop, which is the only place
        that sees turns.

        ``retrieve`` is emitted when a command performs semantic or
        index-backed retrieval — vector/ANN search, embedding generation,
        BM25/TF-IDF, inverted indexes, reranking (see
        :mod:`~src.benchmarks.terminal_bench.retrieval`). Filesystem
        inspection (``grep``/``find``/``cat``) stays ``act``: tagging a ``grep``
        as retrieval would make PhaseProfiler recommend int8 HNSW quantization
        for a file search.

        Phases with no spans are skipped by the analyzer, so the phase count is
        a property of the workload, not of the harness: a shell-only task
        reports 4, while a retrieval task (e.g. ``mteb-retrieve``) reports all 5.
        """
        measured_t0 = time.time()
        env = self._make_environment(task, run_context=run_context)
        try:
            # Phase 01: Admit — container provisioning and task setup. Real,
            # often multi-second work that was previously invisible.
            with self._phase_span(run_context, task, "admit", kind="governance"):
                self._start_environment(env)
                self._run_setup(task, env)
            t0 = time.time()
            try:
                # session_hint = task.id is the contract for runtimes
                # that support KV-cache affinity.  Runtimes that don't
                # support it (LiteLLM CustomLLM path) ignore the field.
                raw_result = agent_invoker.invoke(
                    task.instruction,
                    environment=env,
                    metadata={"task_id": task.id, "category": task.category},
                    session_hint=task.id,
                    run_context=run_context,
                )
            except Exception as e:
                logger.warning("TerminalBenchAdapter.run_task: %s failed: %s",
                               task.id, e, exc_info=True)
                return TaskResult(
                    task_id=task.id,
                    passed=False,
                    reward=0.0,
                    error=f"{type(e).__name__}: {e}",
                    extra={
                        "elapsed_s": time.time() - t0,
                        "measured_s": time.time() - measured_t0,
                    },
                )

            elapsed = time.time() - t0
            # Phase 05: Commit — oracle verification / writeback.
            with self._phase_span(run_context, task, "commit", kind="governance"):
                oracle_passed = self._run_oracle(task, env)
            return self._score(
                task,
                raw_result,
                elapsed_s=elapsed,
                measured_s=time.time() - measured_t0,
                oracle_passed=oracle_passed,
            )
        finally:
            self._teardown_environment(env)

    def teardown(self) -> None:
        """No persistent resources to release at the adapter level."""
        return None

    # ─── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def _phase_span(
        run_context: Optional[RunContext],
        task: TaskSpec,
        phase: str,
        *,
        kind: str,
    ):
        """Open a phase sub-span, or do nothing when running unmeasured.

        ``run_context`` is None for callers that don't measure (and for every
        adapter path exercised without a RunContext), so the phase spans must
        degrade to a plain no-op rather than requiring a context.

        The span id is ``{task.id}/{phase}``, matching the convention the agent
        loop uses for its own sub-spans (``{task_id}/turn_N_llm``) so all of a
        task's phases sort together under the task-level parent span.
        """
        if run_context is None:
            yield None
            return
        with track_span(
            run_context, f"{task.id}/{phase}", kind=kind,
            node_id=phase, phase=phase,
        ) as span:
            yield span

    def _make_environment(
        self,
        task: TaskSpec,
        *,
        run_context: Optional[RunContext] = None,
    ) -> Any:
        """Build the environment handle the agent will operate in.

        If task carries Harbor metadata (task.extra['harbor']), provisions
        a Harbor Docker container. Otherwise falls back to StandaloneEnvironment.
        """
        if self._environment_factory is not None:
            env = self._environment_factory(task)
            self._pin_environment(env, task)
            return env

        # Check if task is Harbor-backed (Phase C). list_tasks() stashes the
        # source TerminalBenchTask under extra["raw"], so the harbor handle
        # lives at extra["raw"].extra["harbor"] — NOT extra["harbor"]. (A prior
        # bug looked only at extra["harbor"], always missed, and silently fell
        # back to host subprocess execution. See
        # docs/INCIDENT_host_exec_venv_pollution.md.)
        harbor_handle = (task.extra or {}).get("harbor")
        if harbor_handle is None:
            raw = (task.extra or {}).get("raw")
            raw_extra = getattr(raw, "extra", None) or {}
            harbor_handle = raw_extra.get("harbor")

        if harbor_handle is not None:
            from src.benchmarks.terminal_bench.harbor_environment import (
                create_harbor_environment,
            )
            session_id = run_context.run_id if run_context is not None else task.id
            env = create_harbor_environment(
                task_path=harbor_handle.task_path,
                session_id=session_id,
                force_build=self._force_build,
            )
            self._pin_environment(env, task)
            return env

        # Default: standalone subprocess env for Phase A/B smoke runs.
        from src.benchmarks.terminal_bench.environment import (
            StandaloneEnvironment,
        )
        env = StandaloneEnvironment()
        self._pin_environment(env, task)
        return env

    def _pin_environment(self, env: Any, task: TaskSpec) -> None:
        """Apply the worker cpuset and this task's declared memory cap."""
        if self._resource_budget is None:
            return
        pin = getattr(env, "pin", None)
        if pin is None:
            raise RuntimeError(
                f"{type(env).__name__} does not support task-sized resource pinning"
            )
        # Imported here, not at module scope: src.streams imports
        # harbor_environment (for sanitize_compose_project_name), which this
        # module also imports, so a top-level `from src.streams.resources
        # import ResourceBudget` closes a cycle and breaks collection with
        # "partially initialized module". The annotation on __init__ is deferred
        # by `from __future__ import annotations`, but this is a runtime call, so
        # the name must genuinely be bound here.
        from src.streams.resources import ResourceBudget

        pin(
            ResourceBudget(
                cpuset=self._resource_budget.cpuset,
                memory_mb=task.memory_mb,
            )
        )

    def _run_setup(self, task: TaskSpec, env: Any) -> None:
        """Run task.setup_commands sequentially in the env. Aborts on failure."""
        raw = (task.extra or {}).get("raw") if task.extra else None
        setup = getattr(raw, "setup_commands", None) if raw is not None else None
        if not setup:
            return
        for cmd in setup:
            res = self._exec_in_env(env, cmd, timeout_sec=30.0)
            if res is None:
                logger.warning(
                    "TerminalBenchAdapter: setup skipped for %s (no exec on env)", task.id,
                )
                return
            if res.return_code != 0:
                raise RuntimeError(
                    f"Setup failed for {task.id}: '{cmd}' exited {res.return_code}\n"
                    f"stderr: {res.stderr[:500]}"
                )

    def _run_oracle(self, task: TaskSpec, env: Any) -> Optional[bool]:
        """Run task oracle: inline oracle_command (Phase B) or Harbor tests/test.sh (Phase C).

        Returns True/False, or None if no oracle.
        """
        # Phase C: Harbor tasks have tests/test.sh
        # Harbor handle is nested: task.extra["raw"] is TerminalBenchTask, which has .extra["harbor"]
        raw = (task.extra or {}).get("raw")
        harbor_handle = getattr(raw, "extra", {}).get("harbor") if raw is not None else None
        if harbor_handle is not None:
            # Harbor test script is at task_path/tests/test.sh (or test.ps1 for
            # Windows). HarborEnvironmentAdapter.start() uploads it to /tests —
            # Harbor does NOT mount it for us, contrary to what this comment
            # used to claim.
            test_script = "/tests/test.sh"
            logger.info(f"Running Harbor oracle for {task.id}: {test_script}")
            res = self._exec_in_env(env, f"bash {test_script}", timeout_sec=300.0)
            if res is None:
                logger.warning(
                    "TerminalBenchAdapter: Harbor oracle exec failed for %s", task.id,
                )
                return None
            # Exit 127 is the shell's "command not found": the test script itself
            # is missing, which is a HARNESS fault, not the agent failing the
            # task. Returning False here is what turned every task into a
            # confident 0 before the upload landed. Return None so _score falls
            # back to the agent's verdict and the cell reads as UNSCORED.
            if res.return_code == 127:
                logger.error(
                    "TerminalBenchAdapter: %s is missing in the container for %s "
                    "(exit 127) — harness fault, NOT a task failure. Scoring falls "
                    "back to the agent's own verdict and this task is UNSCORED.",
                    test_script, task.id,
                )
                return None
            # The verdict lives in the reward file, NOT in test.sh's exit code.
            # Every Terminal-Bench test.sh ends with
            #     if [ $? -eq 0 ]; then echo 1 > .../reward.txt
            #     else echo 0 > .../reward.txt; fi
            # so the script's own exit status is that of a successful `echo` —
            # it is 0 whether the tests passed or failed. Reading return_code
            # here scored every task as passed: measured on an unsolved
            # overfull-hbox with no agent at all, test.sh exited 0 while
            # reward.txt held "0". Harbor's contract is reward.txt/reward.json
            # (harbor.models.trial.paths.reward_text_path).
            return self._read_reward_verdict(task, env, exec_rc=res.return_code)

        # Phase B: inline oracle_command from TerminalBenchTask. Here the exit
        # code IS the contract — a bare shell command, not a Harbor test script.
        raw = (task.extra or {}).get("raw") if task.extra else None
        oracle = getattr(raw, "oracle_command", "") if raw is not None else ""
        if not oracle:
            return None
        res = self._exec_in_env(env, oracle, timeout_sec=60.0)
        if res is None:
            logger.warning(
                "TerminalBenchAdapter: oracle skipped for %s (no exec on env)", task.id,
            )
            return None
        return res.return_code == 0

    # Harbor's verifier writes the verdict to one of these; reward.json wins
    # when both exist, matching harbor.models.trial.paths.
    _REWARD_JSON = "/logs/verifier/reward.json"
    _REWARD_TEXT = "/logs/verifier/reward.txt"
    # Last resort: pytest's CTRF report, which the stock test.sh also writes.
    _CTRF_JSON = "/logs/verifier/ctrf.json"

    def _read_reward_verdict(
        self, task: TaskSpec, env: Any, *, exec_rc: int,
    ) -> Optional[bool]:
        """Return the oracle verdict from the reward file, or None if unreadable.

        Harbor's contract (``harbor.models.trial.paths``) is that the verifier
        writes its reward to ``/logs/verifier/reward.json`` or ``reward.txt``. A
        task passes when every reward is > 0. None means "could not determine" —
        the caller treats that as UNSCORED rather than inventing a verdict.
        """
        res = self._exec_in_env(
            env, f"cat {self._REWARD_JSON} 2>/dev/null", timeout_sec=30.0,
        )
        if res is not None and res.return_code == 0 and (res.stdout or "").strip():
            import json
            try:
                payload = json.loads(res.stdout)
            except ValueError:
                logger.warning(
                    "TerminalBenchAdapter: %s for %s is not valid JSON: %.200s",
                    self._REWARD_JSON, task.id, res.stdout,
                )
            else:
                rewards = (
                    payload if isinstance(payload, Mapping) else {"reward": payload}
                )
                values = [v for v in rewards.values() if isinstance(v, (int, float))]
                if values:
                    passed = all(float(v) > 0 for v in values)
                    logger.info(
                        "Oracle verdict for %s from reward.json: %s (%s)",
                        task.id, passed, dict(rewards),
                    )
                    return passed

        res = self._exec_in_env(
            env, f"cat {self._REWARD_TEXT} 2>/dev/null", timeout_sec=30.0,
        )
        raw = (res.stdout or "").strip() if res is not None else ""
        if res is not None and res.return_code == 0 and raw:
            try:
                reward = float(raw)
            except ValueError:
                logger.warning(
                    "TerminalBenchAdapter: %s for %s is not a number: %r",
                    self._REWARD_TEXT, task.id, raw,
                )
            else:
                passed = reward > 0
                logger.info(
                    "Oracle verdict for %s from reward.txt: %s (reward=%s)",
                    task.id, passed, reward,
                )
                return passed

        # Third tier: the CTRF report. Neither reward file was readable, but the
        # stock test.sh also runs pytest with `--ctrf /logs/verifier/ctrf.json`,
        # so a per-test summary usually survives even when the reward write did
        # not — e.g. the script died between pytest and its `echo N > reward.txt`
        # (a full disk, a kill during teardown, a task whose test.sh omits the
        # reward branch). Recovering the verdict from CTRF is strictly better
        # than UNSCORED, and it is the same evidence a human would read.
        #
        # Adopted from tianmu-li's _run_harbor_oracle on feat/agentic-dataset-
        # export, which found this independently. Reward files still win when
        # present: they are Harbor's declared contract
        # (harbor.models.trial.paths), CTRF is pytest's.
        ctrf = self._read_ctrf_verdict(task, env)
        if ctrf is not None:
            return ctrf

        # Nothing readable. The test script ran (we have its exit code) but left
        # no verdict anywhere, so we genuinely do not know. Do NOT fall back to
        # the exit code: it is 0 even when the tests fail.
        logger.error(
            "TerminalBenchAdapter: no readable verdict for %s (%s / %s / %s); "
            "test.sh exited %d, but its exit code is NOT the verdict. Marking "
            "this task UNSCORED rather than guessing.",
            task.id, self._REWARD_JSON, self._REWARD_TEXT, self._CTRF_JSON, exec_rc,
        )
        return None

    def _read_ctrf_verdict(self, task: TaskSpec, env: Any) -> Optional[bool]:
        """Recover a verdict from pytest's CTRF report, or None if unusable.

        Passes only when the run reported at least one test AND zero failures.
        ``tests > 0`` is load-bearing: a CTRF file with 0 tests and 0 failures is
        a collection error, which would otherwise read as a clean pass.
        """
        res = self._exec_in_env(
            env, f"cat {self._CTRF_JSON} 2>/dev/null", timeout_sec=30.0,
        )
        if res is None or res.return_code != 0 or not (res.stdout or "").strip():
            return None
        import json
        try:
            summary = json.loads(res.stdout)["results"]["summary"]
            total = int(summary.get("tests", 0))
            failed = int(summary.get("failed", 0))
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning(
                "TerminalBenchAdapter: unparsable %s for %s",
                self._CTRF_JSON, task.id,
            )
            return None
        if total <= 0:
            logger.warning(
                "TerminalBenchAdapter: %s for %s reports 0 tests — a collection "
                "error, not a pass. Leaving UNSCORED.",
                self._CTRF_JSON, task.id,
            )
            return None
        passed = failed == 0
        logger.info(
            "Oracle verdict for %s from ctrf.json (no reward file): %s "
            "(%d test(s), %d failed)",
            task.id, passed, total, failed,
        )
        return passed

    @staticmethod
    def _start_environment(env: Any) -> None:
        """Start the environment (provisions the container for Harbor envs).

        Drives the backend's async ``start()`` synchronously. Without this the
        Harbor container is never provisioned and command execution would fall
        back to the host — see docs/INCIDENT_host_exec_venv_pollution.md.
        """
        start = getattr(env, "start", None)
        if start is None:
            return
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(start):
            asyncio.run(start())
        else:
            start()

    @staticmethod
    def _exec_in_env(env: Any, command: str, *, timeout_sec: float) -> Any:
        """Run *command* in *env*, handling sync/async backends transparently.

        Returns the env's result object (with stdout/stderr/return_code),
        or None if the env exposes no exec method.
        """
        from src.benchmarks.terminal_bench.environment import (
            SyncEnvironmentWrapper,
        )
        exec_attr = getattr(env, "exec", None)
        if exec_attr is None:
            return None
        # All current backends expose async exec; wrap for sync use.
        return SyncEnvironmentWrapper(env).exec(command, timeout_sec=timeout_sec)

    def _teardown_environment(self, env: Any) -> None:
        """Best-effort env cleanup: async stop() (Harbor) + teardown/close."""
        # Async stop() — stops + removes the Harbor container/image.
        stop = getattr(env, "stop", None)
        if stop is not None:
            import asyncio
            import inspect
            try:
                if inspect.iscoroutinefunction(stop):
                    asyncio.run(stop())
                else:
                    stop()
            except Exception:  # noqa: BLE001
                logger.debug("TerminalBenchAdapter: env stop failed",
                             exc_info=True)

        teardown = getattr(env, "teardown", None) or getattr(env, "close", None)
        if callable(teardown):
            try:
                teardown()
            except Exception:  # noqa: BLE001
                logger.debug("TerminalBenchAdapter: env teardown failed",
                             exc_info=True)

    def _score(
        self,
        task: TaskSpec,
        raw_result: Any,
        *,
        elapsed_s: float,
        measured_s: float,
        oracle_passed: Optional[bool] = None,
    ) -> TaskResult:
        """Score the agent's output against the task's test oracle.

        If an oracle was supplied (Phase B inline shell oracle, Phase C
        Harbor test container), its result is authoritative. Otherwise
        fall back to the agent's self-reported passed/reward.
        """
        def _get(name: str, default: Any = None) -> Any:
            if hasattr(raw_result, name):
                return getattr(raw_result, name)
            if isinstance(raw_result, Mapping):
                return raw_result.get(name, default)
            return default

        if oracle_passed is not None:
            passed = oracle_passed
            reward = 1.0 if passed else 0.0
        else:
            passed = bool(_get("passed", False))
            reward = float(_get("reward", 1.0 if passed else 0.0))

        error = _get("error")
        traj = _get("trajectory_path")

        return TaskResult(
            task_id=task.id,
            passed=passed,
            reward=reward,
            error=str(error) if error else None,
            trajectory_path=Path(traj) if traj else None,
            extra={
                "elapsed_s": elapsed_s,
                "measured_s": measured_s,
                "raw_result_type": type(raw_result).__name__,
                "oracle_run": oracle_passed is not None,
                "agent_self_reported_passed": bool(_get("passed", False)),
                "submitted": bool(_get("submitted", False)),
                "num_turns": _get("num_turns", 0),
                "num_commands": _get("num_commands", 0),
            },
        )


__all__ = ["TerminalBenchAdapter"]
