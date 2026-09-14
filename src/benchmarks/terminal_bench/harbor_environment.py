#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Harbor-backed environment for Terminal-Bench (Phase C).

Wraps Harbor's DockerEnvironment to satisfy the EnvironmentBackend protocol.
Manages Docker container lifecycle per task.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover — typing only
    # src.streams.orchestrator imports sanitize_compose_project_name
    # from THIS module, so importing streams.resources at module scope closes a
    # cycle: `from src.streams import ...` triggers this module, which
    # re-enters streams before it finishes, and collection fails with
    # "cannot import name sanitize_compose_project_name from partially
    # initialized module". Both uses below are annotations, deferred by
    # `from __future__ import annotations`, so a typing-only import suffices.
    from src.streams.resources import ResourceBudget

logger = logging.getLogger(__name__)


def sanitize_compose_project_name(name: str) -> str:
    """Normalize a run id the same way Harbor's Compose project does."""
    normalized = name.lower()
    if not re.match(r"^[a-z0-9]", normalized):
        normalized = "0" + normalized
    return re.sub(r"[^a-z0-9_-]", "-", normalized)


class HarborEnvironmentAdapter:
    """Wraps a Harbor DockerEnvironment to match EnvironmentBackend protocol.

    Constructed per-task by the adapter when task.extra['harbor'] exists.
    Manages Docker container lifecycle (start → exec commands → stop).

    Parameters
    ----------
    task_path:
        Path to the Harbor task directory (downloaded by harbor_loader).
    session_id:
        Unique session identifier (typically task.id).
    """

    def __init__(
        self,
        task_path: Path,
        session_id: str,
        force_build: bool = True,
    ) -> None:
        self.task_path = task_path
        # session_id is used in tempdir names and docker project names; task
        # ids contain a "/" (e.g. "terminal-bench/mteb-retrieve") which would
        # make mkdtemp try to create a nested path. Sanitize to a flat token.
        self.session_id = sanitize_compose_project_name(session_id)
        self._harbor_env: Optional[Any] = None
        self._trial_dir: Optional[Path] = None
        self._force_build = force_build
        self._budget: Optional[ResourceBudget] = None

    def pin(self, budget: ResourceBudget) -> None:
        """Apply this task's runc CPU and memory limits at creation time."""
        self._budget = budget

    def _write_resource_override(self, trial_dir: Path) -> Optional[Path]:
        if self._budget is None:
            return None
        main: dict[str, Any] = {}
        if self._budget.cpuset:
            main["cpuset"] = self._budget.cpuset_str()
        if self._budget.memory_mb is not None:
            main["mem_limit"] = f"{self._budget.memory_mb}m"
        path = trial_dir / "docker-compose-resource.json"
        path.write_text(json.dumps({"services": {"main": main}}, indent=2))
        return path

    async def start(self) -> None:
        """Provision the Harbor Docker environment."""
        # Lazy import Harbor dependencies
        try:
            from harbor.models.task.task import Task
            from harbor.models.environment_type import EnvironmentType
            from harbor.environments.factory import EnvironmentFactory
            from harbor.models.trial.paths import TrialPaths
        except ImportError as e:
            raise ImportError(
                f"Harbor >=0.8.0 required for Phase C: {e}"
            ) from e

        # Load Harbor Task
        task = Task(self.task_path)

        # Create trial paths (Harbor requires this for bind mounts + logs)
        self._trial_dir = Path(tempfile.mkdtemp(prefix=f"tb_{self.session_id}_"))
        trial_paths = TrialPaths(self._trial_dir)
        trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)

        extra_compose_kwargs: dict[str, Any] = {}
        override = self._write_resource_override(self._trial_dir)
        if override is not None:
            extra_compose_kwargs["extra_docker_compose"] = [override]

        # Create Docker environment
        self._harbor_env = EnvironmentFactory.create_environment(
            type=EnvironmentType.DOCKER,
            environment_dir=task.paths.environment_dir,
            environment_name=task.name,
            session_id=self.session_id,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            **extra_compose_kwargs,
        )

        # Task-sized streams pre-pull images before timing and pass
        # force_build=False. The normal single-run path keeps the historical
        # force_build=True fallback. Harbor 0.8.0 requires this argument.
        logger.info(
            "Starting Harbor environment for %s (force_build=%s)...",
            task.name,
            self._force_build,
        )
        await self._harbor_env.start(force_build=self._force_build)

        # Upload the task's tests/ into the container, and create the /logs tree
        # the test scripts write to.
        #
        # This is REQUIRED for scoring, and its absence made every Harbor task
        # score 0. The adapter's oracle runs `bash /tests/test.sh`, and the
        # comment there claimed "Harbor mounts tests/ by convention" — it does
        # not. Harbor uploads tests/ inside its own Verifier.verify(), which this
        # adapter never calls because it drives the environment directly.
        # Measured before this fix: `bash /tests/test.sh` exited 127 (no such
        # file) in a freshly started container, and _score treats a failed oracle
        # as authoritative over the agent's own verdict, so a CORRECT solution
        # was recorded as passed=False. Verified after: upload tests/, run
        # overfull-hbox's own solution/solve.sh, then test.sh -> 4 passed,
        # /logs/verifier/reward.txt == 1.
        tests_dir = getattr(task.paths, "tests_dir", None) or self.task_path / "tests"
        if Path(tests_dir).is_dir():
            await self._harbor_env.upload_dir(tests_dir, "/tests")
            logger.info("Uploaded %s -> /tests", tests_dir)
        else:
            # Not fatal: a separate-verifier task may score elsewhere. But say
            # so, because the alternative is a silent 0 for every trial.
            logger.warning(
                "No tests/ dir for %s (looked in %s) — /tests will be absent and "
                "the Harbor oracle cannot score this task",
                task.name, tests_dir,
            )
        # Harbor's test scripts redirect reward/CTRF output into /logs/verifier;
        # without these dirs the script dies on the redirect, not the assertion.
        await self._harbor_env.exec(
            "mkdir -p /logs/verifier /logs/agent /logs/artifacts", timeout_sec=30,
        )
        logger.info(f"Harbor environment ready: {task.name}")

    async def stop(self) -> None:
        """Stop and clean up the Harbor environment."""
        if self._harbor_env is not None:
            try:
                await self._harbor_env.stop(delete=self._force_build)
            except Exception as e:
                logger.warning(f"Harbor env stop failed: {e}", exc_info=True)

        # Clean up trial dir
        if self._trial_dir and self._trial_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._trial_dir, ignore_errors=True)
            except Exception:
                pass

    async def exec(
        self,
        command: str,
        timeout_sec: float = 120.0,
        cwd: Optional[str] = None,
    ) -> Any:  # Returns EnvironmentResult
        """Execute a command in the Harbor Docker container.

        Returns
        -------
        EnvironmentResult with stdout, stderr, return_code.
        """
        if self._harbor_env is None:
            raise RuntimeError("Harbor environment not started. Call start() first.")

        # Harbor's exec returns ExecResult (similar shape to EnvironmentResult)
        harbor_result = await self._harbor_env.exec(
            command=command,
            timeout_sec=timeout_sec,
            cwd=cwd,
        )

        # Convert Harbor's ExecResult to our EnvironmentResult
        from src.benchmarks.terminal_bench.environment import EnvironmentResult
        return EnvironmentResult(
            stdout=harbor_result.stdout or "",
            stderr=harbor_result.stderr or "",
            return_code=harbor_result.return_code,
        )


def create_harbor_environment(
    task_path: Path,
    session_id: str,
    force_build: bool = True,
) -> HarborEnvironmentAdapter:
    """Factory function for creating Harbor environments.

    Used by the adapter when task.extra['harbor'] is present.
    """
    return HarborEnvironmentAdapter(
        task_path=task_path,
        session_id=session_id,
        force_build=force_build,
    )


__all__ = [
    "HarborEnvironmentAdapter",
    "create_harbor_environment",
    "sanitize_compose_project_name",
]
