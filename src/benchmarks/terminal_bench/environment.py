#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Environment abstraction for TerminalBench task execution.

Provides a unified interface for running commands that works in both
**Harbor-native mode** (delegating to Harbor's ``BaseEnvironment``) and
**standalone mode** (local subprocess or lightweight Docker container).

Key abstractions:

- :class:`EnvironmentResult`: outcome of an ``exec()`` call.
- :class:`HarborEnvironmentAdapter`: wraps Harbor's BaseEnvironment.
- :class:`StandaloneEnvironment`: local subprocess execution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EnvironmentResult:
    """Outcome of a command execution in an environment."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    @property
    def output(self) -> str:
        """Combined stdout + stderr (separated by newline if both non-empty)."""
        parts = [p for p in (self.stdout.strip(), self.stderr.strip()) if p]
        return "\n".join(parts)

    @property
    def success(self) -> bool:
        return self.return_code == 0


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT PROTOCOL
# ═══════════════════════════════════════════════════════════════════


class EnvironmentBackend(Protocol):
    """Protocol for environment backends.

    Both Harbor-native and standalone environments implement this.
    The agent loop and tool executor are backend-agnostic.
    """

    async def exec(
        self,
        command: str,
        timeout_sec: float = 120.0,
        cwd: Optional[str] = None,
    ) -> EnvironmentResult: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


# ═══════════════════════════════════════════════════════════════════
# HARBOR ADAPTER
# ═══════════════════════════════════════════════════════════════════


class HarborEnvironmentAdapter:
    """Wraps a Harbor ``BaseEnvironment`` to conform to our protocol.

    **REMOVED in Phase A**: Harbor/Docker integration deferred to Phase C.
    This class is a stub to maintain import compatibility.

    Raises
    ------
    NotImplementedError:
        Harbor environments are not available in Phase A.
    """

    def __init__(self, harbor_env: Any) -> None:
        raise NotImplementedError(
            "HarborEnvironmentAdapter is not available in Phase A. "
            "Use StandaloneEnvironment instead."
        )


# ═══════════════════════════════════════════════════════════════════
# STANDALONE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════


class StandaloneEnvironment:
    """Local subprocess execution environment.

    Executes commands via ``subprocess.run()`` in a working directory.
    Suitable for development, testing, and standalone benchmarking
    without the full Harbor framework.

    Parameters
    ----------
    work_dir:
        Working directory for command execution.
    shell_timeout:
        Default timeout in seconds for shell commands.
    """

    def __init__(
        self,
        work_dir: Optional[Path] = None,
        shell_timeout: int = 300,
    ) -> None:
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="tbench_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.shell_timeout = shell_timeout

    async def exec(
        self,
        command: str,
        timeout_sec: float = 0,
        cwd: Optional[str] = None,
    ) -> EnvironmentResult:
        """Execute a command via subprocess."""
        timeout = timeout_sec if timeout_sec > 0 else self.shell_timeout
        exec_cwd = cwd or str(self.work_dir)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._exec_sync, command, timeout, exec_cwd,
        )

    def _exec_sync(
        self,
        command: str,
        timeout: float,
        cwd: str,
    ) -> EnvironmentResult:
        """Synchronous subprocess execution."""
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            return EnvironmentResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return EnvironmentResult(
                stderr=f"Command timed out after {timeout}s: {command[:200]}",
                return_code=124,  # standard timeout exit code
            )
        except Exception as e:
            return EnvironmentResult(
                stderr=f"Execution error: {e}",
                return_code=1,
            )

    async def start(self) -> None:
        """Ensure working directory exists."""
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        """No-op for standalone — caller owns cleanup."""
        pass

    def cleanup(self) -> None:
        """Remove the working directory."""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# SYNC WRAPPER
# ═══════════════════════════════════════════════════════════════════


class SyncEnvironmentWrapper:
    """Synchronous wrapper around an async ``EnvironmentBackend``.

    Allows the synchronous agent loop and tool executor to call
    async environments without managing their own event loop.
    """

    def __init__(self, backend: EnvironmentBackend) -> None:
        self._backend = backend
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def exec(
        self,
        command: str,
        timeout_sec: float = 120.0,
        cwd: Optional[str] = None,
    ) -> EnvironmentResult:
        """Synchronously execute a command."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — use a thread to avoid
            # blocking the event loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._backend.exec(command, timeout_sec=timeout_sec, cwd=cwd),
                )
                return future.result(timeout=timeout_sec + 10)
        else:
            return asyncio.run(
                self._backend.exec(command, timeout_sec=timeout_sec, cwd=cwd),
            )
