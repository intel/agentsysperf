#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""No-op AgentInvoker for testing adapter flow without real LLM.

Provides canned responses that satisfy the AgentInvoker protocol,
allowing benchmark adapters to be tested end-to-end without
spawning actual inference processes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class TrialResult:
    """Minimal TrialResult for NoOpAgentInvoker compatibility.

    Matches the shape expected by TerminalBenchAdapter._score().
    """

    passed: bool = True
    reward: float = 1.0
    error: Optional[str] = None
    trajectory_path: Optional[Path] = None
    steps: int = 3
    elapsed_s: float = 0.0


class NoOpAgentInvoker:
    """AgentInvoker that returns canned success responses.

    Implements the AgentInvoker protocol for testing:
    - invoke() returns a TrialResult with passed=True
    - No actual LLM inference or environment interaction
    - Configurable to simulate failures for error path testing

    Parameters
    ----------
    simulate_failure:
        If True, invoke() returns passed=False with error message.
    sleep_duration:
        Simulated thinking time in seconds.
    """

    def __init__(
        self,
        *,
        simulate_failure: bool = False,
        sleep_duration: float = 0.1,
    ) -> None:
        self.simulate_failure = simulate_failure
        self.sleep_duration = sleep_duration
        self._call_count = 0

    def invoke(
        self,
        instruction: str,
        *,
        environment: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        session_hint: Optional[str] = None,
    ) -> TrialResult:
        """Return a canned TrialResult.

        Sleeps briefly to simulate thinking time, then returns
        success or failure based on configuration.
        """
        self._call_count += 1
        time.sleep(self.sleep_duration)

        if self.simulate_failure:
            return TrialResult(
                passed=False,
                reward=0.0,
                error="Simulated failure from NoOpAgentInvoker",
                steps=1,
                elapsed_s=self.sleep_duration,
            )

        return TrialResult(
            passed=True,
            reward=1.0,
            error=None,
            steps=3,
            elapsed_s=self.sleep_duration,
        )

    @property
    def call_count(self) -> int:
        """Number of times invoke() was called."""
        return self._call_count


__all__ = ["NoOpAgentInvoker", "TrialResult"]
