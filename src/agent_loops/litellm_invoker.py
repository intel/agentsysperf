#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""LiteLLM-backed AgentInvoker for the BenchmarkAdapter protocol.

Wires :class:`LiteLLMTerminalAgentLoop` into AgentSysPerf's
:class:`AgentInvoker` Protocol. Each invoke() call spawns a fresh
loop bound to the per-task environment passed by the adapter.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from src.agent_loops.litellm_terminal_loop import (
    LiteLLMTerminalAgentLoop,
    TrialResult,
)
from src.runner import RunContext

logger = logging.getLogger(__name__)


class LiteLLMAgentInvoker:
    """AgentInvoker that drives terminal tasks with a LiteLLM model.

    Parameters
    ----------
    model:
        LiteLLM model string (e.g. ``"gpt-4o-mini"``).
    max_turns:
        Maximum agent turns per task.
    temperature:
        LLM sampling temperature.
    max_tokens:
        Max completion tokens per turn.
    command_timeout:
        Per-shell-command timeout in seconds.
    use_native_tools:
        Use OpenAI-style native tool calling.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        max_turns: int = 200,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        command_timeout: int = 120,
        use_native_tools: bool = True,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.command_timeout = command_timeout
        self.use_native_tools = use_native_tools

    def invoke(
        self,
        instruction: str,
        *,
        environment: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        session_hint: Optional[str] = None,
        run_context: Optional[RunContext] = None,
    ) -> TrialResult:
        """Run one task end-to-end. Returns a :class:`TrialResult`."""
        if environment is None:
            raise ValueError(
                "LiteLLMAgentInvoker requires an environment to be passed by the adapter."
            )

        task_id = (metadata or {}).get("task_id", "unknown")

        loop = LiteLLMTerminalAgentLoop(
            model=self.model,
            env=environment,
            max_turns=self.max_turns,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            command_timeout=self.command_timeout,
            use_native_tools=self.use_native_tools,
            run_context=run_context,
        )
        return loop.solve(task_id=task_id, instruction=instruction)


__all__ = ["LiteLLMAgentInvoker"]
