#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Shell command classification and metrics for TerminalBench.

In TerminalBench, every agent action is a shell command executed inside
a Docker container.  Unlike SWE-Bench (which has a fixed set of named
tools), TerminalBench commands are arbitrary shell strings.

:func:`classify_command` inspects the command string and assigns a
:class:`~agentflow.core.cost.ResourceTier` based on heuristic prefix
matching.  This follows the same philosophy as :func:`classify_tool`
in ``core/cost.py`` — unknown commands default to ``CPU_HEAVY`` (safe).

:class:`CommandCallRecord` and :class:`CommandMetrics` mirror the
``ToolCallRecord`` / ``ToolMetrics`` pattern from ``swe_bench/tools.py``.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from enum import Enum


class ResourceTier(Enum):
    """Resource tier classification for shell commands.

    Used by :func:`classify_command` to bucket commands by expected
    cost. Phase B uses three coarse tiers; Phase C may add more
    (e.g. NETWORK_HEAVY) when distributed/RAG workloads enter scope.
    """

    CPU_LIGHT = "cpu_light"
    CPU_HEAVY = "cpu_heavy"
    GPU_INFERENCE = "gpu_inference"

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# COMMAND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

# Commands that are lightweight I/O / inspection (typically <50ms).
_CPU_LIGHT_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "echo", "printf", "pwd",
    "whoami", "which", "file", "stat", "readlink", "basename",
    "dirname", "env", "printenv", "id", "date", "hostname",
    "true", "false", "test", "expr", "seq", "yes",
    "touch", "mv", "cp", "rm", "mkdir", "rmdir", "ln",
    "sort", "uniq", "cut", "tr", "tee", "xargs",
    "diff", "comm", "paste", "fold", "expand",
    "md5sum", "sha256sum", "sha1sum",
    "sleep",
})

# Patterns that indicate GPU usage.
_GPU_PATTERNS = [
    re.compile(r"nvidia-smi"),
    re.compile(r"python.*(?:torch|tensorflow|jax).*(?:train|cuda|gpu)", re.IGNORECASE),
    re.compile(r"nvcc\b"),
    re.compile(r"cuda-"),
]


def classify_command(command: str) -> ResourceTier:
    """Classify a shell command into a resource tier.

    Inspects the first token (the binary name) and known patterns
    to determine the resource tier.

    Classification rules:
    1. If the first token is in ``_CPU_LIGHT_COMMANDS`` → CPU_LIGHT
    2. If the command matches a GPU pattern → GPU_INFERENCE
    3. Everything else (make, gcc, python, pip, apt-get, etc.) → CPU_HEAVY

    The fallback to CPU_HEAVY matches the convention in ``core/cost.py``
    where ``classify_tool()`` returns CPU_HEAVY for unknown tools.

    Parameters
    ----------
    command:
        Shell command string (may include pipes and arguments).

    Returns
    -------
    The assigned :class:`ResourceTier`.
    """
    cmd = command.strip()
    if not cmd:
        return ResourceTier.CPU_LIGHT

    # Extract first token, stripping any path prefix.
    # Handle common shell prefixes: sudo, env, nice, time, etc.
    tokens = cmd.split()
    idx = 0
    _PASSTHROUGH = {"sudo", "env", "nice", "time", "timeout", "strace", "nohup"}
    while idx < len(tokens) and tokens[idx] in _PASSTHROUGH:
        idx += 1
    if idx >= len(tokens):
        return ResourceTier.CPU_HEAVY

    first = tokens[idx]
    base = first.rsplit("/", 1)[-1]  # strip path prefix

    if base in _CPU_LIGHT_COMMANDS:
        return ResourceTier.CPU_LIGHT

    # Check for common lightweight builtins invoked via pipes
    # (e.g. "grep foo | wc -l" — the whole pipeline is light)
    # But only if there is no heavy command in the pipeline.
    if "|" in cmd:
        pipe_parts = cmd.split("|")
        all_light = all(
            _first_token(p.strip()) in _CPU_LIGHT_COMMANDS
            for p in pipe_parts
            if p.strip()
        )
        if all_light:
            return ResourceTier.CPU_LIGHT

    # GPU detection
    for pattern in _GPU_PATTERNS:
        if pattern.search(cmd):
            return ResourceTier.GPU_INFERENCE

    # grep is very common and fast
    if base in ("grep", "egrep", "fgrep", "rg", "ag", "ack", "find", "locate"):
        return ResourceTier.CPU_LIGHT

    return ResourceTier.CPU_HEAVY


def _first_token(cmd: str) -> str:
    """Extract the first meaningful token from a command string."""
    tokens = cmd.split()
    _PASSTHROUGH = {"sudo", "env", "nice", "time", "timeout"}
    for t in tokens:
        if t not in _PASSTHROUGH:
            return t.rsplit("/", 1)[-1]
    return ""


# ═══════════════════════════════════════════════════════════════════
# COMMAND METRICS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CommandCallRecord:
    """Record of a single command invocation in a Docker container."""

    command: str  # first 200 chars of the command
    resource_tier: ResourceTier
    wall_clock_ms: float
    exit_code: int
    output_length: int  # chars of stdout+stderr returned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command[:200],
            "tier": self.resource_tier.value,
            "ms": round(self.wall_clock_ms, 2),
            "exit_code": self.exit_code,
            "output_length": self.output_length,
        }


@dataclass
class CommandMetrics:
    """Aggregated metrics across all commands in a trial."""

    calls: List[CommandCallRecord] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_wall_clock_ms(self) -> float:
        return sum(c.wall_clock_ms for c in self.calls)

    @property
    def calls_by_tier(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.calls:
            tier = c.resource_tier.value
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    @property
    def wall_clock_by_tier(self) -> Dict[str, float]:
        """Total wall-clock ms bucketed by resource tier."""
        totals: Dict[str, float] = {}
        for c in self.calls:
            tier = c.resource_tier.value
            totals[tier] = totals.get(tier, 0.0) + c.wall_clock_ms
        return totals

    @property
    def success_count(self) -> int:
        return sum(1 for c in self.calls if c.exit_code == 0)

    @property
    def failure_count(self) -> int:
        return sum(1 for c in self.calls if c.exit_code != 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_wall_clock_ms": round(self.total_wall_clock_ms, 2),
            "calls_by_tier": self.calls_by_tier,
            "wall_clock_by_tier": {
                k: round(v, 2) for k, v in self.wall_clock_by_tier.items()
            },
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "calls": [c.to_dict() for c in self.calls],
        }
