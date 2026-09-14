#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
TerminalBench dataset loading and task representation.

Loads tasks from the Harbor framework's TerminalBench dataset
(``terminal-bench/terminal-bench-2``) or from local JSONL files.
Maps them to :class:`TerminalBenchTask` dataclasses for use by
the agent loop and comparison framework.

TerminalBench tasks are containerised terminal challenges — each task
provides a Docker environment, a natural-language instruction, and
a test script that verifies the outcome.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# TASK DATACLASS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TerminalBenchTask:
    """A single TerminalBench task instance.

    Fields map to the Harbor task format (``task.toml`` +
    ``instruction.md``).  Harbor owns the container lifecycle;
    we own metrics and comparison.
    """

    task_id: str
    instruction: str  # markdown task description

    # Classification
    category: str = ""     # e.g. "coding", "sysadmin", "ml", "crypto"
    difficulty: str = ""   # e.g. "easy", "medium", "hard"

    # Resource budget declared by the task (informational; Harbor enforces)
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 4096
    gpus: int = 0
    allow_internet: bool = False

    # Timeouts
    agent_timeout_sec: float = 900.0
    verifier_timeout_sec: float = 600.0

    # Metadata
    has_solution: bool = False
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Oracle: shell command run in the env after the agent finishes.
    # Exit code 0 → task passed. Empty string → no oracle (skip scoring).
    # In Phase B oracles are inline; in Phase C they come from Harbor's
    # tests/test.sh.
    oracle_command: str = ""

    # Optional: setup commands run in the env before the agent starts
    # (e.g. seed input files). Each is a shell command; failures abort.
    setup_commands: List[str] = field(default_factory=list)

    # Extra metadata (used to attach Harbor handles in Phase C)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ──────────────────────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TerminalBenchTask":
        """Create from a plain dictionary (e.g. loaded from JSONL)."""
        return cls(
            task_id=data["task_id"],
            instruction=data.get("instruction", ""),
            category=data.get("category", ""),
            difficulty=data.get("difficulty", ""),
            cpus=data.get("cpus", 1),
            memory_mb=data.get("memory_mb", 2048),
            storage_mb=data.get("storage_mb", 4096),
            gpus=data.get("gpus", 0),
            allow_internet=data.get("allow_internet", False),
            agent_timeout_sec=data.get("agent_timeout_sec", 900.0),
            verifier_timeout_sec=data.get("verifier_timeout_sec", 600.0),
            has_solution=data.get("has_solution", False),
            authors=data.get("authors", []),
            keywords=data.get("keywords", []),
            oracle_command=data.get("oracle_command", ""),
            setup_commands=data.get("setup_commands", []),
        )

    @classmethod
    def from_harbor_task(cls, harbor_task: Any) -> "TerminalBenchTask":
        """Map a Harbor task object to our internal representation.

        **REMOVED in Phase A**: Harbor integration deferred to Phase C.
        This method is a stub to maintain API compatibility.

        Raises
        ------
        NotImplementedError:
            Harbor loading is not available in Phase A.
        """
        raise NotImplementedError(
            "Harbor integration is not available in Phase A."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary for JSONL export."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "category": self.category,
            "difficulty": self.difficulty,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "gpus": self.gpus,
            "allow_internet": self.allow_internet,
            "agent_timeout_sec": self.agent_timeout_sec,
            "verifier_timeout_sec": self.verifier_timeout_sec,
            "has_solution": self.has_solution,
            "authors": self.authors,
            "keywords": self.keywords,
        }


# ═══════════════════════════════════════════════════════════════════
# LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def load_tasks_from_jsonl(
    path: str,
    limit: Optional[int] = None,
) -> List[TerminalBenchTask]:
    """Load tasks from a local JSONL file.

    Each line must be a JSON object with at least ``task_id`` and
    ``instruction``.
    """
    tasks: List[TerminalBenchTask] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                tasks.append(TerminalBenchTask.from_dict(json.loads(line)))
    logger.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


def load_tasks_from_harbor(
    dataset_name: str = "terminal-bench/terminal-bench-2",
    ref: str = "latest",
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
) -> List[TerminalBenchTask]:
    """Load TerminalBench tasks from the Harbor registry.

    **REMOVED in Phase A**: Harbor integration deferred to Phase C.
    This function is a stub to maintain API compatibility.

    Raises
    ------
    NotImplementedError:
        Harbor loading is not available in Phase A.
    """
    raise NotImplementedError(
        "Harbor integration is not available in Phase A. "
        "Use load_tasks_from_jsonl() or generate_sample_tasks() instead."
    )


# ═══════════════════════════════════════════════════════════════════
# SAMPLE TASKS (for development / testing without Harbor)
# ═══════════════════════════════════════════════════════════════════


def generate_sample_tasks(n: int = 5) -> List[TerminalBenchTask]:
    """Generate synthetic sample tasks for development and testing.

    These do NOT require Docker or Harbor — they describe simple
    terminal tasks that can be tested with :class:`StandaloneEnvironment`.
    Each task includes ``setup_commands`` (seeded files in the env)
    and an ``oracle_command`` (exit 0 = pass) for Phase B scoring.

    Note: tasks operate in the env's working directory (cwd of the
    StandaloneEnvironment temp dir). The ``$PWD`` variable is the
    canonical reference.
    """
    _SAMPLES = [
        TerminalBenchTask(
            task_id="sample/count-lines",
            instruction=(
                "There are several `.py` files in the current working "
                "directory. Count the total number of lines across all of "
                "them and write the integer count (just the number, no "
                "other text) to a file named `result.txt` in the current "
                "working directory."
            ),
            category="coding",
            difficulty="easy",
            cpus=1,
            memory_mb=512,
            setup_commands=[
                # Seed three small .py files; total = 3 + 5 + 2 = 10 lines.
                "printf 'a=1\\nb=2\\nc=3\\n' > foo.py",
                "printf 'def f():\\n    pass\\n\\ndef g():\\n    pass\\n' > bar.py",
                "printf 'x=1\\ny=2\\n' > baz.py",
            ],
            # Oracle: result.txt exists, contains exactly "10".
            oracle_command="test -f result.txt && [ \"$(cat result.txt | tr -d '[:space:]')\" = '10' ]",
        ),
        TerminalBenchTask(
            task_id="sample/find-largest-file",
            instruction=(
                "There are several files in the current working "
                "directory. Identify the file with the largest size in "
                "bytes and write the file's name (just the basename, no "
                "directory prefix) to `largest.txt` in the current "
                "working directory."
            ),
            category="sysadmin",
            difficulty="easy",
            cpus=1,
            memory_mb=512,
            setup_commands=[
                # Seed files of differing sizes; biggest.bin is largest.
                "printf 'small' > small.txt",
                "head -c 200 /dev/urandom > medium.bin",
                "head -c 5000 /dev/urandom > biggest.bin",
            ],
            oracle_command="test -f largest.txt && [ \"$(cat largest.txt | tr -d '[:space:]')\" = 'biggest.bin' ]",
        ),
        TerminalBenchTask(
            task_id="sample/parse-json-logs",
            instruction=(
                "The file `app.log` in the current working directory "
                "contains one JSON object per line. Extract every entry "
                "whose `level` field equals `ERROR` and write them as a "
                "JSON array to `errors.json` in the current working "
                "directory."
            ),
            category="coding",
            difficulty="easy",
            cpus=1,
            memory_mb=512,
            setup_commands=[
                # Two ERROR entries among four lines.
                "printf '%s\\n' "
                "'{\"level\":\"INFO\",\"msg\":\"start\"}' "
                "'{\"level\":\"ERROR\",\"msg\":\"oops\"}' "
                "'{\"level\":\"WARN\",\"msg\":\"hmm\"}' "
                "'{\"level\":\"ERROR\",\"msg\":\"bad\"}' > app.log",
            ],
            # Oracle: errors.json parses as JSON array of length 2 with
            # both entries having level=ERROR.
            oracle_command=(
                "test -f errors.json && "
                "python3 -c \"import json,sys; "
                "d=json.load(open('errors.json')); "
                "assert isinstance(d,list) and len(d)==2 and "
                "all(e['level']=='ERROR' for e in d)\""
            ),
        ),
    ]
    return _SAMPLES[:n]
