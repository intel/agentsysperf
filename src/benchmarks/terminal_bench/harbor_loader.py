#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Harbor-backed task loader for Terminal-Bench (Phase C).

Loads tasks from the Harbor registry and maps them to TerminalBenchTask
dataclasses. Requires Harbor >=0.8.0 and Python >=3.12.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HarborTaskHandle:
    """Opaque handle carrying Harbor task metadata.

    Stored in TerminalBenchTask.extra['harbor'] so the adapter can
    retrieve the Harbor Task + Environment later.
    """
    task_path: Path
    task_id_str: str  # org/name@ref format


def load_tasks_from_harbor_registry(
    dataset_name: str = "terminal-bench/terminal-bench-2",
    ref: str = "latest",
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
) -> Iterable[Any]:  # Returns Iterable[TerminalBenchTask]
    """Load Terminal-Bench tasks from the Harbor registry.

    Phase C implementation using Harbor 0.8.0 API. Downloads tasks to
    ~/.cache/harbor/tasks and returns TerminalBenchTask dataclasses
    with Harbor metadata attached.

    Parameters
    ----------
    dataset_name:
        Harbor dataset identifier (e.g. "terminal-bench/terminal-bench-2").
    ref:
        Dataset version (default: "latest").
    limit:
        Maximum tasks to load.
    categories:
        Filter by category (not yet implemented — Harbor doesn't expose
        task categories in metadata).
    task_names:
        Filter by task name (exact match against task.name).

    Returns
    -------
    Iterable of TerminalBenchTask with Harbor handles attached.
    """
    # Import harbor here so Phase A/B don't require it
    try:
        for module_name in (
            "harbor.registry.client.package",
            "harbor.tasks.client",
            "harbor.models.task.task",
        ):
            importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            "Harbor package required for Phase C. Install with:\n"
            "  pip install 'harbor>=0.8.0'\n"
            f"Original error: {e}"
        ) from e

    # Lazy import to avoid circular dependency
    from src.benchmarks.terminal_bench.dataset import TerminalBenchTask

    # Run async loading in sync context
    return asyncio.run(_load_tasks_async(
        dataset_name=dataset_name,
        ref=ref,
        limit=limit,
        categories=categories,
        task_names=task_names,
        TerminalBenchTask=TerminalBenchTask,
    ))


async def _load_tasks_async(
    dataset_name: str,
    ref: str,
    limit: Optional[int],
    categories: Optional[List[str]],
    task_names: Optional[List[str]],
    TerminalBenchTask: Any,
) -> List[Any]:
    """Async task loading implementation."""
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import TaskClient
    from harbor.models.task.task import Task

    logger.info(f"Loading {dataset_name}@{ref} from Harbor registry...")

    # 1. Get dataset metadata
    ds_client = PackageDatasetClient()
    metadata = await ds_client._get_dataset_metadata(f"{dataset_name}@{ref}")

    task_ids = metadata.task_ids
    if limit is not None:
        task_ids = task_ids[:limit]

    logger.info(f"Dataset has {len(metadata.task_ids)} total tasks, loading {len(task_ids)}")

    # 2. Download all tasks
    task_client = TaskClient()
    download_result = await task_client.download_tasks(
        task_ids=task_ids,
        overwrite=False,
        export=False,  # cache mode
    )

    # 3. Load as Harbor Task objects and convert to TerminalBenchTask
    tasks: List[Any] = []
    for i, (task_id, task_path) in enumerate(zip(task_ids, download_result.paths)):
        harbor_task = Task(task_path)

        # Filter by name if requested
        if task_names and harbor_task.name not in task_names:
            continue

        # Extract config fields
        env_cfg = harbor_task.config.environment
        agent_cfg = harbor_task.config.agent
        verifier_cfg = harbor_task.config.verifier
        task_cfg = harbor_task.config.task

        # Infer category from keywords (best-effort)
        category = ""
        keywords_list = []
        if task_cfg and hasattr(task_cfg, "keywords"):
            keywords_list = task_cfg.keywords or []
            if keywords_list:
                category = keywords_list[0]  # use first keyword as category proxy

        # Extract authors
        authors_list = []
        if task_cfg and hasattr(task_cfg, "authors"):
            authors_list = [
                a.name if hasattr(a, "name") else str(a)
                for a in (task_cfg.authors or [])
            ]

        # Map to TerminalBenchTask
        tb_task = TerminalBenchTask(
            task_id=harbor_task.name,
            instruction=harbor_task.instruction,
            category=category,  # inferred from keywords
            difficulty="",  # not in Harbor 0.8.0 schema
            cpus=env_cfg.cpus if env_cfg else 1,
            memory_mb=env_cfg.memory_mb if env_cfg else 2048,
            storage_mb=env_cfg.storage_mb if env_cfg else 4096,
            gpus=env_cfg.gpus if env_cfg else 0,
            allow_internet=env_cfg.allow_internet if env_cfg else False,
            agent_timeout_sec=agent_cfg.timeout_sec if agent_cfg else 900.0,
            verifier_timeout_sec=verifier_cfg.timeout_sec if verifier_cfg else 600.0,
            has_solution=(task_path / "solution").exists(),
            authors=authors_list,
            keywords=keywords_list,
            # Phase C: no inline oracle_command — Harbor tasks have tests/test.sh
            oracle_command="",
            setup_commands=[],
        )

        # Attach Harbor handle to extra
        if tb_task.extra is None:
            tb_task.extra = {}
        tb_task.extra["harbor"] = HarborTaskHandle(
            task_path=task_path,
            task_id_str=f"{task_id.org}/{task_id.name}@{task_id.ref[:20]}",
        )

        tasks.append(tb_task)

    logger.info(f"Loaded {len(tasks)} Harbor tasks")
    return tasks


__all__ = ["load_tasks_from_harbor_registry", "HarborTaskHandle"]
