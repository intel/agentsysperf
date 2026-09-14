#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""List all Terminal-Bench 2 tasks from Harbor registry with metadata.

Fetches the full task list and extracts computational characteristics
for workload categorization.
"""

import asyncio
import json
import sys
from pathlib import Path


async def list_all_tasks():
    """List all TB2 tasks from Harbor."""
    try:
        from harbor.registry.client.package import PackageDatasetClient
        from harbor.tasks.client import TaskClient
        from harbor.models.task.task import Task
    except ImportError as e:
        print(f"ERROR: Harbor not installed: {e}", file=sys.stderr)
        print("Install with: pip install 'harbor>=0.8.0'", file=sys.stderr)
        return None

    dataset_name = "terminal-bench/terminal-bench-2"
    ref = "latest"

    print(f"Fetching {dataset_name}@{ref} metadata...", file=sys.stderr)

    ds_client = PackageDatasetClient()
    metadata = await ds_client._get_dataset_metadata(f"{dataset_name}@{ref}")

    task_ids = metadata.task_ids
    print(f"Found {len(task_ids)} tasks. Downloading...", file=sys.stderr)

    task_client = TaskClient()
    download_result = await task_client.download_tasks(
        task_ids=task_ids,
        overwrite=False,
        export=False,
    )

    tasks_info = []
    for task_id, task_path in zip(task_ids, download_result.paths):
        harbor_task = Task(task_path)

        # Extract config
        env_cfg = harbor_task.config.environment
        agent_cfg = harbor_task.config.agent
        task_cfg = harbor_task.config.task

        # Get keywords and authors
        keywords = []
        authors = []
        if task_cfg:
            if hasattr(task_cfg, "keywords"):
                keywords = task_cfg.keywords or []
            if hasattr(task_cfg, "authors"):
                authors = [
                    a.name if hasattr(a, "name") else str(a)
                    for a in (task_cfg.authors or [])
                ]

        # Build task info
        info = {
            "task_id": harbor_task.name,
            "instruction": harbor_task.instruction[:150] + "..." if len(harbor_task.instruction) > 150 else harbor_task.instruction,
            "instruction_full": harbor_task.instruction,
            "keywords": keywords,
            "authors": authors,
            "cpus": env_cfg.cpus if env_cfg else 1,
            "memory_mb": env_cfg.memory_mb if env_cfg else 2048,
            "storage_mb": env_cfg.storage_mb if env_cfg else 4096,
            "gpus": env_cfg.gpus if env_cfg else 0,
            "allow_internet": env_cfg.allow_internet if env_cfg else False,
            "agent_timeout_sec": agent_cfg.timeout_sec if agent_cfg else 900.0,
            "has_solution": (task_path / "solution").exists(),
        }

        tasks_info.append(info)

    print(f"Loaded {len(tasks_info)} task metadata records", file=sys.stderr)
    return tasks_info


def main():
    tasks = asyncio.run(list_all_tasks())
    if tasks is None:
        return 1

    # Output as JSON
    print(json.dumps(tasks, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
