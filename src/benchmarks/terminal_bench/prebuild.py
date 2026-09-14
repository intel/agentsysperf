#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Image preflight for reproducible task-sized Docker timing."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from typing import Any


def docker_image_for_task(task: Any) -> str:
    """Return the task's declared image or fail before benchmark timing."""
    handle = (getattr(task, "extra", None) or {}).get("harbor")
    if handle is None:
        raise ValueError(f"task {getattr(task, 'task_id', task)!r} has no Harbor handle")

    from harbor.models.task.task import Task

    image = Task(handle.task_path).config.environment.docker_image
    if not image:
        raise ValueError(
            f"task {getattr(task, 'task_id', task)!r} has no prebuilt docker_image; "
            "task-sized stream benchmarks do not measure local image builds"
        )
    return image


def _image_is_local(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    return proc.returncode == 0


def ensure_task_images(tasks: Iterable[Any]) -> None:
    """Make every selected task image local before any measured task starts."""
    for image in sorted({docker_image_for_task(task) for task in tasks}):
        if _image_is_local(image):
            continue
        try:
            proc = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                check=False,
                text=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("docker is required for task-sized streams") from exc
        if proc.returncode != 0 or not _image_is_local(image):
            raise RuntimeError(
                f"could not pre-pull required image {image!r}: {proc.stderr.strip()}"
            )


__all__ = ["docker_image_for_task", "ensure_task_images"]
