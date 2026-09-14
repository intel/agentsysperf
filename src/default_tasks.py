#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Access the packaged default task selection for task-sized streams."""

from __future__ import annotations

from importlib.resources import files

DEFAULT_TASK_RESOURCE = "data/clean_tasks_23.txt"


def load_default_task_text() -> str:
    """Read the curated task selection from the installed package."""
    return (
        files("src")
        .joinpath(DEFAULT_TASK_RESOURCE)
        .read_text(encoding="utf-8")
    )
