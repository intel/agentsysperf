#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

from __future__ import annotations

import asyncio
import json

import pytest

from src.benchmarks.terminal_bench.adapter import TerminalBenchAdapter
from src.benchmarks.terminal_bench.harbor_environment import (
    HarborEnvironmentAdapter,
    sanitize_compose_project_name,
)
from src.protocols import TaskSpec
from src.streams.resources import ResourceBudget


def test_harbor_resource_override_contains_only_cpuset_and_memory(tmp_path):
    env = HarborEnvironmentAdapter(tmp_path, "stream/task", force_build=False)
    env.pin(ResourceBudget((4, 5), memory_mb=3072))
    path = env._write_resource_override(tmp_path)
    assert path is not None
    assert json.loads(path.read_text()) == {
        "services": {"main": {"cpuset": "4-5", "mem_limit": "3072m"}}
    }


def test_harbor_unpinned_override_retains_memory_without_cpuset(tmp_path):
    env = HarborEnvironmentAdapter(tmp_path, "stream/task", force_build=False)
    env.pin(ResourceBudget((), memory_mb=3072))
    path = env._write_resource_override(tmp_path)
    assert path is not None
    assert json.loads(path.read_text()) == {
        "services": {"main": {"mem_limit": "3072m"}}
    }


def test_adapter_uses_worker_cpus_and_task_memory():
    captured = []

    class Environment:
        def pin(self, budget):
            captured.append(budget)

    adapter = TerminalBenchAdapter(resource_budget=ResourceBudget((8, 9)))
    adapter._pin_environment(
        Environment(),
        TaskSpec(id="task", instruction="", cpu_budget=2, memory_mb=6144),
    )
    assert captured == [ResourceBudget((8, 9), memory_mb=6144)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Run/ABC", "run-abc"),
        ("run.with punctuation", "run-with-punctuation"),
        ("!leading-symbol", "0-leading-symbol"),
    ],
)
def test_compose_project_sanitizer_matches_harbor_session_name(raw, expected):
    assert sanitize_compose_project_name(raw) == expected


def test_harbor_session_uses_shared_compose_project_sanitizer(tmp_path):
    env = HarborEnvironmentAdapter(tmp_path, "!Run/ABC")
    assert env.session_id == "0-run-abc"


@pytest.mark.parametrize(
    ("force_build", "expected_delete"),
    [(True, True), (False, False)],
)
def test_harbor_stop_deletes_only_locally_built_environments(
    tmp_path, force_build, expected_delete
):
    calls = []

    class Environment:
        async def stop(self, *, delete):
            calls.append(delete)

    env = HarborEnvironmentAdapter(tmp_path, "task", force_build=force_build)
    env._harbor_env = Environment()
    asyncio.run(env.stop())
    assert calls == [expected_delete]
