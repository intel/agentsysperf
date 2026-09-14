#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

from __future__ import annotations

import subprocess

import pytest

from src.benchmarks.terminal_bench import prebuild


def test_ensure_task_images_pulls_only_missing_images(monkeypatch):
    monkeypatch.setattr(
        prebuild,
        "docker_image_for_task",
        lambda task: task,
    )
    local = {"present:latest"}
    calls = []

    def fake_inspect(image):
        return image in local

    def fake_run(command, **_):
        calls.append(command)
        local.add(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prebuild, "_image_is_local", fake_inspect)
    monkeypatch.setattr(prebuild.subprocess, "run", fake_run)
    prebuild.ensure_task_images(["present:latest", "missing:latest"])
    assert calls == [["docker", "pull", "missing:latest"]]


def test_ensure_task_images_fails_when_pull_does_not_make_image_local(monkeypatch):
    monkeypatch.setattr(prebuild, "docker_image_for_task", lambda task: task)
    monkeypatch.setattr(prebuild, "_image_is_local", lambda _: False)
    monkeypatch.setattr(
        prebuild.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 1, "", "offline"),
    )
    with pytest.raises(RuntimeError, match="pre-pull"):
        prebuild.ensure_task_images(["missing:latest"])
