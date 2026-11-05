#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""`harbor run` is launched with HARBOR_TELEMETRY off unless the caller set it.

The harbor CLI sends usage telemetry by default. The sweep's subprocess env
defaults it to "0" and leaves an explicit value alone, so a user can still opt
in. subprocess.run is faked; what is under test is the env the sweep builds.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.storage.sqlite_store import SQLiteResultStore
from src.sweep import harbor_sweep
from src.sweep.harbor_sweep import HarborSweep
from src.sweep.spec import SweepSpec


def _harbor_env(tmp_path, monkeypatch) -> dict:
    """Run HarborSweep._harbor_run with a fake subprocess.run; return its env."""
    spec = SweepSpec(
        output_dir=tmp_path,
        tasks=["terminal-bench/hello-world"],
        llm_mode="off",
    )
    sweep = HarborSweep(spec, store=SQLiteResultStore(tmp_path))

    seen = {}

    def fake_run(argv, *, env, **_kwargs):
        seen["argv"] = argv
        seen["env"] = env
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harbor_sweep.subprocess, "run", fake_run)
    jobs_dir = tmp_path / "cell" / "jobs"
    jobs_dir.mkdir(parents=True)
    rc = sweep._harbor_run(jobs_dir, 1, {"OPENAI_API_BASE": "http://127.0.0.1:4001/v1"})

    assert rc == 0
    assert seen["argv"][:2] == ["harbor", "run"]
    return seen["env"]


def test_telemetry_defaults_off(tmp_path, monkeypatch):
    monkeypatch.delenv("HARBOR_TELEMETRY", raising=False)
    assert _harbor_env(tmp_path, monkeypatch)["HARBOR_TELEMETRY"] == "0"


@pytest.mark.parametrize("value", ["1", "true"])
def test_explicit_setting_is_kept(tmp_path, monkeypatch, value):
    monkeypatch.setenv("HARBOR_TELEMETRY", value)
    assert _harbor_env(tmp_path, monkeypatch)["HARBOR_TELEMETRY"] == value
