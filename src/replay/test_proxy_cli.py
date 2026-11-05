#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Command-line guards of the record/replay proxy (`python -m src.replay.proxy`).

Record mode writes every upstream response to --fixture. Started without one,
the proxy used to come up healthy and then fail each call it was asked to
record (it dereferenced the missing path on the first write). It must refuse
at startup instead, with a usage error.
"""
from __future__ import annotations

import pytest

from src.replay import proxy

_GLOBALS = ("_MODE", "_UPSTREAM_URL", "_FIXTURE_PATH", "_STRICT_MISS",
            "_INJECT_LATENCY", "_REPLAY_TRIAL")


@pytest.fixture
def no_server(monkeypatch):
    """Keep main() from binding a port, and restore the globals it sets."""
    for name in _GLOBALS:
        monkeypatch.setattr(proxy, name, getattr(proxy, name))
    calls = []
    monkeypatch.setattr(proxy.uvicorn, "run", lambda *a, **kw: calls.append(kw))
    return calls


def test_record_without_fixture_is_a_usage_error(no_server, capsys):
    with pytest.raises(SystemExit) as exc:
        proxy.main(["--mode", "record"])
    assert exc.value.code == 2
    assert "--fixture is required when --mode record" in capsys.readouterr().err
    assert no_server == [], "the server must not start"


def test_record_with_fixture_starts(no_server, tmp_path):
    fixture = tmp_path / "new.jsonl"
    proxy.main(["--mode", "record", "--fixture", str(fixture), "--port", "0"])
    assert proxy._FIXTURE_PATH == fixture
    assert len(no_server) == 1


def test_off_mode_needs_no_fixture(no_server):
    proxy.main(["--mode", "off", "--port", "0"])
    assert len(no_server) == 1
