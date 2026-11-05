#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Smoke test: live_dashboard.py renders past its first benchmark run.

The script stops at "Click Run Benchmark" until a run has produced results, so
a plain AppTest never reaches the analysis sections. 1.0.0 shipped a NameError
there (`_TMP` was used before it was defined), visible only after the first
run. Seeding one result row drives the script through every section.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_LIVE = Path(__file__).resolve().parents[2] / "live_dashboard.py"


def test_live_dashboard_renders_after_a_run(tmp_path, monkeypatch):
    # Hermetic: no host store and no scratch data from earlier runs.
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    at = AppTest.from_file(str(_LIVE), default_timeout=120)
    at.session_state["benchmark_results"] = [{
        "task": "linalg", "run_id": "r1", "timestamp": time.time(),
        "duration_ms": 2000.0, "cpu_time_s": 1.9, "cpu_pct": 95.0,
        "rss_mb": 80.0, "ipc": 1.4, "cache_miss_pct": 3.0,
    }]
    at.session_state["run_count"] = 1
    at.session_state["is_running"] = False
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    subheaders = [s.value for s in at.subheader]
    assert "Bottleneck Analysis" in subheaders
    assert "Phase Analysis (Agentic Pipeline)" in subheaders
