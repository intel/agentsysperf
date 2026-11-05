#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""The Synth Density Study names the detected host, not a placeholder.

The architecture diagram's label read `PlatformInfo.uarch`, which does not
exist; the AttributeError was swallowed by the fallback, so every host was
labelled "Detected Platform". The field is `microarchitecture`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.platform import detect

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_DEMO = Path(__file__).resolve().parents[2] / "demo_app.py"


def test_density_study_labels_the_detected_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    fake = detect.PlatformInfo(microarchitecture="TestArch", physical_cores=8, numa_nodes=2)
    monkeypatch.setattr(detect, "detect_platform", lambda *a, **kw: fake)

    at = AppTest.from_file(str(_DEMO), default_timeout=120)
    at.session_state["nav_page"] = "Terminal-Bench::Scaling"
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    captions = [c.value for c in at.caption if "Current host" in c.value]
    assert captions, "Synth Density Study caption not rendered"
    assert "Current host: **TestArch (8C, 2 NUMA)**" in captions[0]
