#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the profile capability gate.

The profiles declare an ISA prerequisite (currently AMX). Applying an
AMX profile on a host without AMX would exercise the AVX-512 fallback while
labelling the result "amx_only" — a silently wrong comparison, which is exactly
what abort-don't-degrade exists to prevent. These tests pin that refusal.
"""

from __future__ import annotations

import pytest

from src.optimization_profiles._base import UnsupportedProfileError
from src.platform.detect import PlatformInfo
from src.protocols import discover_optimization_profiles

AMX_HOST = PlatformInfo(
    vendor="GenuineIntel",
    microarchitecture="Granite Rapids (GNR)",
    has_amx=True,
    has_avx512=True,
)
NO_AMX_HOST = PlatformInfo(
    vendor="GenuineIntel",
    microarchitecture="Clearwater Forest (CWF)",
    has_amx=False,
    has_avx512=False,
)


def _profiles():
    profiles = discover_optimization_profiles()
    assert profiles, "no optimization profiles discovered via entry points"
    return profiles


def test_every_amx_profile_declares_the_requirement():
    """A profile whose ISA axis is AMX must declare requires=("amx",).

    Without this, the gate is bypassed and the profile applies anywhere.
    """
    for name, prof in _profiles().items():
        isa = prof.SPEC.axes.get("isa", "")
        if "amx" in isa.lower():
            assert "amx" in prof.requires, f"{name} uses {isa} but does not require amx"


def test_baseline_profile_is_portable():
    """`base` must run on any host — it is the comparison point."""
    base = _profiles()["base"]
    assert base.requires == ()
    assert base.unsupported_on(NO_AMX_HOST) == []
    assert base.apply(sut_config={})["profile"] == "base"


def test_amx_profile_reports_unsupported_without_amx():
    prof = _profiles()["amx_only"]
    assert prof.unsupported_on(NO_AMX_HOST) == ["amx"]
    assert prof.unsupported_on(AMX_HOST) == []


def test_apply_refuses_rather_than_degrading(monkeypatch):
    """The refusal must be an exception, not a warning or a degraded config."""
    import src.platform as platform_mod

    monkeypatch.setattr(platform_mod, "detect_platform", lambda **kw: NO_AMX_HOST)
    prof = _profiles()["amx_only"]
    with pytest.raises(UnsupportedProfileError) as excinfo:
        prof.apply(sut_config={})
    # The message must name the profile and the missing capability, so the
    # failure is actionable from logs alone.
    assert "amx_only" in str(excinfo.value)
    assert "amx" in str(excinfo.value)


def test_apply_succeeds_on_capable_host(monkeypatch):
    """The gate must not block a host that genuinely has the ISA."""
    import src.platform as platform_mod

    monkeypatch.setattr(platform_mod, "detect_platform", lambda **kw: AMX_HOST)
    prof = _profiles()["amx_only"]
    out = prof.apply(sut_config={})
    assert out["profile"] == "amx_only"
    assert out["backend_config"]["isa_target"] == "amx_tdpbssd"
