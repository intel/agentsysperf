#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Deterministic record/replay for agentic CPU benchmarking.

Records one canonical LLM trajectory, then replays it byte-identically on every
CPU under test so run-over-run variance drops from ~±15% to ~±2% and the Xeon
silicon signal becomes visible. Ported from the colleague's agentic-benchmark-4
(AWS) into AgentSysPerf for EMR; the AWS SSM/Bedrock plumbing is intentionally
dropped — only the portable proxy + fixture core is kept.

See :mod:`src.replay.proxy` for the HTTP proxy, :mod:`src.replay.manager`
for the :class:`ReplayProxy` lifecycle wrapper, and :mod:`src.replay.fixture`
for fixture I/O + validation.
"""
from src.replay.fixture import (
    FixtureStats,
    fixture_stats,
    load_fixture,
    validate_fixture,
)
from src.replay.manager import ReplayProxy

__all__ = [
    "ReplayProxy",
    "FixtureStats",
    "load_fixture",
    "fixture_stats",
    "validate_fixture",
]
