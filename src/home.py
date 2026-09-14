#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""AgentSysPerf home directory + canonical store location.

Resolves where the single canonical results store and per-run artifacts live.
This is the seam that lets the same code target a zero-config local file or a
shared server (via a DSN) with no caller change.

Resolution (highest priority first):
- ``AGENTSYSPERF_STORE_DSN`` / ``AGENTSYSPERF_STORE_URL`` — explicit backend URL
  (e.g. ``sqlite:////abs/path/results.db`` or, later, ``postgresql://...``).
- ``AGENTSYSPERF_HOME`` — directory holding ``results.db`` + ``artifacts/``.
- default ``~/.agentsysperf`` (NOT ``/tmp`` — results must survive a reboot).
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME = Path.home() / ".agentsysperf"


def agentsysperf_home() -> Path:
    """The AgentSysPerf home dir (``$AGENTSYSPERF_HOME`` or ``~/.agentsysperf``).

    Created on first use so callers can rely on it existing.
    """
    root = Path(os.environ.get("AGENTSYSPERF_HOME", str(DEFAULT_HOME))).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_db_path() -> Path:
    """Path to the single canonical SQLite store file."""
    return agentsysperf_home() / "results.db"


def runs_dir() -> Path:
    """Root for per-run bulk artifacts (EMON CSVs, plots), addressed by run_id.

    Artifacts live at ``runs_dir()/<run_id>/`` and are referenced from the DB by
    a relative path — never discovered by globbing scattered ``/tmp`` dirs.
    """
    d = agentsysperf_home() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_dsn() -> str | None:
    """The configured backend DSN, if any (``AGENTSYSPERF_STORE_DSN``/``_URL``)."""
    return os.environ.get("AGENTSYSPERF_STORE_DSN") or os.environ.get("AGENTSYSPERF_STORE_URL")


__all__ = ["agentsysperf_home", "default_db_path", "runs_dir", "store_dsn", "DEFAULT_HOME"]
