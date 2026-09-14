#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
AgentSysPerf storage backend plugins.

This module provides result stores that persist benchmark data for offline
analysis and report generation. The reference implementation is
:class:`SQLiteResultStore`, which writes to a local SQLite database.

External packages can ship alternative backends (PostgreSQL, cloud object
storage) by implementing the :class:`~src.protocols.ResultStore`
Protocol and registering via the ``agentsysperf.result_stores`` entry point.
"""

from src.storage.sqlite_store import SQLiteResultStore

__all__ = ["SQLiteResultStore"]
