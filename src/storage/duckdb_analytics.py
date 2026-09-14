#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""DuckDB read-side analytical accelerator (P8, optional).

DuckDB is a columnar embedded engine that ATTACHes the SQLite results store
directly (zero data duplication) and reads Parquet natively. It beats the
row-oriented SQLite store on wide cross-run / cross-sweep aggregation once row
counts are large — measured on real agentsysperf data: SQLite is ~50x faster at
~300 rows (use it), DuckDB wins ~10x above ~10-15k rows. So this is a READ-SIDE
accelerator switched on by scale, NOT a replacement for the system of record.

OPTIONAL: install with ``pip install agentsysperf[analytics]``. This module imports
cleanly without duckdb; the dependency is only required when a function is
actually called (clear message, not an ImportError traceback at import time).
Core and the dashboards never import this module.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence
from urllib.parse import urlparse

from src.home import default_db_path

_INSTALL_HINT = (
    "DuckDB analytics requires the optional 'analytics' extra. Install it with:\n"
    "    pip install agentsysperf[analytics]"
)


def _duckdb():
    """Import duckdb lazily with a clear install hint if it's missing."""
    try:
        import duckdb  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return duckdb


def _proxy_for_duckdb() -> Optional[str]:
    """The proxy env var rewritten into the only form DuckDB accepts.

    DuckDB's ``http_proxy`` setting wants a bare ``host:port``. The environment
    conventionally holds a full URL, and DuckDB does not parse one — on a proxied
    corporate host `http_proxy=http://proxy.example.com:912/` fails every
    extension load with ``Failed to parse port from http_proxy``. Since DuckDB
    autoinstalls the sqlite extension on first LOAD, that aborts the connection
    before any query runs. Credentials in the URL are not forwarded (DuckDB takes
    those as separate settings).
    """
    raw = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.hostname}:{port}"


def _connect():
    """A DuckDB connection with the proxy setting DuckDB can actually use."""
    duckdb = _duckdb()
    con = duckdb.connect()
    proxy = _proxy_for_duckdb()
    if proxy:
        con.execute("SET http_proxy = ?;", [proxy])
    return con


def attach(db_path: Optional[Path] = None, *, alias: str = "perf"):
    """Open a DuckDB connection with the SQLite store ATTACHed (zero copy).

    Returns a duckdb connection where the store's tables are reachable as
    ``{alias}.runs``, ``{alias}.measurements``, etc. The SQLite file is the
    single source of truth — DuckDB only reads it.
    """
    path = Path(db_path) if db_path is not None else default_db_path()
    con = _connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{path}' AS {alias} (TYPE sqlite, READ_ONLY);")
    return con


def query(sql: str, *, db_path: Optional[Path] = None, params: Sequence[Any] = ()) -> List[tuple]:
    """Run an analytical query over the ATTACHed store and return rows."""
    con = attach(db_path)
    try:
        return con.execute(sql, list(params)).fetchall()
    finally:
        con.close()


def measurement_count(*, db_path: Optional[Path] = None) -> int:
    """Total measurement rows — handy for the row-count gate (SQLite vs DuckDB)."""
    rows = query("SELECT count(*) FROM perf.measurements", db_path=db_path)
    return int(rows[0][0]) if rows else 0


def ipc_by_benchmark(*, db_path: Optional[Path] = None) -> List[tuple]:
    """Example wide aggregation: mean IPC per benchmark across ALL runs.

    The kind of cross-run scan DuckDB's columnar layout accelerates — joins
    runs→measurements and groups, scanning only the columns referenced.
    """
    return query(
        """
        SELECT r.benchmark_id,
               count(DISTINCT m.run_id)               AS runs,
               round(avg(m.ipc), 3)                   AS avg_ipc,
               round(avg(m.cache_miss_pct), 2)        AS avg_cache_miss_pct
        FROM perf.measurements m
        JOIN perf.runs r ON r.run_id = m.run_id
        WHERE m.ipc IS NOT NULL
        GROUP BY r.benchmark_id
        ORDER BY avg_ipc DESC
        """,
        db_path=db_path,
    )


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str, what: str) -> str:
    """Gate a bare SQL identifier. Neither table names nor COPY targets can be
    bound as parameters in DuckDB, so they have to be interpolated — which
    means the only safe interpolation is one that cannot contain a quote,
    semicolon, comment marker or whitespace in the first place.
    """
    if not _IDENT_RE.match(name or ""):
        raise ValueError(f"invalid {what}: {name!r}")
    return name


def export_parquet(out_path: Path, *, table: str = "measurements",
                   db_path: Optional[Path] = None) -> Path:
    """Export a store table to a Parquet bundle (publishable / cold tier).

    Parquet is columnar, self-describing, and the lingua franca for sharing
    results across machines. Reads via the ATTACHed SQLite store (no copy).
    """
    _check_identifier(table, "table name")
    con = attach(db_path)
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # `table` is validated above; the destination goes through DuckDB's own
        # single-quote escaping rather than raw f-string interpolation.
        dest = str(out_path).replace("'", "''")
        con.execute(
            f"COPY (SELECT * FROM perf.{table}) TO '{dest}' (FORMAT parquet);"  # nosec B608
        )
        return out_path
    finally:
        con.close()


def read_parquet(path: Path, *, limit: Optional[int] = None) -> List[tuple]:
    """Read a Parquet bundle back (cold-tier query in place, no ingestion)."""
    con = _connect()
    try:
        # read_parquet() is a table function, so its argument DOES bind as a
        # parameter — no interpolation needed here at all.
        if limit:
            return con.execute(
                "SELECT * FROM read_parquet(?) LIMIT ?", [str(Path(path)), int(limit)]
            ).fetchall()
        return con.execute(
            "SELECT * FROM read_parquet(?)", [str(Path(path))]
        ).fetchall()
    finally:
        con.close()


__all__ = [
    "attach", "query", "measurement_count", "ipc_by_benchmark",
    "export_parquet", "read_parquet",
]
