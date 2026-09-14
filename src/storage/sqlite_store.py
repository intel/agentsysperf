#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
SQLite-backed result store for AgentSysPerf benchmark data.

This module implements the :class:`~src.protocols.ResultStore` Protocol
using SQLite as the persistence layer. It stores run metadata, task results,
measurement records, and analyzer verdicts in a local database file.

The SQLite backend is the reference implementation for result storage. It
provides:

- Zero-config setup (no server required)
- ACID guarantees for data integrity
- Efficient querying via indexes
- JSON support for flexible metadata storage
- Backward compatibility with existing JSON-based workflows
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.protocols import AnalysisResult, MeasurementRecord

log = logging.getLogger(__name__)


class SQLiteResultStore:
    """SQLite-backed result store for AgentSysPerf benchmark data.

    This store persists benchmark results in a local SQLite database located
    at ``{output_dir}/agentsysperf_results.db``. The schema supports run metadata,
    task results, measurement records, and analyzer verdicts with efficient
    indexing for common query patterns.

    Example usage:

        >>> store = SQLiteResultStore(output_dir=Path("/tmp/results"))
        >>> store.store_run_metadata(
        ...     run_id="run_001",
        ...     metadata={"hardware_sku": "Xeon 8592+", "model": "claude-3"}
        ... )
        >>> store.store_task_result(
        ...     run_id="run_001",
        ...     task_id="task_001",
        ...     result={"passed": True, "duration_s": 12.5}
        ... )

    The store is thread-safe for reads but uses a single connection for writes.
    For high-concurrency workloads, consider a server-based backend (PostgreSQL).
    """

    name: str = "sqlite"

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        *,
        db_path: Optional[Path] = None,
        read_only: bool = False,
    ) -> None:
        """Initialize the SQLite store.

        Args:
            output_dir: Directory holding the DB at ``{output_dir}/
                agentsysperf_results.db`` (legacy per-dir layout). Back-compat path
                used by existing callers. NOTE this is a *different file* from the
                ``results.db`` that :meth:`open` uses, so passing a directory the
                CLI wrote yields an empty store, not that run's data — logs a
                warning when it detects exactly that.
            db_path: Explicit path to the DB file, overriding ``output_dir``.
                Used by :meth:`open` for the single canonical store
                (``$AGENTSYSPERF_HOME/results.db``).
            read_only: Open the database read-only (``file:...?mode=ro``) and
                skip schema creation/migration. Use for viewers (dashboards)
                that must never mutate or migrate the file.
        """
        if db_path is not None:
            self.db_path = Path(db_path)
            self.output_dir = self.db_path.parent
        elif output_dir is not None:
            self.output_dir = Path(output_dir)
            self.db_path = self.output_dir / "agentsysperf_results.db"
            # Two filenames coexist: this legacy per-dir name, and results.db as
            # written by open() into $AGENTSYSPERF_HOME. Pointing this
            # constructor at a directory the CLI wrote does not read that data —
            # it creates an empty second DB beside it, and every query returns
            # zero rows with no error. Warn rather than switch: the dashboards
            # deliberately open the legacy name to find old sweeps.
            canonical = self.output_dir / "results.db"
            if canonical.exists() and not self.db_path.exists():
                log.warning(
                    "%s already contains results.db (written by "
                    "SQLiteResultStore.open()), but this constructor opens %s — "
                    "queries will return nothing. Use SQLiteResultStore.open() "
                    "or pass db_path=%s to read the existing data.",
                    self.output_dir, self.db_path, canonical,
                )
        else:
            raise ValueError("SQLiteResultStore requires output_dir or db_path")
        self.read_only = read_only
        self._conn: Optional[sqlite3.Connection] = None
        # When inside persist_run(), the individual store_* methods must NOT
        # commit — persist_run owns one transaction across all of them so a
        # crash mid-flush leaves nothing partially committed.
        self._in_transaction = False
        if not read_only:
            self._ensure_db()

    def _commit(self) -> None:
        """Commit unless inside a persist_run() transaction (then a no-op)."""
        if not self._in_transaction:
            self._get_connection().commit()

    def _on_write_error(self, what: str, key: str, exc: Exception) -> None:
        """Handle a write error. Standalone calls log+rollback (degrade);
        inside persist_run() the error is RE-RAISED so the whole transaction
        rolls back (abort-don't-degrade — no half-written run)."""
        if self._in_transaction:
            raise exc
        log.warning("Failed to store %s for %s: %s", what, key, exc)
        self._get_connection().rollback()

    @classmethod
    def open(
        cls,
        *,
        dsn: Optional[str] = None,
        read_only: bool = False,
    ) -> "SQLiteResultStore":
        """Open the canonical store — the single blessed construction seam.

        Resolution order: explicit ``dsn`` arg > ``$AGENTSYSPERF_STORE_DSN`` /
        ``$AGENTSYSPERF_STORE_URL`` > ``$AGENTSYSPERF_HOME/results.db`` (default
        ``~/.agentsysperf``). A bare ``open()`` returns the local SQLite store; a
        non-sqlite DSN is the seam for a future shared-server backend (shipped
        as a separate entry-point plugin), which this class does not implement.

        The same call site therefore targets either the zero-config local file
        or a server — the only difference is the DSN — which is the
        local↔server uniformity the service/multi-user tier hangs off.
        """
        from src.home import default_db_path, store_dsn

        dsn = dsn or store_dsn()
        if dsn:
            return cls.from_dsn(dsn, read_only=read_only)
        return cls(db_path=default_db_path(), read_only=read_only)

    @classmethod
    def from_dsn(cls, dsn: str, *, read_only: bool = False) -> "SQLiteResultStore":
        """Construct from a DSN URL. Only ``sqlite:`` schemes are handled here.

        Accepts ``sqlite:///relative.db`` and ``sqlite:////abs/path.db`` (and a
        bare path). A non-sqlite scheme raises ``NotImplementedError`` with a
        clear message — the seam exists, the backend is a later plugin.
        """
        from urllib.parse import urlparse

        if "://" not in dsn:
            # Bare filesystem path, no scheme.
            return cls(db_path=Path(dsn), read_only=read_only)

        parsed = urlparse(dsn)
        if parsed.scheme != "sqlite":
            raise NotImplementedError(
                f"No result-store backend installed for scheme '{parsed.scheme}://'. "
                f"SQLiteResultStore handles 'sqlite:' only; a '{parsed.scheme}' backend "
                f"ships as a separate agentsysperf.result_stores entry-point plugin."
            )
        # sqlite:///rel/db -> path '/rel/db' (one leading slash = relative);
        # sqlite:////abs/db -> path '//abs/db' (two = absolute). urlparse puts
        # everything after 'sqlite://' into .path here.
        path = parsed.path
        if path.startswith("//"):
            path = path[1:]          # //abs -> /abs (absolute)
        else:
            path = path.lstrip("/")  # /rel -> rel (relative to cwd)
        return cls(db_path=Path(path), read_only=read_only)

    def _ensure_db(self) -> None:
        """Create the database and tables if they don't exist, then migrate."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()

        # Read schema from the embedded SQL file
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            log.warning(
                "SQLite schema file not found at %s — creating minimal schema",
                schema_path,
            )
            self._create_fallback_schema(conn)
        else:
            schema_sql = schema_path.read_text()
            conn.executescript(schema_sql)

        conn.commit()
        self._run_migrations(conn)

    # Directory holding ordered ``000N_*.sql`` migration files. A migration is
    # applied when its leading number is greater than the DB's PRAGMA
    # user_version, inside a transaction, fail-loud (abort-don't-degrade).
    _MIGRATIONS_DIR = Path(__file__).parent / "migrations"

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply ordered SQL migrations newer than ``PRAGMA user_version``.

        Versioning is DB-level via SQLite's built-in ``user_version``. Each
        ``migrations/000N_*.sql`` file whose number ``N`` exceeds the current
        version is executed in its own transaction; on any error we RAISE
        (never swallow) so a half-applied schema aborts the run rather than
        silently degrading. ``schema.sql`` (run above) establishes the baseline
        shape, so a fresh DB starts effectively at the highest baseline version.
        """
        current = conn.execute("PRAGMA user_version").fetchone()[0]

        migrations = []
        if self._MIGRATIONS_DIR.exists():
            for path in sorted(self._MIGRATIONS_DIR.glob("[0-9]*.sql")):
                try:
                    version = int(path.name.split("_", 1)[0])
                except ValueError:
                    log.warning("Skipping unparseable migration filename: %s", path.name)
                    continue
                migrations.append((version, path))

        # Baseline: if the DB has never been versioned (user_version=0) but the
        # core tables already exist (schema.sql just ran), adopt the highest
        # baseline migration version without re-applying its DDL — schema.sql
        # already produced that shape. Marker migration 0001 is the baseline.
        if current == 0 and migrations:
            baseline = max(v for v, _ in migrations if v <= 1) if any(v <= 1 for v, _ in migrations) else 0
            if baseline:
                conn.execute(f"PRAGMA user_version = {baseline}")
                conn.commit()
                current = baseline

        # NOTE on atomicity: sqlite3.executescript() implicitly COMMITs any
        # pending transaction, so a manual BEGIN wrapped around it is a no-op
        # UNLESS it is inside the script text — table-rebuild migrations
        # (DROP/RENAME) therefore embed their own `BEGIN; ... COMMIT;`.
        #
        # NOTE on foreign keys: SQLite's documented table-rebuild procedure
        # requires foreign_keys OFF during the rebuild (so DROP/RENAME doesn't
        # trip child FKs), then a `foreign_key_check` afterward as the integrity
        # gate, then ON. We apply every migration under this protocol; the check
        # is fail-loud (abort-don't-degrade). The pragma is a no-op inside a
        # transaction, so it is set in autocommit before executescript.
        for version, path in migrations:
            if version <= current:
                continue
            sql = path.read_text()
            try:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.executescript(sql)
                conn.execute(f"PRAGMA user_version = {version}")
                conn.commit()
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()
                if violations:
                    raise sqlite3.IntegrityError(
                        f"foreign_key_check found {len(violations)} violation(s): "
                        f"{violations[:5]}"
                    )
            except sqlite3.Error as e:
                conn.rollback()
                conn.execute("PRAGMA foreign_keys = ON")
                raise RuntimeError(
                    f"Migration {path.name} failed (DB left at version {current}): {e}"
                ) from e
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
            current = version
            log.info("Applied migration %s (user_version -> %d)", path.name, version)

    def _create_fallback_schema(self, conn: sqlite3.Connection) -> None:
        """Create a minimal schema if schema.sql is missing."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                start_time INTEGER NOT NULL,
                end_time INTEGER,
                hardware_sku TEXT DEFAULT 'Intel Xeon Platinum 8592+',
                model TEXT,
                total_tasks INTEGER DEFAULT 0,
                passed_tasks INTEGER DEFAULT 0,
                metadata JSON
            );

            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                workload_type TEXT,
                passed BOOLEAN NOT NULL,
                duration_s REAL,
                num_turns INTEGER,
                num_commands INTEGER,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS analyzer_verdicts (
                verdict_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                analyzer_name TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence JSON NOT NULL,
                recommendations JSON,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                parent_span_id TEXT,
                task_id TEXT,
                span_kind TEXT NOT NULL,
                node_id TEXT,
                start_ts_us INTEGER DEFAULT 0,
                end_ts_us INTEGER DEFAULT 0,
                duration_us INTEGER DEFAULT 0,
                model_id TEXT,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                tool_name TEXT,
                resource_tier TEXT,
                status TEXT DEFAULT 'ok',
                error TEXT,
                extra JSON,
                schema_version TEXT DEFAULT '0.3',
                PRIMARY KEY (run_id, span_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_task_workload ON task_results(workload_type);
            CREATE INDEX IF NOT EXISTS idx_verdict_analyzer ON analyzer_verdicts(analyzer_name, verdict);
            CREATE INDEX IF NOT EXISTS idx_spans_task ON spans(run_id, task_id);
            CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(run_id, span_kind);
        """)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection.

        ``PRAGMA foreign_keys`` is connection-scoped, so it must be re-set on
        every new connection — without it the schema's ``ON DELETE CASCADE``
        rules are decorative. WAL gives one-writer/many-readers concurrency.
        Read-only stores open via a ``mode=ro`` URI and never enable WAL
        (which would require a writable file) or DDL.
        """
        if self._conn is None:
            if self.read_only:
                self._conn = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
                )
            else:
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Ensure connection is closed on cleanup."""
        self.close()

    # ── Benchmarks + run discovery (P3) ────────────────────────────────

    def store_benchmark(self, *, benchmark_id: str, metadata: Dict[str, Any]) -> None:
        """Upsert a benchmark (the top-level entity runs belong to)."""
        conn = self._get_connection()
        known = {"display_name", "version", "created_at"}
        extra = {k: v for k, v in metadata.items() if k not in known}
        try:
            conn.execute(
                """
                INSERT INTO benchmarks (benchmark_id, display_name, version, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(benchmark_id) DO UPDATE SET
                    display_name = COALESCE(excluded.display_name, display_name),
                    version = COALESCE(excluded.version, version),
                    metadata = excluded.metadata
                """,
                (
                    benchmark_id,
                    metadata.get("display_name"),
                    metadata.get("version"),
                    metadata.get("created_at", 0),
                    json.dumps(extra),
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            log.warning("Failed to store benchmark %s: %s", benchmark_id, e)
            conn.rollback()

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all known benchmarks (those registered + those runs reference)."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT b.benchmark_id, b.display_name, b.version, b.created_at,
                       COUNT(r.run_id) AS run_count
                FROM benchmarks b
                LEFT JOIN runs r ON r.benchmark_id = b.benchmark_id
                GROUP BY b.benchmark_id
                UNION
                SELECT DISTINCT r.benchmark_id, NULL, NULL, NULL,
                       (SELECT COUNT(*) FROM runs r2 WHERE r2.benchmark_id = r.benchmark_id)
                FROM runs r
                WHERE r.benchmark_id IS NOT NULL
                  AND r.benchmark_id NOT IN (SELECT benchmark_id FROM benchmarks)
                ORDER BY benchmark_id
                """
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            log.warning("list_benchmarks failed: %s", e)
            return []

    def list_runs(
        self, *, benchmark_id: Optional[str] = None, owner_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List runs newest-first (by start_time), optionally filtered.

        ``owner_id=None`` returns everyone's runs (the shared/leaderboard view);
        a value scopes to one owner (the "my experiments" view).
        """
        conn = self._get_connection()
        sql = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []
        if benchmark_id is not None:
            sql += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if owner_id is not None:
            sql += " AND owner_id = ?"
            params.append(owner_id)
        sql += " ORDER BY start_time DESC, run_id DESC LIMIT ?"
        params.append(limit)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            log.warning("list_runs failed: %s", e)
            return []
        return [self._decode_run_row(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one run's full row (metadata JSON decoded), or None."""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        except sqlite3.Error as e:
            log.warning("get_run failed for %s: %s", run_id, e)
            return None
        return self._decode_run_row(row) if row else None

    def latest_run(self, *, benchmark_id: str) -> Optional[str]:
        """Most-recent run_id for a benchmark (newest start_time), or None."""
        runs = self.list_runs(benchmark_id=benchmark_id, limit=1)
        return runs[0]["run_id"] if runs else None

    @staticmethod
    def _decode_run_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (ValueError, TypeError):
                pass
        return d

    # ── Deletion (P3) ──────────────────────────────────────────────────

    def delete_run(self, run_id: str, *, prune_sweep_point: bool = True) -> int:
        """Delete a run and cascade its children. Returns rows deleted (0/1).

        measurements/task_results/analyzer_verdicts cascade via ON DELETE CASCADE
        (requires foreign_keys=ON, which _get_connection sets). spans + sweep_points
        reference run_id WITHOUT a cascade FK, so they are pruned app-level here.
        """
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM spans WHERE run_id = ?", (run_id,))
            if prune_sweep_point:
                conn.execute("DELETE FROM sweep_points WHERE run_id = ?", (run_id,))
            cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return cur.rowcount
        except sqlite3.Error as e:
            log.warning("delete_run failed for %s: %s", run_id, e)
            conn.rollback()
            return 0

    def delete_benchmark(self, benchmark_id: str) -> int:
        """Delete a benchmark and all its runs (+ their children). Returns # runs deleted."""
        conn = self._get_connection()
        try:
            run_ids = [r[0] for r in conn.execute(
                "SELECT run_id FROM runs WHERE benchmark_id = ?", (benchmark_id,)
            ).fetchall()]
            n = 0
            for rid in run_ids:
                n += self.delete_run(rid)
            conn.execute("DELETE FROM benchmarks WHERE benchmark_id = ?", (benchmark_id,))
            conn.commit()
            return n
        except sqlite3.Error as e:
            log.warning("delete_benchmark failed for %s: %s", benchmark_id, e)
            conn.rollback()
            return 0

    def delete_sweep(self, sweep_id: str, *, cascade_runs: bool = True) -> int:
        """Delete a sweep (+ its points). With cascade_runs, also delete the cell
        runs and the sweep_id run row. Returns # cell runs deleted."""
        conn = self._get_connection()
        try:
            cell_run_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT run_id FROM sweep_points WHERE sweep_id = ?", (sweep_id,)
            ).fetchall()]
            conn.execute("DELETE FROM sweep_points WHERE sweep_id = ?", (sweep_id,))
            conn.execute("DELETE FROM sweeps WHERE sweep_id = ?", (sweep_id,))
            conn.commit()
            n = 0
            if cascade_runs:
                for rid in cell_run_ids:
                    n += self.delete_run(rid)
                # the sweep_id itself is also a runs row (the verdict's parent)
                n += self.delete_run(sweep_id)
            return n
        except sqlite3.Error as e:
            log.warning("delete_sweep failed for %s: %s", sweep_id, e)
            conn.rollback()
            return 0

    # ── Artifacts (bulk per-run files, addressed by run_id) (P6) ───────

    def store_artifact(self, *, run_id: str, kind: str, name: str, path: Path) -> None:
        """Register an on-disk artifact (EMON CSV, scaling PNG, ...) for a run.

        Stores a resolvable filesystem path; the file itself stays on disk (the
        store never blobs it). Upserts on (run_id, kind, name).
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO artifacts (run_id, kind, name, path, created_at, metadata)
                VALUES (?, ?, ?, ?, 0, NULL)
                ON CONFLICT(run_id, kind, name) DO UPDATE SET path = excluded.path
                """,
                (run_id, kind, name, str(path)),
            )
            self._commit()
        except sqlite3.Error as e:
            self._on_write_error("artifact", f"{run_id}/{kind}/{name}", e)

    def get_artifact_path(
        self, run_id: str, *, kind: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[Path]:
        """Resolve a registered artifact back to a filesystem Path, or None.

        Returns the first match (most recent by artifact_id) whose file still
        exists on disk — callers (EmonAnalyzer, st.image) need a real path.
        """
        conn = self._get_connection()
        sql = "SELECT path FROM artifacts WHERE run_id = ?"
        params: List[Any] = [run_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        if name is not None:
            sql += " AND name = ?"
            params.append(name)
        sql += " ORDER BY artifact_id DESC"
        try:
            for row in conn.execute(sql, params).fetchall():
                p = Path(row[0])
                if p.exists():
                    return p
        except sqlite3.Error as e:
            log.warning("get_artifact_path failed for %s: %s", run_id, e)
        return None

    def list_artifacts(self, run_id: str, *, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered artifacts for a run, optionally filtered by kind."""
        conn = self._get_connection()
        sql = "SELECT run_id, kind, name, path FROM artifacts WHERE run_id = ?"
        params: List[Any] = [run_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY artifact_id"
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as e:
            log.warning("list_artifacts failed for %s: %s", run_id, e)
            return []

    def store_run_metadata(self, *, run_id: str, metadata: Dict[str, Any]) -> None:
        """Record run-level metadata (hardware SKU, model, start time, etc.).

        Args:
            run_id: Unique identifier for the run.
            metadata: Dictionary containing run metadata. Expected keys:
                - ``start_time`` (int): Unix timestamp when run started.
                - ``end_time`` (int, optional): Unix timestamp when run ended.
                - ``hardware_sku`` (str, optional): CPU SKU identifier.
                - ``model`` (str, optional): LLM model name.
                - ``total_tasks`` (int, optional): Total number of tasks.
                - ``passed_tasks`` (int, optional): Number of passed tasks.
                Additional keys are preserved in the ``metadata`` JSON field.
        """
        conn = self._get_connection()
        try:
            # Promoted columns (incl. P3 provenance/owner). Anything else is
            # preserved in the metadata JSON blob.
            known_fields = {
                "start_time", "end_time", "hardware_sku", "model",
                "total_tasks", "passed_tasks",
                "benchmark_id", "optimization_profile", "numa_policy",
                "agentsysperf_version", "result_digest",
                "owner_id", "owner_kind", "host_id", "status",
            }
            extra_metadata = {k: v for k, v in metadata.items() if k not in known_fields}

            conn.execute(
                """
                INSERT INTO runs (
                    run_id, start_time, end_time, hardware_sku, model,
                    total_tasks, passed_tasks, benchmark_id, optimization_profile,
                    numa_policy, agentsysperf_version, result_digest,
                    owner_id, owner_kind, host_id, status, metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    end_time = COALESCE(excluded.end_time, end_time),
                    hardware_sku = COALESCE(excluded.hardware_sku, hardware_sku),
                    model = COALESCE(excluded.model, model),
                    total_tasks = COALESCE(excluded.total_tasks, total_tasks),
                    passed_tasks = COALESCE(excluded.passed_tasks, passed_tasks),
                    benchmark_id = COALESCE(excluded.benchmark_id, benchmark_id),
                    optimization_profile = COALESCE(excluded.optimization_profile, optimization_profile),
                    numa_policy = COALESCE(excluded.numa_policy, numa_policy),
                    agentsysperf_version = COALESCE(excluded.agentsysperf_version, agentsysperf_version),
                    result_digest = COALESCE(excluded.result_digest, result_digest),
                    owner_id = COALESCE(excluded.owner_id, owner_id),
                    owner_kind = COALESCE(excluded.owner_kind, owner_kind),
                    host_id = COALESCE(excluded.host_id, host_id),
                    status = COALESCE(excluded.status, status),
                    metadata = excluded.metadata
                """,
                (
                    run_id,
                    metadata.get("start_time", 0),
                    metadata.get("end_time"),
                    metadata.get("hardware_sku"),
                    metadata.get("model"),
                    metadata.get("total_tasks"),
                    metadata.get("passed_tasks"),
                    metadata.get("benchmark_id"),
                    metadata.get("optimization_profile"),
                    metadata.get("numa_policy"),
                    metadata.get("agentsysperf_version"),
                    metadata.get("result_digest"),
                    metadata.get("owner_id"),
                    metadata.get("owner_kind"),
                    metadata.get("host_id"),
                    metadata.get("status"),
                    json.dumps(extra_metadata),
                ),
            )
            self._commit()
        except sqlite3.Error as e:
            self._on_write_error("run metadata", run_id, e)

    def store_task_result(
        self, *, run_id: str, task_id: str, result: Dict[str, Any]
    ) -> None:
        """Record the result of one task (pass/fail, timing, turns, etc.).

        Args:
            run_id: Unique identifier for the run.
            task_id: Unique identifier for the task.
            result: Dictionary containing task result. Expected keys:
                - ``passed`` (bool): Whether the task passed.
                - ``workload_type`` (str, optional): Task category/workload type.
                - ``duration_s`` (float, optional): Task duration in seconds.
                - ``elapsed_s`` (float, optional): Agent invocation duration in seconds.
                - ``num_turns`` (int, optional): Number of agent turns.
                - ``num_commands`` (int, optional): Number of commands executed.
        """
        conn = self._get_connection()
        try:
            # logical_task_key = "{benchmark_id}::{task_id}" is the cross-run
            # aggregation key. The run's benchmark_id is the source of truth; look
            # it up so the key is correct even when the caller doesn't pass it.
            row = conn.execute(
                "SELECT benchmark_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            benchmark_id = row[0] if row else None
            logical_task_key = f"{benchmark_id}::{task_id}" if benchmark_id else None

            conn.execute(
                """
                INSERT INTO task_results
                    (run_id, task_id, logical_task_key, workload_type, passed, duration_s, elapsed_s, num_turns, num_commands)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, task_id) DO UPDATE SET
                    logical_task_key = COALESCE(excluded.logical_task_key, logical_task_key),
                    passed = excluded.passed,
                    workload_type = COALESCE(excluded.workload_type, workload_type),
                    duration_s = COALESCE(excluded.duration_s, duration_s),
                    elapsed_s = COALESCE(excluded.elapsed_s, elapsed_s),
                    num_turns = COALESCE(excluded.num_turns, num_turns),
                    num_commands = COALESCE(excluded.num_commands, num_commands)
                """,
                (
                    run_id,
                    task_id,
                    logical_task_key,
                    result.get("workload_type"),
                    result.get("passed", False),
                    result.get("duration_s"),
                    result.get("elapsed_s"),
                    result.get("num_turns"),
                    result.get("num_commands"),
                ),
            )
            self._commit()
        except sqlite3.Error as e:
            self._on_write_error("task result", task_id, e)

    # Keys promoted from the free-form payload into typed columns. The rest of
    # the payload (perfspect TMA, raw events, future keys) stays in the JSON
    # blob. Kept as a tuple so store/query agree on the promoted set.
    _PROMOTED_MEASUREMENT_KEYS = (
        "duration_us",
        "cpu_time_s",
        "cpu_pct_mean",
        "rss_kb_peak",
        "ipc",
        "cache_miss_pct",
    )

    @staticmethod
    def _measurement_task_id(span_id: str) -> str:
        """Derive task_id from span_id.

        Matches the Prometheus exporter's rule verbatim
        (prometheus_exporter.py): ``span_id.split('::')[-1]`` when a ``::`` is
        present (e.g. ``production_run::make-mips``), else the whole span_id
        (e.g. ``terminal-bench/task/turn_0_llm``). Keeping these identical means
        a later store-backed exporter produces the same task_id label set.
        """
        return span_id.split("::")[-1] if "::" in span_id else span_id

    def store_measurements(
        self, *, run_id: str, records: Sequence[MeasurementRecord]
    ) -> None:
        """Persist measurement records (L1/L3/L5/perfspect) for a run.

        Each record becomes one row in ``measurements``: the six promoted hot
        keys are pulled into typed columns (omitted keys stay NULL, never 0),
        and the full payload is preserved as JSON so it round-trips byte-equal
        and new layers/keys need no migration. Idempotent: upserts on
        ``(run_id, span_id, layer)`` so a re-run overwrites rather than
        duplicates. ``seq`` records original insertion order.

        Args:
            run_id: Unique identifier for the run.
            records: Sequence of measurement records to persist.
        """
        conn = self._get_connection()
        try:
            # Continue the seq counter past any rows already stored for this run
            # so re-stored records keep monotonic ordering.
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM measurements WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = (row[0] if row else -1) + 1

            for rec in records:
                payload = dict(rec.payload)
                promoted = [payload.get(k) for k in self._PROMOTED_MEASUREMENT_KEYS]
                conn.execute(
                    """
                    INSERT INTO measurements (
                        run_id, span_id, layer, task_id,
                        duration_us, cpu_time_s, cpu_pct_mean,
                        rss_kb_peak, ipc, cache_miss_pct,
                        payload, seq
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id, span_id, layer) DO UPDATE SET
                        task_id=excluded.task_id,
                        duration_us=excluded.duration_us,
                        cpu_time_s=excluded.cpu_time_s,
                        cpu_pct_mean=excluded.cpu_pct_mean,
                        rss_kb_peak=excluded.rss_kb_peak,
                        ipc=excluded.ipc,
                        cache_miss_pct=excluded.cache_miss_pct,
                        payload=excluded.payload,
                        seq=excluded.seq
                    """,
                    (
                        run_id,
                        rec.span_id,
                        rec.layer,
                        self._measurement_task_id(rec.span_id),
                        *promoted,
                        json.dumps(payload),
                        seq,
                    ),
                )
                seq += 1
            self._commit()
        except sqlite3.Error as e:
            self._on_write_error("measurements", run_id, e)

    def query_measurements(
        self,
        run_id: str,
        *,
        layer: Optional[str] = None,
        span_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch measurement records for a run, optionally filtered.

        Returns rows ordered by ``seq`` (original insertion order), each a dict
        with ``span_id``, ``layer``, ``task_id`` and ``payload`` (decoded from
        JSON). The payload round-trips byte-equal to what was stored, so a row
        reconstructs a :class:`MeasurementRecord` losslessly — the make-or-break
        contract for the dashboards and the Prometheus exporter.
        """
        conn = self._get_connection()
        sql = "SELECT span_id, layer, task_id, payload FROM measurements WHERE run_id = ?"
        params: List[Any] = [run_id]
        if layer is not None:
            sql += " AND layer = ?"
            params.append(layer)
        if span_id is not None:
            sql += " AND span_id = ?"
            params.append(span_id)
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY seq"
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            log.warning("query_measurements failed for %s: %s", run_id, e)
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d["payload"])
            except (ValueError, TypeError):
                d["payload"] = {}
            out.append(d)
        return out

    def store_spans(self, *, run_id: str, spans: Sequence[Any]) -> None:
        """Persist StepTrace rows (step-level execution trace) for a run.

        Idempotent: upserts on the ``(run_id, span_id)`` primary key so a
        re-run of the same task overwrites rather than duplicates. ``spans`` is
        a sequence of :class:`src.trace.StepTrace` (or any object exposing
        the same attributes).
        """
        conn = self._get_connection()
        try:
            for s in spans:
                kind = getattr(s.span_kind, "value", s.span_kind)
                # task_id is the parent for child rows, else the span itself.
                task_id = s.parent_span_id or s.span_id
                conn.execute(
                    """
                    INSERT INTO spans (
                        span_id, run_id, parent_span_id, task_id, span_kind,
                        node_id, start_ts_us, end_ts_us, duration_us,
                        model_id, tokens_in, tokens_out, cost_usd,
                        tool_name, resource_tier, status, error, extra,
                        schema_version
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id, span_id) DO UPDATE SET
                        parent_span_id=excluded.parent_span_id,
                        task_id=excluded.task_id,
                        span_kind=excluded.span_kind,
                        node_id=excluded.node_id,
                        start_ts_us=excluded.start_ts_us,
                        end_ts_us=excluded.end_ts_us,
                        duration_us=excluded.duration_us,
                        model_id=excluded.model_id,
                        tokens_in=excluded.tokens_in,
                        tokens_out=excluded.tokens_out,
                        cost_usd=excluded.cost_usd,
                        tool_name=excluded.tool_name,
                        resource_tier=excluded.resource_tier,
                        status=excluded.status,
                        error=excluded.error,
                        extra=excluded.extra
                    """,
                    (
                        s.span_id, run_id, s.parent_span_id, task_id, kind,
                        s.node_id, s.start_ts_us, s.end_ts_us, s.duration_us,
                        s.model_id, s.tokens_in, s.tokens_out, s.cost_usd,
                        s.tool_name, s.resource_tier, s.status, s.error,
                        json.dumps(dict(s.extra or {})), s.schema_version,
                    ),
                )
            self._commit()
            log.debug("Stored %d spans for run %s", len(list(spans)), run_id)
        except sqlite3.Error as e:
            self._on_write_error("spans", run_id, e)

    def query_spans(
        self,
        run_id: str,
        task_id: Optional[str] = None,
        span_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch step-trace rows for a run, optionally filtered.

        Args:
            run_id: Run identifier.
            task_id: Optional task filter (matches the ``task_id`` column).
            span_kind: Optional kind filter (e.g. ``"llm_call"``).

        Returns rows ordered by span_id (turn order), as dicts with ``extra``
        decoded from JSON.
        """
        conn = self._get_connection()
        sql = "SELECT * FROM spans WHERE run_id = ?"
        params: List[Any] = [run_id]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        if span_kind is not None:
            sql += " AND span_kind = ?"
            params.append(span_kind)
        sql += " ORDER BY span_id"
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            log.warning("query_spans failed for %s: %s", run_id, e)
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("extra"):
                try:
                    d["extra"] = json.loads(d["extra"])
                except (ValueError, TypeError):
                    pass
            out.append(d)
        return out

    def store_analysis_results(
        self, *, run_id: str, results: Sequence[AnalysisResult]
    ) -> None:
        """Persist analyzer verdicts and recommendations.

        Args:
            run_id: Unique identifier for the run.
            results: Sequence of analysis results to persist.
        """
        conn = self._get_connection()
        try:
            for result in results:
                # Extract task_id from span_id if present (assumes span_id == task_id)
                task_id = result.span_id or "unknown"

                conn.execute(
                    """
                    INSERT INTO analyzer_verdicts (run_id, task_id, analyzer_name, verdict, confidence, evidence, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, task_id, analyzer_name) DO UPDATE SET
                        verdict = excluded.verdict,
                        confidence = excluded.confidence,
                        evidence = excluded.evidence,
                        recommendations = excluded.recommendations
                    """,
                    (
                        run_id,
                        task_id,
                        result.analyzer_name,
                        result.verdict,
                        result.confidence,
                        json.dumps(dict(result.evidence)),
                        json.dumps(list(result.recommendations)),
                    ),
                )
            self._commit()
        except sqlite3.Error as e:
            self._on_write_error("analysis results", run_id, e)

    def persist_run(
        self,
        *,
        run_id: str,
        metadata: Dict[str, Any],
        task_results: Sequence[Tuple[str, Dict[str, Any]]] = (),
        records: Sequence[MeasurementRecord] = (),
        spans: Sequence[Any] = (),
        verdicts: Sequence[AnalysisResult] = (),
        artifacts: Sequence[Dict[str, Any]] = (),
    ) -> None:
        """Atomically flush an entire run in ONE transaction.

        Wraps run metadata + task results + measurements + spans + verdicts +
        artifacts so a crash mid-flush leaves NOTHING partially committed
        (abort-don't-degrade). Replaces the scattered sequence of store_* calls
        scripts make today.

        ``task_results`` is a sequence of ``(task_id, result_dict)`` pairs.
        ``artifacts`` is a sequence of ``{"kind", "name", "path"}`` dicts.
        Run metadata must be stored first so task_results' logical_task_key
        lookup (and the artifacts FK parent) see the run row.
        """
        conn = self._get_connection()
        self._in_transaction = True
        try:
            self.store_run_metadata(run_id=run_id, metadata=metadata)
            for task_id, result in task_results:
                self.store_task_result(run_id=run_id, task_id=task_id, result=result)
            if records:
                self.store_measurements(run_id=run_id, records=records)
            if spans:
                self.store_spans(run_id=run_id, spans=spans)
            if verdicts:
                self.store_analysis_results(run_id=run_id, results=verdicts)
            for art in artifacts:
                self.store_artifact(
                    run_id=run_id, kind=art["kind"],
                    name=art["name"], path=Path(art["path"]),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.warning("persist_run failed for %s (rolled back): %s", run_id, e)
            raise
        finally:
            self._in_transaction = False

    def query_tasks(
        self, run_id: str, workload_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch task results for a run, optionally filtered by workload type.

        Args:
            run_id: Unique identifier for the run.
            workload_type: Optional workload type filter (e.g., "linalg", "compile").

        Returns:
            List of task result dictionaries, each containing:
                - ``task_id`` (str)
                - ``workload_type`` (str or None)
                - ``passed`` (bool)
                - ``duration_s`` (float or None)
                - ``elapsed_s`` (float or None)
                - ``num_turns`` (int or None)
                - ``num_commands`` (int or None)
        """
        conn = self._get_connection()
        try:
            if workload_type:
                cursor = conn.execute(
                    """
                    SELECT task_id, workload_type, passed, duration_s, elapsed_s, num_turns, num_commands
                    FROM task_results
                    WHERE run_id = ? AND workload_type = ?
                    ORDER BY task_id
                    """,
                    (run_id, workload_type),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT task_id, workload_type, passed, duration_s, elapsed_s, num_turns, num_commands
                    FROM task_results
                    WHERE run_id = ?
                    ORDER BY task_id
                    """,
                    (run_id,),
                )

            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            log.warning("Failed to query tasks for %s: %s", run_id, e)
            return []

    # ── Concurrency-scaling sweeps ────────────────────────────────────

    def store_sweep_metadata(self, *, sweep_id: str, metadata: Dict[str, Any]) -> None:
        """Record sweep-level metadata (one row per concurrency sweep).

        A sweep groups many per-cell runs; this row carries the shared context
        (hardware, density basis, NUMA policy, replay fixture) the
        ScalingAnalyzer and dashboard need to interpret the points.
        """
        conn = self._get_connection()
        known = {
            "created_at", "hardware_sku", "vcpu_basis", "vcpu_basis_kind",
            "numa_policy", "model", "replay_fixture", "benchmark",
        }
        extra = {k: v for k, v in metadata.items() if k not in known}
        try:
            conn.execute(
                """
                INSERT INTO sweeps (sweep_id, created_at, hardware_sku, vcpu_basis,
                    vcpu_basis_kind, numa_policy, model, replay_fixture, benchmark, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sweep_id) DO UPDATE SET
                    hardware_sku = COALESCE(excluded.hardware_sku, hardware_sku),
                    vcpu_basis = COALESCE(excluded.vcpu_basis, vcpu_basis),
                    vcpu_basis_kind = COALESCE(excluded.vcpu_basis_kind, vcpu_basis_kind),
                    numa_policy = COALESCE(excluded.numa_policy, numa_policy),
                    model = COALESCE(excluded.model, model),
                    replay_fixture = COALESCE(excluded.replay_fixture, replay_fixture),
                    benchmark = COALESCE(excluded.benchmark, benchmark),
                    metadata = excluded.metadata
                """,
                (
                    sweep_id,
                    metadata.get("created_at", 0),
                    metadata.get("hardware_sku"),
                    metadata.get("vcpu_basis"),
                    metadata.get("vcpu_basis_kind"),
                    metadata.get("numa_policy"),
                    metadata.get("model"),
                    metadata.get("replay_fixture"),
                    metadata.get("benchmark"),
                    json.dumps(extra),
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            log.warning("Failed to store sweep metadata for %s: %s", sweep_id, e)
            conn.rollback()

    def store_sweep_point(self, *, sweep_id: str, point: Dict[str, Any]) -> None:
        """Record one density cell's rollup (upsert on (sweep_id, run_id))."""
        conn = self._get_connection()
        known = {
            "run_id", "density", "concurrency", "replicate", "elapsed_s",
            "throughput_per_min", "completed_trials", "p95_trial_latency_s",
            "cpu_avg", "cpu_p95", "cpu_peak", "runqueue_max", "ctx_sw_per_s",
            "mem_avail_mb_min", "iowait_pct_avg",
        }
        extra = {k: v for k, v in point.items() if k not in known}
        try:
            conn.execute(
                """
                INSERT INTO sweep_points (sweep_id, run_id, density, concurrency,
                    replicate, elapsed_s, throughput_per_min, completed_trials,
                    p95_trial_latency_s, cpu_avg, cpu_p95, cpu_peak, runqueue_max,
                    ctx_sw_per_s, mem_avail_mb_min, iowait_pct_avg, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(sweep_id, run_id) DO UPDATE SET
                    density=excluded.density, concurrency=excluded.concurrency,
                    replicate=excluded.replicate, elapsed_s=excluded.elapsed_s,
                    throughput_per_min=excluded.throughput_per_min,
                    completed_trials=excluded.completed_trials,
                    p95_trial_latency_s=excluded.p95_trial_latency_s,
                    cpu_avg=excluded.cpu_avg, cpu_p95=excluded.cpu_p95,
                    cpu_peak=excluded.cpu_peak, runqueue_max=excluded.runqueue_max,
                    ctx_sw_per_s=excluded.ctx_sw_per_s,
                    mem_avail_mb_min=excluded.mem_avail_mb_min,
                    iowait_pct_avg=excluded.iowait_pct_avg, metadata=excluded.metadata
                """,
                (
                    sweep_id, point.get("run_id"), point.get("density"),
                    point.get("concurrency"), point.get("replicate"),
                    point.get("elapsed_s"), point.get("throughput_per_min"),
                    point.get("completed_trials"), point.get("p95_trial_latency_s"),
                    point.get("cpu_avg"), point.get("cpu_p95"), point.get("cpu_peak"),
                    point.get("runqueue_max"), point.get("ctx_sw_per_s"),
                    point.get("mem_avail_mb_min"), point.get("iowait_pct_avg"),
                    json.dumps(extra),
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            log.warning("Failed to store sweep point for %s: %s", sweep_id, e)
            conn.rollback()

    def query_sweeps(self) -> List[Dict[str, Any]]:
        """List all sweeps (most recent first)."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM sweeps ORDER BY created_at DESC"
            ).fetchall()
        except sqlite3.Error as e:
            log.warning("query_sweeps failed: %s", e)
            return []
        out = []
        for r in rows:
            d = dict(r)
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (ValueError, TypeError):
                    pass
            out.append(d)
        return out

    def query_sweep_points(self, sweep_id: str) -> List[Dict[str, Any]]:
        """Fetch all density-cell rollups for a sweep, ordered by density."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM sweep_points WHERE sweep_id = ? ORDER BY density, replicate",
                (sweep_id,),
            ).fetchall()
        except sqlite3.Error as e:
            log.warning("query_sweep_points failed for %s: %s", sweep_id, e)
            return []
        out = []
        for r in rows:
            d = dict(r)
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (ValueError, TypeError):
                    pass
            out.append(d)
        return out

    def query_verdicts(
        self, run_id: str, analyzer_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch analyzer verdicts for a run, optionally filtered by analyzer.

        Args:
            run_id: Unique identifier for the run.
            analyzer_name: Optional analyzer name filter (e.g., "cpu_bound", "cache").

        Returns:
            List of verdict dictionaries, each containing:
                - ``verdict_id`` (int)
                - ``task_id`` (str)
                - ``analyzer_name`` (str)
                - ``verdict`` (str)
                - ``confidence`` (float)
                - ``evidence`` (dict, parsed from JSON)
                - ``recommendations`` (list, parsed from JSON)
        """
        conn = self._get_connection()
        try:
            if analyzer_name:
                cursor = conn.execute(
                    """
                    SELECT verdict_id, task_id, analyzer_name, verdict, confidence, evidence, recommendations
                    FROM analyzer_verdicts
                    WHERE run_id = ? AND analyzer_name = ?
                    ORDER BY verdict_id
                    """,
                    (run_id, analyzer_name),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT verdict_id, task_id, analyzer_name, verdict, confidence, evidence, recommendations
                    FROM analyzer_verdicts
                    WHERE run_id = ?
                    ORDER BY verdict_id
                    """,
                    (run_id,),
                )

            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Parse JSON fields
                row_dict["evidence"] = json.loads(row_dict["evidence"])
                row_dict["recommendations"] = (
                    json.loads(row_dict["recommendations"])
                    if row_dict["recommendations"]
                    else []
                )
                results.append(row_dict)
            return results
        except sqlite3.Error as e:
            log.warning("Failed to query verdicts for %s: %s", run_id, e)
            return []
