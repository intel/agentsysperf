# AgentSysPerf Storage Unification Plan

**Goal:** One store every run pushes to and every consumer (dashboard, reports,
Prometheus) reads from. Benchmarks/runs/tasks with stable IDs, aggregation by
task across runs, a real deletion story, and the robustness bar to open-source.

Synthesized from three architecture reviews (DB / API / cloud-platform), 2026-06-11.

---

## 0. Root cause (proven against the code)

| Symptom | Evidence |
|---|---|
| No single store | `SQLiteResultStore.__init__` → `db_path = output_dir/"agentsysperf_results.db"`. Every runner passes its own `/tmp/...`; ~40 DBs exist. |
| Split-brain | `store_measurements()` is a **no-op** (`sqlite_store.py:271-293`). Measurements land only in per-dir `measurement_records.json`. |
| `tb2_db_to_records.py` | A script that round-trips DB→JSON because the DB lacks measurements — pure symptom. |
| Dashboard bypasses store | `demo_app.py` hardcodes ~20 `/tmp` paths + per-page JSON/CSV parsing; never imports `ResultStore`. |
| Prometheus starts empty | `monitoring/start.sh` re-pushes `docs/*/measurement_records.json` — a third, different read path. |
| FKs are decorative | SQLite enforces FKs only with `PRAGMA foreign_keys = ON` (off by default). `_get_connection()` never sets it → existing cascades silently no-op. |
| Hardcoded hardware | `runs.hardware_sku` DEFAULTs to literal `'Intel Xeon Platinum 8592+'` — mislabels every non-8592 result. |
| No delete, no versioning | No delete method anywhere; `CREATE TABLE IF NOT EXISTS` with no migration path. |

---

## 1. Target entity model

```
benchmark (slug) ─< run (ULID) ─< task_result ─< span ─< measurement
                       │                          (verdict keys run+task)
                       └─< analyzer_verdict
sweep ─< sweep_point ── run (1:1 per cell)
```

### Identity
| Entity | Today | Target |
|---|---|---|
| `benchmark_id` | none | **slug** matching adapter name (`terminal-bench`). First-class column on `runs`. |
| `run_id` | free-text | **ULID** (sortable-by-time; `ORDER BY run_id` = chronological). Keep TEXT PK; add `run_label` for the old human name. A handle, not a fingerprint. |
| `task_result` PK | global `task_id` (**bug**: collides across runs) | composite **`(run_id, task_id)`** + `logical_task_key = benchmark_id::task_id` (the cross-run aggregation key). |
| `span` PK | `(run_id, span_id)` | unchanged (already correct). |
| `measurement` | **missing** | surrogate `measurement_id`. |

---

## 2. The missing `measurements` table

Payload shape varies wildly (l1 ~9 keys, l3 ~8, EMON/perfspect 100+). **Promote
the ~6 hot columns the UI plots; keep the long tail in JSON.** `payload` is the
source of truth; promoted columns are a materialized index.

```sql
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    span_id  TEXT NOT NULL,
    task_id  TEXT,                 -- span_id.split("::")[-1], denormalized
    layer    TEXT NOT NULL,        -- l1 | l3 | emon | perfspect | ...
    node_id  TEXT,
    kind     TEXT,
    duration_us REAL, cpu_time_s REAL, cpu_pct_mean REAL,
    rss_kb_peak REAL, ipc REAL, cache_miss_pct REAL,   -- NULL when layer omits
    payload  JSON NOT NULL,
    schema_version TEXT DEFAULT '0.4',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX idx_meas_run_layer ON measurements(run_id, layer);
CREATE INDEX idx_meas_span      ON measurements(run_id, span_id);
CREATE INDEX idx_meas_run_task  ON measurements(run_id, task_id);
```

`store_measurements()` stops being a no-op: one INSERT per record, six keys
pulled from `payload`, `task_id` derived by splitting `span_id` on `::`.

---

## 3. Deletion

**Hard-delete via FK `ON DELETE CASCADE` for local; soft-delete + tombstone for published.**

Prerequisite (the silent bug): `_get_connection()` MUST run
`PRAGMA foreign_keys = ON`, else cascades no-op.

- **Delete run** → cascades task_results / spans / verdicts / measurements. If it's a
  sweep cell, also `DELETE FROM sweep_points WHERE run_id=?` (FK is on sweep_id → app-level).
- **Delete benchmark** → cascades all its runs (via `runs.benchmark_id ON DELETE CASCADE`).
- **Delete sweep** → cascades sweep_points; with `cascade_runs=True`, also deletes cell runs.
- **Published context** → `status='deleted'` + `deleted_at`/`reason`; keep `run_id`+`result_digest`
  tombstone so a leaderboard shows "withdrawn", not a dangling cite. Purge bulk artifacts immediately.

New `ResultStore` methods (additive → SemVer-safe):
```python
store_benchmark(*, benchmark_id, metadata)
list_benchmarks() -> list
list_runs(*, benchmark_id=None, limit=50) -> list      # newest-first
get_run(run_id) -> dict | None
query_measurements(run_id, *, layer=None, span_id=None, task_id=None) -> list
delete_run(run_id, *, prune_sweep_point=True) -> int
delete_benchmark(benchmark_id) -> int
delete_sweep(sweep_id, *, cascade_runs=True) -> int
latest_run(benchmark_id) -> str | None
store_measurements(*, run_id, records)                 # NO LONGER A NO-OP
```

---

## 4. One store, no daemon

**A single canonical SQLite file under `$AGENTSYSPERF_HOME` (default
`~/.agentsysperf`, XDG-aware), WAL mode.** No long-running service — writers are
minutes apart, WAL gives 1 writer + N readers. The `ResultStore` Protocol stays
the seam; a Postgres backend ships later as an entry-point plugin with no caller
changes.

```
$AGENTSYSPERF_HOME/ (default ~/.agentsysperf)
├── results.db            # the ONE store (WAL)
└── runs/<run_id>/        # bulk artifacts (EMON CSVs, plots) — addressed by run_id, never globbed
```

New module `src/home.py`: `agentsysperf_home()`, `default_db_path()`, `runs_dir()`.
`SQLiteResultStore` gains a `db_path=` arg (defaults to canonical); keeps `output_dir=` for back-compat.

**Stop defaulting output to `/tmp`** — results must survive a reboot.

---

## 5. Unified read path — refactor `demo_app.py`

Add `src/dashboard_data.py` as the dashboard's ONLY data source. UI/charts
unchanged; the `load_*` + path-discovery blocks collapse to named accessors:

```python
# before (repeated ~6×): glob /tmp, json.load, reassemble layers by span_id
# after:
import src.dashboard_data as data
benchmarks = data.benchmark_table("terminal-bench")    # was 30 lines
emon_csv   = data.emon_csv(slug)                        # was a 20-path priority loop
records    = data.phase_records(slug)
```

`dashboard_data` reads via `ResultStore` (`latest_run`, `query_measurements`,
`query_sweep_points`). ~20 hardcoded paths + 6 bespoke parsers → one module.

---

## 6. Script-sprawl collapse → CLI subcommands

| Command | Replaces |
|---|---|
| `agentsysperf run <bench> [--model --tasks --emon --phases --sweep --output]` | all `run_tb2_*`, `run_emon_*`, `run_scaling_*`, `run_*_with_storage` |
| `agentsysperf report <run_id> [--format pptx\|md\|html]` | all `generate_*_report.py` (calls existing `reporting/xeon_pptx.py`) |
| `agentsysperf db ls / rm <run_id> / show <run_id>` | ad-hoc DB pokes; **deletes `tb2_db_to_records.py`** |
| `agentsysperf serve [--dashboard\|--prometheus]` | `monitoring/start.sh` glob loop, `push_to_prometheus.py` (now scrapes the store) |
| `agentsysperf migrate` / `db import docs/*` | one-time backfill |

- **Delete:** `tb2_db_to_records.py` (split-brain artifact), `analyze_existing_deck.py`, `add_breakdown_slide.py`.
- **Move out of core:** pitch-deck builders (`build_pitch_deck.py`, `build_intel_deck.py`,
  `build_visual_pitch_deck.py`) → `decks/` or a sibling repo. Result-driven `report` stays.
- **Keep as examples:** `run_synthetic_l1_l3.py` (CLAUDE.md test), one canonical per-benchmark runner.

---

## 7. Migration / backfill (idempotent `agentsysperf migrate`)

1. Schema upgrade in place: add `benchmarks`, `measurements`; rebuild `task_results`/`spans`/
   `analyzer_verdicts` with cascade FKs; add `PRAGMA user_version` versioning.
2. Seed `benchmarks` from dir names / run metadata / `sweeps.benchmark`.
3. Merge per-dir DBs via `ATTACH` (resolve old global-`task_id` collisions through the rebuild).
4. **Backfill measurements from `docs/*/measurement_records.json`** — net-new into the DB
   (the data the no-op never wrote). `run_id` = the `span_id` prefix before `::`.
5. Provenance: write `migration_manifest.json` (source → run_id → row counts); verify counts;
   **never delete source files**; keep JSON as backup until verified.

---

## 8. OSS robustness bar (what makes a published number defensible)

- **Schema versioning:** `PRAGMA user_version` + ordered `storage/migrations/000N_*.sql`
  (SQLite-native "alembic-lite", zero new dep). Alembic only when Postgres lands.
- **Provenance columns** (promote from free-form JSON): `hardware_fingerprint`
  (from `platform/detect.py` — **kill the hardcoded SKU default**), `optimization_profile` +
  `verify_engaged()` proof, `code_git_sha` (+dirty flag), `agentsysperf_version`, `dataset_version`,
  `replay_fixture_hash`, `env_capture`, `result_digest` (sha256 of canonical result JSON), `status`.
- **Portable export/import:** `agentsysperf export <run_id> -o bundle.tar.zst`
  (manifest + sqlite/parquet rows + artifacts/); `import` validates digest + schema_version.
- **Backend seam:** every path goes through the Protocol; add `ArtifactStore` mini-protocol
  (`local`/`s3`) so blobs move to object storage with zero schema change.

### Tiered (complexity only when needed)
```
Tier 0 LOCAL  sqlite + local FS         pip install; zero-config (default)
Tier 1 TEAM   postgres                  AGENTSYSPERF_STORE_URL=postgresql://...
Tier 2 CLOUD  postgres + S3 artifacts   + Prometheus(live) + Langfuse(traces)
```
Relational = system of record; object store = bulk evidence (EMON CSVs, plots);
Prometheus = live-watch only, never the source of a published number.

### Top day-one embarrassments → minimal fix
1. Dashboard hardcoded `/tmp` paths → read via `list_runs()`/`get_run()`.
2. `hardware_sku` defaults to `'8592+'` → populate from `platform/detect.py`.
3. `store_measurements()` no-op → make it real (the headline data).
4. No schema versioning → `user_version` migrations.
5. No delete + `/tmp` default → `db rm`/`prune` + XDG default location.

---

## 8b. Storage engine — critical assessment (is SQLite future-proof?)

**Question raised:** with measurements, analyzers, benchmarks, and optimization
profiles (DFlash-style deltas) all churning, is a single relational SQLite store
robust enough — or is that why a colleague defaults to JSON files?

**Evidence** (deep research, 2026-06-11; 24/25 claims confirmed, unanimous
adversarial votes, primary vendor/spec sources):

- **Every mature system is a hybrid tiered by access pattern** — Langfuse =
  Postgres (OLTP metadata) + **ClickHouse columnar** (traces) + **S3** (raw blobs).
  *But the tiers exist to solve billions-of-rows / OLTP-OLAP contention* — scale a
  single-researcher local benchmark never hits. Langfuse was Postgres-only (v2)
  until cloud scale forced the split (v3, late-2024).
- **MLPerf/MLCommons uses NO database** — versioned git file trees + per-system
  JSON. Validates the JSON instinct *for the portable archival layer*.
- **Nobody evolves schemas with rigid relational columns.** Proven patterns:
  (A) schemaless JSON/Map columns (ClickHouse, OTel), (B) label-indexed series
  (Prometheus — new metric = new index entry), (C) versioned-schema +
  query-time transformation, never rewrite stored data (OTel). agentsysperf
  **already does (A)+(C)**: `SCHEMA_VERSION="0.3"`, additive-fields-default-None,
  `extra`/`metadata`/`evidence` JSON columns, "large blobs → sidecar".

**Verdict:** the risk was never the SQLite *engine* — it was the no-op
`store_measurements` (data stranded in ungoverned JSON) + no analytical query
path. Fix those and SQLite scales with the project.

1. **SQLite stays the system of record.** Already the correct zero-config choice;
   schema already implements promoted-hot-columns + JSON1-payload. Do NOT adopt
   ClickHouse/Postgres/S3 — overkill at this scale.
2. **Add DuckDB as a read-side analytical accelerator (both, not either).**
   DuckDB `ATTACH`es the SQLite file and reads Parquet natively — no duplication,
   no migration. SQLite for per-span transactional writes; DuckDB for wide
   cross-run/cross-sweep `GROUP BY` (p95 by profile across all runs) — the OLAP
   scans SQLite+`json_extract` is bad at.
3. **Parquet as export/cold-tier format** — columnar, self-describing,
   universally readable; the publishable bundle AND the DuckDB query target.
4. **Promote-when-hot lifecycle:** new metric → `extra`/JSON (no migration) →
   frequent filter → JSON1 expression index → stabilized → `ALTER TABLE ADD COLUMN`.
   Migrate only to *optimize* a metric that earned it, never to *add* one.

**Caveats / open:** the "DuckDB beats SQLite" claim is **medium confidence** —
synthesis, not a head-to-head on agentsysperf data. Highest-value validation:
point DuckDB at the real `docs/*/measurement_records.json`, run an aggregation
in place, measure the crossover (#runs/spans where DuckDB wins). Also open:
retention/compaction policy for months of accumulating runs (drop-old vs
roll-to-Parquet cold tier), and the exact hot/cold cutline (full prompt text,
raw PMU dumps → sidecar vs inline JSON).

---

## 9. Phased sequence (each independently verifiable)

1. **Store foundation** — `home.py`, WAL + `PRAGMA foreign_keys=ON`, `db_path=` arg,
   `measurements` table, real `store_measurements`, `benchmark`/provenance columns.
   *Verify:* `SELECT count(*) FROM measurements > 0`; round-trip equals input.
2. **Read API + runner write path** — new query methods; `RunContext(result_store=)` persists
   run+tasks+measurements+spans+verdicts atomically on stop.
   *Verify:* a fresh `agentsysperf run` is fully readable by the new methods.
3. **Dashboard DAL** — add `dashboard_data.py`; repoint every `load_*`; `db import docs/*`.
   *Verify:* every page renders identically with `/tmp` paths removed.
4. **CLI collapse** — `run`/`report`/`db`/`serve`; delete folded scripts; move deck builders out;
   `serve --prometheus` scrapes the store. *Verify:* Grafana identical without the JSON re-push loop.
5. **Retire JSON emission + dashboard fallback** once 3–4 verified.

**Hard constraints throughout:** never touch `harness/results/` (colleague's baselines);
keep `docs/*` demo data live via additive import; keep `output_dir=` constructor for back-compat.

### Files to change
- *new:* `src/home.py`, `src/dashboard_data.py`, `src/storage/migrations/`
- `src/storage/sqlite_store.py` (WAL, FK pragma, `db_path`, real `store_measurements`, deletes, queries)
- `src/storage/schema.sql` (`benchmarks`, `measurements`, provenance cols, cascade FKs)
- `src/protocols.py` (extend `ResultStore`)
- `src/runner.py` (persist-on-stop)
- `src/cli.py` (`run`/`report`/`db`/`serve`/`migrate`)
- `demo_app.py` (`load_*` → `dashboard_data`)
- `monitoring/start.sh`, `scripts/push_to_prometheus.py` (scrape-from-store)
